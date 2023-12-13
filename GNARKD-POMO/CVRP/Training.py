import os
import sys
import time

import torch
import datetime
from tensorboardX import SummaryWriter
from torch import nn
import os
from ARModel import CVRPModel as AR_Model
from NARmodel import CVRPModel as NAR_Model


# This is for training GNARKD-POMO on CVRP


def get_random_problems(batch_size, problem_size):
    depot_xy = torch.rand(size=(batch_size, 1, 2), device=model_args.device)
    # shape: (batch, 1, 2)

    node_xy = torch.rand(size=(batch_size, problem_size, 2), device=model_args.device)
    # shape: (batch, problem, 2)

    if problem_size == 20:
        demand_scaler = 30
    elif problem_size == 50:
        demand_scaler = 40
    elif problem_size == 100:
        demand_scaler = 50
    else:
        raise NotImplementedError

    node_demand = torch.randint(1, 10, size=(batch_size, problem_size), device=model_args.device) / float(demand_scaler)
    return depot_xy, node_xy, node_demand


def pomo_beam_search(score_stu, node_demand, round_error_epsilon=0.00001):
    batch_size, problem_size = node_demand.size(0), node_demand.size(1)
    pomo_size = problem_size
    BATCH_IDX = torch.arange(batch_size, device=score_stu.device)[:, None].expand(batch_size, pomo_size)
    POMO_IDX = torch.arange(pomo_size, device=score_stu.device)[None, :].expand(batch_size, pomo_size)
    at_the_depot = torch.ones(size=(batch_size, pomo_size), dtype=torch.bool, device=score_stu.device)
    load = torch.ones(size=(batch_size, pomo_size), device=score_stu.device)
    visited_ninf_flag = torch.zeros(size=(batch_size, pomo_size, problem_size + 1), device=score_stu.device).bool()
    ninf_mask = torch.zeros(size=(batch_size, pomo_size, problem_size + 1), device=score_stu.device).bool()
    finished = torch.zeros(size=(batch_size, pomo_size), dtype=torch.bool, device=score_stu.device)
    depot_node_demand = torch.cat((torch.zeros(size=(batch_size, 1), device=score_stu.device), node_demand), dim=1)
    demand_list = depot_node_demand[:, None, :].expand(batch_size, pomo_size, -1)
    selected_node_list = torch.zeros((batch_size, pomo_size, 0), dtype=torch.long, device=depot_xy.device)
    done = False
    selected_count = 0
    while not done:
        if selected_count == 0:
            selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long, device=depot_xy.device)
        elif selected_count == 1:
            selected = torch.arange(start=1, end=pomo_size + 1, device=depot_xy.device)[None, :].expand(batch_size,
                                                                                                        pomo_size)
        else:
            idx_expand = selected.unsqueeze(2).expand(-1, -1, problem_size + 1)
            weight = score_stu.gather(dim=1, index=idx_expand)
            prob_next_node = torch.softmax(weight.masked_fill(ninf_mask, float('-inf')), dim=-1)
            selected = torch.argmax(prob_next_node, dim=2)
        selected_count += 1
        selected_node_list = torch.cat((selected_node_list, selected[:, :, None]), dim=2)
        at_the_depot = (selected == 0)
        gathering_index = selected[:, :, None]
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        load -= selected_demand
        load[at_the_depot] = 1
        visited_ninf_flag[BATCH_IDX, POMO_IDX, selected] = True
        visited_ninf_flag[:, :, 0][~at_the_depot] = False
        ninf_mask = visited_ninf_flag.clone()
        demand_too_large = load[:, :, None] + round_error_epsilon < demand_list
        ninf_mask[demand_too_large] = True
        finished = finished + (visited_ninf_flag == True).all(dim=2)
        ninf_mask[:, :, 0][finished] = False
        done = finished.all()
    return selected_node_list


def compute_tour_length_all_b(depot_node_xy, selected_node_list):
    pomo_size = selected_node_list.size(1)
    gathering_index = selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
    # shape: (batch, pomo, selected_list_length, 2)
    all_xy = depot_node_xy[:, None, :, :].expand(-1, pomo_size, -1, -1)
    # shape: (batch, pomo, problem+1, 2)

    ordered_seq = all_xy.gather(dim=2, index=gathering_index)
    # shape: (batch, pomo, selected_list_length, 2)

    rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
    segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
    # shape: (batch, pomo, selected_list_length)

    travel_distances = segment_lengths.sum(2)
    return travel_distances


class DotDict(dict):
    def __init__(self, **kwds):
        super().__init__()
        self.update(kwds)


model_args = DotDict()
model_args.embedding_dim = 128
model_args.sqrt_embedding_dim = 128 ** (1 / 2)
model_args.encoder_layer_num = 6
model_args.qkv_dim = 16
model_args.head_num = 8
model_args.logit_clipping = 10
model_args.ff_hidden_dim = 512
model_args.eval_type = 'argmax'
model_args.device = torch.device('cuda')

args = DotDict()
args.batch_size = 100
args.problem_size = 100
args.pomo_size = args.problem_size  # POMO
args.aug_factor = 1
args.lr = 1e-5
args.nb_epoch = 500 if args.problem_size == 50 else 1000
args.nb_batch_per_epoch = 500

teacher_model = AR_Model(model_args)

# Note: The pre-training parameters of POMO can be found at the address mentioned in its paper,
checkpoint = torch.load('pretrained/checkpoint-30500.pt')
teacher_model.load_state_dict(checkpoint['model_state_dict'])
teacher_model.eval()
teacher_model.to(model_args.device)

student_model = NAR_Model(model_args)
student_model.to(model_args.device)

loss_func = torch.nn.KLDivLoss(reduction='batchmean')
student_model.encoder.load_state_dict(teacher_model.encoder.state_dict())

ZERO_TO_BATCH = torch.arange(args.batch_size, device=model_args.device)
BATCH_IDX = torch.arange(args.batch_size, device=model_args.device)[:, None].expand(args.batch_size, args.pomo_size)
POMO_IDX = torch.arange(args.pomo_size, device=model_args.device)[None, :].expand(args.batch_size, args.pomo_size)
optimizer = torch.optim.Adam(student_model.parameters(), lr=args.lr)
min_len = torch.inf


# val dataset
val_depot_xy, val_node_xy, val_node_demand = get_random_problems(10000, args.problem_size)
args.nb_batch_eval = (10000 + args.batch_size - 1) // args.batch_size


for epoch in range(args.nb_epochs):
    print('epoch:' + str(epoch))
    student_model.train()
    begin = time.time()
    for step in range(args.nb_batch_per_epoch):
        depot_xy, node_xy, node_demand = get_random_problems(args.batch_size, args.problem_size)
        with torch.no_grad():
            tours, prob_next_node_all = teacher_model(depot_xy, node_xy, node_demand)
        score_stu = student_model(depot_xy, node_xy, node_demand)
        at_the_depot = torch.ones(args.batch_size, dtype=torch.bool, device=model_args.device)
        load = torch.ones(args.batch_size, device=model_args.device)
        visited_ninf_flag = torch.zeros(size=(args.batch_size, args.problem_size + 1), device=model_args.device).bool()
        ninf_mask = torch.zeros(size=(args.batch_size, args.problem_size + 1), device=model_args.device).bool()
        finished = torch.zeros(args.batch_size, dtype=torch.bool, device=model_args.device)
        depot_node_demand = torch.cat((torch.zeros(size=(args.batch_size, 1), device=model_args.device), node_demand),
                                      dim=1)
        demand_list = depot_node_demand
        round_error_epsilon = 0.00001
        prob_next_node_all_NAR = []
        for node_index in range(tours.size(-1)):
            if node_index > 0:
                weight = score_stu[ZERO_TO_BATCH, selected]
                weight = torch.log_softmax(weight.masked_fill(ninf_mask, float('-inf')), dim=-1)
                prob_next_node_all_NAR.append(weight)
            selected = tours[ZERO_TO_BATCH, node_index]
            at_the_depot = (selected == 0)
            gathering_index = selected[:, None]
            selected_demand = demand_list.gather(dim=1, index=gathering_index).squeeze(dim=1)
            load -= selected_demand
            load[at_the_depot] = 1
            visited_ninf_flag[ZERO_TO_BATCH, selected] = True
            visited_ninf_flag[:, 0][~at_the_depot] = False
            ninf_mask = visited_ninf_flag.clone()
            demand_too_large = load[:, None] + round_error_epsilon < demand_list
            ninf_mask[demand_too_large] = True
            finished = finished + (visited_ninf_flag == True).all(dim=1)
            ninf_mask[:, 0][finished] = False
        prob_next_node_all_NAR = torch.stack(prob_next_node_all_NAR, dim=-2)
        loss = loss_func(prob_next_node_all_NAR, prob_next_node_all)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(time.time() - begin)
    student_model.eval()
    tours_length = torch.zeros(size=(0, 1), device=model_args.device)
    for step in range(args.nb_batch_eval):
        depot_xy = val_depot_xy[step * args.batch_size:(step + 1) * args.batch_size, :, :]
        node_xy = val_node_xy[step * args.batch_size:(step + 1) * args.batch_size, :, :]
        node_demand = val_node_demand[step * args.batch_size:(step + 1) * args.batch_size, :]
        with torch.no_grad():
            score_stu = student_model(depot_xy, node_xy, node_demand)
            selected_node_list = pomo_beam_search(score_stu, node_demand)
            L_val_pomo = compute_tour_length_all_b(torch.cat((depot_xy, node_xy), dim=1), selected_node_list)
            L_val, _ = L_val_pomo.min(dim=1)
            tours_length = torch.cat((tours_length, L_val[:, None]), dim=0)
    mean_tour_length = torch.mean(tours_length, dim=0).item()
    print('mean_tour_length:' + str(mean_tour_length))

    checkpoint_dir = os.path.join('checkpoint')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if mean_tour_length < min_len:
        min_len = mean_tour_length
        torch.save({
            'epoch': epoch,
            'CVRP_length': mean_tour_length,
            'parameter': student_model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, '{}.pkl'.format(checkpoint_dir + '/checkpoint' + '-n{}'.format(args.problem_size)))

import os
import sys
import time

import torch
import datetime
from tensorboardX import SummaryWriter
from torch import nn
import os
from ARModel import TSPModel as AR_Model
from NARmodel import TSPModel as NAR_Model


# This is for training GNARKD-POMO on TSP

def compute_tour_length_all_b(x, tour):
    bsz, nb_node, b_width = x.shape[0], x.shape[1], tour.shape[1]
    index = tour.unsqueeze(3).expand(bsz, -1, nb_node, 2)
    seq_expand = x[:, None, :, :].expand(bsz, b_width, nb_node, 2)
    order_seq = seq_expand.gather(dim=2, index=index)
    rolled_seq = order_seq.roll(dims=2, shifts=-1)
    segment_length = ((order_seq - rolled_seq) ** 2).sum(3).sqrt()
    travel_distances = segment_length.sum(2)
    return travel_distances


def pomo_beam_search(score_stu):
    batch_size, problem_size = score_stu.size(0), score_stu.size(1),
    pomo_size = problem_size
    selected_node_list = torch.zeros((batch_size, pomo_size, 0), dtype=torch.int8, device=score_stu.device)
    idx = torch.arange(pomo_size, device=score_stu.device)[None, :].expand(batch_size, pomo_size)
    BATCH_IDX = torch.arange(batch_size, device=score_stu.device)[:, None].expand(batch_size, pomo_size)
    POMO_IDX = torch.arange(pomo_size, device=score_stu.device)[None, :].expand(batch_size, pomo_size)
    ninf_mask = torch.zeros((batch_size, pomo_size, problem_size), device=score_stu.device).bool()
    selected_node_list = torch.cat((selected_node_list, idx[:, :, None]), dim=2)
    ninf_mask[BATCH_IDX, POMO_IDX, idx] = True
    for _ in range(problem_size - 1):
        idx_expand = idx.unsqueeze(2).expand(-1, -1, problem_size)
        weight = score_stu.gather(dim=1, index=idx_expand)
        prob_next_node = torch.softmax(weight.masked_fill(ninf_mask, float('-inf')), dim=-1)
        idx = torch.argmax(prob_next_node, dim=2)
        ninf_mask[BATCH_IDX, POMO_IDX, idx] = True
        selected_node_list = torch.cat((selected_node_list, idx[:, :, None]), dim=2)
    return selected_node_list


class DotDict(dict):
    def __init__(self, **kwds):
        super().__init__()
        self.update(kwds)


# Following POMO
model_args = DotDict()
model_args.embedding_dim = 128
model_args.sqrt_embedding_dim = model_args.embedding_dim ** 0.5
model_args.encoder_layer_num = 6
model_args.qkv_dim = 16
model_args.head_num = 8
model_args.logit_clipping = 10
model_args.ff_hidden_dim = 512
model_args.eval_type = 'argmax'
model_args.device = torch.device('cuda')

args = DotDict()
args.batch_size = 100
args.problem_size = 50  
args.pomo_size = args.problem_size  # POMO
args.aug_factor = 1
args.lr = 1e-5
args.nb_epochs = 500 if args.problem_size == 50 else 1000
args.nb_batch_per_epoch = 500

teacher_model = AR_Model(model_args)

# Note: The pre-training parameters of POMO can be found at the address mentioned in its paper,
checkpoint = torch.load('pretrained/checkpoint-1000.pt') if args.problem_size==50 else torch.load('pretrained/checkpoint-3100.pt')
teacher_model.load_state_dict(checkpoint['model_state_dict'])
teacher_model.eval()
teacher_model.to(model_args.device)
student_model = NAR_Model(model_args)
student_model.to(model_args.device)

student_model.encoder.load_state_dict(teacher_model.encoder.state_dict())
optimizer = torch.optim.Adam(student_model.parameters(), lr=args.lr)



# val dataset
x_10k = torch.rand(size=(10000, args.problem_size, 2), device=model_args.device)

args.nb_batch_eval = (10000 + args.batch_size - 1) // args.batch_size
loss_func = torch.nn.KLDivLoss(reduction='batchmean')
zero_to_bsz = torch.arange(args.batch_size, device=model_args.device)
min_len = torch.inf


BATCH_IDX = torch.arange(args.batch_size, device=model_args.device)[:, None].expand(args.batch_size, args.pomo_size)
POMO_IDX = torch.arange(args.pomo_size, device=model_args.device)[None, :].expand(args.batch_size, args.pomo_size)
for epoch in range(args.nb_epochs):
    print('epoch:' + str(epoch))
    student_model.train()
    begin_time = time.time()
    for step in range(args.nb_batch_per_epoch):
        problems = torch.rand(size=(args.batch_size, args.problem_size, 2), device=model_args.device)
        with torch.no_grad():
            tours, prob_next_node_all = teacher_model(problems)
        score_stu = student_model(problems)
        ninf_mask = torch.zeros((args.batch_size, args.pomo_size, args.problem_size), device=model_args.device).bool()
        idx = torch.arange(args.pomo_size, device=model_args.device)[None, :].expand(args.batch_size, args.pomo_size)
        ninf_mask[BATCH_IDX, POMO_IDX, idx] = True
        prob_next_node_all_NAR = []
        for node_index in range(1, args.problem_size):
            idx_expand = idx.unsqueeze(2).expand(-1, -1, args.problem_size)
            weight = torch.log_softmax(score_stu.gather(dim=1, index=idx_expand).masked_fill(ninf_mask, float('-inf')),
                                       dim=-1)
            prob_next_node_all_NAR.append(weight)
            ninf_mask = ninf_mask.clone()
            idx = tours[BATCH_IDX, POMO_IDX, node_index]
            ninf_mask[BATCH_IDX, POMO_IDX, idx] = True
        prob_next_node_all_NAR = torch.stack(prob_next_node_all_NAR, dim=2)
        prob_next_node_all = prob_next_node_all[:, :, 1:, :]
        loss = loss_func(prob_next_node_all_NAR, prob_next_node_all)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(time.time() - begin_time)

    # val
    student_model.eval()
    tours_length = torch.zeros(size=(0, 1), device=x_10k.device)
    for step in range(args.nb_batch_eval):
        problems = x_10k[step * args.batch_size:(step + 1) * args.batch_size, :, :]
        with torch.no_grad():
            score_stu = student_model(problems)
            selected_node_list = pomo_beam_search(score_stu)
            L_val_pomo = compute_tour_length_all_b(problems, selected_node_list)
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
            'TSP_length': mean_tour_length,
            'parameter': student_model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, '{}.pkl'.format(checkpoint_dir + '/checkpoint' + '-n{}'.format(args.problem_size)))

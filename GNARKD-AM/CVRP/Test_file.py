import os
import torch
from NAR_model import AttentionModel as NAR_model
import time
import sys
import os


# This file is used to test the performance of GNARKD-AM for CVRP

def greedy_search(score_stu, node_demand, round_error_epsilon=0.00001):
    batch_size, problem_size = node_demand.size()
    ZERO_TO_BATCH = torch.arange(batch_size, device=score_stu.device)
    at_the_depot = torch.ones(batch_size, dtype=torch.bool, device=score_stu.device)
    load = torch.ones(batch_size, device=score_stu.device)
    visited_ninf_flag = torch.zeros(size=(batch_size, problem_size + 1), device=score_stu.device).bool()
    ninf_mask = torch.zeros(size=(batch_size, problem_size + 1), device=score_stu.device).bool()
    finished = torch.zeros(batch_size, dtype=torch.bool, device=score_stu.device)
    depot_node_demand = torch.cat((torch.zeros(size=(batch_size, 1), device=score_stu.device), node_demand), dim=1)
    demand_list = depot_node_demand
    round_error_epsilon = 0.00001
    prob_next_node_all_NAR = []
    done = False
    selected_count = 0
    selected_node_list = torch.zeros((batch_size, 0), dtype=torch.long, device=score_stu.device)
    while not done:
        if selected_count == 0:
            selected = torch.zeros(size=(batch_size,), dtype=torch.long, device=score_stu.device)
        else:
            weight = score_stu[ZERO_TO_BATCH, selected]
            weight = torch.softmax(weight.masked_fill(ninf_mask, float('-inf')), dim=-1)
            selected = torch.argmax(weight, dim=1)
        selected_count += 1
        selected_node_list = torch.cat((selected_node_list, selected[:, None]), dim=1)

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
        done = finished.all()
    return selected_node_list


def beam_search(score_stu, node_demand, round_error_epsilon=0.00001, beam_width=1000):
    batch_size, problem_size = node_demand.size(0), node_demand.size(1)
    zero_to_bsz = torch.arange(batch_size, device=score_stu.device)
    depot_node_demand = torch.cat((torch.zeros(size=(batch_size, 1), device=score_stu.device), node_demand),
                                  dim=1)
    done = False
    selected_count = 0
    while not done:
        if selected_count == 0:
            selected = torch.zeros((batch_size, problem_size), device=score_stu.device, dtype=torch.long)
            zero_to_bsz_idx = zero_to_bsz[:, None].expand(batch_size, problem_size)
            zero_to_B_t0 = torch.arange(problem_size, device=score_stu.device)
            zero_to_B_t0_idx = zero_to_B_t0[None, :].expand(batch_size, problem_size)
            load_t0 = torch.ones(size=(batch_size, problem_size), device=score_stu.device)
            visited_ninf_flag_t0 = torch.zeros(size=(batch_size, problem_size, problem_size + 1),
                                               device=score_stu.device).bool()
            finished_t0 = torch.zeros(size=(batch_size, problem_size), dtype=torch.bool, device=score_stu.device)
            demand_list_t0 = depot_node_demand[:, None, :].expand(batch_size, problem_size, -1)
            at_the_depot_t0 = (selected == 0)
            visited_ninf_flag_t0[zero_to_bsz_idx, zero_to_B_t0_idx, selected] = True
            visited_ninf_flag_t0[:, :, 0][~at_the_depot_t0] = False

            mask_visited_nodes = torch.zeros(batch_size, problem_size + 1, device=score_stu.device).bool()
            selected = torch.zeros(batch_size, device=score_stu.device, dtype=torch.long)
            mask_visited_nodes[zero_to_bsz, selected] = True
            weight = score_stu[zero_to_bsz, selected]
            score_t = torch.log_softmax(weight.masked_fill(mask_visited_nodes, float('-inf')), dim=-1)
            sum_scores = score_t
            top_val, top_idx = torch.topk(sum_scores, problem_size, dim=1)
            sum_scores = top_val
            tours = torch.zeros(batch_size, problem_size, problem_size + 1, device=score_stu.device, dtype=torch.long)
            selected = top_idx
            tours[:, :, 1] = top_idx

            selected_count += 1
            at_the_depot_t0 = (selected == 0)
            gathering_index_t0 = selected[:, :, None]
            selected_demand = demand_list_t0.gather(dim=2, index=gathering_index_t0).squeeze(dim=2)
            load_t0 -= selected_demand
            load_t0[at_the_depot_t0] = 1
            visited_ninf_flag_t0[zero_to_bsz_idx, zero_to_B_t0_idx, selected] = True
            visited_ninf_flag_t0[:, :, 0][~at_the_depot_t0] = False
            ninf_mask_t0 = visited_ninf_flag_t0.clone()
            demand_too_large = load_t0[:, :, None] + round_error_epsilon < demand_list_t0
            ninf_mask_t0[demand_too_large] = True
            finished_t0 = finished_t0 + (visited_ninf_flag_t0 == True).all(dim=2)
            ninf_mask_t0[:, :, 0][finished_t0] = False

        elif selected_count == 1:
            top_idx_expand = top_idx.unsqueeze(2).expand(-1, -1, problem_size + 1)
            weight = score_stu.gather(dim=1, index=top_idx_expand)
            score_t = torch.log_softmax(weight.masked_fill(ninf_mask_t0, float('-inf')), dim=-1)

            sum_scores = score_t + sum_scores.unsqueeze(2)
            sum_scores_flatten = sum_scores.view(batch_size, -1)

            top_val, top_idx = torch.topk(sum_scores_flatten, beam_width, dim=1)

            idx_top_beams = torch.div(top_idx, problem_size + 1, rounding_mode='trunc')
            idx_in_beams = top_idx % (problem_size + 1)
            sum_scores = top_val

            selected_count += 1
            zero_to_bsz_expand = zero_to_bsz[:, None].expand(batch_size, beam_width)
            zero_to_B = torch.arange(beam_width, device=score_stu.device)
            zero_to_B_expand = zero_to_B[None, :].expand(batch_size, beam_width)

            at_the_depot = torch.ones(size=(batch_size, beam_width), dtype=torch.bool, device=score_stu.device)
            load = torch.ones(size=(batch_size, beam_width), device=score_stu.device)
            visited_ninf_flag = torch.zeros(size=(batch_size, beam_width, problem_size + 1),
                                            device=score_stu.device).bool()
            ninf_mask = torch.zeros(size=(batch_size, beam_width, problem_size + 1), device=score_stu.device).bool()
            finished = torch.zeros(size=(batch_size, beam_width), dtype=torch.bool, device=score_stu.device)
            demand_list = depot_node_demand[:, None, :].expand(batch_size, beam_width, -1)

            first_node = torch.zeros(size=(batch_size, beam_width), dtype=torch.long, device=score_stu.device)
            tours_all = torch.zeros(batch_size, beam_width, 0, device=score_stu.device, dtype=torch.long)
            tours_all = torch.cat((tours_all, first_node[:, :, None]), dim=2)
            at_the_depot = (first_node == 0)
            visited_ninf_flag[zero_to_bsz_expand, zero_to_B_expand, first_node] = True
            visited_ninf_flag[:, :, 0][~at_the_depot] = False
            ninf_mask = visited_ninf_flag.clone()

            selected = selected[zero_to_bsz_expand, idx_top_beams]
            tours_all = torch.cat((tours_all, selected[:, :, None]), dim=2)
            at_the_depot = (selected == 0)
            gathering_index = selected[:, :, None]
            selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
            load -= selected_demand
            load[at_the_depot] = 1
            visited_ninf_flag[zero_to_bsz_expand, zero_to_B_expand, selected] = True
            visited_ninf_flag[:, :, 0][~at_the_depot] = False
            ninf_mask = visited_ninf_flag.clone()
            finished = finished + (visited_ninf_flag == True).all(dim=2)
            ninf_mask[:, :, 0][finished] = False

            selected = idx_in_beams
            tours_all = torch.cat((tours_all, selected[:, :, None]), dim=2)
            at_the_depot = (selected == 0)
            gathering_index = selected[:, :, None]
            selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
            load -= selected_demand
            load[at_the_depot] = 1
            visited_ninf_flag[zero_to_bsz_expand, zero_to_B_expand, selected] = True
            visited_ninf_flag[:, :, 0][~at_the_depot] = False
            ninf_mask = visited_ninf_flag.clone()
            demand_too_large = load[:, :, None] + round_error_epsilon < demand_list
            ninf_mask[demand_too_large] = True
            finished = finished + (visited_ninf_flag == True).all(dim=2)
            ninf_mask[:, :, 0][finished] = False
            done = finished.all()
        else:
            idx_in_beams_expand = selected.unsqueeze(2).expand(-1, -1, problem_size + 1)
            weight = score_stu.gather(dim=1, index=idx_in_beams_expand)
            score_t = torch.log_softmax(weight.masked_fill(ninf_mask, float('-inf')), dim=-1)
            sum_scores = score_t + sum_scores.unsqueeze(2)
            sum_scores_flatten = sum_scores.view(batch_size, -1)
            top_val, top_idx = torch.topk(sum_scores_flatten, beam_width, dim=1)
            idx_top_beams = torch.div(top_idx, problem_size + 1, rounding_mode='trunc')
            idx_in_beams = top_idx % (problem_size + 1)
            sum_scores = top_val

            selected = idx_in_beams
            selected_count += 1
            tours = tours_all.clone()
            tours_all[zero_to_bsz_expand, zero_to_B_expand] = tours[zero_to_bsz_expand, idx_top_beams]
            tours_all = torch.cat((tours_all, selected[:, :, None]), dim=2)
            at_the_depot = (selected == 0)
            load_tmp = load.clone()
            gathering_index = selected[:, :, None]
            selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
            load = load_tmp[zero_to_bsz_expand, idx_top_beams] - selected_demand
            load[at_the_depot] = 1
            visited_ninf_flag_tmp = visited_ninf_flag.clone()
            visited_ninf_flag[zero_to_bsz_expand, zero_to_B_expand] = visited_ninf_flag_tmp[
                zero_to_bsz_expand, idx_top_beams]
            visited_ninf_flag[zero_to_bsz_expand, zero_to_B_expand, selected] = True
            visited_ninf_flag[:, :, 0][~at_the_depot] = False
            ninf_mask = visited_ninf_flag.clone()
            demand_too_large = load[:, :, None] + round_error_epsilon < demand_list
            ninf_mask[demand_too_large] = True
            finished_tmp = finished.clone()
            finished[zero_to_bsz_expand, zero_to_B_expand] = finished_tmp[zero_to_bsz_expand, idx_top_beams]

            finished = finished + (visited_ninf_flag == True).all(dim=2)
            ninf_mask[:, :, 0][finished] = False
            done = finished.all()

    return tours_all


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


def compute_tour_length(depot_node_xy, selected_node_list):
    index = selected_node_list[:, :, None].expand(-1, -1, 2)
    order_seq = depot_node_xy.gather(dim=1, index=index)
    rolled_seq = order_seq.roll(dims=1, shifts=-1)
    segment_length = ((order_seq - rolled_seq) ** 2).sum(2).sqrt()
    L = segment_length.sum(1)
    return L


class DotDict(dict):
    def __init__(self, **kwds):
        super().__init__()
        self.update(kwds)


args = DotDict()
args.problem_size = 50
args.offset = 0
args.width = 0
args.batch_size = 400
args.lr = 1e-5
args.nb_batch_eval = (10000 + args.batch_size - 1) // args.batch_size

device = torch.device("cuda")

student_model = NAR_model(
    embedding_dim=128,
    hidden_dim=128,
    n_encode_layers=3,
    tanh_clipping=10.0,
    normalization="batch",
)

student_model.to(device)
student_model.eval()
# for test CVRP-50 and CVRP-100
f = open('checkpoint/checkpoint-n50.pkl', 'rb') if args.problem_size == 50 else open('checkpoint/checkpoint-n100.pkl',
                                                                                     'rb')

# for test different temp
# f = open('different_temp/temp_10.pkl','rb')


record = torch.load(f)
student_model.load_state_dict(record['parameter'])

# test dataset
val_problems = problems = torch.load('../../Test_data/CVRP/vrp' + str(args.problem_size) + '_test_seed1234.pt')
args.nb_batch_eval = (10000 + args.batch_size - 1) // args.batch_size
val_depot_xy = problems['depot_xy'].to(device)
val_node_xy = problems['node_xy'].to(device)
val_node_demand = problems['node_demand'].to(device)

tours_length = torch.zeros(size=(0, 1), device=device)
greedy = False
B = 2000

with torch.no_grad():
    begin_time = time.time()
    for step in range(args.nb_batch_eval):
        print(step)
        depot_xy = val_depot_xy[step * args.batch_size:(step + 1) * args.batch_size, :, :]
        node_xy = val_node_xy[step * args.batch_size:(step + 1) * args.batch_size, :, :]
        node_demand = val_node_demand[step * args.batch_size:(step + 1) * args.batch_size, :]
        score_stu = student_model((depot_xy, node_xy, node_demand))
        if greedy:
            selected_node_list = greedy_search(score_stu, node_demand)
            L_val = compute_tour_length(torch.cat((depot_xy, node_xy), dim=1), selected_node_list)
            tours_length = torch.cat((tours_length, L_val[:, None]), dim=0)
        else:
            selected_node_list = beam_search(score_stu, node_demand, beam_width=B)
            L_val = compute_tour_length_all_b(torch.cat((depot_xy, node_xy), dim=1), selected_node_list)
            L_val, _ = L_val.min(dim=1)
            tours_length = torch.cat((tours_length, L_val[:, None]), dim=0)
    print(time.time() - begin_time)
    mean_tour_length = torch.mean(tours_length, dim=0).item()
    print(mean_tour_length)

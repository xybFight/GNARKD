import time

from NAR_model import AttentionModel as NAR_model
import json
import os
import torch
import argparse
from torch.utils.data import DataLoader

# This file is used to test the inference time of GNARKD-AM for TSP


def compute_tour_length_single(x, tour):
    bsz, nb_node = x.shape[0], x.shape[1]
    index = tour.unsqueeze(2).expand(bsz, nb_node, 2)
    order_seq = x.gather(dim=1, index=index)
    rolled_seq = order_seq.roll(dims=1, shifts=-1)
    segment_length = ((order_seq - rolled_seq) ** 2).sum(2).sqrt()
    travel_distances = segment_length.sum(1)
    return travel_distances


def compute_tour_length_all_b(x, tour):
    bsz, nb_node, b_width = x.shape[0], x.shape[1], tour.shape[1]
    index = tour.unsqueeze(3).expand(bsz, -1, nb_node, 2)
    seq_expand = x[:, None, :, :].expand(bsz, b_width, nb_node, 2)
    order_seq = seq_expand.gather(dim=2, index=index)
    rolled_seq = order_seq.roll(dims=2, shifts=-1)
    segment_length = ((order_seq - rolled_seq) ** 2).sum(3).sqrt()
    travel_distances = segment_length.sum(2)
    return travel_distances


def greedy_search(attn_weight_all):
    bsz, nb_nodes = attn_weight_all.shape[0], attn_weight_all.shape[1] - 1
    zero_to_bsz = torch.arange(bsz, device=attn_weight_all.device)
    idx_start_place_holder = torch.tensor([nb_nodes], device=attn_weight_all.device).long().repeat(bsz)
    mask_visited_nodes = torch.zeros(bsz, nb_nodes, device=attn_weight_all.device).bool()
    idx = idx_start_place_holder
    tour_nar = []
    for node_index in range(nb_nodes):
        attn_weight = torch.softmax(attn_weight_all[zero_to_bsz, idx].masked_fill(mask_visited_nodes, float('-1e10')),
                                    dim=-1)
        idx = torch.argmax(attn_weight, dim=-1)
        tour_nar.append(idx)
        mask_visited_nodes = mask_visited_nodes.clone()
        mask_visited_nodes[zero_to_bsz, idx] = True
    tour_nar = torch.stack(tour_nar, dim=1)
    return tour_nar


def beam_search(attn_weight_all, beam_width=1000):
    """
    attn_weight_all:(BSZ, node+1, node)
    """
    bsz, nb_nodes = attn_weight_all.shape[0], attn_weight_all.shape[1] - 1
    zero_to_bsz = torch.arange(bsz, device=attn_weight_all.device)
    zero_to_B = torch.arange(beam_width, device=attn_weight_all.device)
    mask_visited_nodes = torch.zeros(bsz, nb_nodes, device=attn_weight_all.device).bool()
    for t in range(nb_nodes):
        if t == 0:
            idx_start_place_holder = torch.tensor([nb_nodes], device=attn_weight_all.device).long().repeat(bsz)
            B_t0 = min(beam_width, nb_nodes)
            prob_next_node = attn_weight_all[zero_to_bsz, idx_start_place_holder]
            score_t = torch.log_softmax(prob_next_node, dim=-1)
            sum_scores = score_t
            top_val, top_idx = torch.topk(sum_scores, B_t0, dim=1)
            sum_scores = top_val
            zero_to_bsz_idx = zero_to_bsz[:, None].expand(bsz, B_t0)
            zero_to_B_t0 = torch.arange(B_t0, device=attn_weight_all.device)
            zero_to_B_t0_idx = zero_to_B_t0[None, :].expand(bsz, B_t0)
            mask_visited_nodes = torch.zeros(bsz, nb_nodes, device=attn_weight_all.device).bool()
            mask_visited_nodes = mask_visited_nodes.unsqueeze(1)
            mask_visited_nodes = torch.repeat_interleave(mask_visited_nodes, B_t0, dim=1)
            mask_visited_nodes[zero_to_bsz_idx, zero_to_B_t0_idx, top_idx] = True
            tours = torch.zeros(bsz, B_t0, nb_nodes, device=attn_weight_all.device, dtype=torch.long)
            tours[:, :, 0] = top_idx
            attn_weight_all = attn_weight_all[:, :-1, :]
        elif t == 1:
            top_idx_expand = top_idx.unsqueeze(2).expand(-1, -1, nb_nodes)
            prob_next_node = attn_weight_all.gather(dim=1, index=top_idx_expand)
            score_t = torch.log_softmax(prob_next_node.masked_fill(mask_visited_nodes, float('-inf')), dim=-1)
            sum_scores = score_t + sum_scores.unsqueeze(2)  # (bsz, Bt0, nodes)
            sum_scores_flatten = sum_scores.view(bsz, -1)
            top_val, top_idx = torch.topk(sum_scores_flatten, beam_width, dim=1)
            idx_top_beams = torch.div(top_idx, nb_nodes, rounding_mode='trunc')
            idx_in_beams = top_idx % nb_nodes
            sum_scores = top_val
            mask_visited_nodes_tmp = mask_visited_nodes.clone()
            mask_visited_nodes = torch.zeros(bsz, beam_width, nb_nodes, device=attn_weight_all.device).bool()
            zero_to_bsz_idx = zero_to_bsz[:, None].expand(bsz, beam_width)
            zero_to_B_idx = zero_to_B[None, :].expand(bsz, beam_width)
            mask_visited_nodes[zero_to_bsz_idx, zero_to_B_idx] = mask_visited_nodes_tmp[zero_to_bsz_idx, idx_top_beams]
            mask_visited_nodes[zero_to_bsz_idx, zero_to_B_idx, idx_in_beams] = True
            tours_tmp = tours.clone()
            tours = torch.zeros(bsz, beam_width, nb_nodes, device=attn_weight_all.device, dtype=torch.long)
            tours[zero_to_bsz_idx, zero_to_B_idx] = tours_tmp[zero_to_bsz_idx, idx_top_beams]
            tours[:, :, t] = idx_in_beams
            del B_t0, zero_to_B_t0_idx, zero_to_B_t0, top_idx_expand
        else:
            idx_in_beams_expand = idx_in_beams.unsqueeze(2).expand(-1, -1, nb_nodes)
            prob_next_node = attn_weight_all.gather(dim=1, index=idx_in_beams_expand)
            score_t = torch.log_softmax(prob_next_node.masked_fill(mask_visited_nodes, float('-inf')), dim=-1)
            sum_scores = score_t + sum_scores.unsqueeze(2)  # (bsz, B, nodes)
            sum_scores_flatten = sum_scores.view(bsz, -1)
            top_val, top_idx = torch.topk(sum_scores_flatten, beam_width, dim=1)
            idx_top_beams = torch.div(top_idx, nb_nodes, rounding_mode='trunc')
            idx_in_beams = top_idx % nb_nodes
            sum_scores = top_val
            mask_visited_nodes_tmp = mask_visited_nodes.clone()
            mask_visited_nodes[zero_to_bsz_idx, zero_to_B_idx] = mask_visited_nodes_tmp[zero_to_bsz_idx, idx_top_beams]
            mask_visited_nodes[zero_to_bsz_idx, zero_to_B_idx, idx_in_beams] = True
            tours_tmp = tours.clone()
            tours[zero_to_bsz_idx, zero_to_B_idx] = tours_tmp[zero_to_bsz_idx, idx_top_beams]
            tours[:, :, t] = idx_in_beams
    sum_scores = sum_scores[:, 0]  # size(sumScores)=(bsz)
    return tours, sum_scores


class DotDict(dict):
    def __init__(self, **kwds):
        super().__init__()
        self.update(kwds)


args = DotDict()
args.problem_size = 100
args.batch_size = 1
args.offset = 0
args.width = 0
args.softmax_temperature = 0.1
args.nb_epoch = 500 if args.problem_size == 50 else 1000
args.nb_batch_per_epoch = 1000
args.lr = 1e-5
args.nb_batch_eval = (10000 + args.batch_size - 1) // args.batch_size

device = torch.device("cuda")
# x_10k = torch.load('10k_TSP100.pt').to(device)


student_model = NAR_model(
    embedding_dim=128,
    hidden_dim=128,
    n_encode_layers=3,
    normalization="batch",
    tanh_clipping=10.0
)

f = open('checkpoint/checkpoint-n100.pkl', 'rb')
record = torch.load(f)
student_model.load_state_dict(record['parameter'])

student_model.to(device)
student_model.eval()
zero_to_bsz = torch.arange(args.batch_size, device=device)

greedy = True; B = 1000

all_times = []
args.batch_size = 1
with torch.no_grad():
    # time for different problem size
    for args.problem_size in [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900,
                              950, 1000]:
    # time for different width
    # for B in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900,
        #           2000]:
        problems = torch.rand(size=(args.batch_size, args.problem_size, 2), device=device)
        attn_weight_all = student_model(problems)
        begin_time = time.time()
        for step in range(100):
            attn_weight_all = student_model(problems)
            if greedy:
                tours_greedy = greedy_search(attn_weight_all)
                L_greedy = compute_tour_length_single(problems, tours_greedy)
            else:
                tours_beamsearch, _ = beam_search(attn_weight_all, B)
                L_beamsearch = compute_tour_length_all_b(problems, tours_beamsearch)
                L_beamsearch, _ = L_beamsearch.min(dim=1)
        times = (time.time() - begin_time) / 100
        all_times.append(times)
        print(times)
    print(all_times)


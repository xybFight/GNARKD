import torch
from NARmodel import NAR_TSP_net
import numpy as np
import time
import datetime
import os

# This file is used to test the inference time of GNARKD-TM for TSP

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
    bsz, nb_nodes = attn_weight_all.shape[0], attn_weight_all.shape[1]
    zero_to_bsz = torch.arange(bsz, device=attn_weight_all.device)
    idx_start_place_holder = torch.tensor([nb_nodes - 1], device=attn_weight_all.device).long().repeat(bsz)
    mask_visited_nodes = torch.zeros(bsz, nb_nodes, device=attn_weight_all.device).bool()
    mask_visited_nodes[zero_to_bsz, idx_start_place_holder] = True
    idx = idx_start_place_holder
    tour_nar = []
    for node_index in range(nb_nodes - 1):
        attn_weight = torch.softmax(attn_weight_all[zero_to_bsz, idx].masked_fill(mask_visited_nodes, float('-1e10')),
                                    dim=-1)
        idx = torch.argmax(attn_weight, dim=1)
        tour_nar.append(idx)
        mask_visited_nodes = mask_visited_nodes.clone()
        mask_visited_nodes[zero_to_bsz, idx] = True
    tour_nar = torch.stack(tour_nar, dim=1)
    return tour_nar


def beam_search(attn_weight_all, beam_width=1000):
    """
    attn_weight_all:(BSZ, node+1, node+1)
    """
    bsz, nb_nodes = attn_weight_all.shape[0], attn_weight_all.shape[1] - 1
    zero_to_bsz = torch.arange(bsz, device=attn_weight_all.device)
    zero_to_B = torch.arange(beam_width, device=attn_weight_all.device)
    mask_visited_nodes = torch.zeros(bsz, nb_nodes + 1, device=attn_weight_all.device).bool()
    for t in range(nb_nodes):
        if t == 0:
            idx_start_place_holder = torch.tensor([nb_nodes], device=attn_weight_all.device).long().repeat(bsz)
            mask_visited_nodes[zero_to_bsz, idx_start_place_holder] = True
            B_t0 = min(beam_width, nb_nodes)
            prob_next_node = attn_weight_all[zero_to_bsz, idx_start_place_holder]
            score_t = torch.log_softmax(prob_next_node.masked_fill(mask_visited_nodes, float('-inf')), dim=-1)
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
            attn_weight_all = attn_weight_all[:, :-1, :-1]
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


device = torch.device('cuda')
args = DotDict()
args.nb_nodes = 1000  # TSP50
args.bsz = 1  # TSP20 TSP50

# follow the teacher model
args.dim_emb = 128
args.dim_ff = 512
args.dim_input_nodes = 2
args.nb_layers_encoder = 6
args.nb_layers_decoder = 2
args.nb_heads = 8
args.nb_epochs = 1000
args.nb_batch_per_epoch = 200
args.nb_batch_eval = 20
args.lr = 1e-5
args.batch_norm = True  # if batchnorm=True  than batch norm is used
args.max_len_PE = 1000

student_model = NAR_TSP_net(args.dim_input_nodes, args.dim_emb, args.dim_ff, args.nb_layers_encoder,
                            args.nb_layers_decoder,
                            args.nb_heads)
f = open('checkpoint/checkpoint-n100.pkl', 'rb')
record = torch.load(f)
student_model.load_state_dict(record['parameter'])
student_model.to(device)
args.nb_batch_eval = (10000 + args.bsz - 1) // args.bsz

student_model.eval()
tours_length = torch.zeros(size=(0, 1), device=device)
greedy = True
B = 1

all_times = []
args.nb_nodes = 100
with torch.no_grad():
    for args.nb_nodes in [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900,
                              950, 1000]:
    # for B in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900,
    #           2000]:
        x = torch.rand(size=(args.bsz, args.nb_nodes, 2), device=device)
        begin_time = time.time()
        for step in range(100):
            attn_weight_all = student_model(x)
            if greedy:
                tours_greedy = greedy_search(attn_weight_all)
                L_greedy = compute_tour_length_single(x, tours_greedy)
            else:
                tours_beamsearch, _ = beam_search(attn_weight_all, B)
                L_beamsearch = compute_tour_length_all_b(x, tours_beamsearch)
                L_beamsearch, _ = L_beamsearch.min(dim=1)
        times = (time.time() - begin_time) / 100
        all_times.append(times)
        print(times)
print(all_times)

import torch
import time

from NARmodel import CVRPModel as NAR_Model


# This file is used to test the inference time of GNARKD-POMO for CVRP


def get_random_problems(batch_size, problem_size):
    depot_xy = torch.rand(size=(batch_size, 1, 2), device=model_args.device)
    # shape: (batch, 1, 2)

    node_xy = torch.rand(size=(batch_size, problem_size, 2), device=model_args.device)
    # shape: (batch, problem, 2)

    demand_scaler = problem_size // 5 + 30

    node_demand = torch.randint(1, 10, size=(batch_size, problem_size), device=model_args.device) / float(demand_scaler)
    # shape: (batch, problem)

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
    selected_node_list = torch.zeros((batch_size, pomo_size, 0), dtype=torch.long, device=score_stu.device)
    done = False
    selected_count = 0
    while not done:
        if selected_count == 0:
            selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long, device=score_stu.device)
        elif selected_count == 1:
            selected = torch.arange(start=1, end=pomo_size + 1, device=score_stu.device)[None, :].expand(batch_size,
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


def augment_xy_data_by_8_fold(problems):
    # problems.shape: (batch, problem, 2)

    x = problems[:, :, [0]]
    y = problems[:, :, [1]]
    # x,y shape: (batch, problem, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_problems = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, problem, 2)

    return aug_problems


def beam_search(score_stu, node_demand, beam_width):
    round_error_epsilon = 0.00001
    batch_size, problem_size = node_demand.size(0), node_demand.size(1)
    # pomo_size = problem_size
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


class DotDict(dict):
    def __init__(self, **kwds):
        super().__init__()
        self.update(kwds)


# following POMO
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
args.problem_size = 50
args.pomo_size = args.problem_size  # POMO
args.aug_factor = 1
args.lr = 1e-5
args.nb_epochs = 1000
args.nb_batch_per_epoch = 1

student_model = NAR_Model(model_args)
student_model.to(model_args.device)

f = open('checkpoint\checkpoint-n50.pkl', 'rb') if args.problem_size == 50 else open('checkpoint\checkpoint-n100.pkl', 'rb')
record = torch.load(f)
student_model.load_state_dict(record['parameter'])

student_model.eval()
args.batch_size = 1
args.problem_size = 100
args.aug_factor = 1
all_times = []

with torch.no_grad():
    for args.problem_size in [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900,
                              950, 1000]:
    # for B in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800,
    #           1900, 2000]:
        args.pomo_size = args.problem_size
        depot_xy, node_xy, node_demand = get_random_problems(args.batch_size, args.problem_size)
        if args.aug_factor == 8:
            # aug
            depot_xy = augment_xy_data_by_8_fold(depot_xy)
            node_xy = augment_xy_data_by_8_fold(node_xy)
            node_demand = node_demand.repeat(8, 1)
            begin_time = time.time()
            for step in range(100):
                score_stu = student_model(depot_xy, node_xy, node_demand)
                selected_node_list = pomo_beam_search(score_stu, node_demand)
                L_val_aug_pomo = compute_tour_length_all_b(torch.cat((depot_xy, node_xy), dim=1), selected_node_list). \
                    reshape(8, -1, args.pomo_size)
                L_max_aug, _ = L_val_aug_pomo.min(dim=2)
                L_max, _ = L_max_aug.min(dim=0)
        else:
            # no aug
            begin_time = time.time()
            for step in range(100):
                score_stu = student_model(depot_xy, node_xy, node_demand)
                selected_node_list = pomo_beam_search(score_stu, node_demand)
                L_val_pomo = compute_tour_length_all_b(torch.cat((depot_xy, node_xy), dim=1), selected_node_list)
                L_val, _ = L_val_pomo.min(dim=1)
        times = (time.time() - begin_time) / 100
        all_times.append(times)
        print(times)
    print(all_times)


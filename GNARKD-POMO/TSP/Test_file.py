import torch
import time
from NARmodel import TSPModel as NAR_Model


# This file is used to test the performance of GNARKD-POMO for TSP
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
    batch_size, problem_size = score_stu.size(0), score_stu.size(1)
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


def beam_search(score_stu, beam_width=1000):
    batch_size, problem_size = score_stu.size(0), score_stu.size(1)
    zero_to_bsz = torch.arange(batch_size, device=score_stu.device)
    zero_to_B = torch.arange(beam_width, device=score_stu.device)
    tours_all = torch.zeros(batch_size, beam_width, 0, device=score_stu.device, dtype=torch.long)
    for t in range(problem_size - 1):
        if t == 0:
            mask_visited_nodes = torch.zeros(batch_size, problem_size, problem_size, device=score_stu.device).bool()
            zero_to_problem_size_idx = torch.arange(problem_size, device=score_stu.device)[None, :].expand(batch_size,
                                                                                                           problem_size)
            zero_to_bsz_idx = zero_to_bsz[:, None].expand(batch_size, problem_size)
            idx_start_node = torch.arange(problem_size, device=score_stu.device, dtype=torch.long)[None, :].expand(
                batch_size, problem_size)
            mask_visited_nodes[zero_to_bsz_idx, zero_to_problem_size_idx, idx_start_node] = True
            idx_start_node_expand = idx_start_node.unsqueeze(2).expand(-1, -1, problem_size)
            prob_next_node = score_stu.gather(dim=1, index=idx_start_node_expand)
            score_t = torch.log_softmax(prob_next_node.masked_fill(mask_visited_nodes, float('-inf')), dim=-1)
            sum_scores = score_t
            sum_scores_flatten = sum_scores.view(batch_size, -1)
            top_val, top_idx = torch.topk(sum_scores_flatten, beam_width, dim=1)
            sum_scores = top_val
            idx_top_beams = torch.div(top_idx, problem_size, rounding_mode='trunc')
            idx_in_beams = top_idx % problem_size
            tours_all = torch.cat((tours_all, idx_top_beams[:, :, None], idx_in_beams[:, :, None]), dim=2)
            # mask
            zero_to_bsz_idx = zero_to_bsz[:, None].expand(batch_size, beam_width)
            zero_to_B_idx = zero_to_B[None, :].expand(batch_size, beam_width)
            mask_visited_nodes = torch.zeros(batch_size, beam_width, problem_size, device=score_stu.device).bool()
            mask_visited_nodes[zero_to_bsz_idx, zero_to_B_idx, idx_top_beams] = True
            mask_visited_nodes[zero_to_bsz_idx, zero_to_B_idx, idx_in_beams] = True
        else:
            idx_in_beams_expand = idx_in_beams.unsqueeze(2).expand(-1, -1, problem_size)
            prob_next_node = score_stu.gather(dim=1, index=idx_in_beams_expand)
            score_t = torch.log_softmax(prob_next_node.masked_fill(mask_visited_nodes, float('-inf')), dim=-1)
            sum_scores = score_t + sum_scores.unsqueeze(2)
            sum_scores_flatten = sum_scores.view(batch_size, -1)
            top_val, top_idx = torch.topk(sum_scores_flatten, beam_width, dim=1)
            sum_scores = top_val
            idx_top_beams = torch.div(top_idx, problem_size, rounding_mode='trunc')
            idx_in_beams = top_idx % problem_size
            # mask
            mask_visited_nodes_tmp = mask_visited_nodes.clone()
            mask_visited_nodes[zero_to_bsz_idx, zero_to_B_idx] = mask_visited_nodes_tmp[zero_to_bsz_idx, idx_top_beams]
            mask_visited_nodes[zero_to_bsz_idx, zero_to_B_idx, idx_in_beams] = True
            tours_tmp = tours_all.clone()
            tours_all[zero_to_bsz_idx, zero_to_B_idx] = tours_tmp[zero_to_bsz_idx, idx_top_beams]
            tours_all = torch.cat((tours_all, idx_in_beams[:, :, None]), dim=2)
    return tours_all


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


class DotDict(dict):
    def __init__(self, **kwds):
        super().__init__()
        self.update(kwds)


# Following POMO
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
args.batch_size = 1000
args.problem_size = 50
args.pomo_size = args.problem_size  # POMO
args.aug_factor = 1
args.lr = 1e-5
args.nb_epochs = 100
args.nb_batch_per_epoch = 500

student_model = NAR_Model(model_args)
student_model.to(model_args.device)

x_10k = torch.load('../../Test_data/TSP/10k_TSP50.pt').to(model_args.device) if args.problem_size==50 else torch.load('../../Test_data/TSP/10k_TSP100.pt').to(model_args.device)
args.nb_batch_eval = (10000 + args.batch_size - 1) // args.batch_size

f = open('checkpoint/checkpoint-n50.pkl', 'rb')
# Different temperature
# f = open('Temp_trained/temp_10.pkl', 'rb')
record = torch.load(f)
student_model.load_state_dict(record['parameter'])

student_model.eval()
tours_length = torch.zeros(size=(0, 1), device=x_10k.device)

with torch.no_grad():
    begin_time = time.time()
    for step in range(args.nb_batch_eval):
        problems = x_10k[step * args.batch_size:(step + 1) * args.batch_size, :, :]
        if args.aug_factor != 8:
            score_stu = student_model(problems)
            # selected_node_list = beam_search(score_stu)
            selected_node_list = pomo_beam_search(score_stu)
            L_val_pomo = compute_tour_length_all_b(problems, selected_node_list)
            L_val, _ = L_val_pomo.min(dim=1)
            tours_length = torch.cat((tours_length, L_val[:, None]), dim=0)
        else:
            aug_problems = augment_xy_data_by_8_fold(problems)
            score_stu = student_model(aug_problems)
            # selected_node_list = beam_search(score_stu)
            # L_val_aug_pomo = compute_tour_length_all_b(aug_problems, selected_node_list).reshape(8, -1, 1000)
            selected_node_list = pomo_beam_search(score_stu)
            L_val_aug_pomo = compute_tour_length_all_b(aug_problems, selected_node_list).reshape(8, -1, args.pomo_size)
            L_max_aug, _ = L_val_aug_pomo.min(dim=2)
            L_max, _ = L_max_aug.min(dim=0)
            tours_length = torch.cat((tours_length, L_max[:, None]), dim=0)
    print(time.time() - begin_time)
    mean_tour_length = torch.mean(tours_length, dim=0).item()
    print(mean_tour_length)

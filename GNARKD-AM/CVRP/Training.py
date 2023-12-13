import time
from AR_model import AttentionModel as AR_model
import json
import os
from AR_model import CVRP as CVRP_problem
import torch
from NAR_model import AttentionModel as NAR_model
import datetime


# This is for training GNARKD-AM on CVRP

def load_args(filename):
    with open(filename, 'r') as f:
        args = json.load(f)

    # Backwards compatibility
    if 'data_distribution' not in args:
        args['data_distribution'] = None
        probl, *dist = args['problem'].split("_")
        if probl == "op":
            args['problem'] = probl
            args['data_distribution'] = dist[0]
    return args


def _load_model_file(load_path, model):
    """Loads the model with parameters from the file and returns optimizer state dict if it is in the file"""

    # Load the model parameters from a saved state
    load_optimizer_state_dict = None
    print('  [*] Loading model from {}'.format(load_path))

    load_data = torch.load(
        os.path.join(
            os.getcwd(),
            load_path
        ), map_location=lambda storage, loc: storage)

    if isinstance(load_data, dict):
        load_optimizer_state_dict = load_data.get('optimizer', None)
        load_model_state_dict = load_data.get('model', load_data)
    else:
        load_model_state_dict = load_data.state_dict()

    state_dict = model.state_dict()

    state_dict.update(load_model_state_dict)

    model.load_state_dict(state_dict)

    return model, load_optimizer_state_dict


def get_random_problems(batch_size, problem_size):
    depot_xy = torch.rand(size=(batch_size, 1, 2), device=device)
    # shape: (batch, 1, 2)

    node_xy = torch.rand(size=(batch_size, problem_size, 2), device=device)
    # shape: (batch, problem, 2)

    if problem_size == 20:
        demand_scaler = 30.
    elif problem_size == 50:
        demand_scaler = 40.
    elif problem_size == 100:
        demand_scaler = 50.
    else:
        raise NotImplementedError

    node_demand = torch.randint(1, 10, size=(batch_size, problem_size), device=device) / demand_scaler
    # shape: (batch, problem)

    return depot_xy, node_xy, node_demand


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


def compute_tour_length_all_b(depot_node_xy, selected_node_list):
    pomo_size = selected_node_list.size(1)
    gathering_index = selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
    all_xy = depot_node_xy[:, None, :, :].expand(-1, pomo_size, -1, -1)
    ordered_seq = all_xy.gather(dim=2, index=gathering_index)
    rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
    segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
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


# following AM
args = DotDict()
args.problem_size = 100
args.batch_size = 100
args.offset = 0
args.width = 0
args.softmax_temperature = 1
args.nb_epoch = 500 if args.problem_size == 50 else 1000
args.nb_batch_per_epoch = 1000
args.lr = 1e-5
args.nb_batch_eval = (10000 + args.batch_size - 1) // args.batch_size

# Note: The pre-training parameters of AM can be found at the address mentioned in its paper,
path = 'pretrained/cvrp_' + str(args.problem_size)
model_args = load_args(os.path.join(path, 'args.json'))
device = torch.device("cuda")
teacher_model = AR_model(
    model_args['embedding_dim'],
    model_args['hidden_dim'],
    CVRP_problem,
    n_encode_layers=model_args['n_encode_layers'],
    mask_inner=True,
    mask_logits=True,
    normalization=model_args['normalization'],
    tanh_clipping=model_args['tanh_clipping'],
    temp=args.softmax_temperature
)

load_data = torch.load(os.path.join(path, 'epoch-{}.pt'.format(99)), map_location=lambda storage, loc: storage)
teacher_model.load_state_dict({**teacher_model.state_dict(), **load_data.get('model', {})})
teacher_model, *_ = _load_model_file(os.path.join(path, 'epoch-{}.pt'.format(99)), teacher_model)
teacher_model.to(device)
teacher_model.eval()

student_model = NAR_model(
    model_args['embedding_dim'],
    model_args['hidden_dim'],
    n_encode_layers=model_args['n_encode_layers'],
    tanh_clipping=model_args['tanh_clipping'],
    normalization=model_args['normalization'],
)

student_model.to(device)

loss_func = torch.nn.KLDivLoss(reduction='batchmean')
student_model.embedder.load_state_dict(teacher_model.embedder.state_dict())
student_model.init_embed_depot.load_state_dict(teacher_model.init_embed_depot.state_dict())
student_model.init_embed.load_state_dict(teacher_model.init_embed.state_dict())
optimizer = torch.optim.Adam(student_model.parameters(), lr=args.lr)

ZERO_TO_BATCH = torch.arange(args.batch_size, device=device)
zero_tour = torch.zeros(args.batch_size, 1, dtype=torch.long, device=device)

# val dataset
val_depot_xy, val_node_xy, val_node_demand = get_random_problems(10000, args.problem_size)
args.nb_batch_eval = (10000 + args.batch_size - 1) // args.batch_size
min_len = torch.inf

for epoch in range(args.nb_epoch):
    print('epoch:' + str(epoch))
    student_model.train()
    begin = time.time()
    for step in range(args.nb_batch_per_epoch):
        depot_xy, node_xy, node_demand = get_random_problems(args.batch_size, args.problem_size)
        with torch.no_grad():
            tours, prob_next_node_all, _ = teacher_model.sample_many_my((depot_xy, node_xy, node_demand))
        tours = torch.cat((zero_tour, tours), dim=1)
        score_stu = student_model((depot_xy, node_xy, node_demand))
        at_the_depot = torch.ones(args.batch_size, dtype=torch.bool, device=device)
        load = torch.ones(args.batch_size, device=device)
        visited_ninf_flag = torch.zeros(size=(args.batch_size, args.problem_size + 1), device=device).bool()
        ninf_mask = torch.zeros(size=(args.batch_size, args.problem_size + 1), device=device).bool()
        finished = torch.zeros(args.batch_size, dtype=torch.bool, device=device)
        depot_node_demand = torch.cat((torch.zeros(size=(args.batch_size, 1), device=device), node_demand), dim=1)
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
            # ninf_mask[demand_too_large] = float('-inf')
            ninf_mask[demand_too_large] = True
            # finished = finished + (visited_ninf_flag == float('-inf')).all(dim=2)
            finished = finished + (visited_ninf_flag == True).all(dim=1)
            ninf_mask[:, 0][finished] = False
        prob_next_node_all_NAR = torch.stack(prob_next_node_all_NAR, dim=-2)
        loss = loss_func(prob_next_node_all_NAR, prob_next_node_all)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(time.time() - begin)
    # val
    student_model.eval()
    tours_length = torch.zeros(size=(0, 1), device=device)
    for step in range(args.nb_batch_eval):
        depot_xy = val_depot_xy[step * args.batch_size:(step + 1) * args.batch_size, :, :]
        node_xy = val_node_xy[step * args.batch_size:(step + 1) * args.batch_size, :, :]
        node_demand = val_node_demand[step * args.batch_size:(step + 1) * args.batch_size, :]
        with torch.no_grad():
            score_stu = student_model((depot_xy, node_xy, node_demand))
            selected_node_list = greedy_search(score_stu, node_demand)
            L_val = compute_tour_length(torch.cat((depot_xy, node_xy), dim=1), selected_node_list)
            tours_length = torch.cat((tours_length, L_val[:, None]), dim=0)
    mean_tour_length = torch.mean(tours_length, dim=0).item()
    print(mean_tour_length)

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

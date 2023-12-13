import time

from AR_model import AttentionModel as AR_model
from NAR_model import AttentionModel as NAR_model
import json
import os
from AR_model import TSP as TSP_problem
import torch


# This is for training GNARKD-AM on TSP

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


def compute_tour_length(x, tour):
    bsz, nb_node = x.shape[0], x.shape[1]
    index = tour.unsqueeze(2).expand(bsz, nb_node, 2)
    with torch.no_grad():
        order_seq = x.gather(dim=1, index=index)
        rolled_seq = order_seq.roll(dims=1, shifts=-1)
        segment_length = ((order_seq - rolled_seq) ** 2).sum(2).sqrt()
        L = segment_length.sum(1)
    return L


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


class DotDict(dict):
    def __init__(self, **kwds):
        super().__init__()
        self.update(kwds)


# following AM
args = DotDict()
args.problem_size = 50
args.batch_size = 100
args.offset = 0
args.width = 0
args.softmax_temperature = 0.1
args.nb_epoch = 500 if args.problem_size == 50 else 1000
args.nb_batch_per_epoch = 1000
args.batch_size = 100
args.lr = 1e-5
args.nb_batch_eval = (10000 + args.batch_size - 1) // args.batch_size

# Note: The pre-training parameters of AM can be found at the address mentioned in its paper,
path = 'pretrained/tsp_' + str(args.problem_size)
model_args = load_args(os.path.join(path, 'args.json'))
device = torch.device("cuda")

# val dataset
x_10k = torch.rand(size=(10000, args.problem_size, 2), device=device)

teacher_model = AR_model(
    model_args['embedding_dim'],
    model_args['hidden_dim'],
    TSP_problem,
    n_encode_layers=model_args['n_encode_layers'],
    mask_inner=True,
    mask_logits=True,
    normalization=model_args['normalization'],
    tanh_clipping=model_args['tanh_clipping'],
    temp=args.softmax_temperature
)

load_data = torch.load(os.path.join(path, 'epoch-{}.pt'.format(99)), map_location=lambda storage, loc: storage)
teacher_model.load_state_dict({**teacher_model.state_dict(), **load_data.get('model', {})})

# Note: The pre-training parameters of AM can be found at the address mentioned in its paper,

teacher_model, *_ = _load_model_file(os.path.join(path, 'epoch-{}.pt'.format(99)), teacher_model)
teacher_model.to(device)
teacher_model.eval()

student_model = NAR_model(
    model_args['embedding_dim'],
    model_args['hidden_dim'],
    n_encode_layers=model_args['n_encode_layers'],
    normalization=model_args['normalization'],
    tanh_clipping=model_args['tanh_clipping']
)
student_model.embedder.load_state_dict(teacher_model.embedder.state_dict())
student_model.init_embed.load_state_dict(teacher_model.init_embed.state_dict())

student_model.to(device)
zero_to_bsz = torch.arange(args.batch_size, device=device)
loss_func = torch.nn.KLDivLoss(reduction='batchmean')
optimizer = torch.optim.Adam(student_model.parameters(), lr=args.lr)

min_len = torch.inf
for epoch in range(args.nb_epoch):
    print('epoch:' + str(epoch))
    student_model.train()
    begin = time.time()
    for step in range(args.nb_batch_per_epoch):
        problems = torch.rand(size=(args.batch_size, args.problem_size, 2), device=device)
        with torch.no_grad():
            tours, prob_next_node_all = teacher_model.sample_many_my(problems)
        attn_weight_all = student_model(problems)
        idx_start_place_holder = torch.tensor([args.problem_size], device=device).long().repeat(args.batch_size)
        mask_visited_nodes = torch.zeros(args.batch_size, args.problem_size, device=device).bool()
        idx = idx_start_place_holder
        prob_next_node_all_NAR = []
        for node_index in range(args.problem_size):
            attn_weight = attn_weight_all[zero_to_bsz, idx].masked_fill(mask_visited_nodes, float('-1e10'))
            attn_weight = torch.log_softmax(attn_weight, dim=-1)
            prob_next_node_all_NAR.append(attn_weight.squeeze(1))
            mask_visited_nodes = mask_visited_nodes.clone()
            idx = tours[zero_to_bsz, node_index]
            mask_visited_nodes[zero_to_bsz, idx] = True
        prob_next_node_all_NAR = torch.stack(prob_next_node_all_NAR, dim=1)
        loss = loss_func(prob_next_node_all_NAR, prob_next_node_all)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(time.time() - begin)

    student_model.eval()
    tours_length = torch.zeros(size=(0, 1), device=device)
    for step in range(args.nb_batch_eval):
        with torch.no_grad():
            problems = x_10k[step * args.batch_size:(step + 1) * args.batch_size, :, :]
            attn_weight_all = student_model(problems)
            idx_start_place_holder = torch.tensor([args.problem_size], device=device).long().repeat(args.batch_size)
            mask_visited_nodes = torch.zeros(args.batch_size, args.problem_size, device=device).bool()
            idx = idx_start_place_holder
            tour_nar = []
            for node_index in range(args.problem_size):
                attn_weight = attn_weight_all[zero_to_bsz, idx].masked_fill(mask_visited_nodes, float('-1e10'))
                attn_weight = torch.softmax(attn_weight, dim=-1)
                idx = torch.argmax(attn_weight, dim=1)
                tour_nar.append(idx)
                mask_visited_nodes = mask_visited_nodes.clone()
                mask_visited_nodes[zero_to_bsz, idx] = True
            tour_nar = torch.stack(tour_nar, dim=1)
            L_val = compute_tour_length(problems, tour_nar)
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
            'TSP_length': mean_tour_length,
            'parameter': student_model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, '{}.pkl'.format(checkpoint_dir + '/checkpoint' + '-n{}'.format(args.problem_size)))

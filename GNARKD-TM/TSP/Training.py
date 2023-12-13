import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from ARmodel import AR_TSP_net
from NARmodel import NAR_TSP_net
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
import os

# This is for training GNARKD-TM on TSP


def compute_tour_length(x, tour):
    bsz, nb_node = x.shape[0], x.shape[1]
    index = tour.unsqueeze(2).expand(bsz, nb_node, 2)
    with torch.no_grad():
        order_seq = x.gather(dim=1, index=index)
        rolled_seq = order_seq.roll(dims=1, shifts=-1)
        segment_length = ((order_seq - rolled_seq)**2).sum(2).sqrt()
        L = segment_length.sum(1)
    return L


class DotDict(dict):
    def __init__(self, **kwds):
        super().__init__()
        self.update(kwds)


device = torch.device('cuda')
args = DotDict()
args.nb_nodes = 50  # TSP50
args.bsz = 100
args.nb_epochs = 1000
args.nb_batch_per_epoch = 500 if args.nb_nodes == 50 else 1000
args.nb_batch_eval = 100
args.lr = 1e-5


# follow the teacher model
args.batch_norm = True
args.max_len_PE = 1000
args.dim_emb = 128
args.dim_ff = 512
args.dim_input_nodes = 2
args.nb_layers_encoder = 6
args.nb_layers_decoder = 2
args.nb_heads = 8
time_stamp = datetime.datetime.now().strftime("%y-%m-%d--%H-%M-%S")
teacher_model = AR_TSP_net(args.dim_input_nodes, args.dim_emb, args.dim_ff, args.nb_layers_encoder,
                           args.nb_layers_decoder,
                           args.nb_heads, args.max_len_PE)
teacher_model.to(device)

# Note: The pre-training parameters of TM can be found at the address mentioned in its paper,
f = open('pretrained/checkpoint_21-03-01--17-25-00-n50-gpu0.pkl', 'rb')
record = torch.load(f)
teacher_model.load_state_dict(record['model_baseline'])
student_model = NAR_TSP_net(args.dim_input_nodes, args.dim_emb, args.dim_ff, args.nb_layers_encoder,
                            args.nb_layers_decoder,
                            args.nb_heads)
student_model.to(device)
optimizer = torch.optim.Adam(student_model.parameters(), lr=args.lr)
loss_func = nn.KLDivLoss(reduction='batchmean')
args.nb_batch_eval = (10000 + args.bsz - 1) // args.bsz
zero_to_bsz = torch.arange(args.bsz, device=device)
teacher_model.eval()
student_model.encoder_layer.load_state_dict(teacher_model.encoder.state_dict())
min_len = torch.inf

# val dataset
x_10k = torch.rand(size=(10000, args.nb_nodes, 2), device=device)


for epoch in range(args.nb_epochs):
    print('epoch:'+str(epoch))
    student_model.train()
    begin = time.time()
    for step in range(args.nb_batch_per_epoch):
        x = torch.rand(args.bsz, args.nb_nodes, args.dim_input_nodes, device=device)
        with torch.no_grad():
            tours, prob_next_node_all = teacher_model(x)
        attn_weight_all = student_model(x)
        idx_start_place_holder = torch.tensor([args.nb_nodes], device=device).long().repeat(args.bsz)
        mask_visited_nodes = torch.zeros(args.bsz, args.nb_nodes + 1, device=x.device).bool()
        mask_visited_nodes[zero_to_bsz, idx_start_place_holder] = True
        idx = idx_start_place_holder
        prob_next_node_all_NAR = []
        for node_index in range(args.nb_nodes):
            attn_weight = attn_weight_all[zero_to_bsz, idx].masked_fill(mask_visited_nodes, float('-1e10'))
            attn_weight = torch.log_softmax(attn_weight, dim=-1)
            prob_next_node_all_NAR.append(attn_weight.squeeze(1))
            mask_visited_nodes = mask_visited_nodes.clone()
            idx = tours[zero_to_bsz, node_index]
            mask_visited_nodes[zero_to_bsz, idx] = True
        prob_next_node_all_NAR = torch.stack(prob_next_node_all_NAR, dim=1)
        loss = loss_func(prob_next_node_all_NAR, prob_next_node_all)
        sumLogProOfActions = []
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(time.time()-begin)
    student_model.eval()
    tours_length = torch.zeros(size=(0, 1), device=x_10k.device)
    for step in range(args.nb_batch_eval):
        x = x_10k[step * args.bsz:(step + 1) * args.bsz, :, :]
        with torch.no_grad():
            attn_weight_all = student_model(x)
            idx_start_place_holder = torch.tensor([args.nb_nodes], device=device).long().repeat(args.bsz)
            mask_visited_nodes = torch.zeros(args.bsz, args.nb_nodes + 1, device=x.device).bool()
            mask_visited_nodes[zero_to_bsz, idx_start_place_holder] = True
            idx = idx_start_place_holder
            tour_nar = []
            for node_index in range(args.nb_nodes):
                attn_weight = attn_weight_all[zero_to_bsz, idx].masked_fill(mask_visited_nodes, float('-1e10'))
                attn_weight = torch.softmax(attn_weight, dim=-1)
                idx = torch.argmax(attn_weight, dim=1)
                tour_nar.append(idx)
                mask_visited_nodes = mask_visited_nodes.clone()
                mask_visited_nodes[zero_to_bsz, idx] = True
            tour_nar = torch.stack(tour_nar, dim=1)
            L_val = compute_tour_length(x, tour_nar)
            tours_length = torch.cat((tours_length, L_val[:, None]), dim=0)
    mean_tour_length = torch.mean(tours_length, dim=0).item()
    print('mean_tour_length:'+str(mean_tour_length))

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
        }, '{}.pkl'.format(checkpoint_dir + '/checkpoint' + '-n{}'.format(args.nb_nodes)))
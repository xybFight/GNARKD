import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.distributions.categorical import Categorical
from torch.autograd import Variable


def compute_tour_length(x, tour):
    bsz = x.shape[0]
    nb_nodes = x.shape[1]
    arange_vec = torch.arange(bsz, device=x.device)
    first_cities = x[arange_vec, tour[:, 0], :]  # (batchsize, 2)
    previous_cities = first_cities
    L = torch.zeros(bsz, device=x.device)
    with torch.no_grad():
        for i in range(1, nb_nodes):
            current_cities = x[arange_vec, tour[:, i], :]
            L += torch.sum((current_cities - previous_cities) ** 2, dim=1) ** 0.5
            previous_cities = current_cities
        L += torch.sum((current_cities - first_cities) ** 2, dim=1) ** 0.5
    del bsz, nb_nodes, arange_vec, first_cities, previous_cities, current_cities
    return L


def compute_tour_length_all_n(x, tour):
    bsz, nb_node = x.shape[0], x.shape[1]
    index = tour.unsqueeze(3).expand(bsz, -1, nb_node, 2)
    seq_expand = x[:, None, :, :].expand(bsz, nb_node, nb_node, 2)
    order_seq = seq_expand.gather(dim=2, index=index)
    rolled_seq = order_seq.roll(dims=2, shifts=-1)
    segment_length = ((order_seq - rolled_seq) ** 2).sum(3).sqrt()
    travel_distances = segment_length.sum(2)
    return travel_distances


def re_tour(node_begin, pro_matrix, deterministic=True):
    tours = []
    bsz, nb_nodes = pro_matrix.shape[0], pro_matrix.shape[1]
    zero_to_bsz = torch.arange(bsz)
    mask_visited_nodes = torch.zeros(bsz, nb_nodes).bool().to(node_begin.device)
    mask_visited_nodes[zero_to_bsz, node_begin] = True
    idx = node_begin
    tours.append(node_begin)
    for node in range(nb_nodes - 1):
        prob = torch.softmax(pro_matrix[zero_to_bsz, idx].masked_fill(mask_visited_nodes, float('-inf')), dim=-1)
        if deterministic:
            idx = torch.argmax(prob, dim=1)
        else:
            idx = Categorical(prob).sample()
        mask_visited_nodes = mask_visited_nodes.clone()
        mask_visited_nodes[zero_to_bsz, idx] = True
        tours.append(idx)
    tours = torch.stack(tours, dim=1)
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    return tours


class TSP_transformer_encoder(nn.Module):
    def __init__(self, num_layers: int, num_dim: int, num_heads: int, num_ff: int):
        super(TSP_transformer_encoder, self).__init__()
        self.MHA_layers = nn.ModuleList([nn.MultiheadAttention(num_dim, num_heads) for _ in range(num_layers)])
        self.norm1_layers = nn.ModuleList([nn.BatchNorm1d(num_dim) for _ in range(num_layers)])
        self.linear1_layers = nn.ModuleList([nn.Linear(num_dim, num_ff) for _ in range(num_layers)])
        self.linear2_layers = nn.ModuleList([nn.Linear(num_ff, num_dim) for _ in range(num_layers)])
        self.norm2_layers = nn.ModuleList([nn.BatchNorm1d(num_dim) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.activation = nn.ReLU()

    def forward(self, x):
        h = x.transpose(0, 1)  # (node, batch, dim)
        for layer in range(self.num_layers):
            h_rc = h
            h, _ = self.MHA_layers[layer](h, h, h)
            h = h + h_rc  # (node, batch, dim)
            h = self.norm1_layers[layer](h.permute(1, 2, 0).contiguous())  # (batch, dim, node)
            h = h.permute(2, 0, 1).contiguous()
            h_rc = h
            h = self.linear2_layers[layer](self.activation(self.linear1_layers[layer](h)))
            h = h + h_rc
            h = self.norm2_layers[layer](h.permute(1, 2, 0).contiguous())  # (batch, dim, node)
            h = h.permute(2, 0, 1)
        h = h.transpose(0, 1)
        return h


class TSP_transformer_decoder(nn.Module):
    def __init__(self, num_layers: int, num_dim: int, num_heads: int):
        super(TSP_transformer_decoder, self).__init__()
        self.num_dim = num_dim
        self.num_heads = num_heads
        self.decoder_layers = nn.ModuleList([transformer_decoder_layers(num_dim, num_heads) for _ in range(num_layers -1 )])
        self.Wq_final = nn.Linear(num_dim, num_dim)
        self.num_layers = num_layers

    def forward(self, h_t, K_att, V_att):
        """
        h_t: (bsz, node+1, num_dim)
        K_att: (bsz, node+1, num_dim * decoder_layers)
        V_att: (bsz, node+1, num_dim * decoder_layers)
        """
        for layer in range(self.num_layers):
            K_att_l = K_att[:, :, layer * self.num_dim:(layer + 1) * self.num_dim].contiguous()
            if layer < self.num_layers - 1:
                V_att_l = V_att[:, :, layer * self.num_dim:(layer + 1) * self.num_dim].contiguous()
                h_t = self.decoder_layers[layer](h_t, K_att_l, V_att_l)
            else:
                q_final = self.Wq_final(h_t)
                attn_weight_all = torch.bmm(q_final, K_att_l.transpose(1, 2) / self.num_dim ** 0.5)
                attn_weight_all = 10 * torch.tanh(attn_weight_all)
                return attn_weight_all


def myMHA(Q, K, V, nb_heads, mask=None, clip_value=None):
    bsz, nb_nodes, emd_dim = K.size()
    if nb_heads > 1:
        Q = Q.transpose(1, 2).contiguous()  # Q(bsz, dim_emb, 1)
        Q = Q.view(bsz * nb_heads, emd_dim // nb_heads, nb_nodes)  # Q(bsz*nb_heads, dim_emb//nb_heads, 1)
        Q = Q.transpose(1, 2).contiguous()  # Q(bsz*nb_heads, 1, dim_emb//nb_heads)
        K = K.transpose(1, 2).contiguous()  # K(bsz, dim_emb, nb_nodes+1)
        K = K.view(bsz * nb_heads, emd_dim // nb_heads,
                   nb_nodes)  # K(bsz*nb_heads, dim_emb//nb_heads, nb_nodes+1)
        K = K.transpose(1, 2).contiguous()  # K(bsz*nb_heads, nb_nodes+1, dim_emb//nb_heads)
        V = V.transpose(1, 2).contiguous()  # V(bsz, dim_emb, nb_nodes+1)
        V = V.view(bsz * nb_heads, emd_dim // nb_heads,
                   nb_nodes)  # V(bsz*nb_heads, dim_emb//nb_heads, nb_nodes+1)
        V = V.transpose(1, 2).contiguous()  # V(bsz*nb_heads, nb_nodes+1, dim_emb//nb_heads)
    attn_weights = torch.bmm(Q, K.transpose(1, 2) / Q.size(-1)**0.5)
    if clip_value is not None:
        attn_weights = clip_value * torch.tanh(attn_weights)
    if mask is not None:
        if nb_heads > 1:
            mask = torch.repeat_interleave(mask, repeats=nb_heads, dim=0)  # mask(bsz*nb_heads, nb_nodes+1)
        attn_weights = attn_weights.masked_fill(mask.unsqueeze(1), float('-1e9'))
    attn_weights = torch.softmax(attn_weights, dim=-1)  # attn_weights(bsz*nb_heads, 1, nb_nodes+1)
    attn_output = torch.bmm(attn_weights, V)  # attn_output(bsz*nb_heads, 1, dim_emb//nb_heads)
    if nb_heads > 1:
        attn_output = attn_output.transpose(1, 2).contiguous()  # attn_output(bsz*nb_heads, dim_emb//nb_heads, 1)
        attn_output = attn_output.view(bsz, emd_dim, nb_nodes)  # attn_output(bsz, dim_emb, 1)
        attn_output = attn_output.transpose(1, 2).contiguous()  # attn_output(bsz, 1, dim_emb)
        attn_weights = attn_weights.view(bsz, nb_heads, nb_nodes, nb_nodes)  # attn_weights(bsz, nb_heads, 1, nb_nodes+1)
        attn_weights = attn_weights.mean(dim=1)  # attn_weights(bsz, 1, nb_nodes+1)
    return attn_output, attn_weights  # attn_output(bsz, 1, dim_emb)  attn_weights(bsz, 1, nb_nodes+1)


class transformer_decoder_layers(nn.Module):
    def __init__(self, num_dim: int, num_heads: int):
        super(transformer_decoder_layers, self).__init__()
        self.num_dim = num_dim
        self.num_heads = num_heads
        self.Wq_self_att = nn.Linear(num_dim, num_dim)
        self.Wk_self_att = nn.Linear(num_dim, num_dim)
        self.Wv_self_att = nn.Linear(num_dim, num_dim)
        self.W0_self_att = nn.Linear(num_dim, num_dim)
        self.W0_att = nn.Linear(num_dim, num_dim)
        self.Wq_att = nn.Linear(num_dim, num_dim)
        self.W1_MLP = nn.Linear(num_dim, num_dim)
        self.W2_MLP = nn.Linear(num_dim, num_dim)
        self.BN_self_att = nn.LayerNorm(num_dim)
        self.BN_PE_att = nn.LayerNorm(num_dim)
        self.BN_att = nn.LayerNorm(num_dim)
        self.BN_MLP = nn.LayerNorm(num_dim)
        self.K_self_att = None
        self.V_self_att = None
        self.activation = torch.nn.ReLU()

    def forward(self, h_t, K_att, V_att):
        """
        h_t: (bsz, node+1, num_dim)
        K_att: (bsz, node+1, num_dim)
        V_att: (bsz, node+1, num_dim)
        """
        # bsz = h_t.size(0)
        q_self_att = self.Wq_self_att(h_t)
        k_self_att = self.Wk_self_att(h_t)
        v_self_att = self.Wv_self_att(h_t)
        # h_t(bsz, node+1, num_dim)
        h_t = h_t + self.W0_self_att(myMHA(q_self_att, k_self_att, v_self_att, self.num_heads)[0])
        h_t = self.BN_self_att(h_t)
        # cross_attention
        q_att = self.Wq_att(h_t)
        h_t = h_t + self.W0_att(myMHA(q_att, K_att, V_att, self.num_heads)[0])
        h_t = self.BN_self_att(h_t)
        # MLP
        h_t = h_t + self.W2_MLP(self.activation(self.W1_MLP(h_t)))
        h_t = self.BN_self_att(h_t)
        return h_t


class NAR_TSP_net(nn.Module):
    def __init__(self, num_input: int, num_dim: int, num_ff:int, num_encoder_layers: int, num_decoder_layers: int, num_heads: int):
        super(NAR_TSP_net, self).__init__()
        assert num_dim % num_heads == 0
        self.emb = nn.Linear(num_input, num_dim, bias=False)
        self.start_placehoder = nn.Parameter(torch.randn(num_dim))
        self.encoder_layer = TSP_transformer_encoder(num_encoder_layers, num_dim, num_heads, num_ff)
        self.decoder_layer = TSP_transformer_decoder(num_decoder_layers, num_dim, num_heads)
        self.WK_att_decoder = nn.Linear(num_dim, num_decoder_layers * num_dim)
        self.WV_att_decoder = nn.Linear(num_dim, (num_decoder_layers - 1) * num_dim)
        self.num_decoder_layers = num_decoder_layers
        self.num_dim = num_dim

    def forward(self, x):
        """
        x(batch，node, 2)
        """
        batch_size = x.shape[0]
        h = self.emb(x)  # (batch, node, num_dim)
        h = torch.cat([h, self.start_placehoder.repeat(batch_size, 1, 1)], dim=1)  # (batch, node+1, dim)
        h_encoder = self.encoder_layer(h)
        h_encoder = h_encoder
        K_att_decoder = self.WK_att_decoder(h_encoder)
        V_att_decoder = self.WV_att_decoder(h_encoder)
        attn_weight_all = self.decoder_layer(h_encoder, K_att_decoder, V_att_decoder)
        return attn_weight_all

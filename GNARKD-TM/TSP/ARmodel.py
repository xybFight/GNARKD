import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

device = torch.device('cuda')
gpu_id = 0


# This is from publicly available TM model.
class Transformer_encoder_net(nn.Module):
    def __init__(self, nb_layers, dim_emb, nb_heads, dim_ff, batchnorm):
        super(Transformer_encoder_net, self).__init__()
        assert dim_emb == nb_heads * (dim_emb // nb_heads)
        self.MHA_layers = nn.ModuleList([nn.MultiheadAttention(dim_emb, nb_heads) for _ in range(nb_layers)])
        self.linear1_layers = nn.ModuleList([nn.Linear(dim_emb, dim_ff) for _ in range(nb_layers)])
        self.linear2_layers = nn.ModuleList([nn.Linear(dim_ff, dim_emb) for _ in range(nb_layers)])
        if batchnorm:
            # BN
            self.norm1_layers = nn.ModuleList([nn.BatchNorm1d(dim_emb) for _ in range(nb_layers)])
            self.norm2_layers = nn.ModuleList([nn.BatchNorm1d(dim_emb) for _ in range(nb_layers)])
        else:
            # LN
            self.norm1_layers = nn.ModuleList([nn.LayerNorm(dim_emb) for _ in range(nb_layers)])
            self.norm2_layers = nn.ModuleList([nn.LayerNorm(dim_emb) for _ in range(nb_layers)])
        self.nb_layers = nb_layers
        self.nb_head = nb_heads
        self.batchnorm = batchnorm

    def forward(self, h):
        """
        h(bsz, nb_nodes, dim_emb)
        """
        h = h.transpose(0, 1)  # (nb_nodes, bsz, dim_emb)
        for i in range(self.nb_layers):
            h_rc = h  # h_rc(nb_nodes, bsz, dim_emb)
            # h(nb_nodes, bsz, dim_emb), score(bsz, nb_nodes,nb_nodes)
            h, score = self.MHA_layers[i](h, h, h)
            h = h_rc + h  # h(nb_nodes, bsz, dim_emb)
            if self.batchnorm:  # BN
                h = h.permute(1, 2, 0).contiguous()  # h(bsz, dim_emb, nb_nodes)
                h = self.norm1_layers[i](h)  # h(bsz, dim_emb, nb_nodes)
                h = h.permute(2, 0, 1).contiguous()  # h(nb_nodes, bsz, dim_emb)
            else:
                # Layer norm
                h = self.norm1_layers[i](h)
            h_rc = h
            h = self.linear2_layers[i](torch.relu(self.linear1_layers[i](h)))
            h = h_rc + h
            if self.batchnorm:
                # BN
                h = h.permute(1, 2, 0).contiguous()  # h(bsz, dim_emb, nb_nodes)
                h = self.norm2_layers[i](h)  # h(bsz, dim_emb, nb_nodes)
                h = h.permute(2, 0, 1).contiguous()  # h(nb_nodes, bsz, dim_emb)
            else:
                # LN
                h = self.norm2_layers[i](h)
        h = h.transpose(0, 1)  # h(bsz, nb_nodes, dim_emb)
        return h, score


def myMHA(Q, K, V, nb_heads, mask=None, clip_value=None):
    bsz, nb_nodes, emd_dim = K.size()
    if nb_heads > 1:
        Q = Q.transpose(1, 2).contiguous()  # Q(bsz, dim_emb, 1)
        Q = Q.view(bsz * nb_heads, emd_dim // nb_heads, 1)  # Q(bsz*nb_heads, dim_emb//nb_heads, 1)
        Q = Q.transpose(1, 2).contiguous()  # Q(bsz*nb_heads, 1, dim_emb//nb_heads)
        K = K.transpose(1, 2).contiguous()  # K(bsz, dim_emb, nb_nodes+1)
        K = K.view(bsz * nb_heads, emd_dim // nb_heads,
                   nb_nodes)  # K(bsz*nb_heads, dim_emb//nb_heads, nb_nodes+1)
        K = K.transpose(1, 2).contiguous()  # K(bsz*nb_heads, nb_nodes+1, dim_emb//nb_heads)
        V = V.transpose(1, 2).contiguous()  # V(bsz, dim_emb, nb_nodes+1)
        V = V.view(bsz * nb_heads, emd_dim // nb_heads,
                   nb_nodes)  # V(bsz*nb_heads, dim_emb//nb_heads, nb_nodes+1)
        V = V.transpose(1, 2).contiguous()  # V(bsz*nb_heads, nb_nodes+1, dim_emb//nb_heads)
    attn_weights = torch.bmm(Q, K.transpose(1, 2) / Q.size(-1) ** 0.5)
    if clip_value is not None:
        attn_weights = clip_value * torch.tanh(attn_weights)
    if mask is not None:
        if nb_heads > 1:
            mask = torch.repeat_interleave(mask, repeats=nb_heads, dim=0)  # mask(bsz*nb_heads, nb_nodes+1)
        # attn_weights(bsz*nb_heads, 1, nb_nodes+1)
        attn_weights = attn_weights.masked_fill(mask.unsqueeze(1), float('-1e9'))
    attn_weights = torch.softmax(attn_weights, dim=-1)  # attn_weights(bsz*nb_heads, 1, nb_nodes+1)
    attn_output = torch.bmm(attn_weights, V)  # attn_output(bsz*nb_heads, 1, dim_emb//nb_heads)
    if nb_heads > 1:
        attn_output = attn_output.transpose(1, 2).contiguous()  # attn_output(bsz*nb_heads, dim_emb//nb_heads, 1)
        attn_output = attn_output.view(bsz, emd_dim, 1)  # attn_output(bsz, dim_emb, 1)
        attn_output = attn_output.transpose(1, 2).contiguous()  # attn_output(bsz, 1, dim_emb)
        attn_weights = attn_weights.view(bsz, nb_heads, 1, nb_nodes)  # attn_weights(bsz, nb_heads, 1, nb_nodes+1)
        attn_weights = attn_weights.mean(dim=1)  # attn_weights(bsz, 1, nb_nodes+1)
    return attn_output, attn_weights  # attn_output(bsz, 1, dim_emb)  attn_weights(bsz, 1, nb_nodes+1)


def final_myMHA(Q, K, V, nb_heads, mask=None, clip_value=None):
    bsz, nb_nodes, emd_dim = K.size()
    if nb_heads > 1:
        Q = Q.transpose(1, 2).contiguous()  # Q(bsz, dim_emb, 1)
        Q = Q.view(bsz * nb_heads, emd_dim // nb_heads, 1)  # Q(bsz*nb_heads, dim_emb//nb_heads, 1)
        Q = Q.transpose(1, 2).contiguous()  # Q(bsz*nb_heads, 1, dim_emb//nb_heads)
        K = K.transpose(1, 2).contiguous()  # K(bsz, dim_emb, nb_nodes+1)
        K = K.view(bsz * nb_heads, emd_dim // nb_heads,
                   nb_nodes)  # K(bsz*nb_heads, dim_emb//nb_heads, nb_nodes+1)
        K = K.transpose(1, 2).contiguous()  # K(bsz*nb_heads, nb_nodes+1, dim_emb//nb_heads)
        V = V.transpose(1, 2).contiguous()  # V(bsz, dim_emb, nb_nodes+1)
        V = V.view(bsz * nb_heads, emd_dim // nb_heads,
                   nb_nodes)  # V(bsz*nb_heads, dim_emb//nb_heads, nb_nodes+1)
        V = V.transpose(1, 2).contiguous()  # V(bsz*nb_heads, nb_nodes+1, dim_emb//nb_heads)
    attn_weights = torch.bmm(Q, K.transpose(1, 2) / Q.size(-1) ** 0.5)
    if clip_value is not None:
        attn_weights = clip_value * torch.tanh(attn_weights)
    if mask is not None:
        if nb_heads > 1:
            mask = torch.repeat_interleave(mask, repeats=nb_heads, dim=0)  # mask(bsz*nb_heads, nb_nodes+1)
        attn_weights = attn_weights.masked_fill(mask.unsqueeze(1), float('-1e9'))
    attn_weights = torch.softmax(attn_weights / 0.1, dim=-1)  # attn_weights(bsz*nb_heads, 1, nb_nodes+1)
    attn_output = torch.bmm(attn_weights, V)  # attn_output(bsz*nb_heads, 1, dim_emb//nb_heads)
    if nb_heads > 1:
        attn_output = attn_output.transpose(1, 2).contiguous()  # attn_output(bsz*nb_heads, dim_emb//nb_heads, 1)
        attn_output = attn_output.view(bsz, emd_dim, 1)  # attn_output(bsz, dim_emb, 1)
        attn_output = attn_output.transpose(1, 2).contiguous()  # attn_output(bsz, 1, dim_emb)
        attn_weights = attn_weights.view(bsz, nb_heads, 1, nb_nodes)  # attn_weights(bsz, nb_heads, 1, nb_nodes+1)
        attn_weights = attn_weights.mean(dim=1)  # attn_weights(bsz, 1, nb_nodes+1)
    return attn_output, attn_weights  # attn_output(bsz, 1, dim_emb)  attn_weights(bsz, 1, nb_nodes+1)


class AutoRegressiveDecoderLayer(nn.Module):
    def __init__(self, dim_emb, nb_heads):
        super(AutoRegressiveDecoderLayer, self).__init__()
        self.dim_emb = dim_emb
        self.nb_heads = nb_heads
        self.Wq_selfatt = nn.Linear(dim_emb, dim_emb)
        self.Wk_selfatt = nn.Linear(dim_emb, dim_emb)
        self.Wv_selfatt = nn.Linear(dim_emb, dim_emb)
        self.W0_selfatt = nn.Linear(dim_emb, dim_emb)
        self.W0_att = nn.Linear(dim_emb, dim_emb)
        self.Wq_att = nn.Linear(dim_emb, dim_emb)
        self.W1_MLP = nn.Linear(dim_emb, dim_emb)
        self.W2_MLP = nn.Linear(dim_emb, dim_emb)
        self.BN_selfatt = nn.LayerNorm(dim_emb)
        self.BN_att = nn.LayerNorm(dim_emb)
        self.BN_MLP = nn.LayerNorm(dim_emb)
        self.K_sa = None
        self.V_sa = None

    def reset_selfatt_keys_values(self):
        self.K_sa = None
        self.V_sa = None

    def forward(self, h_t, K_att, V_att, mask):
        bsz = h_t.size(0)
        h_t = h_t.view(bsz, 1, self.dim_emb)  # h_t(bsz, 1, dim_emb)
        q_sa = self.Wq_selfatt(h_t)  # q_sa(bsz, 1, dim_emb)
        k_sa = self.Wk_selfatt(h_t)  # k_sa(bsz, 1, dim_emb)
        v_sa = self.Wv_selfatt(h_t)  # v_sa(bsz, 1, dim_emb)
        if self.K_sa is None:
            self.K_sa = k_sa  # self.K_sa=(bsz, 1, dim_emb)
            self.V_sa = v_sa  # self.V_sa=(bsz, 1, dim_emb)
        else:
            self.K_sa = torch.cat([self.K_sa, k_sa], dim=1)
            self.V_sa = torch.cat([self.V_sa, v_sa], dim=1)
        h_t = h_t + self.W0_selfatt(myMHA(q_sa, self.K_sa, self.V_sa, self.nb_heads)[0])  # h_t(bsz, 1, dim_emb)
        h_t = self.BN_selfatt(h_t.squeeze())  # h_t(bsz, dim_emb)
        h_t = h_t.view(bsz, 1, self.dim_emb)  # h_t(bsz, 1, dim_emb)
        q_a = self.Wq_att(h_t)  # h_t(bsz, 1, dim_emb)
        h_t = h_t + self.W0_att(myMHA(q_a, K_att, V_att, self.nb_heads, mask)[0])  # h_t(bsz, 1, dim_emb)
        h_t = self.BN_att(h_t.squeeze())  # h_t(bsz, dim_emb)
        h_t = h_t.view(bsz, 1, self.dim_emb)  # h_t(bsz, 1, dim_emb)
        h_t = h_t + self.W2_MLP(torch.relu(self.W1_MLP(h_t)))
        h_t = self.BN_MLP(h_t.squeeze(1))  # h_t(bsz, dim_emb)
        return h_t


class Transformer_decoder_net(nn.Module):
    def __init__(self, dim_emb, nb_heads, nb_layers_decoder):
        super(Transformer_decoder_net, self).__init__()
        self.dim_emb = dim_emb
        self.nb_heads = nb_heads
        self.nb_layers_decoder = nb_layers_decoder
        self.decoder_layers = nn.ModuleList([AutoRegressiveDecoderLayer(dim_emb, nb_heads)
                                             for _ in range(nb_layers_decoder - 1)])
        self.Wq_final = nn.Linear(dim_emb, dim_emb)

    def reset_selfatt_keys_values(self):
        for n in range(self.nb_layers_decoder - 1):
            self.decoder_layers[n].reset_selfatt_keys_values()

    def forward(self, h_t, K_att, V_att, mask):
        for layer in range(self.nb_layers_decoder):
            K_att_l = K_att[:, :, layer * self.dim_emb:(layer + 1) * self.dim_emb].contiguous()
            V_att_l = V_att[:, :, layer * self.dim_emb:(layer + 1) * self.dim_emb].contiguous()
            if layer < self.nb_layers_decoder - 1:
                h_t = self.decoder_layers[layer](h_t, K_att_l, V_att_l, mask)
            else:
                q_final = self.Wq_final(h_t)
                bsz = h_t.size(0)
                q_final = q_final.view(bsz, 1, self.dim_emb)
                # attn_weight = myMHA(q_final, K_att_l, V_att_l, 1, mask, 10)[1]
                attn_weight = final_myMHA(q_final, K_att_l, V_att_l, 1, mask, 10)[1]
        prob_next_node = attn_weight.squeeze(1)
        return prob_next_node


def generate_positional_encoding(d_model, max_len):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class AR_TSP_net(nn.Module):
    def __init__(self, dim_input_nodes, dim_emb, dim_ff, nb_layers_encoder, nb_layers_decoder, nb_heads, max_len_PE,
                 batch_norm=True):
        super(AR_TSP_net, self).__init__()
        self.dim_emb = dim_emb
        self.input_emb = nn.Linear(dim_input_nodes, dim_emb)
        self.encoder = Transformer_encoder_net(nb_layers_encoder, dim_emb, nb_heads, dim_ff, batch_norm)
        self.start_placeholder = nn.Parameter(torch.randn(dim_emb))
        self.decoder = Transformer_decoder_net(dim_emb, nb_heads, nb_layers_decoder)
        self.WK_att_decoder = nn.Linear(dim_emb, nb_layers_decoder * dim_emb)
        self.WV_att_decoder = nn.Linear(dim_emb, nb_layers_decoder * dim_emb)
        self.PE = generate_positional_encoding(dim_emb, max_len_PE)
        self.nb_heads = nb_heads
        self.nb_layers_decoder = nb_layers_decoder

    def forward(self, x):
        # x(bsz, nb_nodes, 2)
        bsz = x.shape[0]
        nb_nodes = x.shape[1]
        zero_to_bsz = torch.arange(bsz, device=x.device)  # [0, 1, ..., bsz-1]
        h = self.input_emb(x)  # h(bsz, nb_nodes, dim_emb)
        h = torch.cat([h, self.start_placeholder.repeat(bsz, 1, 1)], dim=1)  # h(bsz, nb_nodes+1, dim_emb)]
        h_encoder, score = self.encoder(h)  # h_encoder(bsz, nb_nodes+1, dim_emb)
        tours, sumLogProbOfActions = [], []
        K_att_decoder = self.WK_att_decoder(h_encoder)  # K_att_decoder(bsz, nb_nodes+1, dim_emb*nb_layer_decoder)
        V_att_decoder = self.WV_att_decoder(h_encoder)  # V_att_decoder(bsz, nb_nodes+1, dim_emb*nb_layers_decoder)
        self.PE = self.PE.to(x.device)
        idx_start_placehoder = torch.Tensor([nb_nodes]).long().repeat(bsz).to(x.device)  # idx_start_placehoder(bsz,)
        h_start = h_encoder[zero_to_bsz, idx_start_placehoder, :] + self.PE[0].repeat(bsz, 1)  # h_start(bsz, dim_emb)
        mask_visited_nodes = torch.zeros(bsz, nb_nodes + 1, device=x.device).bool()  # False
        mask_visited_nodes[zero_to_bsz, idx_start_placehoder] = True  
        self.decoder.reset_selfatt_keys_values()

        h_t = h_start
        prob_next_node_all = []
        for t in range(nb_nodes):
            prob_next_node = self.decoder(h_t, K_att_decoder, V_att_decoder, mask_visited_nodes)
            prob_next_node_all.append(prob_next_node)
            idx = torch.argmax(prob_next_node, dim=1)  # idx(bsz,)
            h_t = h_encoder[zero_to_bsz, idx, :]  # h_t(bsz, dim_emb)
            h_t = h_t + self.PE[t + 1].expand(bsz, self.dim_emb)
            tours.append(idx)
            mask_visited_nodes = mask_visited_nodes.clone()
            mask_visited_nodes[zero_to_bsz, idx] = True
        prob_next_node_all = torch.stack(prob_next_node_all, dim=1)
        tours = torch.stack(tours, dim=1)  # tours(bsz, nb_nodes)
        return tours, prob_next_node_all

import torch
import torch.nn as nn
import torch.nn.functional as F


# This is from publicly available POMO model.
class TSPModel(nn.Module):

    def __init__(self, model_args):
        super().__init__()
        self.model_args = model_args

        self.encoder = TSP_Encoder(model_args)
        self.decoder = TSP_Decoder(model_args)
        self.encoded_nodes = None
        # shape: (batch, problem, EMBEDDING_DIM)

    def pre_forward(self, problems):
        self.encoded_nodes = self.encoder(problems)
        # shape: (batch, problem, EMBEDDING_DIM)
        self.decoder.set_kv(self.encoded_nodes)

    def forward(self, problems):
        selected_count = 0
        problem_size = problems.size(1)
        # aug_problems = augment_xy_data_by_8_fold(problems)
        aug_problems = problems
        self.pre_forward(aug_problems)

        batch_size = aug_problems.size(0)
        pomo_size = aug_problems.size(1)
        ZERO_TO_BATCH = torch.arange(batch_size, device=problems.device)
        BATCH_IDX = torch.arange(batch_size, device=problems.device)[:, None].expand(batch_size, pomo_size)
        POMO_IDX = torch.arange(pomo_size, device=problems.device)[None, :].expand(batch_size, pomo_size)

        selected_node_list = torch.zeros((batch_size, pomo_size, 0), dtype=torch.long, device=problems.device)
        ninf_mask = torch.zeros((batch_size, pomo_size, problem_size), device=problems.device)

        selected = torch.arange(pomo_size, device=problems.device)[None, :].expand(batch_size, pomo_size)
        # prob = torch.ones(size=(batch_size, pomo_size), device=problems.device)

        encoded_first_node = _get_encoding(self.encoded_nodes, selected)
        # shape: (batch, pomo, embedding)
        self.decoder.set_q1(encoded_first_node)

        selected_count += 1
        current_node = selected
        selected_node_list = torch.cat((selected_node_list, current_node[:, :, None]), dim=2)

        ninf_mask[BATCH_IDX, POMO_IDX, current_node] = float('-inf')

        initial_probs = torch.eye(pomo_size, device=problems.device).unsqueeze(0).repeat_interleave(repeats=batch_size,
                                                                                                    dim=0)
        prob_next_node_all = [initial_probs]

        while selected_count != problem_size:
            encoded_last_node = _get_encoding(self.encoded_nodes, current_node)
            # shape: (batch, pomo, embedding)
            probs = self.decoder(encoded_last_node, ninf_mask=ninf_mask)
            # shape: (batch, pomo, problem)
            #
            # if self.training or self.model_args.eval_type == 'softmax':
            #     while True:
            #         selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1).squeeze(dim=1)\
            #             .reshape(batch_size, pomo_size)
            #         # shape: (batch, pomo)
            #
            #         prob = probs[BATCH_IDX, POMO_IDX, selected].reshape(batch_size, pomo_size)
            #         # shape: (batch, pomo)
            #
            #         if (prob != 0).all():
            #             break
            # else:
            # selected = probs.argmax(dim=2)
            # shape: (batch, pomo)
            # prob = None

            selected = probs.argmax(dim=2)

            selected_count += 1
            current_node = selected
            selected_node_list = torch.cat((selected_node_list, current_node[:, :, None]), dim=2)
            ninf_mask[BATCH_IDX, POMO_IDX, current_node] = float('-inf')
            prob_next_node_all.append(probs)

        prob_next_node_all = torch.stack(prob_next_node_all, dim=-2)
        # reward = _get_travel_distance(selected_node_list, aug_problems)
        # max_pomo_reward, max_idx = reward.min(dim=-1)
        # max_select_node_list = selected_node_list[ZERO_TO_BATCH, max_idx]
        # max_prob_next_node_all = prob_next_node_all[ZERO_TO_BATCH, max_idx]
        # return max_select_node_list, max_prob_next_node_all
        return selected_node_list, prob_next_node_all


def _get_travel_distance(selected_node_list, problems):
    batch_size, pomo_size = selected_node_list.size(0), selected_node_list.size(1)
    problem_size = pomo_size
    gathering_index = selected_node_list.unsqueeze(3).expand(batch_size, -1, problem_size, 2)
    # shape: (batch, pomo, problem, 2)
    seq_expanded = problems[:, None, :, :].expand(batch_size, pomo_size, problem_size, 2)

    ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
    # shape: (batch, pomo, problem, 2)

    rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
    segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
    # shape: (batch, pomo, problem)

    travel_distances = segment_lengths.sum(2)
    # shape: (batch, pomo)
    return travel_distances


def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_nodes


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


########################################
# ENCODER
########################################

class TSP_Encoder(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.model_args = model_args
        embedding_dim = self.model_args.embedding_dim
        encoder_layer_num = self.model_args.encoder_layer_num

        self.embedding = nn.Linear(2, embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(model_args) for _ in range(encoder_layer_num)])

    def forward(self, data):
        # data.shape: (batch, problem, 2)

        embedded_input = self.embedding(data)
        # shape: (batch, problem, embedding)

        out = embedded_input
        for layer in self.layers:
            out = layer(out)

        return out


class EncoderLayer(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.model_args = model_args
        embedding_dim = self.model_args.embedding_dim
        head_num = self.model_args.head_num
        qkv_dim = self.model_args.qkv_dim

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.addAndNormalization1 = Add_And_Normalization_Module(model_args)
        self.feedForward = Feed_Forward_Module(model_args)
        self.addAndNormalization2 = Add_And_Normalization_Module(model_args)

    def forward(self, input1):
        # input.shape: (batch, problem, EMBEDDING_DIM)
        head_num = self.model_args.head_num

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)

        out_concat = multi_head_attention(q, k, v)
        # shape: (batch, problem, HEAD_NUM*KEY_DIM)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem, EMBEDDING_DIM)

        out1 = self.addAndNormalization1(input1, multi_head_out)
        out2 = self.feedForward(out1)
        out3 = self.addAndNormalization2(out1, out2)

        return out3
        # shape: (batch, problem, EMBEDDING_DIM)


########################################
# DECODER
########################################

class TSP_Decoder(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.model_args = model_args
        embedding_dim = self.model_args.embedding_dim
        head_num = self.model_args.head_num
        qkv_dim = self.model_args.qkv_dim

        self.Wq_first = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_last = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        self.q_first = None  # saved q1, for multi-head attention

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem, embedding)
        head_num = self.model_args.head_num

        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem)

    def set_q1(self, encoded_q1):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_args.head_num

        self.q_first = reshape_by_heads(self.Wq_first(encoded_q1), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def forward(self, encoded_last_node, ninf_mask):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # ninf_mask.shape: (batch, pomo, problem)

        head_num = self.model_args.head_num

        #  Multi-Head Attention
        #######################################################
        q_last = reshape_by_heads(self.Wq_last(encoded_last_node), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)

        q = self.q_first + q_last
        # shape: (batch, head_num, pomo, qkv_dim)

        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        # shape: (batch, pomo, head_num*qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, pomo, problem)

        sqrt_embedding_dim = self.model_args.sqrt_embedding_dim
        logit_clipping = self.model_args.logit_clipping

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, problem)

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked / 0.1, dim=2)
        # shape: (batch, pomo, problem)

        return probs


########################################
# NN SUB CLASS / FUNCTIONS
########################################

def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    # score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)

    # score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    score_scaled = torch.matmul(q, k.transpose(2, 3) / key_dim ** 0.5)
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)

    return out_concat


class Add_And_Normalization_Module(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        embedding_dim = model_args.embedding_dim
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        added = input1 + input2
        # shape: (batch, problem, embedding)

        transposed = added.transpose(1, 2)
        # shape: (batch, embedding, problem)

        normalized = self.norm(transposed)
        # shape: (batch, embedding, problem)

        back_trans = normalized.transpose(1, 2)
        # shape: (batch, problem, embedding)

        return back_trans


class Feed_Forward_Module(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        embedding_dim = model_args.embedding_dim
        ff_hidden_dim = model_args.ff_hidden_dim

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))

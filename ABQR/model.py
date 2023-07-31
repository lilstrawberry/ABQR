import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

import glo

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_feature, out_feature, alpha, dropout):
        super(GraphAttentionLayer, self).__init__()

        self.out_feature = out_feature

        self.W = nn.Parameter(torch.empty((in_feature, out_feature)))
        self.a = nn.Parameter(torch.empty((2 * out_feature, 1)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(p=dropout)
        self.leakRelu = nn.LeakyReLU(alpha)

    def prepare(self, h):
        # h: n out_feature

        h_i = torch.matmul(h, self.a[:self.out_feature, :])
        # n 1
        h_j = torch.matmul(h, self.a[self.out_feature:, :])
        # n 1
        e = h_i + h_j.T
        # n n
        e = self.leakRelu(e)
        return e

    def forward(self, x, adj):
        #   x: n in_feature
        # adj: n n

        # adj = adj.to_dense()
        adj = glo.get_value('regist_pos_matrix').to_dense()




        batch_size = 10000
        device = x.device
        num_nodes = x.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        indices = torch.arange(0, num_nodes).to(device)

        h = torch.matmul(x, self.W)
        # n out_feature
        dd = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]

            h_i = torch.matmul(h[mask], self.a[:self.out_feature, :])  # mask 1
            h_j = torch.matmul(h, self.a[self.out_feature:, :])  # n 1
            e = h_i + h_j.T  # mask n
            e = self.leakRelu(e)

            now_mask = (adj[mask] <= 0)  # mask n
            attn = torch.masked_fill(e, now_mask, -1e9)

            # attn = e

            attn = torch.softmax(attn, dim=-1)
            # mask n
            res = torch.matmul(attn, h)
            res = F.elu(res)  # mask n

            res = self.dropout(res)

            dd.append(res)
        dd = torch.vstack(dd)

        return dd

class GCNConv(nn.Module):
    def __init__(self, in_dim, out_dim, p):
        super(GCNConv, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.w = nn.Parameter(torch.rand((in_dim, out_dim)))
        nn.init.xavier_uniform_(self.w)

        self.b = nn.Parameter(torch.rand((out_dim)))
        nn.init.zeros_(self.b)

        self.dropout = nn.Dropout(p=p)

    def forward(self, x, adj):
        adj = glo.get_value('matrix')

        x = self.dropout(x)
        x = torch.matmul(x, self.w)
        x = torch.sparse.mm(adj.float(), x)
        x = x + self.b
        return x

class MLP_Predictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP_Predictor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=True),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, output_size, bias=True)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def forward(self, x):
        return self.net(x)

def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1),),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0
    return x

def normalize_graph(A):
    eps = 1e-8
    A = A.to_dense()
    deg_inv_sqrt = (A.sum(dim=-1).clamp(min=0.) + eps).pow(-0.5)
    if A.size()[0] != A.size()[1]:
        A = deg_inv_sqrt.unsqueeze(-1) * (deg_inv_sqrt.unsqueeze(-1) * A)
    else:
        A = deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(-2)
    return A.to_sparse()

def drop_adj(edge_index, drop_prob):
    begin_size = edge_index.size()
    use_edge = edge_index._indices()
    drop_mask = torch.empty(
        (use_edge.size(1),),
        dtype=torch.float32,
        device=use_edge.device).uniform_(0, 1) >= drop_prob
    y = use_edge.clone()
    res = y[:, drop_mask]
    values = torch.ones(res.shape[1]).to(device)
    size = begin_size
    graph = torch.sparse.FloatTensor(res, values, size)
    graph = normalize_graph(graph)
    return graph

def augment_graph(x, feat_drop, edge, edge_drop):
    drop_x = drop_feature(x, feat_drop)
    drop_edge = drop_adj(edge, edge_drop)
    return drop_x, drop_edge

def drop_adj_gat(edge_index, drop_prob):
    begin_size = edge_index.size()
    use_edge = edge_index._indices()
    drop_mask = torch.empty(
        (use_edge.size(1),),
        dtype=torch.float32,
        device=use_edge.device).uniform_(0, 1) >= drop_prob
    y = use_edge.clone()
    res = y[:, drop_mask]
    values = torch.ones(res.shape[1]).to(device)
    size = begin_size
    graph = torch.sparse.FloatTensor(res, values, size)
    return graph

def augment_graph_gat(x, feat_drop, edge, edge_drop):
    drop_x = drop_feature(x, feat_drop)
    drop_edge = drop_adj_gat(edge, edge_drop)
    return drop_x, drop_edge

class CosineDecayScheduler:
    def __init__(self, max_val, warmup_steps, total_steps):
        self.max_val = max_val
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get(self, step):
        if step < self.warmup_steps:
            return self.max_val * step / self.warmup_steps
        elif self.warmup_steps <= step <= self.total_steps:
            return self.max_val * (1 + np.cos((step - self.warmup_steps) * np.pi /
                                              (self.total_steps - self.warmup_steps))) / 2
        else:
            raise ValueError('Step ({}) > total number of steps ({}).'.format(step, self.total_steps))

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()
    return loss

class BGRL(nn.Module):
    def __init__(self, d, p, drop_feat1, drop_feat2, drop_edge1, drop_edge2):
        super(BGRL, self).__init__()

        self.drop_feat1, self.drop_feat2, self.drop_edge1, self.drop_edge2 = drop_feat1, drop_feat2, drop_edge1, drop_edge2

        # self.online_encoder = GraphAttentionLayer(d, d, 0.2, p)

        self.online_encoder = GCNConv(d, d, p)

        self.decoder = GCNConv(d, d, p)

        self.predictor = MLP_Predictor(d, d, d)

        self.target_encoder = copy.deepcopy(self.online_encoder)

        # self.GMAE = GMAE(d, p, mask_rate, alpha)

        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)

        # self.target_encoder.reset_parameters()

        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.enc_mask_token = nn.Parameter(torch.zeros(1, d))
        self.encoder_to_decoder = nn.Linear(d, d, bias=False)

    def encoding_mask_noise(self, x, mask_rate=0.3):
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)

        mask_nodes = perm[:num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        num_noise_nodes = int(0.1 * num_mask_nodes)
        perm_mask = torch.randperm(num_mask_nodes, device=x.device)
        token_nodes = mask_nodes[perm_mask[: int(0.9 * num_mask_nodes)]]
        noise_nodes = mask_nodes[perm_mask[-int(0.1 * num_mask_nodes):]]
        noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

        out_x = x.clone()
        out_x[token_nodes] = 0.0
        out_x[noise_nodes] = x[noise_to_be_chosen]

        out_x[token_nodes] += self.enc_mask_token

        return out_x, mask_nodes, keep_nodes

    def trainable_parameters(self):
        r"""Returns the parameters that will be updated via an optimizer."""
        return list(self.online_encoder.parameters()) + list(self.predictor.parameters())

    @torch.no_grad()
    def update_target_network(self, mm):
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm)

    def project(self, z):
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def compute_batch_loss(x, y):

        z1 = self.project(z1)
        z2 = self.project(z2)

        c1 = F.normalize(z1, dim=-1, p=2)
        c2 = F.normalize(z2, dim=-1, p=2)

        batch_size = 15000
        device = x.device
        num_nodes = x.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            batch_pos_matrix = glo.get_value('regist_pos_matrix')[mask]  # batch n
            item_loss = torch.matmul(c1[mask], c2.T)  # batch n
            item_loss = 2 - 2 * item_loss  # batch n
            need_loss = item_loss * batch_pos_matrix  # batch n

            need_sum = need_loss.sum(dim=-1, keepdims=True)  # batch 1
            # need_mean = need_sum / (item_pos_sum[mask] + 1e-8)  # batch n
            need_mean = need_sum

            losses.append(need_mean)

        return -torch.cat(losses).mean()

    def getGraphMAE_loss(self, x, adj):
        mask_rate = 0.3
        use_x, mask_nodes, keep_nodes = self.encoding_mask_noise(x, mask_rate)

        enc_rep = self.online_encoder(use_x, adj)

        rep = self.encoder_to_decoder(enc_rep)

        rep[mask_nodes] = 0

        recon = self.decoder(rep, adj)

        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]
        loss = sce_loss(x_rec, x_init, 3)
        return enc_rep, loss

    def forward(self, x, adj, perb=None):

        #         if perb is not None:
        #             x1, adj1 = x, adj
        #             x2, adj2 = x1 + perb, copy.deepcopy(adj1)
        #             embed = x2 + self.online_encoder(x2, regist_pos_matrix)
        #         else:
        #             x1, adj1 = augment_graph_gat(x, self.drop_feat1, adj, self.drop_edge1)
        #             x2, adj2 = augment_graph_gat(x, self.drop_feat2, adj, self.drop_edge2)
        #             embed = x + self.online_encoder(x, regist_pos_matrix)
        if perb is None:
            return x + self.online_encoder(x, glo.get_value('regist_pos_matrix')), 0

        x1, adj1 = x, copy.deepcopy(adj)
        x2, adj2 = x + perb, copy.deepcopy(adj)

        embed = x2 + self.online_encoder(x2, adj2)

        online_x = self.online_encoder(x1, adj1)
        online_y = self.online_encoder(x2, adj2)

        with torch.no_grad():
            #             detach_gat = self.target_encoder(x, adj).detach()
            #             target_y = self.project(detach_gat)
            #             target_x = self.project(detach_gat + perb)
            target_y = self.target_encoder(x1, adj1).detach()
            target_x = self.target_encoder(x2, adj2).detach()

        online_x = self.predictor(online_x)
        online_y = self.predictor(online_y)

        loss = (loss_fn(online_x, target_x) + loss_fn(online_y, target_y)).mean()

        return embed, loss


class ABQR(nn.Module):
    def __init__(self, skill_max, drop_feat1, drop_feat2, drop_edge1, drop_edge2, positive_matrix, pro_max, lamda,
                 contrast_batch, tau, lamda1,
                 top_k, d, p, head=1, graph_aug='knn', gnn_mode='gcn'):
        super(ABQR, self).__init__()

        self.lamda = lamda

        self.head = head

        # self.gcl = Multi_level_GCL(positive_matrix, contrast_batch, tau, lamda1, top_k, d, p, head, graph_aug, gnn_mode)

        self.gcl = BGRL(d, p, drop_feat1, drop_feat2, drop_edge1, drop_edge2)

        self.gcn = GCNConv(d, d, p)

        self.pro_embed = nn.Parameter(torch.ones((pro_max, d)))
        nn.init.xavier_uniform_(self.pro_embed)

        self.ans_embed = nn.Embedding(2, d)

        self.attn = nn.MultiheadAttention(d, 8, dropout=p)
        self.attn_dropout = nn.Dropout(p)
        self.attn_layer_norm = nn.LayerNorm(d)

        self.FFN = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(d, d),
            nn.Dropout(p),
        )
        self.FFN_layer_norm = nn.LayerNorm(d)

        self.pred = nn.Linear(d, 1)

        self.lstm = nn.LSTM(d, d, batch_first=True)

        self.origin_lstm = nn.LSTM(2 * d, 2 * d, batch_first=True)
        self.oppo_lstm = nn.LSTM(d, d, batch_first=True)

        self.origin_lstm2 = nn.LSTM(d, d, batch_first=True)
        self.oppo_lstm2 = nn.LSTM(d, d, batch_first=True)

        self.dropout = nn.Dropout(p=p)

        self.origin_out = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.Linear(d, 1)
        )
        self.oppo_out = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.ReLU(),
            nn.Linear(d, 1)
        )

        self.encoder_lstm = nn.LSTM(d, d, batch_first=True)
        self.decoder_lstm = nn.LSTM(d, d, batch_first=True)

        self.enc_token = nn.Parameter(torch.rand(1, d))
        self.enc_dec = nn.Linear(d, d)

        self.classify = nn.Sequential(
            nn.Linear(d, skill_max)
        )
        # nn.Linear(d, skill_max)

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def compute_loss(self, pro_clas, true_clas):
        pro_clas = pro_clas.view(-1, pro_clas.shape[-1])
        true_clas = true_clas.view(-1)
        loss = F.cross_entropy(pro_clas, true_clas)
        return loss

    def encoding_mask_seq(self, x, mask_rate=0.3):
        # batch seq d
        num_nodes = x.shape[1]
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[:num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]
        out_x = x.clone()
        token_nodes = mask_nodes
        out_x[:, token_nodes] = self.enc_token
        return out_x, mask_nodes, keep_nodes

    def forward(self, last_pro, last_ans, last_skill, next_pro,
                next_skill, matrix, perb=None):

        device = last_pro.device

        batch, seq = last_pro.shape[0], last_pro.shape[1]

        pro_embed, contrast_loss = self.gcl(self.pro_embed, matrix, perb)

        contrast_loss = 0.1 * contrast_loss

        last_pro_embed = F.embedding(last_pro, pro_embed)
        next_pro_embed = F.embedding(next_pro, pro_embed)

        ans_embed = self.ans_embed(last_ans)

        X = last_pro_embed + ans_embed

        X = self.dropout(X)

        X, _ = self.lstm(X)

        #         origin_X, oppo_X = self.ContrastiveAttn(X, X, X)
        #         origin_X = origin_X + X

        P = torch.sigmoid(self.origin_out(torch.cat([X, next_pro_embed], dim=-1))).squeeze(-1)
        false_P = P

        return P, false_P, contrast_loss
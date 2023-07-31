import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import copy
import torch.utils.data as data
import math
import argparse

import glo
from model import ABQR
from run import run_epoch

data2path = {
    'static11': {
        'ques_skill_path': 'data/static11/static11_ques_skill.csv',
        'train_path': 'data/static11/static11_train_question.txt',
        'test_path': 'data/static11/static11_test_question.txt',
        'train_skill_path': 'data/static11/static11_train_skill.txt',
        'test_skill_path': 'data/static11/static11_test_skill.txt',
        'pre_load_gcn': 'data/static11/static_ques_skill_gcn_adj.pt',
        'pre_load_gat': 'data/static11/STATIC11_GAT_edge_adj',
        'positive_matrix_path': 'data/static11/Static11_Q_Q_sparse.pt',
        'skill_max': 106
    }
}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = 'static11'

ques_skill_path = data2path[dataset]['ques_skill_path']
train_path = data2path[dataset]['train_path']
test_path = data2path[dataset]['test_path']
train_skill_path = data2path[dataset]['train_skill_path']
test_skill_path = data2path[dataset]['test_skill_path']

pre_load_gcn = data2path[dataset]['pre_load_gcn']
pre_load_gat = data2path[dataset]['pre_load_gat']
positive_matrix_path = data2path[dataset]['positive_matrix_path']
skill_max = data2path[dataset]['skill_max']

positive_matrix = torch.load(positive_matrix_path).to(device)

pro_max = 1 + max(pd.read_csv(ques_skill_path).values[:, 0])
lamda = 5
contrast_batch = 1000
tau = 0.8
lamda1 = 20

p = 0.4

d = 128

head = 8
graph_aug = 'knn'
gnn_mode = 'gcn'
learning_rate = 0.002
epochs = 50
batch_size = 80
min_seq = 3

max_seq = 200
grad_clip = 15.0
patience = 30

if gnn_mode == 'gcn':
    matrix = torch.load(pre_load_gcn).to(device)
    num_edge = matrix._nnz()

elif gnn_mode == 'gat':
    matrix = torch.load(pre_load_gat).to(device)
    num_edge = matrix.shape[1]
top_k = 25

drop_feat1 = 0.2
drop_feat2 = 0.3
drop_edge1 = 0.3
drop_edge2 = 0.2

lr = 5e-3

mm = 0.99

lr_warmup_steps = 1000
weight_decay = 1e-5
steps = 10000

mask_rate = 0.75
alpha = 3

regist_pos_matrix = torch.load(positive_matrix_path).to(device)

glo._init()
glo.set_value('d', d)
glo.set_value('mm', mm)
glo.set_value('matrix', matrix)
glo.set_value('regist_pos_matrix', regist_pos_matrix)



avg_auc = 0
avg_acc = 0

for now_step in range(5):

    best_acc = 0
    best_auc = 0
    state = {'auc': 0, 'acc': 0, 'loss': 0}

    model = ABQR(skill_max, drop_feat1, drop_feat2, drop_edge1, drop_edge2, positive_matrix, pro_max, lamda,
                   contrast_batch, tau, lamda1,
                   top_k, d, p, head, graph_aug, gnn_mode)
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-5)
    # , weight_decay=1e-5
    one_p = 0

    for epoch in range(60):

        one_p += 1

        train_loss, train_acc, train_auc = run_epoch(train_skill_path, matrix, pro_max, train_path, batch_size,
                                                     True, min_seq, max_seq, model, optimizer, criterion, device,
                                                     grad_clip)
        print(f'epoch: {epoch}, train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, train_auc: {train_auc:.4f}')

        test_loss, test_acc, test_auc = run_epoch(test_skill_path, matrix, pro_max, test_path, batch_size, False,
                                                  min_seq, max_seq, model, optimizer, criterion, device, grad_clip)

        print(f'epoch: {epoch}, test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}, test_auc: {test_auc:.4f}')

        if test_auc >= best_auc:
            one_p = 0
            best_auc = test_auc
            best_acc = test_acc
            torch.save(model.state_dict(), f"./ASSIST17_GCN_ABQR_model{now_step}.pkl")
            state['auc'] = test_auc
            state['acc'] = test_acc
            state['loss'] = test_loss
            torch.save(state, f'./ASSIST17_GCN_ABQR_model{now_step}.ckpt')

    #         if one_p >= patience:
    #             break

    print(f'*******************************************************************************')
    print(f'best_acc: {best_acc:.4f}, best_auc: {best_auc:.4f}')
    print(f'*******************************************************************************')

    avg_auc += best_auc
    avg_acc += best_acc

avg_auc = avg_auc / 5
avg_acc = avg_acc / 5
print(f'*******************************************************************************')
print(f'*******************************************************************************')
print(f'*******************************************************************************')
print(f'*******************************************************************************')
print(f'*******************************************************************************')
print(f'final_avg_acc: {avg_acc:.4f}, final_avg_auc: {avg_auc:.4f}')
print(f'*******************************************************************************')
print(f'*******************************************************************************')
print(f'*******************************************************************************')
print(f'*******************************************************************************')
print(f'*******************************************************************************')

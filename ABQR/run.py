from sklearn import metrics
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

from load_data import getLoader
import glo

def run_epoch(skill_path, matrix, max_problem, path, batch_size, is_train, min_problem_num, max_problem_num,
              model, optimizer, criterion, device, grad_clip):
    total_correct = 0
    total_num = 0
    total_loss = []

    labels = []
    outputs = []

    if is_train:
        model.train()
    else:
        model.eval()

    data_loader = getLoader(skill_path, max_problem, path, batch_size, is_train, min_problem_num, max_problem_num)

    for i, data in tqdm(enumerate(data_loader), desc='加载中...'):
        # x_vec: batch n-1

        last_problem, last_ans, last_skill, next_skill, next_problem, next_ans, mask = data

        if is_train:

            forward = lambda perturb: model(last_problem, last_ans, last_skill, next_problem, next_skill, matrix,
                                            perturb)

            perturb_shape = (max_problem, glo.get_value('d'))

            step_size = 3e-2
            step_m = 3

            perturb = torch.FloatTensor(*perturb_shape).uniform_(-step_size, step_size).to(device)
            perturb.requires_grad_()

            predict, _, contrast_loss = forward(perturb)
            next_predict = torch.masked_select(predict, mask)
            next_true = torch.masked_select(next_ans, mask)
            loss = criterion(next_predict, next_true) + contrast_loss
            loss /= step_m

            optimizer.zero_grad()

            for _ in range(step_m - 1):
                loss.backward()
                perturb_data = perturb.detach() + step_size * torch.sign(perturb.grad.detach())
                perturb.data = perturb_data.data
                perturb.grad[:] = 0

                predict, _, contrast_loss = forward(perturb)
                next_predict = torch.masked_select(predict, mask)
                next_true = torch.masked_select(next_ans, mask)
                loss = criterion(next_predict, next_true) + contrast_loss
                loss /= step_m

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            model.gcl.update_target_network(glo.get_value('mm'))

            labels.extend(next_true.view(-1).data.cpu().numpy())
            outputs.extend(next_predict.view(-1).data.cpu().numpy())

            total_loss.append(loss.item())
            total_num += len(next_true)

            to_pred = (next_predict >= 0.5).long()
            total_correct += (next_true == to_pred).sum()

        else:
            with torch.no_grad():
                predict, _, _ = model(last_problem, last_ans, last_skill, next_problem, next_skill, matrix)
                next_predict = torch.masked_select(predict, mask)
                next_true = torch.masked_select(next_ans, mask)

                loss = criterion(next_predict, next_true)

                labels.extend(next_true.view(-1).data.cpu().numpy())
                outputs.extend(next_predict.view(-1).data.cpu().numpy())

                total_loss.append(loss.item())
                total_num += len(next_true)

                to_pred = (next_predict >= 0.5).long()
                total_correct += (next_true == to_pred).sum()

    avg_loss = np.average(total_loss)
    acc = total_correct * 1.0 / total_num
    auc = metrics.roc_auc_score(labels, outputs)
    return avg_loss, acc, auc
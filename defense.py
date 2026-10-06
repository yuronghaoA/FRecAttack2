import copy
import torch
from parse import args


def avg(param_list, m_item, m_user):
    flag = False
    number = 0

    if args.select_model in ['FedNCF', 'FedGMF', "FedMLP"]:
        gradient_item = torch.zeros(m_item, args.embedding_dim).to(args.device)
        gradient_user = None

        for parameter in param_list:
            [model_grad, item_grad, item] = parameter
            num = len(item)
            number += num
            if not flag:
                flag = True
                gradient_item[item, :] += item_grad * num
                gradient_model = []
                for i in range(len(model_grad)):
                    gradient_model.append(model_grad[i] * num)
            else:
                gradient_item[item, :] += item_grad * num
                for i in range(len(model_grad)):
                    gradient_model[i] += model_grad[i] * num
        gradient_item /= number

        for i in range(len(gradient_model)):
            gradient_model[i] = gradient_model[i] / number

    elif args.select_model in ['FedSoG', 'FedGNN']:
        gradient_item = torch.zeros(m_item, args.embedding_dim).to(args.device)
        gradient_user = torch.zeros(m_user, args.embedding_dim).to(args.device)
        for parameter in param_list:
            [model_grad, item_grad, user_grad, returned_items, returned_users] = parameter
            num = len(returned_items)
            number += num
            if not flag:
                flag = True
                gradient_model = []
                gradient_item[returned_items, :] += item_grad * num
                gradient_user[returned_users, :] += user_grad * num
                for i in range(len(model_grad)):
                    gradient_model.append(model_grad[i] * num)
            else:
                gradient_item[returned_items, :] += item_grad * num
                gradient_user[returned_users, :] += user_grad * num
                for i in range(len(model_grad)):
                    gradient_model[i] += model_grad[i] * num
        gradient_item /= number
        gradient_user /= number
        for i in range(len(gradient_model)):
            gradient_model[i] = gradient_model[i] / number

    return gradient_model, gradient_item, gradient_user


def ratio_defense(batch_clients_idx, dict_, param_list, m_item, pred_item, sample_user_emb, item_emb, model):
    pred_pop_item_AS, pred_unpop_item_AS, pred_pop_item_VB, pred_unpop_item_VB, pred_pop_item_RI = pred_item

    rating = []
    for user_emb_ in sample_user_emb:
        rating.append(model(user_emb_.repeat(m_item, 1), item_emb.weight))
    rating = sum(rating)

    T_ratio_pop_item_AS = []
    T_ratio_unpop_item_AS = []
    T_ratio_pop_item_VB = []
    T_ratio_unpop_item_VB = []
    T_ratio_pop_item_RI = []

    for i, parameter in enumerate(param_list):
        beni_user = []
        item_emb_ =copy.deepcopy(item_emb)
        model_ = copy.deepcopy(model)

        [model_grad, item_grad, item] = parameter
        inter_pop_item_AS = set(item.tolist()) & set(pred_pop_item_AS)
        inter_unpop_item_AS = set(item.tolist()) & set(pred_unpop_item_AS)
        inter_pop_item_VB = set(item.tolist()) & set(pred_pop_item_VB)
        inter_unpop_item_VB = set(item.tolist()) & set(pred_unpop_item_VB)
        inter_pop_item_RI = set(item.tolist()) & set(pred_pop_item_RI)

        # if intersection is null, then label the user is normal user
        if len(inter_pop_item_AS) <= 3 and len(inter_pop_item_VB) <= 3 and len(inter_pop_item_RI)<=3:
            beni_user.append(i)

        gradient_item = torch.zeros(m_item, args.embedding_dim).to(args.device)
        gradient_item[item, :] += item_grad
        item_emb_.weight.data.add_(gradient_item, alpha=-args.lr)
        ls_model_param = list(model_.parameters())
        for j in range(len(ls_model_param)):
            ls_model_param[j].data = ls_model_param[j].data - args.lr * model_grad[j]

        rating_new = []
        for user_emb_ in sample_user_emb:
            rating_new.append(model_(user_emb_.repeat(m_item, 1), item_emb_.weight))
        rating_new = sum(rating_new)

        diff = torch.sign(rating_new - rating)
        ratio_item = torch.tensor([[sum(diff[list(inter_pop_item_AS)] == 1)+1, len(inter_pop_item_AS)+1],
                                   [sum(diff[list(inter_unpop_item_AS)] == -1)+1, len(inter_unpop_item_AS)+1],
                                   [sum(diff[list(inter_pop_item_VB)] == 1)+1, len(inter_pop_item_VB)+1],
                                   [sum(diff[list(inter_unpop_item_VB)] == -1)+1, len(inter_unpop_item_VB)+1],
                                   [sum(diff[list(inter_pop_item_RI)] == 1)+1, len(inter_pop_item_RI)+1]])

        dict_[batch_clients_idx[i]].append(ratio_item)
        T_sum_pop_item = sum(dict_[batch_clients_idx[i]][-args.wind_sz:])

        T_ratio_pop_item_AS.append(T_sum_pop_item[0][0] / T_sum_pop_item[0][1])
        T_ratio_unpop_item_AS.append(T_sum_pop_item[1][0] / T_sum_pop_item[1][1])
        T_ratio_pop_item_VB.append(T_sum_pop_item[2][0] / T_sum_pop_item[2][1])
        T_ratio_unpop_item_VB.append(T_sum_pop_item[3][0] / T_sum_pop_item[3][1])
        T_ratio_pop_item_RI.append(T_sum_pop_item[4][0] / T_sum_pop_item[4][1])

    return dict_, T_ratio_pop_item_AS, T_ratio_unpop_item_AS, T_ratio_pop_item_VB, T_ratio_unpop_item_VB, T_ratio_pop_item_RI, beni_user


def ratio_defense_GNN(batch_clients_idx, dict_, param_list, m_item, pred_item, sample_user_emb, item_emb, model):
    pred_pop_item_AS, pred_unpop_item_AS, pred_pop_item_VB, pred_unpop_item_VB, pred_pop_item_RI = pred_item
    rating = []
    for user_emb_ in sample_user_emb:
        rating.append(torch.matmul(user_emb_.squeeze(0), item_emb.t()).detach())
    rating = sum(rating)

    T_ratio_pop_item_AS = []
    T_ratio_unpop_item_AS = []
    T_ratio_pop_item_VB = []
    T_ratio_unpop_item_VB = []
    T_ratio_pop_item_RI = []

    for i, parameter in enumerate(param_list):
        beni_user = []
        item_emb_ =copy.deepcopy(item_emb)
        [model_grad, item_grad, user_grad, item, returned_users] = parameter

        inter_pop_item_AS = set(item) & set(pred_pop_item_AS)
        inter_unpop_item_AS = set(item) & set(pred_unpop_item_AS)
        inter_pop_item_VB = set(item) & set(pred_pop_item_VB)
        inter_unpop_item_VB = set(item) & set(pred_unpop_item_VB)
        inter_pop_item_RI = set(item) & set(pred_pop_item_RI)

        if len(inter_pop_item_AS) <= 3 and len(inter_pop_item_VB) <= 3 and len(inter_pop_item_RI) <= 3:
            beni_user.append(i)

        # computer temp global update
        gradient_item = torch.zeros(m_item, args.embedding_dim).to(args.device)
        gradient_item[item, :] += item_grad
        item_emb_ = item_emb_ - args.lr * gradient_item - args.weight_decay * item_emb
        # computer temp rating
        rating_new = []
        for user_emb_ in sample_user_emb:
            rating_new.append(torch.matmul(user_emb_.squeeze(0), item_emb_.t()))
        rating_new = sum(rating_new)

        # computer diff
        diff = torch.sign(rating_new - rating)
        # computer ratio
        ratio_item = torch.tensor([[sum(diff[list(inter_pop_item_AS)] == 1)+1, len(inter_pop_item_AS)+1],
                                   [sum(diff[list(inter_unpop_item_AS)] == -1)+1, len(inter_unpop_item_AS)+1],
                                   [sum(diff[list(inter_pop_item_VB)] == 1)+1, len(inter_pop_item_VB)+1],
                                   [sum(diff[list(inter_unpop_item_VB)] == -1)+1, len(inter_unpop_item_VB)+1],
                                   [sum(diff[list(inter_pop_item_RI)] == 1)+1, len(inter_pop_item_RI)+1]])

        dict_[batch_clients_idx[i]].append(ratio_item)
        T_sum_pop_item = sum(dict_[batch_clients_idx[i]][-args.wind_sz:])

        T_ratio_pop_item_AS.append(T_sum_pop_item[0][0] / T_sum_pop_item[0][1])
        T_ratio_unpop_item_AS.append(T_sum_pop_item[1][0] / T_sum_pop_item[1][1])
        T_ratio_pop_item_VB.append(T_sum_pop_item[2][0] / T_sum_pop_item[2][1])
        T_ratio_unpop_item_VB.append(T_sum_pop_item[3][0] / T_sum_pop_item[3][1])
        T_ratio_pop_item_RI.append(T_sum_pop_item[4][0] / T_sum_pop_item[4][1])

    return dict_, T_ratio_pop_item_AS, T_ratio_unpop_item_AS, T_ratio_pop_item_VB, T_ratio_unpop_item_VB, T_ratio_pop_item_RI, beni_user

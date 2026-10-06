import copy
import torch
import random
import numpy as np
import torch.nn as nn
from parse import args
from model import FedNCF, FedGNN, FedSoG, FedMLP
from defense import avg, ratio_defense, ratio_defense_GNN
from collections import Counter


class FedRecServer(nn.Module):
    def __init__(self, m_item, clients):
        super().__init__()
        self.clients = clients
        self.m_item = m_item
        self.model = eval(args.select_model)()
        self.items_emb = nn.Embedding(self.m_item, args.embedding_dim)
        nn.init.normal_(self.items_emb.weight, std=0.01)

        self.top_k = 1/20
        self.num_samples = 20
        self.item_emb_rec = [torch.zeros(self.m_item, args.embedding_dim, device=args.device)]
        self.dict_ = {i:[torch.zeros((5, 2))] for i in range(len(clients))}
        self.beta = 1
        self.last_opt_item = []
        self.state = 0
        self.pop_dec = 0
        self.unpop_dec = 0
        self.max_pop_dec = 6
        self.max_unpop_dec = 6
        self.max_pop_insec = 0
        self.max_unpop_insec = 0

    def distribute(self, user):
        user.update_local_model(self.model, self.items_emb)

    def detect_by_AS(self, top_num):
        rating = []
        for user_emb_ in self.sample_user_emb:
            rating.append(self.model(user_emb_.repeat(self.m_item, 1), self.items_emb.weight))
        rating = sum(rating)
        pred_pop_item = torch.as_tensor(rating).topk(top_num)[1].tolist()
        pred_unpop_item = torch.as_tensor(-rating).topk(top_num)[1].tolist()
        return pred_pop_item, pred_unpop_item

    def detect_by_VB(self, top_num):
        rating = []
        last_rating_1ord = []
        if len(self.item_emb_rec) < args.window_size:
            self.item_emb_rec.append(self.items_emb.weight.data)
        elif len(self.item_emb_rec) == args.window_size:
            self.item_emb_rec.pop(0)
            self.item_emb_rec.append(self.items_emb.weight.data)

        for user_emb_ in self.sample_user_emb:
            rating.append(self.model(user_emb_.repeat(self.m_item, 1), self.item_emb_rec[-1]))
            last_rating_1ord.append(self.model(user_emb_.repeat(self.m_item, 1), self.item_emb_rec[0]))
        rating = sum(rating)
        last_rating_1ord = sum(last_rating_1ord)
        pred_pop_item, pred_unpop_item = self.get_new_train_data_by_rating_change(rating, last_rating_1ord, top_num)
        return pred_pop_item, pred_unpop_item

    def detect_by_RI(self, all_return_item):
        all_elements = torch.cat(all_return_item).tolist()
        element_counts = Counter(all_elements)
        topk_RI = int(self.top_k * len(set(all_elements)))
        pred_pop_item_RI = [element for element, _ in Counter(element_counts).most_common(topk_RI)]

        return pred_pop_item_RI

    def detect_for_index(self, ratio_list):
        ratio_diff = []
        ratio_list = torch.tensor(ratio_list)
        valid_ratio_list = ratio_list[~torch.isnan(ratio_list)]
        ratio_list[torch.isnan(ratio_list)] = max(valid_ratio_list)
        # sort ratio; calculate difference
        ratio_sort, ratio_index = torch.as_tensor(ratio_list).topk(len(ratio_list))
        for i in range(len(ratio_list) - 1):
            ratio_diff.append(ratio_sort[i] - ratio_sort[i + 1])

        if max(ratio_diff) < 0.1:
            mali_index = torch.tensor([])
        else:
            max_ratio_index = np.array(ratio_diff).argmax()
            mali_index = ratio_index[max_ratio_index + 1:]
        return mali_index

    def train_(self, mali_client, epoch):
        total_loss = []
        total_loss_ = []
        all_return_item = []

        rand_clients = np.arange(len(self.clients))
        np.random.shuffle(rand_clients)
        frac_client_num = int(args.frac * len(self.clients))
        frac_clients_list = random.sample(list(rand_clients), k=frac_client_num)

        for i in range(0, len(frac_clients_list), args.batch_size):
            param_list = []
            param_list_ = []
            batch_clients_idx = frac_clients_list[i: i + args.batch_size]

            for client_id in batch_clients_idx:
                this_client = self.clients[client_id]
                self.distribute(this_client)
                if client_id in mali_client:
                    param, loss = eval('this_client.'+args.attack_user+'()')  
                else:
                    param, loss = this_client.train_()

                # temp parameter
                all_return_item.append(param[2])
                total_loss_.append(loss.item())
                param_list_.append(param)

            if args.is_detect == 1 and epoch > args.start_detect:
                # get random user
                self.sample_user_emb = [torch.randn((1, args.embedding_dim)).to(args.device) * 0.01 for _ in range(self.num_samples)]
                # by absolute score
                pred_pop_item_AS, pred_unpop_item_AS = self.detect_by_AS(int(self.m_item*self.top_k))
                # by velocity_based
                pred_pop_item_VB, pred_unpop_item_VB = self.detect_by_VB(int(self.m_item * self.top_k))
                # by returned item
                pred_pop_item_RI = self.detect_by_RI(all_return_item)

                # computer ratio  
                pred_item = [pred_pop_item_AS, pred_unpop_item_AS, pred_pop_item_VB, pred_unpop_item_VB, pred_pop_item_RI]
                self.dict_, T_ratio_pop_item_AS, T_ratio_unpop_item_AS, T_ratio_pop_item_VB, T_ratio_unpop_item_VB, T_ratio_pop_item_RI, beni_user = \
                    ratio_defense(batch_clients_idx, self.dict_, param_list_, self.m_item, pred_item, self.sample_user_emb, self.items_emb, self.model)

                mali_index_AS_pop = self.detect_for_index(T_ratio_pop_item_AS).tolist()
                mali_index_VB_pop = self.detect_for_index(T_ratio_pop_item_VB).tolist()
                mali_index_RI_pop = self.detect_for_index(T_ratio_pop_item_RI).tolist()

                mali_union_index = set(mali_index_AS_pop) & set(mali_index_VB_pop) & set(mali_index_RI_pop)-set(beni_user)

                for ii in range(len(batch_clients_idx)):
                    if ii not in mali_union_index:
                        total_loss.append(total_loss_[ii])
                        param_list.append(param_list_[ii])
            else:
               total_loss = total_loss_
               param_list = param_list_

            if len(param_list) != 0:
                if args.agg == "avg":
                    gradient_model, gradient_item, _ = avg(param_list, self.m_item, len(self.clients))

                self.items_emb.weight.data.add_(gradient_item, alpha=-args.lr)
                ls_model_param = list(self.model.parameters())
                for j in range(len(ls_model_param)):
                    ls_model_param[j].data = ls_model_param[j].data - args.lr * gradient_model[j]

        return np.mean(total_loss)

    def train_col(self, mali_client, col_server, epoch): 
        total_loss = []
        total_loss_ = []
        all_return_item = []

        rand_clients = np.arange(len(self.clients))
        np.random.shuffle(rand_clients)
        frac_client_num = int(args.frac * len(self.clients))
        frac_clients_list = random.sample(list(rand_clients), k=frac_client_num)  

        if args.attack_user == "ColSH":
            cluster_centers, cluster_sizes = col_server.get_center_emb()

        for i in range(0, len(frac_clients_list), args.batch_size):
            param_list = []
            param_list_ = []
            batch_clients_idx = frac_clients_list[i: i + args.batch_size]

            for client_id in batch_clients_idx:
                this_client = self.clients[client_id]
                self.distribute(this_client)
                if client_id in mali_client:
                    param, loss = eval('this_client.'+args.attack_user+'(cluster_centers, cluster_sizes)')
                else: 
                    param, loss = this_client.train_()

                # temp parameter
                all_return_item.append(param[2])
                total_loss_.append(loss.item())
                param_list_.append(param)

            if args.is_detect == 1 and epoch > args.start_detect:
                # get random user
                self.sample_user_emb = [torch.randn((1, args.embedding_dim)).to(args.device) * 0.01 for _ in range(self.num_samples)]
                # by absolute score
                pred_pop_item_AS, pred_unpop_item_AS = self.detect_by_AS(int(self.m_item*self.top_k))
                # by velocity_based
                pred_pop_item_VB, pred_unpop_item_VB = self.detect_by_VB(int(self.m_item * self.top_k))
                # by returned item
                pred_pop_item_RI = self.detect_by_RI(all_return_item)

                # computer ratio
                pred_item = [pred_pop_item_AS, pred_unpop_item_AS, pred_pop_item_VB, pred_unpop_item_VB, pred_pop_item_RI]
                self.dict_, T_ratio_pop_item_AS, T_ratio_unpop_item_AS, T_ratio_pop_item_VB, T_ratio_unpop_item_VB, T_ratio_pop_item_RI, beni_user = \
                    ratio_defense(batch_clients_idx, self.dict_, param_list_, self.m_item, pred_item, self.sample_user_emb, self.items_emb, self.model)

                mali_index_AS_pop = self.detect_for_index(T_ratio_pop_item_AS).tolist()
                mali_index_VB_pop = self.detect_for_index(T_ratio_pop_item_VB).tolist()
                mali_index_RI_pop = self.detect_for_index(T_ratio_pop_item_RI).tolist()

                mali_union_index = set(mali_index_AS_pop) & set(mali_index_VB_pop) & set(mali_index_RI_pop)-set(beni_user)

                for ii in range(len(batch_clients_idx)):
                    if ii not in mali_union_index:
                        total_loss.append(total_loss_[ii])
                        param_list.append(param_list_[ii])
            else:
               total_loss = total_loss_
               param_list = param_list_
            if len(param_list) != 0:
                if args.agg == "avg":
                    gradient_model, gradient_item, _ = avg(param_list, self.m_item, len(self.clients))

                self.items_emb.weight.data.add_(gradient_item, alpha=-args.lr)
                ls_model_param = list(self.model.parameters())  
                for j in range(len(ls_model_param)):
                    ls_model_param[j].data = ls_model_param[j].data - args.lr * gradient_model[j]
        return np.mean(total_loss)

    def test_(self):
        items_emb = self.items_emb.weight
        test_cnt, test_results = 0, 0.
        with torch.no_grad():
            for client in self.clients:
                test_result = client.test_(items_emb, self.model)
                if test_result is not None:
                    test_cnt += 1
                    test_results += test_result
        return test_results / test_cnt

    def get_new_train_data_by_rating_change(self, rating, last_rating_1ord, top_num):  # rating + 1ord
        # 0ord
        sorted_indices_0ord = torch.argsort(rating, descending=True)
        ranks_0ord = torch.argsort(sorted_indices_0ord) + 1
        # 1ord
        rating_diff_1ord = (rating - last_rating_1ord)
        sorted_indices_1ord = torch.argsort(rating_diff_1ord, descending=True)
        ranks_1ord = torch.argsort(sorted_indices_1ord) + 1
        rating_agg = (1 - self.beta) * (1 / ranks_0ord) + self.beta * (1 / ranks_1ord)

        new_pos_item = rating_agg.topk(top_num)[1]
        new_neg_item = (-rating_agg).topk(top_num)[1]
        new_train_data_ = new_pos_item.tolist() + new_neg_item.tolist()

        if len(self.last_opt_item) <= 2:
            self.last_opt_item.append(new_train_data_)
            self.opt_pop_item = new_pos_item.tolist()
            self.opt_unpop_item = new_neg_item.tolist()
        else:
            if self.pop_dec < self.max_pop_dec or self.unpop_dec < self.max_unpop_dec:
                if self.state == 0:
                    self.last_opt_item.pop(0)
                    self.last_opt_item.append(new_train_data_)
                    self.optim_pop_item(self.last_opt_item[-2], self.last_opt_item[-1], top_num)
                elif self.state == 1:
                    self.last_opt_item.pop(0)
                    self.last_opt_item.append(new_train_data_)
                    self.optim_pop_item(self.last_opt_item[-1], self.opt_new_train_data, top_num)
                else:
                    self.last_opt_item.pop(0)
                    self.last_opt_item.append(new_train_data_)
                    self.optim_pop_item(self.last_opt_item[-2], self.last_opt_item[-1], top_num)
            else:
                self.state += 1
                self.beta = 0
                self.pop_dec = 0
                self.unpop_dec = 0
                self.max_pop_insec = 0
                self.max_unpop_insec = 0
                self.max_pop_dec = self.max_unpop_dec = 3

        self.opt_new_train_data = self.opt_pop_item + self.opt_unpop_item
        return self.opt_pop_item, self.opt_unpop_item

    def optim_pop_item(self, last_item, current_item, top_num):
        pop_insec = len(set(last_item[:top_num]) & set(current_item[:top_num]))
        unpop_insec = len(set(last_item[-top_num:]) & set(current_item[-top_num:]))

        if self.pop_dec < self.max_pop_dec:
            if pop_insec > self.max_pop_insec:
                self.max_pop_insec = pop_insec
                self.opt_pop_item = current_item[:top_num]
                self.pop_dec = 0
            else:
                self.pop_dec += 1

        if self.unpop_dec < self.max_unpop_dec:
            if unpop_insec > self.max_unpop_insec:
                self.max_unpop_insec = unpop_insec
                self.opt_unpop_item = current_item[-top_num:]
                self.unpop_dec = 0
            else:
                self.unpop_dec += 1


class FedGraphServer(nn.Module):
    def __init__(self, clients, true_uid_list, uid_list, iid_list, rating_max, rating_min):
        super().__init__()
        self.clients = clients
        self.true_uid_list = true_uid_list
        self.iid_list = iid_list
        self.m_item = len(iid_list)
        self.uid_list = uid_list
        self.user_embedding = torch.randn(len(uid_list), args.embedding_dim).share_memory_().to(args.device)
        self.item_embedding = torch.randn(len(iid_list), args.embedding_dim).share_memory_().to(args.device)
        self.model = eval(args.select_model)().to(args.device)
        self.rating_max = rating_max
        self.rating_min = rating_min
        self.distribute(self.clients)

        self.top_k = 1/20
        self.num_samples = 20
        self.item_emb_rec = [torch.zeros(self.m_item, args.embedding_dim, device=args.device)]
        self.dict_ = {i:[torch.zeros((5, 2))] for i in range(len(clients))}
        self.beta = 1
        self.last_opt_item = []
        self.state = 0
        self.pop_dec = 0
        self.unpop_dec = 0
        self.max_pop_dec = 8
        self.max_unpop_dec = 8
        self.max_pop_insec = 0
        self.max_unpop_insec = 0

    def distribute(self, users):
        for user in users:
            user.update_local_GNN(self.model, self.item_embedding, self.user_embedding, self.rating_max, self.rating_min)

    def detect_by_AS(self, top_num):
        rating = []
        for user_emb_ in self.sample_user_emb:
            rating.append(torch.matmul(user_emb_.squeeze(0), self.item_embedding.t()).detach())
        rating = sum(rating)
        pred_pop_item = torch.as_tensor(rating).topk(top_num)[1].tolist()
        pred_unpop_item = torch.as_tensor(-rating).topk(top_num)[1].tolist()
        return pred_pop_item, pred_unpop_item

    def detect_by_VB(self, top_num):
        rating = []
        last_rating_1ord = []
        embedding_item = torch.clone(self.item_embedding).detach().to(args.device)

        if len(self.item_emb_rec) < args.window_size:
            self.item_emb_rec.append(embedding_item)
        elif len(self.item_emb_rec) == args.window_size:
            self.item_emb_rec.pop(0)
            self.item_emb_rec.append(embedding_item)
        for user_emb_ in self.sample_user_emb:
            rating.append(torch.matmul(user_emb_.squeeze(0), self.item_emb_rec[-1].t()).detach())
            last_rating_1ord.append(torch.matmul(user_emb_.squeeze(0), self.item_emb_rec[0].t()))

        rating = sum(rating)
        last_rating_1ord = sum(last_rating_1ord)
        pred_pop_item, pred_unpop_item = self.get_new_train_data_by_rating_change(rating, last_rating_1ord, top_num)
        return pred_pop_item, pred_unpop_item

    def detect_by_RI(self, all_return_item):
        all_elements = torch.cat(all_return_item).tolist()
        element_counts = Counter(all_elements)
        topk_RI = int(self.top_k * len(set(all_elements)))
        pred_pop_item_RI = [element for element, _ in Counter(element_counts).most_common(topk_RI)]
        return pred_pop_item_RI

    def detect_for_index(self, ratio_list):
        ratio_diff = []

        ratio_list = torch.tensor(ratio_list)
        valid_ratio_list = ratio_list[~torch.isnan(ratio_list)]
        ratio_list[torch.isnan(ratio_list)] = max(valid_ratio_list)

        max_value = max(tensor.item() for tensor in ratio_list if tensor.item() != 1.0)
        ratio_list = [torch.where(tensor == 1.0, torch.tensor(max_value), tensor) for tensor in ratio_list]
        # sort ratio; calculate difference
        ratio_sort, ratio_index = torch.as_tensor(ratio_list).topk(len(ratio_list))
        for i in range(len(ratio_list) - 1):
            ratio_diff.append(ratio_sort[i] - ratio_sort[i + 1])

        max_ratio_index = np.array(ratio_diff).argmax()
        if max_ratio_index <= int(len(ratio_list)*0.6):
            for i in range(max_ratio_index+1):
                ratio_diff[i] = 0
            max_ratio_index = np.array(ratio_diff).argmax()
        mali_index = ratio_index[max_ratio_index + 1:]
        return mali_index

    def train_(self, mali_client, epoch):
        total_loss = []
        param_list = []
        param_list_ = []
        total_loss_ = []
        all_return_item = []

        np.random.shuffle(self.true_uid_list)
        frac_client_num = int(args.frac * len(self.uid_list))
        frac_clients_list = random.sample(list(self.true_uid_list), k=frac_client_num)

        self.distribute([self.clients[uid] for uid in frac_clients_list])
        for client_id in frac_clients_list:
            this_client = self.clients[client_id]
            if client_id in mali_client:
                param, loss = eval('this_client.' + args.attack_user + '()')
            else:
                param, loss = this_client.train_()

            num_layer = len(param[0])
            # temp parameter
            all_return_item.append(torch.tensor(param[3]))
            total_loss_.append(loss.item())
            param_list_.append(param)

        if args.is_detect == 1 and epoch > args.start_detect:
            # get random user  self.user_embedding
            self.sample_user_emb = copy.deepcopy(self.user_embedding)
            # by absolute score
            pred_pop_item_AS, pred_unpop_item_AS = self.detect_by_AS(int(self.m_item * self.top_k))
            # by velocity_based
            pred_pop_item_VB, pred_unpop_item_VB = self.detect_by_VB(int(self.m_item * self.top_k))
            # by returned item
            pred_pop_item_RI = self.detect_by_RI(all_return_item)

            # computer ratio
            pred_item = [pred_pop_item_AS, pred_unpop_item_AS, pred_pop_item_VB, pred_unpop_item_VB, pred_pop_item_RI]
            self.dict_, T_ratio_pop_item_AS, T_ratio_unpop_item_AS, T_ratio_pop_item_VB, T_ratio_unpop_item_VB, T_ratio_pop_item_RI, beni_user = \
                ratio_defense_GNN(frac_clients_list, self.dict_, param_list_, self.m_item, pred_item, self.sample_user_emb,
                              self.item_embedding, self.model)

            # computer gap
            mali_index_AS_pop = self.detect_for_index(T_ratio_pop_item_AS).tolist()
            mali_index_VB_pop = self.detect_for_index(T_ratio_pop_item_VB).tolist()
            mali_index_RI_pop = self.detect_for_index(T_ratio_pop_item_RI).tolist()

            mali_union_index = set(mali_index_AS_pop) & set(mali_index_VB_pop) & set(mali_index_RI_pop) - set(beni_user)
            if len(mali_union_index) >= int(len(frac_clients_list) * 0.6):
                mali_union_index = list(range(len(frac_clients_list)))
            for ii in range(len(frac_clients_list)):
                if ii not in mali_union_index:
                    total_loss.append(total_loss_[ii])
                    param_list.append(param_list_[ii])
        else:
            total_loss = total_loss_
            param_list = param_list_

        if len(param_list) != 0:
            if args.agg == "avg":
                gradient_model, gradient_item, gradient_user = avg(param_list, self.m_item, len(self.clients))
            ls_model_param = list(self.model.parameters())

            for i in range(len(ls_model_param)):
                ls_model_param[i].data = ls_model_param[i].data - args.lr * gradient_model[i].cpu() - args.weight_decay * ls_model_param[i].data
            self.item_embedding = self.item_embedding - args.lr * gradient_item - args.weight_decay * self.item_embedding
            self.user_embedding = self.user_embedding - args.lr * gradient_user - args.weight_decay * self.user_embedding
            return np.mean(total_loss)
        else:
            return 0

    def train_col(self, mali_client, col_server, epoch):
        total_loss_ = []
        param_list_ = []
        total_loss = []
        param_list = []
        all_return_item = []

        np.random.shuffle(self.true_uid_list)
        frac_client_num = int(args.frac * len(self.true_uid_list))
        frac_clients_list = random.sample(list(self.true_uid_list), k=frac_client_num)

        mali_local_user_emb_list = []
        for client in mali_client:
            this_client = self.clients[client]
            mali_local_user_emb_list.append(this_client.local_user_emb.unsqueeze(0).to(args.device))

        mali_local_user_emb_list = torch.cat(mali_local_user_emb_list)

        if args.attack_user == "Noisy_Col":
            if args.Noisy_pat == "IndDP":
                cluster_centers, cluster_variances, cluster_sizes = col_server.get_center_emb_GNN(self.user_embedding)
            elif args.Noisy_pat == "ColDP":
                cluster_centers, cluster_variances, cluster_sizes = col_server.get_center_emb_GNN(self.user_embedding)
                distances = torch.cdist(cluster_centers, mali_local_user_emb_list, p=2)
                closest_indices = torch.argmin(distances, dim=0).cpu().numpy()

                index_dict = {}
                for i, value in enumerate(closest_indices):
                    if value not in index_dict:
                        index_dict[value] = [i]
                    else:
                        index_dict[value].append(i)

                new_cluster_centers_list = []
                for key, value in index_dict.items():
                    new_cluster_centers = 0
                    for i in index_dict[key]:
                        new_cluster_centers += self.user_embedding[mali_client[i]]
                    new_cluster_centers += cluster_centers[key]
                    new_cluster_centers = torch.div(new_cluster_centers, len(cluster_centers)+1)
                    new_cluster_centers_list.append(new_cluster_centers.unsqueeze(0))
                cluster_centers = torch.cat(new_cluster_centers_list)

        self.distribute([self.clients[uid] for uid in frac_clients_list])
        for client_id in frac_clients_list:
            this_client = self.clients[client_id]
            if client_id in mali_client:
                param, loss = eval('this_client.'+args.attack_user+'(cluster_centers, cluster_variances, cluster_sizes)')
            else:
                param, loss = this_client.train_()

            num_layer = len(param[0])
            all_return_item.append(torch.tensor(param[3]))
            total_loss_.append(loss.item())
            param_list_.append(param)
        if args.is_detect == 1 and epoch > args.start_detect:
            # get random user
            self.sample_user_emb = copy.deepcopy(self.user_embedding)
            # by absolute score
            pred_pop_item_AS, pred_unpop_item_AS = self.detect_by_AS(int(self.m_item * self.top_k))
            # by velocity_based
            pred_pop_item_VB, pred_unpop_item_VB = self.detect_by_VB(int(self.m_item * self.top_k))
            # by returned item
            pred_pop_item_RI = self.detect_by_RI(all_return_item)

            # computer ratio
            pred_item = [pred_pop_item_AS, pred_unpop_item_AS, pred_pop_item_VB, pred_unpop_item_VB, pred_pop_item_RI]
            self.dict_, T_ratio_pop_item_AS, T_ratio_unpop_item_AS, T_ratio_pop_item_VB, T_ratio_unpop_item_VB, T_ratio_pop_item_RI, beni_user = \
                ratio_defense_GNN(frac_clients_list, self.dict_, param_list_, self.m_item, pred_item,
                                  self.sample_user_emb,
                                  self.item_embedding, self.model)

            mali_index_AS_pop = self.detect_for_index(T_ratio_pop_item_AS).tolist()
            mali_index_VB_pop = self.detect_for_index(T_ratio_pop_item_VB).tolist()
            mali_index_RI_pop = self.detect_for_index(T_ratio_pop_item_RI).tolist()

            mali_union_index = set(mali_index_AS_pop) & set(mali_index_VB_pop) & set(mali_index_RI_pop) - set(beni_user)

            if len(mali_union_index) >= int(len(frac_clients_list) * 0.6):
                mali_union_index = list(range(len(frac_clients_list)))
            for ii in range(len(frac_clients_list)):
                if ii not in mali_union_index:
                    total_loss.append(total_loss_[ii])
                    param_list.append(param_list_[ii])

        else:
            total_loss = total_loss_
            param_list = param_list_
        if len(param_list) != 0:
            if args.agg == "avg":
                grad_model, grad_item, grad_user = avg(param_list, self.m_item, len(self.clients))
            ls_model_param = list(self.model.parameters())

            for i in range(len(ls_model_param)):
                ls_model_param[i].data = ls_model_param[i].data - args.lr * grad_model[i].cpu() - args.weight_decay * ls_model_param[i].data
            self.item_embedding = self.item_embedding - args.lr * grad_item - args.weight_decay * self.item_embedding
            self.user_embedding = self.user_embedding - args.lr * grad_user - args.weight_decay * self.user_embedding

            return np.mean(total_loss)
        else:
            return 0

    def predict(self, valid_data):
        users = valid_data[:, 0].astype(int)
        items = valid_data[:, 1].astype(int)
        label = valid_data[:, -1]

        test_cnt, test_results = 0, 0.
        for i in range(len(users)):
            test_result = self.clients[users[i]].test_(items[i], label[i], self.item_embedding)
            if test_result is not None:
                test_cnt += 1
                test_results += test_result

        return test_results/test_cnt

    def get_new_train_data_by_rating_change(self, rating, last_rating_1ord, top_num):  # rating + 1ord
        # 0ord
        sorted_indices_0ord = torch.argsort(rating, descending=True)
        ranks_0ord = torch.argsort(sorted_indices_0ord) + 1
        # 1ord
        rating_diff_1ord = (rating - last_rating_1ord)
        sorted_indices_1ord = torch.argsort(rating_diff_1ord, descending=True)
        ranks_1ord = torch.argsort(sorted_indices_1ord) + 1
        rating_agg = (1 - self.beta) * (1 / ranks_0ord) + self.beta * (1 / ranks_1ord)

        new_pos_item = rating_agg.topk(top_num)[1]
        new_neg_item = (-rating_agg).topk(top_num)[1]
        new_train_data_ = new_pos_item.tolist() + new_neg_item.tolist()

        if len(self.last_opt_item) <= 2:
            self.last_opt_item.append(new_train_data_)
            self.opt_pop_item = new_pos_item.tolist()
            self.opt_unpop_item = new_neg_item.tolist()
        else:
            if self.pop_dec < self.max_pop_dec or self.unpop_dec < self.max_unpop_dec:
                if self.state == 0:
                    self.last_opt_item.pop(0)
                    self.last_opt_item.append(new_train_data_)
                    self.optim_pop_item(self.last_opt_item[-2], self.last_opt_item[-1], top_num)
                elif self.state == 1:
                    self.last_opt_item.pop(0)
                    self.last_opt_item.append(new_train_data_)
                    self.optim_pop_item(self.last_opt_item[-1], self.opt_new_train_data, top_num)
                else:
                    self.last_opt_item.pop(0)
                    self.last_opt_item.append(new_train_data_)
                    self.optim_pop_item(self.last_opt_item[-2], self.last_opt_item[-1], top_num)
            else:
                self.state += 1
                self.beta = 0
                self.pop_dec = 0
                self.unpop_dec = 0
                self.max_pop_insec = 0
                self.max_unpop_insec = 0
                self.max_pop_dec = self.max_unpop_dec = 3

        self.opt_new_train_data = self.opt_pop_item + self.opt_unpop_item
        return self.opt_pop_item, self.opt_unpop_item

    def optim_pop_item(self, last_item, current_item, top_num):
        pop_insec = len(set(last_item[:top_num]) & set(current_item[:top_num]))
        unpop_insec = len(set(last_item[-top_num:]) & set(current_item[-top_num:]))

        if self.pop_dec < self.max_pop_dec:
            if pop_insec > self.max_pop_insec:
                self.max_pop_insec = pop_insec
                self.opt_pop_item = current_item[:top_num]
                self.pop_dec = 0
            else:
                self.pop_dec += 1

        if self.unpop_dec < self.max_unpop_dec:
            if unpop_insec > self.max_unpop_insec:
                self.max_unpop_insec = unpop_insec
                self.opt_unpop_item = current_item[-top_num:]
                self.unpop_dec = 0
            else:
                self.unpop_dec += 1
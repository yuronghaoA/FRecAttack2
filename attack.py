import dgl
import copy
import torch
import torch.nn as nn
import numpy as np
from numpy.random import randint
from parse import args
from random import sample
from model import FedNCF, FedMLP, FedGNN, FedSoG


class FedGraphAttack(nn.Module):
    def __init__(self, id_self, items, ratings, neighbors, iid_list):
        super().__init__()
        self.id_self = id_self
        self.items = items
        self.iid_list = iid_list
        self.m_item = len(iid_list)
        self.ratings = ratings
        self.neighbors = neighbors
        self.model = eval(args.select_model)().to(args.device)
        self.graph = self.build_local_graph(items, neighbors)
        self.graph = dgl.add_self_loop(self.graph).to(args.device)
        self.user_feature = torch.randn(args.embedding_dim,device=args.device)

        self.local_user_emb = torch.randn(args.embedding_dim,device=args.device)
        self.item_emb_rec = [torch.zeros(self.m_item, args.embedding_dim, device=args.device)]
        self.beta = 1
        self.last_opt_item = []

        self.pop_dec = 0
        self.unpop_dec = 0
        self.max_pop_insec = 0
        self.max_unpop_insec = 0
        self.state = 0
        self.max_pop_dec = 8
        self.max_unpop_dec = 8
        self.num = int(args.alpha * self.m_item)

    def user_embedding(self, embedding):
        return embedding[torch.tensor(self.neighbors)].to(args.device), embedding[torch.tensor(self.id_self)].to(args.device)

    def item_embedding(self, embedding):
        return embedding[torch.tensor(self.items)]

    def update_local_GNN(self, global_model, embedding_item, embedding_user, rating_max, rating_min):
        self.model = copy.deepcopy(global_model.cpu()).to(args.device)
        self.rating_max = rating_max
        self.rating_min = rating_min
        self.embedding_user = embedding_user
        self.embedding_item = embedding_item

    def build_local_graph(self, items, neighbors):
        G = dgl.DGLGraph()
        dic_user = {self.id_self: 0}
        dic_item = {}
        count = 1
        for n in neighbors:
            dic_user[n] = count
            count += 1
        for item in items:
            dic_item[item] = count
            count += 1
        G.add_edges([i for i in range(1, len(dic_user))], 0)
        G.add_edges(list(dic_item.values()), 0)
        G.add_edges(0, 0)
        return G

    def GNN(self, embedding_user, embedding_item, new_train_items):
        # get use_emb
        neighbor_embedding, self_embedding = self.user_embedding(embedding_user)
        items_embedding = self.item_embedding(embedding_item)
        features = torch.cat((self_embedding.unsqueeze(0), neighbor_embedding, items_embedding), 0)
        user_feature = self.model(self.graph, features, len(self.neighbors) + 1)
        self.user_feature = user_feature.detach()
        # get item_emb
        new_train_items_embedding = embedding_item[torch.tensor(new_train_items)]
        # computer pred
        predicted = torch.matmul(user_feature.view(-1, args.embedding_dim), new_train_items_embedding.view(-1, args.embedding_dim).t())
        return predicted

    def SoG(self, embedding_user, embedding_item, new_train_items):
        # get use_emb
        neighbor_embedding, self_embedding = self.user_embedding(embedding_user)
        items_embedding = self.item_embedding(embedding_item)
        user_feature = self.model(self_embedding, neighbor_embedding, items_embedding)
        self.user_feature = user_feature.detach()
        # get item_emb
        new_train_items_embedding = embedding_item[torch.tensor(new_train_items)]
        # computer pred
        predicted = torch.matmul(user_feature.view(-1, args.embedding_dim), new_train_items_embedding.view(-1, args.embedding_dim).t())
        return predicted

    def gnn_loss(self, predicted, sampled_rating):
        return torch.mean((predicted.to(args.device) - sampled_rating.to(args.device)) ** 2)

    def sog_loss(self, predicted, sampled_rating):
        return torch.sqrt(torch.mean((predicted.to(args.device) - sampled_rating.to(args.device)) ** 2))

    def predict(self, item_id):
        item_embedding = self.embedding_item[item_id]
        return torch.matmul(self.user_feature, item_embedding.t())

    def sog_negative_sample_item(self, embedding_item):
        item_num = embedding_item.shape[0]
        ls = [i for i in range(item_num) if i not in self.items]
        sampled_items = sample(ls, args.num_neg * len(self.items))

        sampled_item_embedding = embedding_item[torch.tensor(sampled_items)]
        sampled_rating = torch.matmul(self.user_feature.detach(), sampled_item_embedding.t())
        sampled_rating = torch.round(torch.clip(sampled_rating, min=self.rating_min, max=self.rating_max))
        return sampled_items, sampled_rating

    def LDP(self, tensor):
        tensor = torch.clamp(tensor, min=-args.clip, max=args.clip)
        loc = torch.zeros_like(tensor)
        scale = torch.ones_like(tensor) * args.laplace_lambda
        tensor = tensor + torch.distributions.laplace.Laplace(loc, scale).sample()
        return tensor

    # --------------------------Attack------------------------

    def Noisy_Col(self, cluster_centers, cluster_variances, cluster_sizes):
        total_loss = []
        sample_user_emb_list = []
        rating, last_rating_1ord = [], []

        embedding_user = torch.clone(self.embedding_user).detach().to(args.device)
        embedding_item = torch.clone(self.embedding_item).detach().to(args.device)

        # save item_emb of last round, add/delete by queue rule (max window_size)
        if len(self.item_emb_rec) < args.window_size:
            self.item_emb_rec.append(embedding_item)
        elif len(self.item_emb_rec) == args.window_size:
            self.item_emb_rec.pop(0)
            self.item_emb_rec.append(embedding_item)

        # select attack_user
        neighbor_embedding = embedding_user[torch.tensor(self.neighbors)]
        items_embedding = embedding_item[torch.tensor(self.items)]
        for i in range(len(cluster_centers)):
            var = cluster_variances[i]
            for _ in range(int(cluster_sizes[i] / sum(cluster_sizes.values()) * args.sample_size)):
                sample_user_emb = cluster_centers[i] + torch.randn((1, args.embedding_dim)).to(args.device) * var
                if args.select_model == "FedSoG":
                    user_feature = self.model(sample_user_emb.squeeze(0), neighbor_embedding, items_embedding).detach()
                elif args.select_model == "FedGNN":
                    features = torch.cat((sample_user_emb, neighbor_embedding, items_embedding), 0)
                    user_feature = self.model(self.graph, features, len(self.neighbors) + 1).detach()
                sample_user_emb_list.append(user_feature)

                rating.append(torch.matmul(user_feature, embedding_item.t()).detach())
                last_rating_1ord.append(torch.matmul(user_feature, self.item_emb_rec[0].t()))

        rating = sum(rating)
        last_rating_1ord = sum(last_rating_1ord)

        # select attack_item
        if args.attack_item == "Rating_only":
            new_train_data, new_train_label = self.get_new_train_data_by_rating(rating)
        elif args.attack_item == "RatingOfChange":
            new_train_data, new_train_label = self.get_new_train_data_by_rating_change(rating, last_rating_1ord)

        # loss
        embedding_user.grad = torch.zeros_like(embedding_user)
        embedding_item.grad = torch.zeros_like(embedding_item)
        embedding_user.requires_grad = False
        embedding_item.requires_grad = True
        for param in self.model.parameters():
            param.requires_grad = False

        new_train_items_embedding = embedding_item[new_train_data]

        for sample_user_emb_ in sample_user_emb_list:
            prediction = torch.matmul(sample_user_emb_, new_train_items_embedding.view(-1, args.embedding_dim).t())
            if args.select_model == "FedSoG":
                loss_n = self.sog_loss(prediction, new_train_label)
            else:
                loss_n = self.gnn_loss(prediction, new_train_label)
            total_loss.append(loss_n)

        loss = sum(total_loss) / args.sample_size
        loss.backward()
        model_grad, user_grad = self.train_user_emb_benign()
        return self.get_grad_(embedding_item, loss, new_train_data, user_grad, model_grad)

    def train_user_emb_benign(self):
        embedding_user = torch.clone(self.embedding_user).detach().to(args.device)
        embedding_item = torch.clone(self.embedding_item).detach().to(args.device)
        embedding_user.grad = torch.zeros_like(embedding_user)
        embedding_user.requires_grad = True
        embedding_item.requires_grad = False
        for param in self.model.parameters():
            param.requires_grad = True

        sampled_items, sampled_rating = self.sog_negative_sample_item(embedding_item)
        items_with_sampled = sampled_items + self.items
        rating_with_sampled = torch.Tensor(sampled_rating.tolist() + self.ratings).to(args.device)

        if args.select_model == "FedSoG":
            predicted = self.SoG(embedding_user, embedding_item, items_with_sampled)
            loss = self.sog_loss(predicted, rating_with_sampled)
        elif args.select_model == "FedGNN":
            predicted = self.GNN(embedding_user, embedding_item, items_with_sampled)
            loss = self.gnn_loss(predicted, rating_with_sampled)

        self.model.zero_grad()
        loss.backward()
        model_param = self.model.parameters()
        model_grad = [param.grad for param in list(model_param)]

        returned_users = self.neighbors + [self.id_self]
        user_grad = embedding_user.grad[returned_users, :]
        self.local_user_emb = torch.Tensor(embedding_user[self.id_self].cpu() - args.lr * embedding_user.grad[self.id_self, :].cpu()).detach()

        return model_grad, user_grad

    def train_user_emb_mali(self):  # malicious
        embedding_user = torch.clone(self.embedding_user).detach().to(args.device)
        embedding_item = torch.clone(self.embedding_item).detach().to(args.device)
        embedding_user.grad = torch.zeros_like(embedding_user)

        embedding_user.requires_grad = True
        embedding_item.requires_grad = False
        for param in self.model.parameters():
            param.requires_grad = True

        sampled_items, sampled_rating = self.sog_negative_sample_item(embedding_item)
        items_with_sampled = sampled_items + self.items

        rating_max = self.rating_max
        sampled_rating_flip = rating_max - sampled_rating
        rating_flip = [rating_max-x for x in self.ratings]
        rating_with_sampled = torch.Tensor(sampled_rating_flip.tolist() + rating_flip,device=args.device)

        if args.select_model == "FedSoG":
            predicted = self.SoG(embedding_user, embedding_item, items_with_sampled)
            loss = self.sog_loss(predicted, rating_with_sampled)
        elif args.select_model == "FedGNN":
            predicted = self.GNN(embedding_user, embedding_item, items_with_sampled)
            loss = self.gnn_loss(predicted, rating_with_sampled)

        self.model.zero_grad()
        loss.backward()
        model_param = self.model.parameters()
        model_grad = [param.grad for param in list(model_param)]
        returned_users = self.neighbors + [self.id_self]
        user_grad = embedding_user.grad[returned_users, :]
        return model_grad, user_grad

    def get_new_train_data_by_rating(self, rating):
        new_neg_item = rating.topk(int(self.m_item * args.alpha))[1]
        new_pos_item = (-rating).topk(int(self.m_item * args.alpha))[1]

        new_train_data = torch.cat((new_pos_item, new_neg_item)).to(args.device)
        new_train_label = torch.as_tensor([self.rating_max] * len(new_pos_item) + [self.rating_min] * len(new_neg_item),device=args.device)

        return new_train_data.tolist(), new_train_label

    def get_new_train_data_by_rating_change(self, rating, last_rating_1ord):           # rating + 1ord
        # 0ord
        sorted_indices_0ord = torch.argsort(rating, descending=True)
        ranks_0ord = torch.argsort(sorted_indices_0ord) + 1

        # 1ord
        rating_diff_1ord = (rating - last_rating_1ord)
        sorted_indices_1ord = torch.argsort(rating_diff_1ord, descending=True)
        ranks_1ord = torch.argsort(sorted_indices_1ord) + 1

        rating_agg = (1 - self.beta) * (1 / ranks_0ord) + self.beta * (1 / ranks_1ord)
        new_neg_item = rating_agg.topk(int(self.m_item * args.alpha))[1]
        new_pos_item = (-rating_agg).topk(int(self.m_item * args.alpha))[1]
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
                    self.optim_pop_item(self.last_opt_item[-2], self.last_opt_item[-1])
                elif self.state == 1:
                    self.last_opt_item.pop(0)
                    self.last_opt_item.append(new_train_data_)
                    self.optim_pop_item(self.last_opt_item[-1], self.opt_new_train_data)
                else:
                    self.last_opt_item.pop(0)
                    self.last_opt_item.append(new_train_data_)
                    self.optim_pop_item(self.last_opt_item[-2], self.last_opt_item[-1])
            else:
                self.state += 1
                self.beta = 0
                self.pop_dec = 0
                self.unpop_dec = 0
                self.max_pop_insec = 0
                self.max_unpop_insec = 0
                self.max_pop_dec = self.max_unpop_dec = 3

        self.opt_new_train_data = self.opt_pop_item + self.opt_unpop_item
        new_train_label = torch.as_tensor([self.rating_max] * len(new_pos_item.squeeze(axis=0)) + [self.rating_min] * len(new_neg_item.squeeze(axis=0)),device=args.device)
        return self.opt_new_train_data, new_train_label

    def optim_pop_item(self, last_item, current_item):
        pop_insec = len(set(last_item[:self.num]) & set(current_item[:self.num]))
        unpop_insec = len(set(last_item[-self.num:]) & set(current_item[-self.num:]))

        if self.pop_dec < self.max_pop_dec:
            if pop_insec > self.max_pop_insec:
                self.max_pop_insec = pop_insec
                self.opt_pop_item = current_item[:self.num]
                self.pop_dec = 0
            else:
                self.pop_dec += 1

        if self.unpop_dec < self.max_unpop_dec:
            if unpop_insec > self.max_unpop_insec:
                self.max_unpop_insec = unpop_insec
                self.opt_unpop_item = current_item[-self.num:]
                self.unpop_dec = 0
            else:
                self.unpop_dec += 1

    def get_grad(self, embedding_item, loss, returned_items, user_grad):
        item_grad = embedding_item.grad[returned_items, :]
        returned_users = self.neighbors + [self.id_self]

        model_grad = []
        for param in list(self.model.parameters()):
            model_grad.append(param.grad)
        res = (model_grad, item_grad, user_grad, returned_items, returned_users)
        return res, loss

    def get_grad_(self, embedding_item, loss, returned_items, user_grad, model_grad):
        item_grad = embedding_item.grad[returned_items, :]
        returned_users = self.neighbors + [self.id_self]
        res = (model_grad, item_grad, user_grad, returned_items, returned_users)
        return res, loss

    def test_(self, *w):
        return None


class FedRecAttack(nn.Module):
    def __init__(self, train_data, valid_data, test_data, m_item):
        super().__init__()
        self.m_item = m_item
        self.test_data = test_data
        self.valid_data = valid_data
        self.train_data, self.train_label = self.add_neg_item(train_data)
        self.user_emb = nn.Embedding(1, args.embedding_dim,device=args.device)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        self.model = eval(args.select_model)().to(args.device) 

        self.local_user_emb = torch.randn(args.embedding_dim, device=args.device)
        self.item_emb_rec = [torch.zeros(self.m_item, args.embedding_dim, device=args.device)]

        self.beta = 1
        self.last_opt_item = []

        self.state = 0
        self.pop_dec = 0
        self.unpop_dec = 0
        self.max_pop_dec = 6
        self.max_unpop_dec = 6
        self.max_pop_insec = 0
        self.max_unpop_insec = 0
        self.num = int(args.alpha * self.m_item)

    def add_neg_item(self, train_data):
        items, labels = [], []
        for pos_item in train_data:
            items.append(pos_item)
            labels.append(1)
            for _ in range(args.num_neg):
                neg_item = randint(self.m_item)
                while neg_item in train_data + self.test_data + self.valid_data:
                    neg_item = randint(self.m_item)
                items.append(neg_item)
                labels.append(0)

        train_item = torch.Tensor(items).long()
        train_label = torch.Tensor(labels).to(args.device)
        return train_item, train_label

    def update_local_model(self, global_model, item_emb):
        self.model = copy.deepcopy(global_model).to(args.device)
        self.item_emb = copy.deepcopy(item_emb).to(args.device)

    # ------------------------Attack------------------------
    def IndSH(self):
        var = args.sigma
        total_loss = []
        sample_user_emb_list = []
        rating, last_rating_1ord = [], []

        if len(self.item_emb_rec) < args.window_size:
            self.item_emb_rec.append(self.item_emb.weight.data)
        elif len(self.item_emb_rec) == args.window_size:
            self.item_emb_rec.pop(0)
            self.item_emb_rec.append(self.item_emb.weight.data)

        # select args.attack_user
        item_emb_rec = self.item_emb_rec
        m_item = self.m_item
        embedding_dim = args.embedding_dim

        for _ in range(args.sample_size):
            sample_user_emb = self.user_emb.weight.clone().detach() + torch.randn((1, embedding_dim),device=args.device) * var
            sample_user_emb_list.append(sample_user_emb)
            rating.append(self.model(sample_user_emb.repeat(m_item, 1), item_emb_rec[-1]))
            last_rating_1ord.append(self.model(sample_user_emb.repeat(m_item, 1), item_emb_rec[0]))

        rating = sum(rating)
        last_rating_1ord = sum(last_rating_1ord)

        # select args.attack_item
        if args.attack_item == "Rating_only":
            new_train_data, new_train_label = self.get_new_train_data_by_rating(rating)
        elif args.attack_item == "RatingOfChange":
            new_train_data, new_train_label = self.get_new_train_data_by_rating_change(rating, last_rating_1ord)

        # loss
        for sample_user_id in range(args.sample_size):
            prediction = self.model(sample_user_emb_list[sample_user_id].requires_grad_(False).repeat(len(new_train_data), 1),
                                    self.item_emb.weight[new_train_data].requires_grad_(True))
            loss_n = nn.BCELoss()(prediction, new_train_label)
            total_loss.append(loss_n)
        loss = sum(total_loss) / args.sample_size

        # update user_embedding
        self.item_emb.zero_grad()
        self.model.zero_grad()
        loss.backward()
        for _ in range(20):
            self.train_user_emb()
        return self.get_grad(loss, new_train_data)

    def ColSH(self, cluster_centers, cluster_sizes):
        total_loss = []
        sample_user_emb_list = []
        rating, last_rating_1ord = [], []

        if len(self.item_emb_rec) < args.window_size:
            self.item_emb_rec.append(self.item_emb.weight.data)
        elif len(self.item_emb_rec) == args.window_size:
            self.item_emb_rec.pop(0)
            self.item_emb_rec.append(self.item_emb.weight.data)

        # attack_user
        m_item = self.m_item
        sigma = args.sigma
        embedding_dim = args.embedding_dim

        for i in range(len(cluster_centers)):
            var = sigma * float(cluster_sizes[i] / max(cluster_sizes.values()))
            for _ in range(int(cluster_sizes[i] / sum(cluster_sizes.values()) * args.sample_size)):
                sample_user_emb = cluster_centers[i].to(args.device) + torch.randn((1, embedding_dim),device=args.device) * var
                sample_user_emb_list.append(sample_user_emb)
                rating.append(self.model(sample_user_emb.repeat(m_item, 1), self.item_emb.weight))
                last_rating_1ord.append(self.model(sample_user_emb.repeat(m_item, 1), self.item_emb_rec[0]))

        rating = sum(rating)
        last_rating_1ord = sum(last_rating_1ord)

        # attack_item
        if args.attack_item == "Rating_only":
            new_train_data, new_train_label = self.get_new_train_data_by_rating(rating)
        elif args.attack_item == "RatingOfChange":
            new_train_data, new_train_label = self.get_new_train_data_by_rating_change(rating, last_rating_1ord)

        # loss
        for sample_user_id in range(len(sample_user_emb_list)):
            prediction = self.model(sample_user_emb_list[sample_user_id].requires_grad_(False).repeat(len(new_train_data), 1),
                                    self.item_emb.weight[new_train_data].requires_grad_(True))
            loss_n = nn.BCELoss()(prediction, new_train_label)
            total_loss.append(loss_n)

        loss = sum(total_loss) / len(sample_user_emb_list)
        self.item_emb.zero_grad()
        self.model.zero_grad()
        loss.backward()

        return self.get_grad(loss, new_train_data)

    # -------------------------------------------------
    def train_user_emb(self):
        # update user_emb
        user_emb = self.user_emb.weight.repeat(len(self.train_data), 1).requires_grad_(True)
        item_emb = self.item_emb.weight[self.train_data].clone().detach().requires_grad_(False)
        predictions = self.model(user_emb, item_emb, False)

        loss = nn.BCELoss()(predictions, self.train_label)
        self.user_emb.zero_grad()
        loss.backward()
        self.user_emb.weight.data.add_(self.user_emb.weight.grad, alpha=-args.lr)

    def get_new_train_data_by_rating(self, rating):
        new_neg_item = torch.as_tensor(rating).topk(int(self.m_item * args.alpha))[1]

        rating = rating.detach().cpu().numpy()
        new_pos_candidate_item = np.delete(np.arange(self.m_item), new_neg_item.cpu().numpy())
        prob = 1 / rating[new_pos_candidate_item] / (np.sum(1 / rating[new_pos_candidate_item]))
        new_pos_item = torch.Tensor(np.random.choice(new_pos_candidate_item, size=int(self.m_item * args.alpha), replace=False, p=prob))

        new_train_data = new_pos_item.tolist() + new_neg_item.tolist()
        new_train_label = torch.as_tensor([1.0] * len(new_pos_item) + [0.0] * len(new_neg_item),device=args.device)
        return new_train_data, new_train_label

    def get_new_train_data_by_rating_change(self, rating, last_rating_1ord):  # rating + 1ord
        # 0ord
        sorted_indices_0ord = torch.argsort(rating, descending=True)
        ranks_0ord = torch.argsort(sorted_indices_0ord) + 1
        # 1ord
        rating_diff_1ord = (rating - last_rating_1ord)
        sorted_indices_1ord = torch.argsort(rating_diff_1ord, descending=True)
        ranks_1ord = torch.argsort(sorted_indices_1ord) + 1

        rating_agg = (1 - self.beta) * (1 / ranks_0ord) + self.beta * (1 / ranks_1ord)
        new_neg_item = rating_agg.topk(int(self.m_item * args.alpha))[1]
        new_pos_item = (-rating_agg).topk(int(self.m_item * args.alpha))[1]
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
                    self.optim_pop_item(self.last_opt_item[-2], self.last_opt_item[-1])
                elif self.state == 1:
                    self.last_opt_item.pop(0)
                    self.last_opt_item.append(new_train_data_)
                    self.optim_pop_item(self.last_opt_item[-1], self.opt_new_train_data)
                else:
                    self.last_opt_item.pop(0)
                    self.last_opt_item.append(new_train_data_)
                    self.optim_pop_item(self.last_opt_item[-2], self.last_opt_item[-1])
            else:
                self.state += 1
                self.beta = 0
                self.pop_dec = 0
                self.unpop_dec = 0
                self.max_pop_insec = 0
                self.max_unpop_insec = 0
                self.max_pop_dec = self.max_unpop_dec = 3

        self.opt_new_train_data = self.opt_pop_item + self.opt_unpop_item
        new_train_label = torch.as_tensor([1.0] * len(new_pos_item) + [0.0] * len(new_neg_item),device=args.device)
        return self.opt_new_train_data, new_train_label

    def optim_pop_item(self, last_item, current_item):
        pop_insec = len(set(last_item[:self.num]) & set(current_item[:self.num]))
        unpop_insec = len(set(last_item[-self.num:]) & set(current_item[-self.num:]))

        if self.pop_dec < self.max_pop_dec:
            if pop_insec > self.max_pop_insec:
                self.max_pop_insec = pop_insec
                self.opt_pop_item = current_item[:self.num]
                self.pop_dec = 0
            else:
                self.pop_dec += 1

        if self.unpop_dec < self.max_unpop_dec:
            if unpop_insec > self.max_unpop_insec:
                self.max_unpop_insec = unpop_insec
                self.opt_unpop_item = current_item[-self.num:]
                self.unpop_dec = 0
            else:
                self.unpop_dec += 1

    def get_grad(self, loss, returned_items):
        item_grad = self.item_emb.weight.grad
        model_grad = [param.grad for param in list(self.model.parameters())]
        res = (model_grad, item_grad[returned_items], torch.Tensor(returned_items).long())
        return res, loss

    def test_(self, items_emb, model):
        return None

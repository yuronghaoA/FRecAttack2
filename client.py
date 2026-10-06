import dgl
import copy
import torch
import numpy as np
import torch.nn as nn
from parse import args
from random import sample
from numpy.random import randint
from model import FedNCF, FedGNN, FedSoG, FedMLP
from utils import evaluate_precision, evaluate_recall, evaluate_ndcg


class FedRecClient(nn.Module):
    def __init__(self, train_data, valid_data, test_data, m_item):
        super().__init__()
        self.m_item = m_item
        self.test_data = test_data
        self.valid_data = valid_data
        self.train_data, self.train_label = self.add_neg_item(train_data)
        self.user_emb = nn.Embedding(1, args.embedding_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        self.model = eval(args.select_model)()
        self.user_emb = self.user_emb.to(args.device)

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

    def train_(self):
        items_emb = self.item_emb.weight[self.train_data].clone().detach().requires_grad_(True)
        user_emb = self.user_emb.weight.repeat(len(self.train_data), 1)
        predictions = self.model(user_emb, items_emb)
        loss = nn.BCELoss()(predictions, self.train_label)
        self.item_emb.zero_grad()
        self.user_emb.zero_grad()
        self.model.zero_grad()
        loss.backward()

        # get gradient
        user_grad = self.user_emb.weight.grad
        item_grad = items_emb.grad
        model_grad = [param.grad for param in list(self.model.parameters())]

        # update and upload
        self.user_emb.weight.data.add_(user_grad, alpha=-args.lr)
        returned_items = self.train_data
        res = (model_grad, item_grad, returned_items)
        return res, loss

    def test_(self, items_emb, model):
        user_emb = self.user_emb.weight.repeat(self.m_item, 1)
        rating = model(user_emb.to(args.device), items_emb.to(args.device))

        if self.test_data:
            test_loss = nn.BCELoss()(rating[self.test_data], torch.ones(len(self.test_data)).to(args.device)).cpu().item()
            rating[self.train_data] = -(1<<10)
            hr_result, prec_result, ndcg_result = evaluate_recall(rating, self.test_data),\
                                                  evaluate_precision(rating, self.test_data),\
                                                  evaluate_ndcg(rating, self.test_data)
            Data_type = object
            test_result = np.array([test_loss, hr_result, prec_result, ndcg_result], dtype=Data_type)
        else:
            test_result = None

        return test_result


class FedGraphClient(nn.Module):
    def __init__(self, id_self, items, ratings, neighbors):
        super().__init__()
        self.id_self = id_self
        self.items = items
        self.ratings = ratings
        self.neighbors = neighbors
        self.model = eval(args.select_model)().to(args.device)
        self.graph = self.build_local_graph(items, neighbors)
        self.graph = dgl.add_self_loop(self.graph).to(args.device)
        self.user_feature = torch.randn(args.embedding_dim).to(args.device)

    def user_embedding(self, embedding):
        return embedding[torch.tensor(self.neighbors)], embedding[torch.tensor(self.id_self)]

    def item_embedding(self, embedding):
        return embedding[torch.tensor(self.items)]

    def update_local_GNN(self, global_model, embedding_item, embedding_user, rating_max, rating_min):
        self.model = copy.deepcopy(global_model.cpu()).to(args.device)
        self.embedding_user = embedding_user
        self.embedding_item = embedding_item
        self.rating_max = rating_max
        self.rating_min = rating_min

        if args.select_model == 'FedSoG':
            neighbor_embedding, self_embedding = self.user_embedding(self.embedding_user)
            if len(self.items) > 0:
                items_embedding = self.item_embedding(embedding_item)
            else:
                items_embedding = False
            user_feature = self.model(self_embedding, neighbor_embedding, items_embedding)
            self.user_feature = user_feature.detach()

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

    def GNN(self, embedding_user, embedding_item, items_with_sample):
        # use_emb
        neighbor_embedding, self_embedding = self.user_embedding(embedding_user)
        items_embedding = self.item_embedding(embedding_item)
        features = torch.cat((self_embedding.unsqueeze(0), neighbor_embedding, items_embedding), 0)
        user_feature = self.model(self.graph, features, len(self.neighbors) + 1)
        self.user_feature = user_feature.detach()
        # item_emb
        items_embedding_with_sampled = embedding_item[torch.tensor(items_with_sample)]
        # predict
        predicted = torch.matmul(user_feature, items_embedding_with_sampled.t())
        return predicted

    def SoG(self, embedding_user, embedding_item, items_with_sample):
        # user_emb
        neighbor_embedding, self_embedding = self.user_embedding(embedding_user)
        items_embedding = self.item_embedding(embedding_item)
        user_feature = self.model(self_embedding, neighbor_embedding, items_embedding)
        self.user_feature = user_feature.detach()
        # item_emb
        items_embedding_with_sampled = embedding_item[torch.tensor(items_with_sample)]
        # predict
        predicted = torch.matmul(user_feature, items_embedding_with_sampled.t())
        return predicted

    def gnn_loss(self, predicted, rating_with_sample):
        return torch.mean((predicted - rating_with_sample) ** 2)

    def sog_loss(self, predicted, rating_with_sample):
        return torch.sqrt(torch.mean((predicted - rating_with_sample) ** 2))

    def sog_negative_sample_item(self, embedding_item):
        item_num = embedding_item.shape[0]
        ls = [i for i in range(item_num) if i not in self.items]

        sampled_items = sample(ls, args.num_neg * len(self.items))
        sampled_item_embedding = embedding_item[torch.tensor(sampled_items)]
        predicted = torch.matmul(self.user_feature, sampled_item_embedding.t())
        predicted = torch.round(torch.clip(predicted, min=self.rating_min, max=self.rating_max))
        return sampled_items, predicted

    def LDP(self, tensor):
        tensor = torch.clamp(tensor, min=-args.clip, max=args.clip)
        loc = torch.zeros_like(tensor)
        scale = torch.ones_like(tensor) * args.laplace_lambda
        tensor = tensor + torch.distributions.laplace.Laplace(loc, scale).sample()
        return tensor

    def train_(self):
        embedding_user = torch.clone(self.embedding_user).detach().to(args.device)
        embedding_item = torch.clone(self.embedding_item).detach().to(args.device)
        embedding_user.requires_grad = True
        embedding_item.requires_grad = True
        embedding_user.grad = torch.zeros_like(embedding_user)
        embedding_item.grad = torch.zeros_like(embedding_item)

        sampled_items, sampled_rating = self.sog_negative_sample_item(embedding_item)
        items_with_sample = self.items + sampled_items
        rating_with_sample = torch.Tensor(self.ratings + sampled_rating.tolist()).to(args.device)

        if args.select_model == "FedSoG":
            predicted = self.SoG(embedding_user, embedding_item, items_with_sample)
            loss = self.sog_loss(predicted, rating_with_sample)
        elif args.select_model == "FedGNN":
            predicted = self.GNN(embedding_user, embedding_item, items_with_sample)
            loss = self.gnn_loss(predicted, rating_with_sample)

        self.model.zero_grad()
        loss.backward()

        model_grad = [self.LDP(param.grad) for param in list(self.model.parameters())]
        returned_items = self.items + sampled_items
        item_grad = self.LDP(embedding_item.grad[returned_items, :])

        returned_users = self.neighbors + [self.id_self]
        user_grad = self.LDP(embedding_user.grad[returned_users, :])

        res = (model_grad, item_grad, user_grad, returned_items, returned_users)
        return res, loss

    def predict(self, item_id, embedding_item):
        self.model.eval()
        item_embedding = embedding_item[item_id]
        return torch.matmul(self.user_feature, item_embedding.t())

    def test_(self, item_id, label, embedding_item):
        self.model.eval()
        if item_id:
            predicted = torch.matmul(self.user_feature, embedding_item.t()).cpu()
            if args.loss == "mae":

                test_loss = torch.mean(abs(predicted[item_id] - torch.tensor(int(label))))
            else:
                test_loss = torch.sqrt(torch.mean((predicted[item_id] - torch.tensor(label)) ** 2)).item()

            predicted[self.items] = -(1<<10)
            hr_result, prec_result, ndcg_result = evaluate_recall(predicted, [item_id.tolist()]), \
                                                  evaluate_precision(predicted, [item_id.tolist()]), \
                                                  evaluate_ndcg(predicted, [item_id.tolist()])
            Data_type = object
            test_result = np.array([test_loss, hr_result, prec_result, ndcg_result], dtype=Data_type)
        else:
            test_result = None
        return test_result
import torch
import torch.nn as nn
from parse import args
from GAT import GraphAttentionLayer
from dgl.nn.pytorch.conv import GATConv


class FedNCF(nn.Module):
    def __init__(self):
        super().__init__()

        # MLP
        self.layers_dim = [2 * args.embedding_dim] + eval(args.layers) + [args.embedding_dim]
        self.linear_layers = nn.ModuleList([nn.Linear(self.layers_dim[i - 1], self.layers_dim[i])
                                                      for i in range(1, len(self.layers_dim))])
        # NCF
        self.linear_layers.append(nn.Linear(2 * args.embedding_dim, 1))
        self._logistic = nn.Sigmoid().to(args.device)

        for layer in self.linear_layers:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)

        self.linear_layers = self.linear_layers.to(args.device)

    def forward(self, user_emb, items_emb, flag=True):
        # GMF
        gmf_vector = torch.mul(user_emb, items_emb)
        # MLP
        mlp_vector = torch.cat((user_emb, items_emb), dim=-1)

        for i, linear in enumerate(self.linear_layers):
            w = linear.weight.requires_grad_(flag)
            b = linear.bias.requires_grad_(flag)
            if i < len(self.linear_layers) - 1:
                # MLP
                mlp_vector = mlp_vector @ w.t() + b
                mlp_vector = mlp_vector.relu()
            else:
                # Neural MF
                vector = torch.cat([mlp_vector, gmf_vector], dim=-1)
                logits = vector @ w.t() + b
                rating = self._logistic(logits)

        return rating.view(-1)


class FedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self._logistic = nn.Sigmoid()

        layers_dim = [2 * args.embedding_dim] + eval(args.layers) + [1]
        self.linear_layers = nn.ModuleList([nn.Linear(layers_dim[i - 1], layers_dim[i]) for i in range(1, len(layers_dim))])
        self._logistic = nn.Sigmoid()

        for layer in self.linear_layers:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)

    def forward(self, user_emb, items_emb, flag=True):
        # MLP
        mlp_vector = torch.cat((user_emb, items_emb), dim=-1)

        for i, linear in enumerate(self.linear_layers):
            w = linear.weight.requires_grad_(flag)
            b = linear.bias.requires_grad_(flag)
            if i < len(self.linear_layers) - 1:
                # MLP
                mlp_vector = mlp_vector @ w.t() + b
                mlp_vector = mlp_vector.relu()
            else:
                mlp_vector = mlp_vector @ w.t() + b
                rating = self._logistic(mlp_vector)
        return rating.view(-1)


class FedSoG(nn.Module):
    def __init__(self):
        super().__init__()
        self.GAT_neighbor = GraphAttentionLayer(args.embedding_dim, args.embedding_dim)
        self.GAT_item = GraphAttentionLayer(args.embedding_dim, args.embedding_dim)
        self.relation_neighbor = nn.Parameter(torch.randn(args.embedding_dim))
        self.relation_item = nn.Parameter(torch.randn(args.embedding_dim))
        self.relation_self = nn.Parameter(torch.randn(args.embedding_dim))
        self.c = nn.Parameter(torch.randn(2 * args.embedding_dim))

    def predict(self, user_embedding, item_embedding):
        return torch.matmul(user_embedding, item_embedding.t())

    def forward(self, feature_self, feature_neighbor, feature_item):
        if type(feature_item) == torch.Tensor:
            f_n = self.GAT_neighbor(feature_self, feature_neighbor)
            f_i = self.GAT_item(feature_self, feature_item)
            e_n = torch.matmul(self.c, torch.cat((f_n, self.relation_neighbor)))
            e_i = torch.matmul(self.c, torch.cat((f_i, self.relation_item)))
            e_s = torch.matmul(self.c, torch.cat((feature_self, self.relation_self)))
            m = nn.Softmax(dim=-1)
            e_tensor = torch.stack([e_n, e_i, e_s])
            e_tensor = m(e_tensor)
            r_n, r_i, r_s = e_tensor
            user_embedding = r_s * feature_self + r_n * f_n + r_i * f_i
        else:
            f_n = self.GAT_neighbor(feature_self, feature_neighbor)
            e_n = torch.matmul(self.c, torch.cat((f_n, self.relation_neighbor)))
            e_s = torch.matmul(self.c, torch.cat((feature_self, self.relation_self)))
            m = nn.Softmax(dim=-1)
            e_tensor = torch.stack([e_n, e_s])
            e_tensor = m(e_tensor)
            r_n, r_s = e_tensor
            user_embedding = r_s * feature_self + r_n * f_n

        return user_embedding


class FedGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.GAT_layer = GATConv(args.embedding_dim, args.embedding_dim, args.head_num).to(args.device)

    def predict(self, user_embedding, item_embedding):
        return torch.matmul(user_embedding, item_embedding.t())

    def forward(self, graph, features_in, item_index):
        features = self.GAT_layer(graph, features_in)
        n = features.shape[0]
        features = features.reshape(n, -1)
        user_embedding = features[0, :]
        return user_embedding

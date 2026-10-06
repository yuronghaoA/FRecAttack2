import torch
import warnings
import numpy as np
import torch.nn as nn
from parse import args
from sklearn.cluster import KMeans   # for FedNCF/FedMLP
from joblib import parallel_backend
from sklearn.metrics import silhouette_score
import cupy as cp
from cuml.cluster import KMeans  # FedSoG
warnings.filterwarnings("ignore")
from cuml.metrics import pairwise_distances


class ColServer(nn.Module):
    def __init__(self, clients, mali_client):
        super().__init__()
        self.clients = clients
        self.mali_client = mali_client

    # For FedNCF/FedMLP
    def get_center_emb(self):
        mali_users_emb = [self.clients[client_id].user_emb.weight.detach() for client_id in self.mali_client]
        mali_users_emb = np.array([x.squeeze(0).cpu().numpy() for x in mali_users_emb])

        score = []
        with parallel_backend('multiprocessing'):
            for i in range(10):
                k_cluster = KMeans(n_clusters=i+2, max_iter=300, init='k-means++', n_init=10, tol=1e-4, random_state=0).fit(mali_users_emb)
                score.append(silhouette_score(mali_users_emb, k_cluster.labels_, metric='euclidean'))
            k_cluster = KMeans(n_clusters=score.index(max(score))+2, max_iter=300, init='k-means++', n_init=10, tol=1e-4, random_state=0).fit(mali_users_emb)

        cluster_centers = torch.Tensor(k_cluster.cluster_centers_).to(args.device)
        unique_labels, counts = np.unique(k_cluster.labels_, return_counts=True)
        cluster_sizes = dict(zip(unique_labels, counts))

        return cluster_centers, cluster_sizes

    # For FedSoG/FedGNN
    def get_center_emb_GNN(self, user_embedding):
        user_embedding_numpy = np.array(user_embedding.cpu())
        user_embedding_gpu = cp.asarray(user_embedding_numpy)
        score = []
        with parallel_backend('multiprocessing'):
            for i in range(20):
                k_cluster = KMeans(n_clusters=i + 2, max_iter=300, init='k-means++', n_init=10, tol=1e-4, random_state=0).fit(user_embedding_gpu)
                distances = pairwise_distances(user_embedding_gpu, metric='euclidean')
                score.append(silhouette_score(cp.asnumpy(distances), cp.asnumpy(k_cluster.labels_), metric='precomputed'))
            k_cluster = KMeans(n_clusters=score.index(max(score)) + 2, max_iter=300, init='k-means++', n_init=10, tol=1e-4, random_state=0).fit(user_embedding_gpu)
       
        cluster_centers = torch.Tensor(k_cluster.cluster_centers_).to(args.device)
        labels = torch.Tensor(k_cluster.labels_).to(args.device)
        unique_labels, counts = np.unique(k_cluster.labels_.tolist(), return_counts=True)
        cluster_sizes = dict(zip(unique_labels, counts))

        cluster_variances = []
        for i in range(len(cluster_centers)):
            cluster_samples = torch.Tensor(user_embedding)[labels == i].to(args.device)
            cluster_var = torch.mean(torch.norm(cluster_samples - cluster_centers[i], dim=1) ** 2)
            cluster_variances.append(cluster_var)
        return cluster_centers, torch.Tensor(cluster_variances), cluster_sizes

    # For FedNCF/FedMLP
    def train_(self, server):
        self.model = server.model
        self.items_emb = server.items_emb
        for client_id in self.mali_client:
            self.clients[client_id].update_local_model(self.model, self.items_emb)
            self.clients[client_id].train_user_emb()


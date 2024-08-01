import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import numpy as np
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import matplotlib.pyplot as plt
import warnings
import random

from evaluate import *
from relations import *
from llm import *


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.determinivstic = True


def read_dataset(path):
    with open(path, 'r') as file:
        first_line = file.readline().strip()

    columns = first_line.split('\t')
    num_features = len(columns) - 2

    return num_features


def extract_features(split):
    features = []
    for i in range(0, size):
        features.append(float(split[i]))

    return features


def train_get_format_data(path):
    with open(path, 'r') as file:
        for line in file:
            split = line.split()
            y_train.append(int(split[size + 1]))
            x_train.append(extract_features(split[1:]))

    return x_train, y_train


def test_get_format_data(path):
    with open(path, 'r') as file:
        for line in file:
            split = line.split()
            y_test.append(int(split[size + 1]))
            x_test.append(extract_features(split[1:]))

    return x_test, y_test


def train_build_graph(path):
    data_x, data_y = train_get_format_data(path)
    num_samples = len(data_x)
    similarity_matrix = np.zeros((num_samples, num_samples))

    neigh = NearestNeighbors(n_neighbors=near_k)
    neigh.fit(data_x)
    indices = neigh.kneighbors(data_x, return_distance=False)

    for i in range(num_samples):
        for j in range(num_samples):
            if j == i:
                similarity_matrix[i, j] = 1
            elif j in indices[i]:
                sigma_i = np.std(data_x[i])
                sigma_j = np.std(data_x[j])
                similarity_matrix[i, j] = gaussian_kernel(data_x[i], data_x[j], sigma_i, sigma_j)
            else:
                similarity_matrix[i, j] = 0

    similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2
    G = dgl.from_networkx(nx.Graph(nx.from_numpy_matrix(similarity_matrix)))
    G.ndata['feat'] = torch.tensor(data_x)
    G.ndata['label'] = torch.tensor(data_y)
    nx_G = G.to_networkx().to_undirected()
    pos = nx.spring_layout(nx_G)
    nx.draw(nx_G, pos, with_labels=True, font_weight='bold', node_color='skyblue', node_size=800, font_size=8)
    plt.show()
    return data_x, data_y, G


def test_build_graph(path):
    data_x, data_y = test_get_format_data(path)
    num_samples = len(data_x)
    similarity_matrix = np.zeros((num_samples, num_samples))

    neigh = NearestNeighbors(n_neighbors=near_k)
    neigh.fit(data_x)
    indices = neigh.kneighbors(data_x, return_distance=False)

    for i in range(num_samples):
        for j in range(num_samples):
            if j == i:
                similarity_matrix[i, j] = 1
            elif j in indices[i]:
                sigma_i = np.std(data_x[i])
                sigma_j = np.std(data_x[j])
                similarity_matrix[i, j] = gaussian_kernel(data_x[i], data_x[j], sigma_i, sigma_j)
            else:
                similarity_matrix[i, j] = 0

    similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2
    G = dgl.from_networkx(nx.Graph(nx.from_numpy_matrix(similarity_matrix)))
    G.ndata['feat'] = torch.tensor(data_x)
    G.ndata['label'] = torch.tensor(data_y)
    nx_G = G.to_networkx().to_undirected()
    pos = nx.spring_layout(nx_G)
    nx.draw(nx_G, pos, with_labels=True, font_weight='bold', node_color='skyblue', node_size=800, font_size=8)
    plt.show()
    return data_x, data_y, G


class GNNModel(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GNNModel, self).__init__()
        self.raw_weights = nn.Parameter(torch.ones(in_feats))
        self.conv1 = dglnn.SAGEConv(in_feats, h_feats, "mean")
        self.conv2 = dglnn.SAGEConv(h_feats, h_feats, "mean")
        self.skip_connection1 = nn.Linear(in_feats, h_feats) if in_feats != h_feats else nn.Identity()
        self.skip_connection2 = nn.Identity()

    def forward(self, g, x):
        criteria_weights = F.softmax(self.raw_weights, dim=0)
        weighted_x = x * criteria_weights
        out1 = F.relu(self.conv1(g, weighted_x)) + self.skip_connection1(weighted_x)
        out2 = F.relu(self.conv2(g, out1)) + self.skip_connection2(out1)
        return out2


class CriteriaRelationNetwork(nn.Module):
    def __init__(self, num_criteria, num_features, num_relations):
        super(CriteriaRelationNetwork, self).__init__()
        self.num_criteria = num_criteria
        self.conv1 = dglnn.GraphConv(num_features, num_features * 3)
        self.conv2 = dglnn.GraphConv(num_features * 3, num_relations)

    def forward(self, g, features):
        x = F.relu(self.conv1(g, features))
        x = self.conv2(g, x)
        return x.view(-1, self.num_criteria, self.num_criteria)


class AttentionMechanism(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(AttentionMechanism, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim // 2)
        self.fc2 = nn.Linear(input_dim // 2, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


class GMLM(nn.Module):
    def __init__(self, in_feats, h_feats, num_criteria, dm_weights):
        super(GMLM, self).__init__()
        self.gnn = GNNModel(in_feats, h_feats)
        self.criteria_relation_net = CriteriaRelationNetwork(num_criteria, in_feats, num_criteria * num_criteria)

        self.fc = nn.Linear(h_feats, 1)
        self.decision_maker_weights = nn.Parameter(dm_weights, requires_grad=False)
        self.attention = AttentionMechanism(h_feats)

    def forward(self, g, x, pairs):

        embeddings = self.gnn(g, x)
        emb1 = embeddings[pairs[:, 0]]
        emb2 = embeddings[pairs[:, 1]]
        score_diff = self.fc(emb1) - self.fc(emb2)
        primary_output = torch.sigmoid(score_diff)
        aux_output = self.criteria_relation_net(g, x)
        attention_weights = self.attention(embeddings)
        return primary_output, aux_output, attention_weights

    def predict(self, g, x):

        embeddings = self.gnn(g, x)
        scores = self.fc(embeddings)
        return scores.squeeze().cpu().numpy()


def traintest():

    train_pairs, train_labels = paire_compaire(train_y)
    train_crlabels = criteria_label(torch.tensor(train_x), train_y)

    in_size = torch.tensor(train_x).shape[1]
    model = GMLM(in_feats=in_size, h_feats=in_size*3, num_criteria=in_size, dm_weights=dm_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    primary_loss_function = nn.BCELoss()
    aux_loss_function = nn.MSELoss()
    lambda_reg = 1e-1

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        primary_output, aux_output, attention_weights = model(train_g, torch.tensor(train_x), train_pairs)
        loss_primary = primary_loss_function(primary_output.squeeze(), train_labels)
        loss_aux = aux_loss_function(aux_output, train_crlabels)
        reg_loss = lambda_reg * torch.norm(model.gnn.raw_weights - model.decision_maker_weights)
        loss = loss_primary + (attention_weights * loss_aux).mean() + reg_loss
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    model.eval()
    with torch.no_grad():
        scores = model.predict(test_g, torch.tensor(test_x))
        score = np.array(test_y)[np.argsort(scores)[::-1]]
        ndcg = nDCG(score)
        ndcgk1 = nDCG_k(score, nk1)
        ndcgk2 = nDCG_k(score, nk2)
        ndcgk3 = nDCG_k(score, nk3)
        ndcgk4 = nDCG_k(score, nk4)
        ndcgk5 = nDCG_k(score, nk5)
        ndcgk6 = nDCG_k(score, nk6)
        ap = aP(score)
        apk1 = aPK(score, ak1)
        apk2 = aPK(score, ak2)
        apk3 = aPK(score, ak3)
        apk4 = aPK(score, ak4)
        apk5 = aPK(score, ak5)
        apk6 = aPK(score, ak6)
        cindex = cINDEX(scores, test_y)
        spear = sPEAR(scores, test_y)
        return ndcg, ndcgk1, ndcgk2, ndcgk3, ndcgk4, ndcgk5, ndcgk6, ap, apk1, apk2, apk3, apk4, apk5, apk6, cindex, spear



if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    setup_seed(100)
    torch.use_deterministic_algorithms(True)

    num_runs = 1
    num_epochs = 100
    near_k = 3
    nk1 = 5
    nk2 = 10
    nk3 = 15
    nk4 = 20
    nk5 = 25
    nk6 = 30
    ak1 = 5
    ak2 = 10
    ak3 = 15
    ak4 = 20
    ak5 = 25
    ak6 = 30

    ndcg_list = [0 for x in range(0, num_runs)]
    ndcg1_list = [0 for x in range(0, num_runs)]
    ndcg2_list = [0 for x in range(0, num_runs)]
    ndcg3_list = [0 for x in range(0, num_runs)]
    ndcg4_list = [0 for x in range(0, num_runs)]
    ndcg5_list = [0 for x in range(0, num_runs)]
    ndcg6_list = [0 for x in range(0, num_runs)]
    ap_list = [0 for x in range(0, num_runs)]
    ap1_list = [0 for x in range(0, num_runs)]
    ap2_list = [0 for x in range(0, num_runs)]
    ap3_list = [0 for x in range(0, num_runs)]
    ap4_list = [0 for x in range(0, num_runs)]
    ap5_list = [0 for x in range(0, num_runs)]
    ap6_list = [0 for x in range(0, num_runs)]
    cindex_list = [0 for x in range(0, num_runs)]
    spear_list = [0 for x in range(0, num_runs)]


    for i in range(num_runs):

        x_train = []
        x_test = []
        y_train = []
        y_test = []

        train_path = 'dataset/xx.txt'
        test_path = 'dataset/xx.txt'
        size = read_dataset(train_path)
        train_x, train_y, train_g = train_build_graph(train_path)
        test_x, test_y, test_g = test_build_graph(test_path)
        dm_weights = torch.tensor(dm_weight(), dtype=torch.float32)
        ndcg_list[i], ndcg1_list[i], ndcg2_list[i], ndcg3_list[i], ndcg4_list[i], ndcg5_list[i], ndcg6_list[i], \
        ap_list[i], ap1_list[i], ap2_list[i], ap3_list[i], ap4_list[i], ap5_list[i], ap6_list[i], cindex_list[i], \
        spear_list[i] = traintest()

    mean_ndcg = np.mean(ndcg_list)
    mean_ndcg1 = np.mean(ndcg1_list)
    mean_ndcg2 = np.mean(ndcg2_list)
    mean_ndcg3 = np.mean(ndcg3_list)
    mean_ndcg4 = np.mean(ndcg4_list)
    mean_ndcg5 = np.mean(ndcg5_list)
    mean_ndcg6 = np.mean(ndcg6_list)
    mean_ap = np.mean(ap_list)
    mean_ap1 = np.mean(ap1_list)
    mean_ap2 = np.mean(ap2_list)
    mean_ap3 = np.mean(ap3_list)
    mean_ap4 = np.mean(ap4_list)
    mean_ap5 = np.mean(ap5_list)
    mean_ap6 = np.mean(ap6_list)
    mean_cindex = np.mean(cindex_list)
    mean_spear = np.mean(spear_list)

    std_ndcg = np.std(ndcg_list)
    std_ndcg1 = np.std(ndcg1_list)
    std_ndcg2 = np.std(ndcg2_list)
    std_ndcg3 = np.std(ndcg3_list)
    std_ndcg4 = np.std(ndcg4_list)
    std_ndcg5 = np.std(ndcg5_list)
    std_ndcg6 = np.std(ndcg6_list)
    std_ap = np.std(ap_list)
    std_ap1 = np.std(ap1_list)
    std_ap2 = np.std(ap2_list)
    std_ap3 = np.std(ap3_list)
    std_ap4 = np.std(ap4_list)
    std_ap5 = np.std(ap5_list)
    std_ap6 = np.std(ap6_list)
    std_cindex = np.std(cindex_list)
    std_spear = np.std(spear_list)

    print("Res:", " NDCG {:.4f} | NDCG@{} {:.4f} | NDCG@{} {:.4f} | NDCG@{} {:.4f} | NDCG@{} {:.4f} |"
                  " NDCG@{} {:.4f} | NDCG@{} {:.4f} | AP {:.4f} | AP@{} {:.4f} | AP@{} {:.4f} | AP@{} {:.4f} |"
                  " AP@{} {:.4f} | AP@{} {:.4f} | AP@{} {:.4f} | CINDEX {:.4f} | SPEAR {:.4f}".format(mean_ndcg, nk1,
                                                                                                      mean_ndcg1, nk2,
                                                                                                      mean_ndcg2,
                                                                                                      nk3, mean_ndcg3,
                                                                                                      nk4, mean_ndcg4,
                                                                                                      nk5,
                                                                                                      mean_ndcg5, nk6,
                                                                                                      mean_ndcg6,
                                                                                                      mean_ap, ak1,
                                                                                                      mean_ap1, ak2,
                                                                                                      mean_ap2, ak3,
                                                                                                      mean_ap3, ak4,
                                                                                                      mean_ap4, ak5,
                                                                                                      mean_ap5,
                                                                                                      ak6, mean_ap6,
                                                                                                      mean_cindex,
                                                                                                      mean_spear))

    print("Dev:", " NDCG {:.4f} | NDCG@{} {:.4f} | NDCG@{} {:.4f} | NDCG@{} {:.4f} | NDCG@{} {:.4f} |"
                  " NDCG@{} {:.4f} | NDCG@{} {:.4f} | AP {:.4f} | AP@{} {:.4f} | AP@{} {:.4f} | AP@{} {:.4f} |"
                  " AP@{} {:.4f} | AP@{} {:.4f} | AP@{} {:.4f} | CINDEX {:.4f} | SPEAR {:.4f}".format(std_ndcg, nk1,
                                                                                                      std_ndcg1, nk2,
                                                                                                      std_ndcg2,
                                                                                                      nk3, std_ndcg3,
                                                                                                      nk4, std_ndcg4,
                                                                                                      nk5,
                                                                                                      std_ndcg5, nk6,
                                                                                                      std_ndcg6, std_ap,
                                                                                                      ak1, std_ap1, ak2,
                                                                                                      std_ap2, ak3,
                                                                                                      std_ap3, ak4,
                                                                                                      std_ap4, ak5,
                                                                                                      std_ap5,
                                                                                                      ak6, std_ap6,
                                                                                                      std_cindex,
                                                                                                      std_spear))
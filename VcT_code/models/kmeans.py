import numpy as np
import torch


def initialize(X, num_clusters):
    """
    initialize cluster centers
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :return: (np.array) initial state
    """
    # indices_row = np.random.choice(X.shape[1], num_clusters*X.shape[0], replace=False)
    # np.random.seed(1)
    # indices_row = np.sort(np.random.choice(X.shape[1], num_clusters, replace=False))

    indices_row = np.linspace(0, X.shape[1] - 1, num=num_clusters, endpoint=True, retstep=False, dtype=int)
    indices_row = np.tile(indices_row,8)
    # print(indices_row)

    # indices_row = np.array([100, 300, 500, 700]*8)
    indices_col = np.arange(X.shape[0]).repeat(num_clusters)

    initial_state = X[indices_col, indices_row,:].reshape(X.shape[0],num_clusters,-1)
    return initial_state


def kmeans(
        X,
        num_clusters,
        distance='euclidean',
        tol=1e-4,
):
    """
    perform kmeans
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param tol: (float) threshold [default: 0.0001]
    :param device: (torch.device) device [default: cpu]
    :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
    """

    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    elif distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # initialize
    initial_state = initialize(X, num_clusters)

    while True:
        dis = pairwise_distance_function(X, initial_state)
        choice_cluster = torch.argmin(dis, dim=2)#------tidudiushi
        initial_state_pre = initial_state.clone()
        from torch_scatter import scatter_mean, scatter_add
        initial_state = scatter_mean(X, choice_cluster, dim=1, dim_size=num_clusters)

        # for index in range(num_clusters):
        #     selected = torch.nonzero(choice_cluster == index).squeeze()

        #     selected = torch.index_select(X, 0, selected)
        #     initial_state[index] = selected.mean(dim=0)
        
        center_shift = torch.sum(
            torch.sqrt(
                torch.sum((initial_state - initial_state_pre) ** 2, dim=2)
            ),1)
        if torch.all(~(center_shift ** 2).gt(tol)):
            break
    
    return choice_cluster, initial_state




def pairwise_distance(data1, data2):
    # transfer to device
    # data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=2)

    # 1*N*M
    B = data2.unsqueeze(dim=1)

    dis = (A - B) ** 2.0
    # return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1).squeeze()
    return dis


def pairwise_cosine(data1, data2):
    # transfer to device

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)

    cosine = A_normalized * B_normalized

    # return N*N matrix for pairwise distance
    cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
    return cosine_dis


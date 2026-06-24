import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import subspace_angles
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

# includes self as the first neighbor
# data is either X or distance matrix d_e
def nearest_neighbors(data, k_nn, metric, n_jobs=-1, sort_results=True):
    n = data.shape[0]
    if k_nn > 1:
        neigh = NearestNeighbors(n_neighbors=k_nn-1, metric=metric, n_jobs=n_jobs)
        neigh.fit(data)
        neigh_dist, neigh_ind = neigh.kneighbors()
        neigh_dist = np.insert(neigh_dist, 0, np.zeros(n), axis=1)
        neigh_ind = np.insert(neigh_ind, 0, np.arange(n), axis=1)
        if sort_results:
            inds = np.argsort(neigh_dist, axis=-1)
            for i in range(neigh_ind.shape[0]):
                neigh_ind[i,:] = neigh_ind[i,inds[i,:]]
                neigh_dist[i,:] = neigh_dist[i,inds[i,:]]
    else:
        neigh_dist = np.zeros((n,1))
        neigh_ind = np.arange(n).reshape((n,1)).astype('int')
    return neigh_ind, neigh_dist

def normalize_diameter(X):
    X = X.copy()
    mu = X.mean(axis=0)
    X = X - mu[None,:]
    max_norm = np.max(np.linalg.norm(X, axis=1))
    X /= (2*max_norm*(1 + 1e-6))
    return X

def compute_principal_angles(tang_basis_1, tang_basis_2):
    n_samples, emb_dim, _ = tang_basis_1.shape
    principal_angles = np.zeros((n_samples, emb_dim))
    for k in range(n_samples):
        tang_basis_1_at_k = tang_basis_1[k,:]
        tang_basis_2_at_k = tang_basis_2[k,:]
        theta = subspace_angles( # compares columns so take transpose
            tang_basis_1_at_k.T, tang_basis_2_at_k.T
        )
        principal_angles[k,:] = theta
    return principal_angles

def tang_basis_estimate_error(tang_basis_1, tang_basis_2):
    principal_angles = compute_principal_angles(tang_basis_1, tang_basis_2)
    error = np.sum(1-np.cos(principal_angles), axis=1)
    return error
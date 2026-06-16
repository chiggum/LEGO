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

def point_cloud_diameter(points, iterations=20):
    points = np.asarray(points)
    if len(points) < 2:
        return 0.0

    # Start with a random point
    current_point = points[0]
    max_diameter = 0.0

    for _ in range(iterations):
        # Calculate squared distances from the current point to all other points
        # (Calculating squared distances avoids the costly square root operation)
        diffs = points - current_point
        sq_dists = np.sum(diffs**2, axis=1) 
        
        # Find the index of the furthest point
        furthest_idx = np.argmax(sq_dists)
        
        current_max_dist = np.sqrt(sq_dists[furthest_idx])
        if current_max_dist > max_diameter:
            max_diameter = current_max_dist
            
        # Jump to the furthest point for the next iteration
        current_point = points[furthest_idx]

    return max_diameter

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
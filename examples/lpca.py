import time
import numpy as np
from lego import util
from scipy.linalg import svd
from scipy.sparse.linalg import svds

def lpca(
    X,
    opts = {}, # see default_opts below
):
    default_opts = {
        'emb_dim': 2,
        'explain_var': 0,
        'k_nn': 9,
        'metric': 'euclidean',
        'kernel': None,
        'bandwidth': None,
        'print_time': True
    }
    default_opts.update(opts)
    opts = default_opts

    n_samples, ambient_dim = X.shape
    assert opts['emb_dim'] <= ambient_dim, "opts['emb_dim'] > ambient_dim."

    total_start = time.perf_counter() if opts['print_time'] else None

    # --- 1. Nearest Neighbors ---
    t0 = time.perf_counter() if opts['print_time'] else None

    # find nearest neighbors (includes self-loops at first index)
    neigh_ind, neigh_dist = util.nearest_neighbors(X, opts['k_nn'], opts['metric'])

    if opts['print_time']:
        print(f"[LPCA] Nearest neighbors: {time.perf_counter() - t0:.4f} s")

    # --- 2. Tangent Space Basis Estimation ---
    t0 = time.perf_counter() if opts['print_time'] else None

    # estimate basis of tangent spaces 
    emb_dim = opts['emb_dim']
    var_explained = np.zeros((n_samples, emb_dim))
    local_mean = np.zeros((n_samples, ambient_dim))
    tang_basis = np.zeros((n_samples, emb_dim, ambient_dim))
    
    for k in range(n_samples):
        nbrs = neigh_ind[k,:]
        X_k = X[nbrs,:]
        local_mean[k,:] = np.mean(X_k, axis=0)
        X_k = X_k - local_mean[k,:][None,:]
        X_k = X_k.T

        if opts['kernel'] is not None:
            dist_k = np.linalg.norm(X_k, axis=0)
            if opts['bandwidth'] is not None:
                ndist_k = dist_k/opts['bandwidth']
            else:
                ndist_k = dist_k/np.max(dist_k)
            
            if opts['kernel'] == 'epanechnikov':
                ker_k = 1-ndist_k**2
            elif opts['kernel'] =='gaussian':
                ker_k = np.exp(-ndist_k**2)
            else:
                raise NotImplementedError("Only epanechnikov or gaussian kernels are available.")
            
            ker_k = np.sqrt(ker_k)
            X_k = X_k * ker_k[None,:]
        
        if emb_dim == ambient_dim:
            Q_k, Sigma_k, _ = svd(X_k) # Q_k.shape = (ambient_dim, ambient_dim)
        else:
            Q_k, Sigma_k, _ = svds(
                X_k,
                k=emb_dim,
                which='LM',
                random_state=42
            )  # Q_k.shape = (ambient_dim, emb_dim)
        
        var_explained[k,:] = Sigma_k**2
        
        # Guard against zero-division
        sum_var = np.sum(var_explained[k,:])
        if sum_var > 0:
            var_explained[k,:] /= sum_var

        if opts['explain_var'] > 0:
            emb_dim_k = min(emb_dim, np.argmax(np.cumsum(var_explained[k,:]) >= opts['explain_var'])+1)
        else:
            emb_dim_k = emb_dim
            
        Q_k = Q_k[:,:emb_dim_k]
        tang_basis[k,:emb_dim_k,:] = Q_k.T

    if opts['print_time']:
        print(f"[LPCA] Tangent basis estimation: {time.perf_counter() - t0:.4f} s")
        print(f"[LPCA] Total execution time: {time.perf_counter() - total_start:.4f} s\n")

    return {
        'local_mean': local_mean,
        'var_explained': var_explained,
        'tang_basis': tang_basis
    }
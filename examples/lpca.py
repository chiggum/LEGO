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
        'bandwidth': None
    }
    default_opts.update(opts)
    opts = default_opts

    n_samples, ambient_dim = X.shape
    assert opts['emb_dim'] <= ambient_dim, "opts['emb_dim'] > ambient_dim."

    # find nearest neighbors (includes self-loops at first index)
    neigh_ind, neigh_dist = util.nearest_neighbors(X, opts['k_nn'], opts['metric'])

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
                raise NotImplementedError("Only epanechnikov kernel is available.")
            
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
        var_explained[k,:] /= np.sum(var_explained[k,:])

        if opts['explain_var'] > 0:
            emb_dim_k = min(emb_dim, np.argmax(np.cumsum(var_explained[k,:]) >= opts['explain_var'])+1)
        else:
            emb_dim_k = emb_dim
        Q_k = Q_k[:,:emb_dim_k]
        tang_basis[k,:emb_dim_k,:] = Q_k.T

    return {
        'local_mean': local_mean,
        'var_explained': var_explained,
        'tang_basis': tang_basis
    }
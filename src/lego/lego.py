import time
import numpy as np
from . import util
from . import gl

from scipy.linalg import svd
from scipy.sparse.linalg import svds

def lego(
    X,
    opts = {}, # see default_opts below
    gl_opts = {}, # see default_gl_opts below
    return_aux_info = False
):
    default_opts = {
        'emb_dim': 2,
        'explain_var': None,
        'k_nn': 9,
        'metric': 'euclidean',
        'n_eig_for_grad': 40,
        'kernel_for_grad': None,
        # Higher value (e.g. 1e-1) may be needed for clean data
        # to prevent blow up during pinv if not using tikhonov
        'r_tol': 1e-2,
        'reg_grad': True,
        'tikhonov': True,
        'tikhonov_power': 1/2,
        'print_time': True
    }
    default_gl_opts = {
        'which': 'diffusion',
        'tuning': 'self',
        'k_tune': None,
        'kernel': 'gaussian',
        'ds_max_iter': 0,
        'n_eig': 100,
        'n_ignore': 0
    }
    default_opts.update(opts)
    opts = default_opts

    default_gl_opts.update(gl_opts)
    gl_opts = default_gl_opts

    n_samples, ambient_dim = X.shape
    if gl_opts['k_tune'] is None:
        gl_opts['k_tune'] = opts['k_nn']-1 # minus 1 because k_nn includes self but k_tune does not
    if opts['n_eig_for_grad'] == -1:
        opts['n_eig_for_grad'] = gl_opts['n_eig']
    if not opts['reg_grad']:
        gl_opts['n_eig'] = opts['n_eig_for_grad']
        
    assert gl_opts['n_eig'] >= opts['n_eig_for_grad'], "gl_opts['n_eig'] < opts['n_eig_for_grad']."
    assert opts['emb_dim'] <= ambient_dim, "opts['emb_dim'] > ambient_dim."
    
    total_start = time.perf_counter() if opts['print_time'] else None

    # --- 1. Nearest Neighbors ---
    t0 = time.perf_counter() if opts['print_time'] else None
    
    neigh_ind, neigh_dist = util.nearest_neighbors(X, opts['k_nn'], opts['metric'])
    diam = util.point_cloud_diameter(X)
    
    if opts['print_time']:
        print(f"[LEGO] Nearest neighbors & diameter: {time.perf_counter() - t0:.4f} s")

    # --- 2. Graph Laplacian Spectrum ---
    t0 = time.perf_counter() if opts['print_time'] else None
    
    # compute eigenvectors of the graph Laplacian
    # these also contain trivial eigenvectors if n_ignore is zero
    _, phi = gl.spectrum_of_laplacian_from_neighbors(
        neigh_ind[:,1:], neigh_dist[:,1:], # remove self-loops
        opts = gl_opts
    )
    
    if opts['print_time']:
        print(f"[LEGO] Graph Laplacian spectrum: {time.perf_counter() - t0:.4f} s")

    # --- 3. First Order Approximation of Gradients ---
    t0 = time.perf_counter() if opts['print_time'] else None
    
    n_eig_for_grad = opts['n_eig_for_grad']
    grad_eig_foa = np.zeros((n_eig_for_grad, n_samples, ambient_dim))
    n_survived_dim_in_pinv = np.zeros(n_samples)
    
    for k in range(n_samples):
        nbrs = neigh_ind[k,:]
        X_k = X[nbrs,:] - X[k,:][None,:]
        phi_k = phi[nbrs,:n_eig_for_grad] - phi[k,:n_eig_for_grad][None,:]

        if opts['kernel_for_grad'] is not None:
            if opts['kernel_for_grad'] == 'epanechnikov':
                dist_k = np.linalg.norm(X_k, axis=1)
                ndist_k = dist_k/np.max(dist_k)
                ker_k = 1-ndist_k**2
                ker_k = np.sqrt(ker_k)
                X_k = X_k * ker_k[:,None]
                phi_k = phi_k * ker_k[:,None]
            else:
                raise NotImplementedError("Only epanechnikov kernel is available.")

        # Compute common SVD values for regularization
        Uk, Sk, Vk_T = svd(X_k, full_matrices=False) # X_k = U_kS_kV_k^T
        if opts['tikhonov']:
            reg_param = np.sum(Sk**(2*(1+opts['tikhonov_power'])))
            reg_param = reg_param/(diam**(2*opts['tikhonov_power']))
            Sk_pinv = Sk/(Sk**2 + reg_param)
            n_survived_dim_in_pinv[k] = ambient_dim
        else:
            Sk_pinv = np.zeros_like(Sk)
            mask = Sk**2 <= opts['r_tol']*(Sk[0]**2)
            n_survived_dim_in_pinv[k] = np.sum(~mask)
            Sk_pinv[~mask] = 1/Sk[~mask] 

        X_k_pinv = Vk_T.T.dot(Sk_pinv[:,None] * Uk.T) # X_k_pinv = V_k S_k_pinv U_k^T
        grad_eig_foa[:,k,:] = X_k_pinv.dot(phi_k).T

    if opts['print_time']:
        print(f"[LEGO] Gradient first-order approx: {time.perf_counter() - t0:.4f} s")

    # --- 4. Regularize Gradients ---
    if opts['reg_grad']:
        t0 = time.perf_counter() if opts['print_time'] else None
        
        # regularize gradients by projecting on the eigenvectors
        # orthogonalize eigenvectors first
        U_phi = svd(phi, full_matrices=False)[0] # (n_samples, n_eig)
        reg_grad_eig = np.zeros_like(grad_eig_foa)
        for i in range(n_eig_for_grad):
            temp = U_phi.T.dot(grad_eig_foa[i,:,:]) # n_eig x ambient_dim
            reg_grad_eig[i,:,:] = U_phi.dot(temp) # n_samples x ambient_dim
            
        if opts['print_time']:
            print(f"[LEGO] Gradient regularization: {time.perf_counter() - t0:.4f} s")
    else:
        reg_grad_eig = grad_eig_foa

    # --- 5. Tangent Space Basis Estimation ---
    t0 = time.perf_counter() if opts['print_time'] else None
    
    # finally orthogonalize gradients to estimate basis of tangent spaces 
    emb_dim = opts['emb_dim']
    var_explained = np.zeros((n_samples, emb_dim))
    local_mean = np.zeros((n_samples, ambient_dim))
    tang_basis = np.zeros((n_samples, emb_dim, ambient_dim))
    
    for k in range(n_samples):
        nbrs = neigh_ind[k,:]
        local_mean[k,:] = np.mean(X[nbrs,:], axis=0)
        if emb_dim == ambient_dim:
            Q_k, Sigma_k, _ = svd(reg_grad_eig[:,k,:].T) # Q_k.shape = (ambient_dim, ambient_dim)
        else:
            Q_k, Sigma_k, _ = svds(
                reg_grad_eig[:,k,:].T,
                k=emb_dim,
                which='LM',
                random_state=42
            )  # Q_k.shape = (ambient_dim, emb_dim)
        
        var_explained[k,:] = Sigma_k**2
        
        # Guard against zero-division if variance is absolutely zero
        sum_var = np.sum(var_explained[k,:])
        if sum_var > 0:
            var_explained[k,:] /= sum_var

        if opts['explain_var'] is not None:
            emb_dim_k = min(emb_dim, np.argmax(np.cumsum(var_explained[k,:]) >= opts['explain_var'])+1)
        else:
            emb_dim_k = emb_dim

        Q_k = Q_k[:,:emb_dim_k]
        tang_basis[k,:emb_dim_k,:] = Q_k.T

    if opts['print_time']:
        print(f"[LEGO] Tangent basis estimation: {time.perf_counter() - t0:.4f} s")
        print(f"[LEGO] Total execution time: {time.perf_counter() - total_start:.4f} s\n")

    output = {
        'local_mean': local_mean,
        'var_explained': var_explained,
        'tang_basis': tang_basis,
    }

    if return_aux_info:
        output.update({
            'phi': phi,
            'reg_grad_eig': reg_grad_eig,
            'n_survived_dim_in_pinv': n_survived_dim_in_pinv
        })

    return output
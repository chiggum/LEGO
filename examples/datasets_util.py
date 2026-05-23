import numpy as np
import scipy.integrate as integrate
from scipy.optimize import fsolve
from scipy.stats import truncnorm
from PIL import Image, ImageOps
from sklearn.decomposition import PCA
import os


def noisyswissroll(
    RES=100,
    noise=0.01,
    noise_type = 'ortho',
    theta0=3*np.pi/2,
    nturns=2,
    rmax=2*1e-2,
    seed=42
):
    sideL1 = integrate.quad(lambda x: rmax*np.sqrt(1+x**2), theta0, theta0*(1+nturns))[0]
    sideL2 = 1/sideL1
    RESt = int(np.ceil(sideL1*RES+1))
    tdistv = np.linspace(0,sideL1,RESt)
    tv = []
    for tdist in tdistv.tolist():
        tt = fsolve(lambda x: (0.5*rmax*(x*np.sqrt(1+x**2)+np.arcsinh(x)))-\
                                0.5*rmax*(theta0*np.sqrt(1+theta0**2)+np.arcsinh(theta0))-\
                                tdist,theta0*(1+nturns/2))
        tv.append(tt)
    tv = np.array(tv)    
    RESh = int(np.ceil(sideL2*RES+1))
    heightv = np.linspace(0,sideL2,RESh)[:,None]
    heightv = np.tile(heightv,[RESt,1])
    heightv = heightv.flatten('F')[:,None]
    tv = np.repeat(tv,RESh)[:,None]

    X=np.concatenate([rmax*tv*np.cos(tv), heightv, rmax*tv*np.sin(tv)], axis=1)
    
    np.random.seed(seed)
    if noise_type == 'gaussian':
        X = X+noise*np.random.normal(0,1,[X.shape[0],3])
    elif noise_type == 'uniform':
        X = X+noise*np.random.uniform(0,1,[X.shape[0],3])
    elif 'ortho' in noise_type:
        # the swiss roll rolls around y axis
        temp = X.copy()
        temp[:,1] = 0
        temp = temp/np.linalg.norm(temp, axis=1)[:,None]
        if noise_type == 'ortho-uniform':
            X = X + noise*np.random.uniform(-1,1,(X.shape[0],1))*temp
        else: # ortho-gaussian
            X = X + noise*np.random.normal(0,1,(X.shape[0],1))*temp
    
    labels = np.concatenate([tv, X[:,[1]]], axis=1)
    return X, labels

def wave_on_curvedtorus3d(
    n=10000, noise=0, noise_type='ortho', Rmax=0.25, seed=42, freq=4, 
    density='uniform', wave_amp_r=0.2, wave_freq_r=5, wave_amp_R=0.1, 
    wave_freq_R=3, rmax=None
):
    np.random.seed(seed)
    rmax = rmax if rmax is not None else 1 / (4 * np.pi**2 * Rmax)

    # Helper to calculate surface points, derivatives, and area elements all at once
    def get_surface(theta, phi):
        r = rmax * (1 + wave_amp_r * np.sin(wave_freq_r * phi))
        R = Rmax * (1 + wave_amp_R * np.sin(wave_freq_R * theta))
        r_p = rmax * wave_amp_r * wave_freq_r * np.cos(wave_freq_r * phi)
        R_p = Rmax * wave_amp_R * wave_freq_R * np.cos(wave_freq_R * theta)

        X  = np.c_[(R + r*np.cos(theta))*np.cos(phi), (R + r*np.cos(theta))*np.sin(phi), r*np.sin(theta)]
        Xt = np.c_[(R_p - r*np.sin(theta))*np.cos(phi), (R_p - r*np.sin(theta))*np.sin(phi), r*np.cos(theta)]
        Xp = np.c_[r_p*np.cos(theta)*np.cos(phi) - (R + r*np.cos(theta))*np.sin(phi),
                   r_p*np.cos(theta)*np.sin(phi) + (R + r*np.cos(theta))*np.cos(phi), r_p*np.sin(theta)]
        
        norm = np.cross(Xt, Xp)
        return X, Xt, Xp, norm, np.linalg.norm(norm, axis=1)

    # 1. Estimate maximum area using a quick 100x100 grid
    tg, pg = np.mgrid[0:2*np.pi:100j, 0:2*np.pi:100j]
    max_area = get_surface(tg.ravel(), pg.ravel())[4].max() * 1.05 # safety buffer

    # 2. Vectorized Rejection Sampling (Batches of 2*n for speed)
    th_v, ph_v = np.empty(0), np.empty(0)
    while len(th_v) < n:
        th = np.random.uniform(0, 2*np.pi, n*2) if density == 'uniform' else \
             truncnorm.rvs(a=-2, b=2, scale=0.5*np.pi, size=n*2)
        ph = np.random.uniform(0, 2*np.pi, n*2)
        
        area = get_surface(th, ph)[4]
        valid = np.random.uniform(0, 1, n*2) <= (area / max_area)
        
        th_v, ph_v = np.append(th_v, th[valid]), np.append(ph_v, ph[valid])

    th_v, ph_v = th_v[:n, None], ph_v[:n, None]

    # 3. Generate final clean surface points and unit vectors
    X, Xt, Xp, normal, _ = get_surface(th_v, ph_v)
    norm_unit = normal / np.linalg.norm(normal, axis=1)[:, None]

    # 4. Generate and apply noise
    if noise > 0:
        if noise_type == 'ortho-uniform':        n_vals = np.random.uniform(-1, 1, (n, 1))
        elif noise_type == 'ortho-gaussian':     n_vals = np.random.normal(0, 1, (n, 1))
        elif noise_type == 'ortho-hetero-uniform':n_vals = np.random.uniform(-np.cos(freq*ph_v)**2, np.cos(freq*ph_v)**2)
        elif noise_type == 'ortho-hetero-gaussian':n_vals = np.random.normal(0, np.cos(freq*ph_v)**2)
        else:                              n_vals = 0

        X_noisy = X + (noise * n_vals) * norm_unit
    else:
        X_noisy = X
    Xt_unit = Xt / np.linalg.norm(Xt, axis=1)[:, None]
    Xp_unit = Xp / np.linalg.norm(Xp, axis=1)[:, None]

    return X_noisy, X, np.c_[np.mod(th_v, 2*np.pi), ph_v], None, (Xt_unit, Xp_unit, norm_unit)

# generates a clean torus
def curvedtorus3d_with_normal_dir(n=10000, density='uniform', seed=42):
    _, X, labelsMat, _, (_, _, normal_dir) = wave_on_curvedtorus3d(
        n=n, noise=0, noise_type='ortho-uniform',
        seed=seed, density=density,
        wave_amp_r=0, wave_freq_r=0, wave_amp_R=0,
        wave_freq_R=0, rmax=None
    )
    return X, labelsMat, normal_dir

def read_img(fpath, grayscale=False, bbox=None):
    if grayscale:
        img = ImageOps.grayscale(Image.open(fpath))
    else:
        img = Image.open(fpath)
    if bbox is not None:
        return np.asarray(img.crop(bbox).reduce(2))
    else:
        return np.asarray(img.reduce(2))
    
def do_pca(X, n_pca):
    print('Applying PCA')
    pca = PCA(n_components=n_pca, random_state=42)
    pca.fit(X)
    print('sum(explained_variance_ratio):', np.sum(pca.explained_variance_ratio_))
    X = pca.fit_transform(X)
    return X

def puppets_data(
    dirpath, prefix='s1', n=None, bbox=None,
    grayscale=False, normalize = False, n_pca=0
):
    X = []
    labels = []
    fnames = []
    for fname in sorted(os.listdir(dirpath)):
        if prefix in fname:
            fnames.append(fname)
    
    if n is not None:
        fnames = fnames[:n]
        
    for fname in fnames:
        X_k = read_img(dirpath+'/'+fname, bbox=bbox, grayscale=grayscale)
        X.append(X_k.flatten())
        labels.append(int(fname.split('.')[0].split('_')[1])-100000)
    
    img_shape = X_k.shape
    X = np.array(X)
    labels = np.array(labels)[:,None]-1
    labelsMat = np.concatenate([labels,labels], axis=1)
    if normalize:
        X = X - np.mean(X,axis=0)[None,:]
        X = X / (np.std(X,axis=0)[None,:] + 1e-12)
        
    if n_pca:
        X_new = do_pca(X, n_pca)
    else:
        X_new = X
    
    X_new = X_new / np.max(np.abs(X_new))
    print('X.shape = ', X_new.shape)
    return X_new, labelsMat, X, img_shape
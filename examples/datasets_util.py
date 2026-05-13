import numpy as np
import scipy.integrate as integrate
from scipy.optimize import fsolve
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
    if noise_type == 'normal':
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
        else:
            X = X + noise*np.random.normal(0,1,(X.shape[0],1))*temp
    
    labels = np.concatenate([tv, X[:,[1]]], axis=1)
    return X, labels

def wave_on_curvedtorus3d(
    n=10000, noise=0, noise_type='ortho',
    Rmax=0.25, seed=42, freq=4, density='uniform',
    wave_amp_r=0.2, wave_freq_r=5, wave_amp_R=0.1, wave_freq_R=3, rmax=None
):
    if rmax is None:
        rmax=1/(4*(np.pi**2)*Rmax)

    theta = np.pi
    phi = (3*np.pi)/(2*wave_freq_r + 1e-12)
    r_ = rmax + wave_amp_r * rmax * np.sin(wave_freq_r * phi)
    R_ = Rmax + wave_amp_R * Rmax * np.sin(wave_freq_R * theta)

    r_prime_ = wave_freq_r * wave_amp_r * np.cos(wave_freq_r * phi)
    R_prime_ = wave_freq_R * wave_amp_R * np.cos(wave_freq_R * theta)
    
    X_theta_ = np.array([(R_prime_ - r_*np.sin(theta))*np.cos(phi),
                            (R_prime_ - r_*np.sin(theta))*np.sin(phi),
                            r_*np.cos(theta)])
    X_phi_ = np.array([r_prime_*np.cos(theta)*np.cos(phi) - (R_+r_*np.cos(theta))*np.sin(phi),
                        r_prime_*np.cos(theta)*np.sin(phi) + (R_+r_*np.cos(theta))*np.cos(phi),
                        r_prime_*np.sin(theta)])
    X_theta_cross_X_phi_max = np.linalg.norm(np.cross(X_theta_, X_phi_))

    X = []
    thetav = []
    phiv = []
    np.random.seed(seed)
    k = 0
    sigma = 0.75*np.pi
    while k < n:
        rU = np.random.uniform(0,1,3)
        if density != 'uniform':
            #theta = np.mod(sigma*np.random.normal(0, 1), 2*np.pi)
            theta = truncnorm.rvs(a=-np.pi/sigma,b=np.pi/sigma,scale=sigma)
        else:
            theta = 2*np.pi*rU[0]
        phi = 2*np.pi*rU[1]

        r_ = rmax + wave_amp_r * rmax * np.sin(wave_freq_r * phi)
        R_ = Rmax + wave_amp_R * Rmax * np.sin(wave_freq_R * theta)

        r_prime_ = wave_freq_r * wave_amp_r * np.cos(wave_freq_r * phi)
        R_prime_ = wave_freq_R * wave_amp_R * np.cos(wave_freq_R * theta)
        
        X_theta_ = np.array([(R_prime_ - r_*np.sin(theta))*np.cos(phi),
                                (R_prime_ - r_*np.sin(theta))*np.sin(phi),
                                r_*np.cos(theta)])
        X_phi_ = np.array([r_prime_*np.cos(theta)*np.cos(phi) - (R_+r_*np.cos(theta))*np.sin(phi),
                            r_prime_*np.cos(theta)*np.sin(phi) + (R_+r_*np.cos(theta))*np.cos(phi),
                            r_prime_*np.sin(theta)])
        X_theta_cross_X_phi = np.linalg.norm(np.cross(X_theta_, X_phi_))
        X_theta_cross_X_phi = X_theta_cross_X_phi/X_theta_cross_X_phi_max

        #if rU[2] <= (Rmax + rmax*np.cos(theta))/(Rmax + rmax):
        #if rU[2] <= (R_ + r_*np.cos(theta))/(R_ + r_):
        if rU[2] <= X_theta_cross_X_phi:
            thetav.append(theta)
            phiv.append(phi)
            k = k + 1
    
    thetav = np.array(thetav)[:,None]
    phiv = np.array(phiv)[:,None]
    dX = None

    r = rmax + wave_amp_r * rmax * np.sin(wave_freq_r * phiv)
    R = Rmax + wave_amp_R * Rmax * np.sin(wave_freq_R * thetav)
    X = np.concatenate([(R+r*np.cos(thetav))*np.cos(phiv),
                            (R+r*np.cos(thetav))*np.sin(phiv),
                            r*np.sin(thetav)], axis=1)
    
    np.random.seed(42)
    if 'uniform' in noise_type:
        noise = noise*np.random.uniform(-1,1,(X.shape[0],1))
    elif 'gaussian' in noise_type:
        noise = noise*np.random.normal(0,1,(X.shape[0],1))
    else:
        #noise_u = 0.01 + 0.3*(1+np.cos(freq*phiv))/2
        noise_u = np.cos(freq*phiv)**2
        noise_u = np.random.uniform(-noise_u,noise_u)
        noise = noise*noise_u

    r_prime = wave_freq_r * wave_amp_r * rmax * np.cos(wave_freq_r * phiv)
    R_prime = wave_freq_R * wave_amp_R * Rmax * np.cos(wave_freq_R * thetav)
    X_theta = np.concatenate([(R_prime - r*np.sin(thetav))*np.cos(phiv),
                            (R_prime - r*np.sin(thetav))*np.sin(phiv),
                                r*np.cos(thetav)], axis=1)
    X_phi = np.concatenate([r_prime*np.cos(thetav)*np.cos(phiv) - (R+r*np.cos(thetav))*np.sin(phiv),
                            r_prime*np.cos(thetav)*np.sin(phiv) + (R+r*np.cos(thetav))*np.cos(phiv),
                            r_prime*np.sin(thetav)], axis=1)
    normal_dir = np.cross(X_theta, X_phi)
    normal_dir = normal_dir/np.linalg.norm(normal_dir, axis=1)[:,None]
    X_noisy = X + noise * normal_dir

    X_theta = X_theta/np.linalg.norm(X_theta,axis=1)[:,None]
    X_phi = X_phi/np.linalg.norm(X_phi,axis=1)[:,None]

    labelsMat = np.concatenate([np.mod(thetav, 2*np.pi), phiv], axis=1)
    return X_noisy, X, labelsMat, dX, (X_theta, X_phi, normal_dir)

def curvedtorus3d_with_normal_dir(n=10000, density='uniform', seed=42):
    _, X, labelsMat, _, (_, _, normal_dir) = wave_on_curvedtorus3d(
        n=n, noise=0, noise_type='ortho',
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
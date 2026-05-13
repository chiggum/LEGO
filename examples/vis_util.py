import pdb
import sys
import os

import numpy as np
import seaborn as sns
import pandas as pd

import math
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import scipy

import matplotlib
print('matplotlib.get_backend() = ', matplotlib.get_backend())
#matplotlib.use('Qt4Agg')
from matplotlib import pyplot as plt
plt.rcParams.update({'scatter.marker':'.'})
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
from scipy.spatial.distance import pdist, squareform

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

import imageio

OUTPUT_EXT = '.png'

def maximize_window():
    manager = plt.get_current_fig_manager()
    try:
        # TkAgg backend
        manager.window.state('zoomed')  # Windows
    except AttributeError:
        try:
            # Qt backend
            manager.window.showMaximized()
        except AttributeError:
            try:
                # Wx backend
                manager.frame.Maximize(True)
            except AttributeError:
                pass  # Fallback — do nothing

def get_orientations(step_size = 15, max_angle = 360*7):
    orientation = []
    for angle in range(0, max_angle + 1, step_size):
        # Normalize the angle to the range [-180, 180] for display
        angle_norm = (angle + 180) % 360 - 180

        # Cycle through a full rotation of elevation, then azimuth, roll, and all
        elev = azim = roll = 0
        if angle <= 360:
            elev = angle_norm
        elif angle <= 360*2:
            azim = angle_norm
        elif angle <= 360*3:
            roll = angle_norm
        elif angle <= 360*4:
            elev = azim = angle_norm
        elif angle <= 360*5:
            azim = roll = angle_norm
        elif angle <= 360*6:
            elev = roll = angle_norm
        else:
            elev = azim = roll = angle_norm
        orientation.append((elev, azim, roll))
    return orientation

def save_gif(y, labels, vis_obj, fpath, cmap='jet', FPS = 10, step_size = 15, max_angle = 360*8, s=1, figsize=(8,8)):
    frames = []
    vis_obj.global_embedding(y, labels, cmap, s=s, figsize=figsize)
    ax = plt.gca()
    orientation = get_orientations(step_size, max_angle)
    for i in range(len(orientation)):
        elev, azim, roll = orientation[i]
        # Update the axis view and title
        ax.view_init(elev, azim, roll)
        set_axes_equal(ax)

        fig = plt.gcf()
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)

    plt.close()
    imageio.mimsave(fpath, frames, duration=1000/FPS)

def combine_cmaps(zeta, U_k):
    min_zeta = np.min(zeta)
    max_zeta = np.max(zeta)
    c1 = plt.cm.jet((zeta - min_zeta)/(max_zeta-min_zeta))
    c1[U_k,:-1] = 0
    return c1

def eval_param(phi, Psi_gamma, Psi_i, k, mask, beta=None, T=None, v=None):
    if beta is None:
        return Psi_gamma[k,:][np.newaxis,:] * phi[np.ix_(mask,Psi_i[k,:])]
    else:
        if T is not None and v is not None:
            return np.dot(beta[k]*Psi_gamma[k,:][np.newaxis,:] * phi[np.ix_(mask,Psi_i[k,:])], T[k,:,:]) + v[[k],:]
        else:
            return beta[k]*Psi_gamma[k,:][np.newaxis,:] * phi[np.ix_(mask,Psi_i[k,:])]

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)

def arrow3d(ax, x, y, z, u, v, w, mutation_scale=0.1, color='r', lw=1):
    a = Arrow3D([x, u], [y, v], 
                [z, w], mutation_scale=mutation_scale, 
                lw=lw, arrowstyle="-|>", color=color)
    ax.add_artist(a)

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"

# colorcube colormap taken from matlab
def colorcube(m):
    nrgsteps = np.fix(np.power(m,1/3)+np.finfo(np.float32).eps)
    extra = m-np.power(nrgsteps,3)
    if (extra == 0) and (nrgsteps > 2):
        nbsteps = nrgsteps - 1
    else:
        nbsteps = nrgsteps

    rgstep = 1/(nrgsteps-1)
    bstep  = 1/(nbsteps-1)
    [r,g,b] = np.meshgrid(np.arange(nrgsteps)*rgstep,
                          np.arange(nrgsteps)*rgstep,
                          np.arange(nbsteps)*bstep)
    r = r.flatten('F')[:,np.newaxis]
    g = g.flatten('F')[:,np.newaxis]
    b = b.flatten('F')[:,np.newaxis]
    mymap = np.concatenate([r,g,b], axis=1)
    
    

    diffmap = np.diff(mymap.T, axis=0).T
    summap = np.sum(np.abs(diffmap),1)
    notgrays = (summap != 0)
    mymap = mymap[notgrays,:]

    summap = np.concatenate([np.sum(mymap[:,[0,1]],1)[:,np.newaxis],
                             np.sum(mymap[:,[1,2]],1)[:,np.newaxis],
                             np.sum(mymap[:,[0,2]],1)[:,np.newaxis]], axis=1)
    mymap = mymap[np.min(summap,axis=1) != 0,:]
    
    remlen = m - mymap.shape[0] - 1

    rgbnsteps = np.floor(remlen / 4)
    knsteps   = remlen - 3*rgbnsteps

    rgbstep = 1/(rgbnsteps)
    kstep   = 1/(knsteps  )

    rgbramp = np.arange(0,rgbnsteps)*rgbstep + rgbstep
    rgbzero = np.zeros((rgbramp.shape[0], 1))
    kramp   = np.arange(0,knsteps)*kstep + kstep
    
    rgbramp = rgbramp[:,np.newaxis]
    kramp = kramp[:,np.newaxis]

    mymap = np.concatenate([mymap,
                            np.concatenate([rgbramp, rgbzero, rgbzero], axis=1),
                            np.concatenate([rgbzero, rgbramp, rgbzero], axis=1),
                            np.concatenate([rgbzero, rgbzero, rgbramp], axis=1),
                            np.zeros((1,3)),
                            np.concatenate([kramp, kramp, kramp,], axis=1)], axis=0)
    return mymap

def on_close(event):
    raise RuntimeError("Figure closed.")

def imscatter(x, y, image, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists

def closest_pt(x_min, x_max, y_min, y_max, p):
    i = np.argmin([p[0]-x_min,x_max-p[0], p[1]-y_min, y_max-p[1]])
    if i == 0:
        return [x_min,p[1]]
    elif i==1:
        return [x_max,p[1]]
    elif i==2:
        return [p[0],y_min]
    elif i==3:
        return [p[0],y_max]

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

class Visualize:
    def __init__(self, save_dir=''):
        self.save_dir = save_dir
        if self.save_dir:
            if (not os.path.isdir(self.save_dir)) and (not os.path.islink(self.save_dir)):
                os.makedirs(self.save_dir)
        pass
    
    def init_3d_figure(self, figsize=None):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection='3d')
        return ax
    
    def init_3d_subplots(self, n_row, n_col, subplot_h=1, subplot_w=1):
        fig, ax = plt.subplots(n_row, n_col, figsize=(n_col*subplot_w, n_row*subplot_h),
                                subplot_kw={'projection': '3d'})
        return fig, ax

    def data(self, X, labels=None, title='', figsize=None, s=5,
              cmap='jet', azim=None, elev=None, roll=None, colorbar=False, axis_image=True,
              focus_pts=None, s_focus_scale=5, n_c_bar_ticks=4, make_label_mask=False,
              vmin=None, vmax=None, show_axis=True, save_fn=''):
        
        if type(X)==list:
            X = np.concatenate([X[0].flatten()[:,None],
                                X[1].flatten()[:,None]], axis=1)
        if labels is None:
            labels = np.zeros(X.shape[0])
            
        if make_label_mask:
            labels = labels.copy()
            temp = np.zeros(X.shape[0])
            temp[labels] = 1
            labels = temp
            
        assert X.shape[1] <= 3, 'X.shape[1] must be either 2 or 3.'
        
        fig = plt.figure(figsize=figsize)
        if matplotlib.get_backend().startswith('Qt'):
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            
        if X.shape[1] == 2:
            # Capture the mappable object 'p' for the colorbar
            p = plt.scatter(X[:,0], X[:,1], s=s, c=labels, cmap=cmap, vmin=vmin, vmax=vmax)
            
            if focus_pts is not None:
                plt.scatter(X[focus_pts,0], X[focus_pts,1], s=s_focus_scale*s, c='r')
            if axis_image:
                plt.axis('image')
                
            if colorbar:
                if (vmin is not None) and (vmax is not None):
                    cbar_ticks = np.linspace(vmin, vmax, n_c_bar_ticks, endpoint=True)
                else:
                    cbar_ticks = np.linspace(np.nanmin(labels), np.nanmax(labels), n_c_bar_ticks, endpoint=True)
                
                # Use fraction and pad to match colorbar height to the plot
                plt.colorbar(p, fraction=0.046, pad=0.04, ticks=cbar_ticks)
                
        elif X.shape[1] == 3:
            ax = fig.add_subplot(projection='3d')
            if azim and elev:
                ax.view_init(azim=azim, elev=elev, roll=roll)
                
            p = ax.scatter(X[:,0], X[:,1], X[:,2], s=s, c=labels, cmap=cmap, vmin=vmin, vmax=vmax)
            
            if focus_pts is not None:
                ax.scatter(X[focus_pts,0], X[focus_pts,1], X[focus_pts,2], s=s_focus_scale*s, c='r')
            if axis_image:
                # Assuming set_axes_equal is defined elsewhere in your code
                set_axes_equal(ax) 
                
            if colorbar:
                cbar_ticks = np.linspace(np.nanmin(labels), np.nanmax(labels), n_c_bar_ticks, endpoint=True)
                # Use shrink for 3D plots to scale it down, and fraction/pad for positioning
                cbar = fig.colorbar(p, ax=ax, shrink=0.7, fraction=0.046, pad=0.04)
                cbar.set_ticks(cbar_ticks)
                
        plt.title(title)
        plt.tight_layout()
        
        if not show_axis:
            plt.axis('off')
            
        if hasattr(self, 'save_dir') and self.save_dir:
            if save_fn == '':
                save_fn = title
            # Assuming OUTPUT_EXT is defined globally
            plt.savefig(self.save_dir + '/' + save_fn + OUTPUT_EXT)

    def plot_tangent_vectors_2d_in_3d(self, Psi, X, X0, subsample_inds, title='', save_fn='', length=0.025, alpha=0.1):
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(*X.T, alpha=alpha, s=10)
        ax.scatter(*X0.T, color='k', alpha=2*alpha, s=2)
        for i in subsample_inds.tolist():
            plt.gca().quiver(X0[i,0], X0[i,1], X0[i,2], Psi[i,0,0], Psi[i,0,1], Psi[i,0,2], color='red', length=length)
            plt.gca().quiver(X0[i,0], X0[i,1], X0[i,2], Psi[i,1,0], Psi[i,1,1], Psi[i,1,2], color='green', length=length)
        plt.title(title)
        plt.tight_layout()
        plt.axis('image')
        if save_fn == '':
            save_fn = title
        plt.savefig(self.save_dir + '/' + save_fn + OUTPUT_EXT)

    def plot_tangent_vectors_1d_in_2d(
        self, Psi, X, X0, subsample_inds, title='', save_fn='', figsize=(4,4),
        scale=15, width=0.005, color=None, c=None, arrow_color='red', alpha=1, s=1
    ):
        plt.figure(figsize=figsize)
        if X is not None:
            plt.scatter(*X.T, c=c, cmap='jet', s=s, alpha=alpha)
            
        plt.scatter(*X0.T, color='k', alpha=0.5, s=2)
        color_ = arrow_color
        for i in subsample_inds.tolist():
            if color is not None:
                color_ = color[i,:]
            plt.gca().quiver(X0[i,0], X0[i,1], Psi[i,0,0], Psi[i,0,1], color=color_, scale=scale, width=width)
        plt.tight_layout()
        plt.title(title)
        plt.axis('image')
        plt.axis('off')
        if save_fn=='':
            save_fn = title
        plt.savefig(self.save_dir + '/' + save_fn + OUTPUT_EXT, dpi=600)

    def compare_principal_angle_discrep(
        self, X, descrep_list, s=5, title='', cmap='plasma',
        vmin=0, vmax=1, figsize=(12, 4), elev=None, azim=None,
        save_fn=''
    ):
        n_descrep = len(descrep_list)
        if X.shape[1] == 2:
            fig, axes = plt.subplots(1, n_descrep, figsize=figsize)
        else:
            fig = plt.figure(figsize=plt.figaspect(0.5))
            axes = []
            for i in range(n_descrep):
                axes.append(fig.add_subplot(1, n_descrep, i+1, projection='3d'))
            
            for i in range(n_descrep):
                axes[i].view_init(elev=elev, azim=azim)

        plt.subplots_adjust(wspace=0, hspace=0)

        for i in range(n_descrep):
            p = axes[i].scatter(*X.T, c=descrep_list[i], cmap=cmap, vmin=vmin, vmax=vmax, s=s)
            axes[i].axis('off')
            axes[i].axis('equal')
        
        fig.colorbar(p, ax=axes, orientation='vertical', shrink=0.5)
        plt.title(title)
        if save_fn == '':
            save_fn = title
        plt.savefig(self.save_dir + '/' + save_fn + OUTPUT_EXT, dpi=1200)

    def nbrhd(self, X, inds, title='', figsize=None, s=20,
              cmap='jet', azim=None, elev=None, colorbar=False, axis_image=True):
        if len(inds) < X.shape[0]:
            mask = np.zeros(X.shape[0], dtype=bool)
            mask[inds] = 1
        else:
            mask = inds

        self.data(X, mask, title=title, figsize=figsize, s=s,
              cmap=cmap, azim=azim, elev=elev, colorbar=colorbar, axis_image=axis_image)
    
    def plot_var_exp_violin_plot(self, var_exp, color='k', save_fn='', figsize=(4,1.5)):
        vmin = np.min(var_exp)
        vmax = np.max(var_exp)
        plt.figure(figsize=figsize)
        for i in range(var_exp.shape[1]):
            med_i = np.median(var_exp[:,i])
            plt.plot([0.5,i+1],[med_i,med_i], 'k--', linewidth=1, alpha=0.5)
        parts = plt.violinplot(var_exp, showextrema=False, showmedians=True)
        plt.ylim(0,1)
        plt.xlim(0.5,var_exp.shape[1]+0.5)
        for kk in ['cmaxes', 'cmins', 'cmeans', 'cbars', 'cmedians']:
            if kk not in parts:
                continue
            parts[kk].set_color('black')

        for pc in parts['bodies']:
            #pc.set_linewidth(lw)
            pc.set_alpha(1)
            if color is not None:
                pc.set_edgecolor(color)
            pc.set_facecolor(color)
        plt.xticks(np.arange(var_exp.shape[1]) + 1)
        plt.yticks([0,0.25,0.5,0.75,1])
        plt.gca().spines[['right', 'top']].set_visible(False)
        plt.tight_layout()
        if save_fn == '':
            save_fn = 'var_exp'
        plt.savefig(self.save_dir + '/' + save_fn + OUTPUT_EXT, dpi=600)
        
    def eigenvalues(self, lmbda, figsize=None):
        fig = plt.figure(figsize=figsize)
        if matplotlib.get_backend().startswith('Qt'):
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
        plt.plot(lmbda, 'o-')
        plt.ylabel('$\lambda_i$')
        plt.xlabel('i')
        plt.title('Eigenvalues')
        if self.save_dir:
            plt.savefig(self.save_dir+'/eigenvalues.png') 
        plt.show()
        
    def gamma(self, X, gamma, i, figsize=None, s=20):
        assert X.shape[1] <= 3, 'X.shape[1] must be either 2 or 3.'
        fig = plt.figure(figsize=figsize)
        if matplotlib.get_backend().startswith('Qt'):
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
        if X.shape[1] == 2:
            plt.scatter(X[:,0], X[:,1], s=s, c=gamma[:,i], cmap='jet')
            plt.axis('image')
            plt.colorbar()
        elif X.shape[1] == 3:
            ax = fig.add_subplot(projection='3d')
            p = ax.scatter(X[:,0], X[:,1], X[:,2], s=s, c=gamma[:,i], cmap='jet')
            set_axes_equal(ax)
            fig.colorbar(p)
        plt.title('$\gamma_{%d}$'%i)
        if self.save_dir:
            if not os.path.isdir(self.save_dir+'/gamma'):
                os.makedirs(self.save_dir+'/gamma')
            plt.savefig(self.save_dir+'/gamma/'+str(i)+OUTPUT_EXT) 
        plt.show()
    
    def eigenvector(self, X, phi, i, figsize=None, s=20):
        assert X.shape[1] <= 3, 'X.shape[1] must be either 2 or 3.'
        fig = plt.figure(figsize=figsize)
        if matplotlib.get_backend().startswith('Qt'):
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
        if X.shape[1] == 2:
            plt.scatter(X[:,0], X[:,1], s=s, c=phi[:,i], cmap='jet')
            plt.axis('image')
            plt.colorbar()
        elif X.shape[1] == 3:
            ax = fig.add_subplot(projection='3d')
            p = ax.scatter(X[:,0], X[:,1], X[:,2], s=s, c=phi[:,i], cmap='jet')
            set_axes_equal(ax)
            fig.colorbar(p)
        plt.title('$\phi_{%d}$'%i)
        if self.save_dir:
            if not os.path.isdir(self.save_dir+'/eigvecs'):
                os.makedirs(self.save_dir+'/eigvecs')
            plt.savefig(self.save_dir+'/eigvecs/'+str(i)+OUTPUT_EXT) 
        plt.show()
    
    def grad_phi(self, X, phi, grad_phi, i, prop=0.01, figsize=None, s=20):
        assert X.shape[1] <= 3, 'X.shape[1] must be either 2 or 3.'
        fig = plt.figure(figsize=figsize)
        if matplotlib.get_backend().startswith('Qt'):
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
        n = X.shape[0]
        np.random.seed(42)
        
        mask = np.random.uniform(0,1,n)<prop
        #mod = int(n*prop)
        #mask = np.mod(np.arange(n),mod) == 0
        
        if X.shape[1] == 2:
            plt.subplot(121)
            plt.scatter(X[:,0], X[:,1], s=s, c=phi[:,i], cmap='jet')
            plt.axis('image')
            #plt.colorbar()
            plt.title('$\phi_{%d}$'%i)
            plt.subplot(122)
            plt.quiver(X[mask,0], X[mask,1],grad_phi[mask,i,0], grad_phi[mask,i,1])
            plt.axis('image')
            plt.title('$\\nabla\\phi_{%d}$'%i)
        elif X.shape[1] == 3:
            ax = fig.add_subplot(121,projection='3d')
            p = ax.scatter(X[:,0], X[:,1], X[:,2], s=s, c=phi[:,i], cmap='jet')
            set_axes_equal(ax)
            fig.colorbar(p, ax=ax)
            plt.title('$\phi_{%d}$'%i)
            ax = fig.add_subplot(122,projection='3d')
            p = ax.quiver(X[mask,0], X[mask,1], X[mask,2], grad_phi[mask,i,0], grad_phi[mask,i,1], grad_phi[mask,i,2])
            set_axes_equal(ax)
            plt.title('$\\nabla\phi_{%d}$'%i)
        
        if self.save_dir:
            if not os.path.isdir(self.save_dir+'/grad_phi'):
                os.makedirs(self.save_dir+'/grad_phi')
            plt.savefig(self.save_dir+'/grad_phi/'+str(i)+OUTPUT_EXT) 
        plt.show()
    
    def n_eigvecs_w_grad_lt(self, X, Atilde, thresh_prctile=None, figsize=(16,8), s=20):
        assert X.shape[1] <= 3, 'X.shape[1] must be either 2 or 3.'
        fig = plt.figure(figsize=figsize)
        if matplotlib.get_backend().startswith('Qt'):
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
        
        if X.shape[1] == 3:
            ax = fig.add_subplot(122,projection='3d')
        elif X.shape[1] == 2:
            ax = fig.add_subplot(122)
        cb = None
        
        Atilde_diag = np.diagonal(Atilde, axis1=1, axis2=2)
        
        prctiles = np.arange(100)
        plt.subplot(121)
        plt.plot(prctiles, np.percentile(Atilde_diag.flatten(), prctiles), 'bo-')
        plt.xlabel('percentiles')
        plt.title('$\\widetilde{A}_{kii}$\nDouble click = Select threshold\nPress button = quit')
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        while True:
            plt.subplot(121)
            
            if thresh_prctile is None:
                to_exit = plt.waitforbuttonpress(timeout=20)
                if to_exit is None:
                    print('Timed out')
                    break

                if to_exit:
                    plt.close()
                    return
            
            
                thresh = plt.ginput(1)
                thresh = thresh[0][1]
            else:
                thresh = np.percentile(Atilde_diag.flatten(), thresh_prctile)
            
            plt.cla()
            
            plt.plot(prctiles, np.percentile(Atilde_diag.flatten(), prctiles), 'bo-')
            plt.plot([0,100], [thresh]*2, 'r-')
            plt.xlabel('percentiles')
            plt.title('$\\widetilde{A}_{kii}$, threshold = %f\nDouble click = Select threshold\nPress button = quit' % thresh)
            fig.canvas.draw()
            fig.canvas.flush_events()
        
            n_grad_lt = np.sum(Atilde_diag < thresh, 1)
        
            if X.shape[1] == 2:
                ax.cla()
                p = ax.scatter(X[:,0], X[:,1], s=s, c=n_grad_lt, cmap='jet')
                ax.axis('image')
                if cb is not None:
                    cb.remove()
                cb = fig.colorbar(p, ax=ax)
                ax.set_title('$n_k = \sum_{i}\widetilde{A}_{kii} < %f$'% thresh)
            elif X.shape[1] == 3:
                ax.cla()
                ax.autoscale()
                p = ax.scatter(X[:,0], X[:,1], X[:,2], s=s, c=n_grad_lt, cmap='jet')
                set_axes_equal(ax)
                if cb is not None:
                    cb.remove()
                cb = fig.colorbar(p, ax=ax)
                ax.set_title('$n_k = \sum_{i}\widetilde{A}_{kii} < %f$'% thresh)   
            if self.save_dir:
                if not os.path.isdir(self.save_dir+'/n_eigvecs_w_grad_lt'):
                    os.makedirs(self.save_dir+'/n_eigvecs_w_grad_lt')
                plt.savefig(self.save_dir+'/n_eigvecs_w_grad_lt/'+str(thresh)+OUTPUT_EXT)
            plt.show()
            
            if thresh_prctile is not None:
                break
    
    def distortion(self, X, zeta, title, figsize=None, s=20):
        assert X.shape[1] <= 3, 'X.shape[1] must be either 2 or 3.'
        fig = plt.figure(figsize=figsize)
        if matplotlib.get_backend().startswith('Qt'):
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
        if X.shape[1] == 1:
            plt.scatter(X[:,0], X[:,0], s=s, c=zeta, cmap='jet')
            plt.axis('image')
            plt.colorbar()
        elif X.shape[1] == 2:
            plt.scatter(X[:,0], X[:,1], s=s, c=zeta, cmap='jet')
            plt.axis('image')
            plt.colorbar()
        elif X.shape[1] == 3:
            ax = fig.add_subplot(projection='3d')
            p = ax.scatter(X[:,0], X[:,1], X[:,2], s=s, c=zeta, cmap='jet')
            set_axes_equal(ax)
            fig.colorbar(p)
        plt.title(title)
        if self.save_dir:
            plt.savefig(self.save_dir+'/'+title+OUTPUT_EXT) 
        plt.show()
    
    def distortion_boxplot(self, zeta, title, figsize=None):
        fig = plt.figure(figsize=figsize)
        plt.boxplot([zeta],labels=[title], notch=True, patch_artist=True)
        if self.save_dir:
            plt.savefig(self.save_dir+'/box_'+title+OUTPUT_EXT) 
        plt.show()
        
    def global_distortion_viloinplot_overlay(self, dist_dicts, label=r'$\log(\mathcal{G}_k)$',
                                     title='',
                                     log_scale=True, figsize=None,
                                     color=None, widths=0.5, lw=3,
                                     offset1=1, offset2=0.01,
                                     legend=True, rotation=0, filled=False, loc_='lower right',
                                     bbox_to_anchor_=(1,-0.05,0.35,0.35),
                                     vert=True, ncol=1, columnspacing=2, prop={'size':26},
                                     remove_annotation = True,
                                     showmedians=False, showmeans=False,showextrema=False,
                                     plotlinethrghmedians=False, plotlinethrghmeans=False,
                                     stat_color='red', save_fn=''):
        lw = 1
        # create figure and axes
        fig, ax = plt.subplots(figsize=figsize)
        global_ticks = []
        ticks = []
        ticklabels = []
        x0 = 0
        for ex_name,dist_dict in dist_dicts.items():
            ticks.append(x0)
            ticklabels.append(ex_name)
            local_ticks = []
            local_medians = []
            local_means = []
            for k in dist_dict.keys():
                if log_scale:
                    data=pd.DataFrame(dist_dict[k]).applymap(lambda x: np.log(x))
                else:
                    data=pd.DataFrame(dist_dict[k])
                parts = ax.violinplot(data, [x0], showmeans=showmeans,
                                       showmedians=showmedians,
                                       showextrema=showextrema,
                                       widths=widths, vert=vert)

                for kk in ['cmaxes', 'cmins', 'cmeans', 'cbars', 'cmedians']:
                    if kk not in parts:
                        continue
                    parts[kk].set_color(stat_color)

                global_ticks.append(x0)
                local_ticks.append(x0)
                if showmedians:
                    local_medians.append(np.median(data))
                if showmeans:
                    local_means.append(np.mean(data))
                if vert:
                    x0 += offset2
                else:
                    x0 -= offset2
                for pc in parts['bodies']:
                    pc.set_linewidth(lw)
                    pc.set_alpha(1)
                    if color is not None:
                        pc.set_edgecolor(color[k])
                    if not filled:
                        pc.set_facecolor('none')
                    else:
                        pc.set_facecolor(color[k])
                    pc.set_label(k)
            if vert:
                x0 += offset1
            else:
                x0 -= offset1
            if showmedians and plotlinethrghmedians:
                if vert:
                    ax.plot(local_ticks,local_medians,'k--', alpha=0.75, lw=1)
                else:
                    ax.plot(local_medians,local_ticks,'k--', alpha=0.75, lw=1)
            if showmeans and plotlinethrghmeans:
                if vert:
                    ax.plot(local_ticks,local_means,'k--', alpha=0.75, lw=1)
                else:
                    ax.plot(local_means,local_ticks,'k--', alpha=0.75, lw=1)

        if legend:
            custom_lines = [
                Line2D([0], [0], color=color[k], lw=lw, alpha=1) 
                for k in color.keys()
            ]
            if bbox_to_anchor_ is None:
                ax.legend(
                    custom_lines, 
                    [k for k in color.keys()],
                    loc=loc_,
                    ncol=ncol,
                    columnspacing=columnspacing,
                    prop=prop
                )
            else:
                ax.legend(
                    custom_lines, 
                    [k for k in color.keys()],
                    loc=loc_,
                    bbox_to_anchor=bbox_to_anchor_,
                    ncol=ncol,
                    columnspacing=columnspacing,
                    prop=prop
                )
        
        if vert:
            ax.set_xticks(ticks)
            ax.set_xticklabels(ticklabels, rotation=rotation)
            plt.ylabel(label)
        else:
            ax.set_yticks(ticks)
            ax.set_yticklabels(ticklabels, rotation=rotation)
            plt.xlabel(label)
        plt.title(title)
        plt.tight_layout()
        
        if remove_annotation:
            plt.ylabel('')
            plt.title('')
            plt.gca().spines[['right', 'top']].set_visible(False)
            yticks = np.arange(1+int(np.round(plt.gca().get_ylim()[1])))
            yticklabels = yticks.copy().tolist()

            yticklabels[0] = str(yticklabels[0])
            yticklabels[-1] = str(yticklabels[-1])
            for i in range(1,len(yticklabels)-1):
                yticklabels[i] = ''
            plt.gca().xaxis.set_ticklabels([])
            plt.yticks(yticks, yticklabels)
            
        if self.save_dir:
            if save_fn == '':
                save_fn = title
            plt.savefig(self.save_dir+'/'+save_fn+OUTPUT_EXT)
        return global_ticks
        
    def global_distortion_viloinplot(self, dist_dict, ylabel='$\log(D_k)$',
                                     title='violinplot for $\log(D_k)$',
                                     log_scale=True, figsize=None):
        fig = plt.figure(figsize=figsize)
        if log_scale:
            sns.violinplot(data=pd.DataFrame(dist_dict).applymap(lambda x: np.log(x)))
            plt.ylabel('$\log(D_k)$')
            plt.title('Violinplot for $\log(D_k)$')
        else:
            sns.violinplot(data=pd.DataFrame(dist_dict))
            plt.ylabel('$D_k$')
            plt.title('Violinplot for $D_k$')
        if self.save_dir:
            plt.savefig(self.save_dir+'/global_distortion_violinplot.eps', format='eps') 
        plt.show()

    def global_embedding(
        self, y, labels, cmap0, color_of_pts_on_tear=None, cmap1=None,
        title=None, figsize=(2,2), s=30, set_title=False, elev=None, azim=None, roll=None,
        vmin=None, vmax=None, show_axis=False, ax=None, colorbar=False, save_fn=''
    ):
        d = y.shape[1]
        if d == 1:
            y = np.concatenate([y,y],axis=1)
        d = y.shape[1]
        if d > 3:
            return
        
        if ax is None:
            fig = plt.figure(figsize=figsize)
        if matplotlib.get_backend().startswith('Qt'):
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
        
        if d == 2:
            if ax is None:
                ax = fig.add_subplot()
            p_handle = ax.scatter(y[:,0], y[:,1], s=s, c=labels, cmap=cmap0, vmin=vmin, vmax=vmax)
            ax.axis('image')
        elif d == 3:
            if ax is None:
                ax = fig.add_subplot(projection='3d')
            p_handle = ax.scatter(y[:,0], y[:,1], y[:,2], s=s, c=labels, cmap=cmap0, vmin=vmin, vmax=vmax)
            set_axes_equal(ax)

        if colorbar:
            plt.colorbar(p_handle, ax=ax)
        
        if (color_of_pts_on_tear is not None):
            if len(color_of_pts_on_tear.shape)==1:
                pts_on_tear = ~np.isnan(color_of_pts_on_tear)
            else:
                pts_on_tear = ~np.isnan(color_of_pts_on_tear[:,0])
            if cmap1 == 'colorcube' and (len(color_of_pts_on_tear.shape)==1):
                uniq_vals = np.sort(np.unique(color_of_pts_on_tear[pts_on_tear]))
                mymap = {}
                ctr = 0
                for i in range(uniq_vals.shape[0]):
                    mymap[uniq_vals[i]] = i

                cc_map = colorcube(uniq_vals.shape[0])
                n_ = color_of_pts_on_tear.shape[0]
                c_ = np.ones((n_,4))
                for i in range(n_):
                    if ~np.isnan(color_of_pts_on_tear[i]):
                        c_[i,:3] = cc_map[mymap[color_of_pts_on_tear[i]],:]
                    else:
                        c_[i,3] = 0

                if d == 2:
                    ax.scatter(y[:,0], y[:,1], s=s, c=c_)
                elif d==3:
                    ax.scatter(y[:,0], y[:,1], y[:,2], s=s, c=c_)
                    set_axes_equal(ax)
            else:
                if d == 2:
                    ax.scatter(y[pts_on_tear,0], y[pts_on_tear,1],
                               s=s, c=color_of_pts_on_tear[pts_on_tear], cmap=cmap1, vmin=None, vmax=None)
                elif d==3:
                    ax.scatter(y[pts_on_tear,0], y[pts_on_tear,1], y[pts_on_tear,2],
                               s=s, c=color_of_pts_on_tear[pts_on_tear], cmap=cmap1, vmin=None, vmax=None)
                    set_axes_equal(ax)
        
        if set_title:
            ax.set_title(title)
        if elev is not None:
            ax.view_init(elev=elev, azim=azim, roll=roll)
            title += f"_%d,%d,%d"%(elev, azim, roll)
        
        if not show_axis:
            ax.axis('off')
            plt.gca().set_axis_off()
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                        hspace = 0, wspace = 0)
            plt.margins(0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
        if self.save_dir:
            if save_fn == '':
                save_fn = title
            plt.savefig(self.save_dir+'/'+save_fn+OUTPUT_EXT, bbox_inches = 'tight')
        #plt.show()
        return ax

def plot_detected_boundary(
    X, bx_estimates, labels, max_prctile = 10, elev=None, azim=None, save_fpath='', s=5, figsize=(3,3)
):
    #max_prctile = 100*np.sum(ddX<0.015)/len(ddX)
    #colors = np.array([[82, 143, 178],[255,0,0]])/255
    colors = np.array([[255,200,0],[0,0,255]])/255

    n_estimates = len(bx_estimates)
    assert len(labels) == n_estimates, 'len(labels) != len(bx_estimates).'

    ambient_dim = X.shape[1]
    assert ambient_dim in [2,3], 'ambient_dim must be in [2,3].'

    fig = plt.figure(figsize=figsize)
    axes = []
    for i in range(n_estimates):
        if ambient_dim == 3:
            axes.append(fig.add_subplot(1, n_estimates, i+1, projection='3d'))
        else:
            axes.append(fig.add_subplot(1, n_estimates, i+1))
    
    for i in range(n_estimates):
        bdry = bx_estimates[i]<np.percentile(bx_estimates[i], max_prctile)
        axes[i].scatter(*X[~bdry,:].T, color=colors[0], s=s)
        axes[i].scatter(*X[bdry,:].T, color=colors[1], s=2*s)
        if ambient_dim == 3:
            axes[i].view_init(elev=elev, azim=azim)
        axes[i].axis('off')
        axes[i].axis('image')
        axes[i].set_title(labels[i])

    plt.tight_layout()
    if save_fpath:
        plt.savefig(save_fpath)
    plt.show()
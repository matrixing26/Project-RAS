import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_data(vx: np.ndarray,           # (N, )
              grid: np.ndarray,         # (N, 2)
              uxt: np.ndarray,          # (N, )
              out: np.ndarray = None,   # (N, )
              ext: np.ndarray = None,   # (N, )
              ):
    # plot-data
    plot_number = 2
    if out is not None:
        plot_number += 2
    if ext is not None:
        plot_number += 1
        
    fig, axs = plt.subplots(1, plot_number, figsize=(plot_number * 5, 5))

    v_space = np.linspace(0, 1, len(vx))

    for i, ax in enumerate(axs):
        ax.set_xlim(0, 1)
        if i != 0:
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
    
    axs[0].scatter(v_space, vx, s=1)
    sc = axs[1].scatter(grid[:, 0], grid[:, 1], c=uxt, cmap='viridis')
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=sc.norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax = axs[1])
    if out is not None:
        sc = axs[2].scatter(grid[:, 0], grid[:, 1], c=out, cmap='viridis')
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=sc.norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax = axs[2])
        sc = axs[3].scatter(grid[:, 0], grid[:, 1], c=np.abs(uxt-out), cmap='viridis')
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=sc.norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax = axs[3])
    if ext is not None:
        sc = axs[4].scatter(grid[:, 0], grid[:, 1], c=ext, cmap='viridis')
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=sc.norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax = axs[4])

    plt.tight_layout()
    plt.show()
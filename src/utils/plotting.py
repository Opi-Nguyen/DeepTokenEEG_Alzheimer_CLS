import os
import numpy as np
import matplotlib.pyplot as plt

def savefig(path: str, dpi=150):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()

def plot_timeseries_compare(x_list, labels, title, out_path, max_points=None):
    """
    x_list: list of 1D arrays (same length)
    """
    plt.figure()
    for x, lab in zip(x_list, labels):
        if max_points is not None and len(x) > max_points:
            idx = np.linspace(0, len(x)-1, max_points).astype(int)
            x = x[idx]
        plt.plot(x, label=lab)
    plt.legend()
    plt.title(title)
    savefig(out_path)

def plot_heatmap(mat, title, out_path, xlabel="Time", ylabel="Feature"):
    plt.figure()
    plt.imshow(mat, aspect="auto")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar()
    savefig(out_path)

def plot_vector_compare(v1, v2, labels, title, out_path):
    plt.figure()
    plt.plot(v1, label=labels[0])
    plt.plot(v2, label=labels[1])
    plt.legend()
    plt.title(title)
    savefig(out_path)

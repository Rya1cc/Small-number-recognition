import numpy as np
import matplotlib.pyplot as plt

def plot_cost(costs, show=True, save_path=None):
    if not costs: return
    plt.figure()
    plt.plot(range(len(costs)), costs)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()

def save_model(path, theta, meta: dict):
    np.savez(path, theta=theta, meta=np.array([meta], dtype=object))

def load_model(path):
    d = np.load(path, allow_pickle=True)
    return d['theta'], d['meta'][0].item()

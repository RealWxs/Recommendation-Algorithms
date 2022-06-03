import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


def show(train, val):
    fig, axes = plt.subplots(1, 1, figsize=(8, 4))

    x = range(1, len(train) + 1)
    axes.plot(x, train, linestyle='-', color='#DE6B58', linewidth=1.5,label='train')
    axes.plot(x, val, linestyle='-', color='#E1A084', linewidth=1.5,label='validate')

    axes.grid(which='minor', c='lightgrey')

    axes.set_ylabel("RMSE loss")
    axes.set_xlabel("Train Epochs")
    plt.legend(loc=0)
    plt.show()


if __name__ == '__main__':
    a = np.linspace(1, 0.02, 300)
    b = np.linspace(1, 0.03, 300)
    show(a, b)

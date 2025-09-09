import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def plot_matrix(array, ax=None, title=None, xlabel=None, ylabel=None, fontsize=16, **kwargs):
    if ax is None:
        plt.figure()
        ax = plt.gca()

    kw = {'origin': 'upper',
          'interpolation': 'nearest',
          'aspect': 'equal',  # (already the imshow default)
          **kwargs,
          }
    im = ax.imshow(array, **kw)
    ax.title.set_y(1.05)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=fontsize)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    return ax, im

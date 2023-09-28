import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def nan_intep_1d(y):
    "interplate a 1d np array with nan values"

    def _nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    if np.isnan(y).all():
        return y

    if not np.isnan(y).any():
        return y

    nans, x = _nan_helper(y)
    y[nans] = np.interp(x(nans), x(~nans), y[~nans])

    return y


def nan_intep_2d(y, axis):
    "interplate a 2d np array with nan values"

    if np.isnan(y).all():
        return y

    if not np.isnan(y).any():
        return y

    h, w = y.shape

    if axis == 0:
        for i in range(w):
            y[:, i] = nan_intep_1d(y[:, i])
    elif axis == 1:
        for i in range(h):
            y[i, :] = nan_intep_1d(y[i, :])

    return y


def randn_like(x):
    y = []
    for i in range(x.size):
        y.append(random.gauss(0, 1))
    return np.array(y).reshape(x.shape)


def visualize_training(x, pred_true, pred_mean, pred_std, vis_path):
    "Visualize batch prediction and ground truth"

    batch_size, m_tokens, input_dim = x.shape
    _, _, output_dim = pred_true.shape

    fig, axs = plt.subplots(4, 4, figsize=(32, 32))

    for bid in range(min(16, batch_size)):  # batches
        ax = axs[bid // 4][bid % 4]
        ax.set_xlim(0, 150)
        ax.set_ylim(-100, 0)

        for vid in range(m_tokens):  # vehicles

            nt = int(input_dim / 2)
            x_lat = x[bid, vid, 0:nt].cpu().numpy()
            x_lon = x[bid, vid, nt:].cpu().numpy()

            nt = int(output_dim / 2)
            true_lat = pred_true[bid, vid, 0:nt].cpu().numpy()
            true_lon = pred_true[bid, vid, nt:].cpu().numpy()

            pred_lat_mean = pred_mean[bid, vid, 0:nt].cpu().numpy()
            pred_lon_mean = pred_mean[bid, vid, nt:].cpu().numpy()

            pred_lat_std = pred_std[bid, vid, 0:nt].cpu().numpy()
            pred_lon_std = pred_std[bid, vid, nt:].cpu().numpy()

            if (x_lat == 0).sum() >= 1:
                continue

            x_lat = x_lat[x_lat != 0]
            x_lon = x_lon[x_lon != 0]
            true_lat = true_lat[true_lat != 0]
            true_lon = true_lon[true_lon != 0]

            ells = [Ellipse(xy=(pred_lat_mean[ii], pred_lon_mean[ii]),
                            width=6 * pred_lat_std[ii], height=6 * pred_lon_std[ii])
                    for ii in range(len(pred_lat_mean))]
            for e in ells:
                ax.add_artist(e)
                e.set_fill(False)
                e.set_edgecolor('green')

            ax.plot(np.concatenate([x_lat, true_lat]), np.concatenate([x_lon, true_lon]), linewidth=2, color='red')
            ax.plot(np.concatenate([x_lat, pred_lat_mean]), np.concatenate([x_lon, pred_lon_mean]), linewidth=2,
                    color='green')
            ax.plot(x_lat, x_lon, linewidth=2, color='black')
            ax.legend(['truth', 'pred', 'input'], loc=1)

    plt.savefig(vis_path)
    plt.close('all')


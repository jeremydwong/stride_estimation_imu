import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Dict

def plt_ltrl_frwd_strides(strides: Dict[str, Any]):
    """Plot lateral vs forward strides."""
    ltrl = strides['ltrl']
    frwd = strides['frwd']
    n = ltrl.shape[0]
    colors = plt.cm.jet(np.linspace(0, 1, n))
    plt.figure()
    for i in range(n):
        plt.plot(ltrl[i, :], frwd[i, :], color=colors[i])
        plt.plot(ltrl[i, -1], frwd[i, -1], 'ko', markerfacecolor='k', markersize=5)
    plt.grid(True)
    plt.ylabel('Frwd [m]')
    plt.xlabel('Ltrl [m]')
    plt.show()

def plt_frwd_elev_strides(strides: Dict[str, Any]):
    """Plot forward vs elevation strides."""
    frwd = strides['frwd']
    elev = strides['elev']
    n = frwd.shape[0]
    colors = plt.cm.jet(np.linspace(0, 1, n))
    plt.figure()
    for i in range(n):
        elev_i = elev[i, :]
        delev = np.diff(elev_i)
        NN = np.where(delev != 0)[0]
        if NN.size > 0:
            NN = NN[-1] + 1
            elev_i = elev_i[:NN]
            err = elev_i[-1]
            elev_i = elev_i - np.linspace(0, err, NN)
            elev_i = np.pad(elev_i, (0, frwd.shape[1] - NN), 'constant')
        plt.plot(-elev_i, color=colors[i])
        plt.plot(frwd[i, -1], -elev_i[-1], 'ko', markerfacecolor='k', markersize=5)
    plt.grid(True)
    plt.ylabel('Elevation [m]')
    plt.xlabel('Forward [m]')
    plt.show()

def plt_stride_var(strides: Dict[str, Any]):
    """Plot stride variability ellipse and points."""
    ltrl = strides['ltrl']
    frwd = strides['frwd']
    
    # Handle empty strides
    if ltrl.size == 0 or frwd.size == 0:
        plt.figure()
        plt.grid(True)
        plt.ylabel('Frwd [m]')
        plt.xlabel('Ltrl [m]')
        plt.title('No strides detected')
        plt.show()
        return
    
    # Extract final position for each stride
    if ltrl.ndim == 2:
        ltrl = ltrl[:, -1]
        frwd = frwd[:, -1]
    
    cx, cy, ex, ey, covar = compute_cov(ltrl, frwd)
    colors = plt.cm.jet(np.linspace(0, 1, len(ltrl)))
    plt.figure()
    for i in range(len(ltrl)):
        plt.plot(cx[i], cy[i], 'o', markeredgecolor='k', markerfacecolor=colors[i], markersize=5)
    plt.plot(ex, ey, color='r', linewidth=3)
    plt.grid(True)
    plt.ylabel('Frwd [m]')
    plt.xlabel('Ltrl [m]')
    plt.axis('equal')
    plt.show()

def compute_cov(x, y, shift_to_zero=True):
    x = np.asarray(x)
    y = np.asarray(y)
    data = np.column_stack([x, y])
    data_center = np.mean(data, axis=0)
    data = data - data_center
    sigma = 1
    from scipy.stats import chi2, norm
    confidence_value = norm.cdf(sigma) - norm.cdf(-sigma)
    scale = chi2.ppf(confidence_value, 2)
    covar = np.cov(data, rowvar=False) * scale
    vals, vecs = np.linalg.eigh(covar)
    t = np.linspace(0, 2 * np.pi, 100)
    e = np.array([np.cos(t), np.sin(t)])
    VV = vecs @ np.diag(np.sqrt(vals))
    e = VV @ e
    ex = e[0, :]
    ey = e[1, :]
    cx = data[:, 0]
    cy = data[:, 1]
    if not shift_to_zero:
        ex = ex + np.mean(x)
        ey = ey + np.mean(y)
        cx = cx + np.mean(x)
        cy = cy + np.mean(y)
    return cx, cy, ex, ey, covar 
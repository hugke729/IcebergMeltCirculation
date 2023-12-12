import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from IcebergMeltCirculation import melt_induced_circulation


def define_consts():
    f = 1.37e-4
    g = 9.81
    rho0 = 1000
    betaS = 8e-4
    Nz = 24
    return f, g, rho0, betaS, Nz


def define_grid():
    width = 5400
    xmax = 100e3
    x_edge = np.linspace(-xmax, xmax, 301)
    x_center = (x_edge[1:] + x_edge[:-1])/2

    Ny = 19
    y_edge = np.linspace(-width/2, width/2, Ny)
    y_center = (y_edge[1:] + y_edge[:-1])/2

    H = 300
    Nz = 24
    z_edge = np.linspace(0, -H, Nz + 1)
    z_center = (z_edge[1:] + z_edge[:-1])/2

    return x_center, x_edge, y_center, y_edge, z_center, z_edge, H


def define_starting_densities():
    drho = 0.015
    drho_dz = -0.005
    rhoR = rho0 + drho_dz*z_center
    rhoL = rhoR.copy()
    rhoL[:Nz//3] -= 2*drho
    return rhoL, rhoR


def define_oblique_matrix(zscale):
    # Create a matrix that projects 2D slices into 3D
    yscale = np.sqrt(2)/4  # Value for cabinet projection
    yscale *= 6  # Empirically better value for plotting
    M = np.matrix([[1, yscale, 0],
                   [0, 0, 0],
                   [0, yscale, zscale]])
    return M


def get_xy_from_rotation(unrot, zscale):
    M = define_oblique_matrix(zscale)
    xproj, yproj = (M*unrot)[[0, -1], :]
    xproj = np.array(xproj).flatten()
    yproj = np.array(yproj).flatten()
    return xproj, yproj


def red_yellow_white_cyan_blue():
    N = 21
    cols = ['#000055', '#0000f5', '#008cff',
            '#9ee6e6', '#ececec', '#eaea99',
            '#ff9d0c', '#ff0500', '#5f0000']
    return LinearSegmentedColormap.from_list('custom', cols, N)


def pcolor_3D(U, x_edge, y_edge, z_edge):
    fig, ax = plt.subplots(figsize=[4, 2.5], constrained_layout=True)
    X, Y = np.meshgrid(x_edge, y_edge)
    # r stands for rotated
    Xr, Yr = [np.zeros_like(X) for _ in range(2)]

    zscale = 200
    vmax = 0.03
    cmap = red_yellow_white_cyan_blue()
    pcol_opts = dict(cmap=cmap, vmax=vmax, vmin=-vmax)
    line_opts = dict(lw=0.5, color='k')

    # Vertical slice at back
    cax = ax.pcolormesh(x_edge, z_edge*zscale, U[:, -1, :], zorder=-1, **pcol_opts)

    # Three horizontal slices
    for zi, z in zip([23, 12, 0], [z_edge[-1], z_edge[12], z_edge[0]]):
        for yi, xi in np.ndindex(Xr.shape):
            M = np.matrix([X[yi, xi], Y[yi, xi] - y_edge.max(), z]).T
            Xr[yi, xi], Yr[yi, xi] = get_xy_from_rotation(M, zscale)

        ax.plot(Xr[0, :], Yr[0, :], zorder=1, **line_opts)
        ax.plot(Xr[-1, :], Yr[-1, :], zorder=1, **line_opts)
        cax = ax.pcolormesh(Xr, Yr, U[zi, ...], zorder=0, **pcol_opts)

    xticks = np.linspace(-100e3, 100e3, 5)
    xticklabels = [str(int(x/1e3)) for x in xticks]
    ax.set(xticks=xticks, xticklabels=xticklabels, title='x (km)',
           yticks=[])
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    cbar = fig.colorbar(cax, ax=ax, orientation='horizontal', shrink=0.5)
    cbar.set_label('Velocity (m/s)')


if __name__ == '__main__':
    f, g, rho0, betaS, Nz = define_consts()
    x_center, x_edge, y_center, y_edge, z_center, z_edge, H = define_grid()
    rhoL, rhoR = define_starting_densities()

    t0 = 1.5*86400
    U_3D = melt_induced_circulation(
        rhoL, rhoR, t0, x=x_center, z=z_center, y=y_center, tau=None, f=f, L=None)
    pcolor_3D(U_3D, x_edge, y_edge, z_edge)

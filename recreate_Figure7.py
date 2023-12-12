import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from IcebergMeltCirculation import melt_induced_circulation


def define_consts():
    g = 9.81  # m/s^2
    rho0 = 1000  # kg/m3
    betaS = 8e-4  # per PSU
    Nz = 24
    return g, rho0, betaS, Nz


def define_grid():
    xmax = 300e3
    L = 10e3
    x_edge = np.linspace(-L, xmax, 601)
    x_center = (x_edge[1:] + x_edge[:-1])/2
    # For plotting sake, I want to exaggerate size of x < 0 region
    x_plot = x_edge.copy()
    x_plot[x_plot < 0] *= 2.5

    H = 300
    z_edge = np.linspace(0, -H, Nz + 1)
    z_center = (z_edge[1:] + z_edge[:-1])/2
    return x_center, x_plot, L, z_edge, z_center, H


def define_starting_densities():
    drho_dz = -0.01333
    rhoR = rho0 + drho_dz*z_center
    rhoL = rhoR.copy()
    drho = 0.3
    rhoL[:Nz//3] -= 2*drho
    return rhoL, rhoR


def create_figure():
    fig, ax = plt.subplots(
        nrows=2, figsize=[4, 5], sharey=True, sharex=True)
    return fig, ax


def red_yellow_white_cyan_blue():
    N = 256
    cols = ['#000055', '#0000f5', '#008cff',
            # '#7affff', '#ffffff', '#ffff83',
            '#9ee6e6', '#ececec', '#eaea99',
            '#ff9d0c', '#ff0500', '#5f0000']
    return LinearSegmentedColormap.from_list('custom', cols, N)


def plot_results(U_top, U_bot):
    fig, ax = create_figure()
    # Panel (a)
    vmax = 0.2
    cax = ax[0].pcolormesh(
        x_plot/1e3, z_edge, U_top,
        vmax=vmax, vmin=-vmax, cmap=red_yellow_white_cyan_blue())
    cbar = fig.colorbar(cax, ax=ax[0])
    cbar.set_label('U (m/s)')

    # Panel (c)
    vmax = 0.02
    cax = ax[1].pcolormesh(
        x_plot/1e3, z_edge, U_bot,
        vmax=vmax, vmin=-vmax, cmap=red_yellow_white_cyan_blue())
    ax[1].set(xlabel='x (km)', ylabel='z (m)')
    cbar = fig.colorbar(cax, ax=ax[1])
    cbar.set_label('U (m/s)')


if __name__ == '__main__':
    g, rho0, betaS, Nz = define_consts()
    x_center, x_plot, L, z_edge, z_center, H = define_grid()
    rhoL, rhoR = define_starting_densities()

    # Panel a
    t0 = 260e3
    U_top = melt_induced_circulation(
        rhoL, rhoR, t0, x_center, z_center, L=L, tau=None, y=None)

    tau = 86400*10  # 10 days
    U_bot = melt_induced_circulation(
        rhoL, rhoR, t0, x_center, z_center, L=L, tau=tau, y=None)

    plot_results(U_top, U_bot)

# Analytical model associated with Hughes (2023)
# Main function 'melt_induced_circulation' is at the bottom of the script
from numpy import (
    cosh, max, roll, sum, allclose, arange, argsort, cos, errstate, flip,
    gradient, insert, real, atleast_1d, diag, diff, exp, isclose, pi, sign,
    sqrt, zeros, zeros_like)

from scipy.linalg import eig
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid

# Constants used throughout
g = 9.81
rho0 = 1025


def check_inputs(x, y, z, L, tau):
    # Ensure all grid inputs are numpy arrays
    x = atleast_1d(x)
    z = atleast_1d(z)
    nx = x.size
    nz = z.size

    if y is not None:
        y = atleast_1d(y)
        ny = y.size
        yIsSymmetric = allclose(y[y < 0], -y[y > 0][::-1])
        assert yIsSymmetric, 'If set, y must be symmetric about 0'
    else:
        ny = 1

    def isEvenSpaced(vec):
        return (vec.size == 1) or all(isclose(diff(vec), diff(vec)[0]))

    assert isEvenSpaced(x), 'x must be evenly spaced'
    assert isEvenSpaced(z), 'z must be evenly spaced'
    assert all(z < 0), 'z must be negative'

    if L is not None:
        assert L > 0, 'If set, L must be positive'

    if tau is not None:
        assert tau > 0, 'If set, tau must be positive'

    return x, y, z, nx, ny, nz


def sech(x):
    return 1/cosh(x)


def define_d2_dz2_matrix(zp1):
    """
    Create the matrix operator for d^2/dz^2 with boundary conditions of y = 0
    """
    nz = len(zp1)
    D2 = zeros([nz, nz])
    # Centered difference where possible
    for i in range(1, nz - 1):
        D2[i, i-1:i+2] = [1, -2, 1]

    # Boundary conditions
    D2[0, 0] = 1
    D2[-1, -1] = 1

    # Scale
    dz = calc_dz(zp1)
    D2 /= dz**2

    return D2


def calc_depth(z):
    """
    Return positive depth H
    """
    dz = calc_dz(z)
    H = -z[-1] - dz/2
    return H


def calc_half_width(y):
    """
    Return positive half-width W
    """
    dy = calc_dy(y)
    W = y[-1] + dy/2
    return W


def calc_dz(z):
    """
    Return single negative value for dz

    Input can be either z or zp1
    """
    dz = diff(z).mean()
    return dz


def calc_dy(y):
    """
    Return single positive value for dy
    """
    dy = diff(y).mean()
    return dy


def calc_density_gradient(rho, z):
    """
    If stratification is constant, return a single value for drho/dz
    If stratification is variable, return drho/dz on same grid as z

    Output is negative
    """
    drho_dz = gradient(rho)/gradient(z)

    assert all(drho_dz <= 0), 'Density gradient is not negative everywhere'

    if allclose(drho_dz, drho_dz[0]):
        return drho_dz[0]
    else:
        return drho_dz


def interp_z_to_zp1(arr, z, zp1):
    opts = dict(kind='linear', bounds_error=False, fill_value='extrapolate')
    return interp1d(z, arr, **opts)(zp1)


def calc_mode_shapes_and_speeds(drho_dz, z):
    """
    Output
    ------
    c: Nz-element array with wave speeds for each mode with fastest first
    phi: (Nz, Nz) array of mode shapes
    """
    N2 = -g/rho0*drho_dz

    if isinstance(N2, float):
        # Constant stratification so we can use analytical expressions
        N = sqrt(N2)
        n = arange(len(z))
        with errstate(divide='ignore'):
            c = N*H/(n*pi)
            c[0] = sqrt(g*H)  # Not actually used anywhere

        arg = (pi/H)*z[:, None]*n[None, :]
        phi = cos(arg)

    else:
        # Arbitrary stratification so we must solve numerically
        # Solve on zp1 grid and interpolate back at the end
        dz = calc_dz(z)
        zp1 = insert(z + dz/2, 0, 0)
        ddz2 = define_d2_dz2_matrix(zp1)
        N2 = interp_z_to_zp1(N2, z, zp1)
        A = diag(-N2)
        # Solve the eigenvalue problem for vertical velocity mode shape
        cinv2, phi_w = eig(ddz2, A)
        cinv = sqrt(cinv2)
        c = real(1/cinv)
        # Sort fastest modes first
        idx = flip(argsort(c))
        c = c[idx]
        phi_w = phi_w[:, idx]

        # Make all w mode shapes initially increase downward from surface
        # Not necessary, but makes things tidier
        for mode in range(phi_w.shape[1]):
            if diff(phi_w[:2, mode]) < 0:
                phi_w[:, mode] *= -1

        # We want vertical derivative of vertical velocity mode shape
        phi = gradient(phi_w, axis=0)

        # Now convert back from zp1 grid to z grid
        phi = (phi[1:, :] + phi[:-1, :])/2

        # And make output match analytical form, which means
        # there's a mode-0 shape at the start
        phi = roll(phi[:, :-1], 1, axis=-1)
        phi[:, 0] = 1
        c = roll(c[:-1], 1)
        c[0] = sqrt(g*H)

        # Normalize mode shapes so all have a maximum of 1
        phi = phi/max(phi, axis=0)

    return c, phi


def calc_rho_and_p_prime(rhoL, rhoR, z):
    """
    Return p' on z grid
    """
    # We're integrating from surface to grid centers, so we need to temporarily
    # add the surface points
    rho_prime = (rhoR - rhoL)/2
    rho_prime_w_surf = insert(rho_prime, 0, rho_prime[0])
    z_w_surf = insert(z, 0, 0)
    p_prime = g/rho0*cumulative_trapezoid(rho_prime_w_surf, x=z_w_surf)
    return rho_prime, p_prime


def calc_p_prime_coefficients(p_prime, phi, z):
    dz = calc_dz(z)
    H = calc_depth(z)
    N_coeffs = phi.shape[1]
    A_P = zeros(N_coeffs)
    for n in range(1, N_coeffs):
        A_P[n] = (2/H)*sum(p_prime*phi[:, n]*abs(dz))

    return A_P


def convert_p_prime_coefficients_to_U(A_P, phi, z, L, rhoL, rhoR):
    """
    A_Un = C n A_P

    where C comes from equating Ep(t=0) to Ek(t=inf)
    """
    dz = calc_dz(z)
    ns = arange(len(A_P))
    A_U = ns*A_P

    # Calculate final kinetic energy without the constant C
    U_end = zeros_like(z)
    Ek_end = 0
    for n in ns:
        U_n = A_U[n]*phi[:, n]
        U_end += U_n

        if L is not None:
            # For the closed case, we want to sum as we go
            Ek_end += sum(rho0*L*U_n**2*abs(dz))

    if L is None:
        # For the open case, we calc Ek after adding all modes
        Ek_end = sum(0.5*rho0*U_end**2*abs(dz))

    # Now do potential energy at the start
    # Expression below is same as g^2/(2 rho0) * integral(rho'^2/N^2 dz)
    rho_prime = calc_rho_and_p_prime(rhoL, rhoR, z)[0]
    drho_dz = calc_density_gradient(rhoR, z)

    Ep_start = 0.5*g*sum(-rho_prime**2/drho_dz)*abs(dz)
    if L is not None:
        Ep_start *= 2*L

    # Scale velocity coefficients such that Ek = Ep
    Escale = Ek_end/Ep_start
    Uscale = sqrt(Escale)
    A_U /= Uscale
    return A_U


def calc_velocity_at_x(A_U, phi, z, x, t, f, c, L, y=None):
    is2D = y is None
    isClosed = L is not None

    if is2D:
        U = zeros_like(z)
        W = 0
    else:
        U = zeros([z.size, y.size])
        W = calc_half_width(y)

    if isClosed and (x < -L):
        # Outside the domain. No need to calculate anything
        return U

    n_E = sum(c[1:]*t > abs(x))
    for n in range(1, n_E+1):
        U_n = A_U[n]*phi[:, n]
        if is2D:
            U += U_n
        else:
            Lr = c[n]/f
            decay_term = sech(W/Lr)*exp(-sign(x)*y/Lr)
            U += U_n[:, None]*decay_term[None, :]

    if isClosed:
        n_W = sum(c[1:]*t > x + 2*L + 2*W)
        for n in range(1, n_W + 1):
            U_n = A_U[n]*phi[:, n]
            if is2D:
                U -= U_n
            else:
                Lr = c[n]/f
                decay_term = sech(W/Lr)*exp(-y/Lr)
                U -= U_n[:, None]*decay_term[None, :]

    return U


def scale_rhoL_for_gradual_release(rhoL, rhoR, m, t, tau):
    rhoL = rhoR - t/(tau*m)*(rhoR - rhoL)
    return rhoL


def melt_induced_circulation(
        rhoL, rhoR, t, x, z, y=None, L=None, tau=None, f=1.37e-4, nmax=None):
    """
    The analytical model described by Hughes (2023)

    - If ``y`` is specified, the domain is 3D, otherwise it is 2D
    - If ``L`` is specified, the domain is closed on the left, otherwise it is open
    - If ``tau`` is specified, the forcing is gradual, otherwise it is a 'dam-break'

    Inputs
    ------
    rhoL, rhoR : array-like
        Density in kg/m^3 at points z
    t : float
        Time at which to evaluate the velocity field
    x : float or array-like
        Point or evenly-spaced horizontal grid in meters
    z : array-like
        Evenly-spaced depths in meters (negative, center of cells, see Notes)
    y : None or array-like, optional
        Evenly-spaced cross-channel distance in meters (center of cells, see Notes)
    L : float, optional
        If this length is set, then the channel is closed on the left end
    f : float, optional
        Coriolis frequency in s^-1 (defaults to f(70) = 1.37e-4)
    tau : float, optional
        Time scale it would take rhoR to reach rhoL if reduced at constant rate
    nmax : int, optional
        Include only first nmax modes

    Output
    ------
    U: 2D or 3D array in units of m/s

    - (Nz, Ny, Nx) if y is specified and len(x) > 1
    - (Nz, Ny) if y is specified and len(x) == 1
    - (Nz, Nx) if y is not specified

    Notes
    -----
    - The vertical grid input z implicitly defines H. For example, if H is 100,
      then z could be -5, -15, ..., -85, -95
    - Similarly, the cross-channel grid (if specified) implicitly defines W.
      For example, if W = 1000, then y could be -950, -850, ..., 850, 950
    - Code assumes rhoL <= rhoR
    - Assumes rhoL switches to rhoL at x = 0

    Ken Hughes, September 2023

    Example
    -------
    >>> # 2D open-channel, abrupt-release, constant stratification case
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> H = 600
    >>> zp1 = -np.linspace(0, H, 51)
    >>> z = (zp1[:-1] + zp1[1:])/2
    >>> t = 3*86400
    >>> x = 20e3
    >>> rhoR = 1025 - 3*z/H
    >>> rhoL = rhoR.copy()
    >>> rhoL[z > -100] -= 2e-3*(z[z > -100] + 100)
    >>> U = melt_induced_circulation(rhoL, rhoR, t, x, z, L=None)
    >>> # Plot result
    >>> plt.plot(U, z)
    >>> plt.xlabel('Velocity (cm/s)')
    >>> plt.ylabel('Depth (m)')

    >>> # 3D closed-channel, gradual-release case, with nonlinear stratification
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> H = 600
    >>> zp1 = -np.linspace(0, H, 51)
    >>> z = (zp1[:-1] + zp1[1:])/2
    >>> L = 10e3
    >>> t = 7*86400
    >>> tau = 7*86400
    >>> x = 20e3
    >>> f = 1.37e-4
    >>> W = 2.5e3
    >>> yp1 = W*np.linspace(-1, 1, 51)
    >>> y = (yp1[:-1] + yp1[1:])/2
    >>> rhoR = 1025 + 3*np.sqrt(-z/H)
    >>> rhoL = rhoR.copy()
    >>> rhoL[z > -100] -= 10e-3*(z[z > -100] + 100)
    >>> U = melt_induced_circulation(rhoL, rhoR, t, x, z, y=y, L=L, tau=tau, f=f)
    >>> # Plot result
    >>> contour_opts = dict(levels=0.1*np.linspace(-1, 1, 21), cmap='RdBu')
    >>> con = plt.contourf(y, z, U, **contour_opts)
    >>> cbar = plt.colorbar(con)
    >>> cbar.set_label('Velocity (m/s)')

    >>> # 3D closed-channel with melting ice from Section 5.1 of the paper
    >>> import numpy as np
    >>> from seawater.eos80 import pden
    >>> H = 600
    >>> zp1 = -np.linspace(0, H, 51)
    >>> z = (zp1[:-1] + zp1[1:])/2
    >>> L = 8e3
    >>> t = 7*86400
    >>> tau = 7*86400
    >>> x = 20e3
    >>> f = 1.37e-4
    >>> W = 2.5e3
    >>> yp1 = W*np.linspace(-1, 1, 51)
    >>> y = (yp1[:-1] + yp1[1:])/2
    >>> deltaS = 3
    >>> SR = 35 - deltaS*(1 + z/H)
    >>> TR = 2 + 0*z  # Let T denote theta
    >>> rhoR = pden(SR, TR, 0)
    >>> M = 2.5e-8*W*L*(1 + np.tanh((z+100)/25))
    >>> deltaA = M*tau
    >>> A0 = 2*W*L
    >>> SL = A0*SR/(A0 + deltaA)
    >>> T_eff = -85
    >>> TL = (A0*TR + deltaA*T_eff)/(A0 + deltaA)
    >>> rhoL = pden(SL, TL, 0)
    >>> U = melt_induced_circulation(rhoL, rhoR, t, x, z, y=y, L=L, tau=tau, f=f)
    >>> dy, dz = np.diff(yp1).mean(), np.diff(-zp1).mean()
    >>> Qout = np.sum(U[U > 0]*dy*dz)
    >>> print(f'Qout = {Qout.astype(int)} m^3/s')
    """
    x, y, z, nx, ny, nz = check_inputs(x, y, z, L, tau)
    is2D = y is None

    if tau is not None:
        m = 50
        rhoL = scale_rhoL_for_gradual_release(rhoL, rhoR, m, t, tau)
    else:
        m = 1

    drho_dz = calc_density_gradient(rhoR, z)
    c, phi = calc_mode_shapes_and_speeds(drho_dz, z)
    rho_prime, p_prime = calc_rho_and_p_prime(rhoL, rhoR, z)
    A_P = calc_p_prime_coefficients(p_prime, phi, z)
    A_U = convert_p_prime_coefficients_to_U(A_P, phi, z, L, rhoL, rhoR)

    if nmax is not None:
        A_U[nmax+1:] = 0

    if is2D:
        U = zeros([nz, nx])
        for mi in range(1, m+1):
            for i, xi in enumerate(x):
                U[:, i] += calc_velocity_at_x(
                    A_U, phi, z, xi, (mi/m)*t, f, c, L)

    else:
        U = zeros([nz, ny, nx])
        for mi in range(1, m+1):
            for i, xi in enumerate(x):
                U[:, :, i] += calc_velocity_at_x(
                    A_U, phi, z, xi, (mi/m)*t, f, c, L, y)

    U = U.squeeze()
    return U


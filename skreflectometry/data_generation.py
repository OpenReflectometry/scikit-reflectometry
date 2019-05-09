from __future__ import print_function, division, absolute_import
# from builtins import range
import numpy as np


def find_nearest_index(array, value):
    """Return the index of the closest element to `value` in `array`."""
    return (np.abs(array - value)).argmin()


def find_nearest(array, value):
    """Return the closest element to `value` in `array`."""
    return array[find_nearest_index(array, value)]


def density_profile(radius_arr=None, n_points=100, dens_central=5e19, m=2, n=5,
                    r_hfs=1.15, r_central=1.65, r_lfs=2.15, r_vacuum=0.):
    """
    Generate a typical density profile for the provided radius.

    The density is of shape
    :math:`n(r) = n_0 (1-(\\ (R_0 - r)/(R_0 - R_{edge})\\,)^m)^n` .


    Parameters
    ----------
    radius_arr : array_like, optional
        Radius array to be used. If is None, an array from 0.8 `r_hfs` to 1.2
        `r_lfs` with `n_points` will be created.
    n_points : number, optional
        Number of points to create the radius array. Ignored if `radius_arr` is
        given.
    dens_central : number, optional
        Value of the density at its highest value.
    m : int, optional
    n : int, optional
    r_hfs : number, optional
        Position of the High-Field Side edge.
    r_central : number, optional
        Position with the highest density; position at the center of the
        distribution.
    r_lfs : number, optional
        Position of the Low-Field Side edge.
    r_vacuum : number, optional
        Vacuum distance to the plasma.

    Returns
    -------
    radius_arr : ndarray
        The radius array used.
    dens_prof : ndarray
        The density profile created.

    """

    hfs_edge = r_hfs + r_vacuum
    lfs_edge = r_lfs - r_vacuum

    if radius_arr is None:
        radius_arr = np.linspace(0.8*r_hfs, 1.1*r_lfs, n_points)
    else:
        radius_arr = np.array(radius_arr)

    dens_prof = np.zeros(radius_arr.shape)

    hfs_inds = radius_arr <= r_central
    lfs_inds = ~hfs_inds

    def dens_fun(r, r_edge):
        return dens_central * np.power(1. - np.power((r_central - r) / (r_central - r_edge), m), n)

    dens_prof[hfs_inds] = dens_fun(radius_arr[hfs_inds], hfs_edge)
    dens_prof[lfs_inds] = dens_fun(radius_arr[lfs_inds], lfs_edge)

    dens_prof[np.where(dens_prof < 0)] = 0

    #print(dens_prof)

    return radius_arr, dens_prof


def density_add_bump(radius_arr, dens_prof,
                     bump_pos, bump_size=0.005, bump_height=0.95, bug=False):
    """
    Add a parabolic bump to the density profile.

    Parameters
    ----------
    radius_arr : ndarray
        Radius array to be used.
    dens_prof : ndarray
        Density profile to be modified.
    bump_pos : number
        Central position of the bump.
    bump_size : number, optional
        Diameter of the bump.
    bump_height : number, optional
        Scaling value for the central point of the bump
        (:math:`n(r_0)=\mathtt{height}*n_0(r_0)`).
    bug

    Returns
    -------
    ndarray
        Modified density profile.

    """

    bump_ind = find_nearest_index(radius_arr, bump_pos)
    bump_min_ind = find_nearest_index(radius_arr, bump_pos - bump_size/2)
    bump_max_ind = find_nearest_index(radius_arr, bump_pos + bump_size/2)
    inds = [bump_min_ind, bump_ind, bump_max_ind]

    dens_prof_alt = np.copy(dens_prof)
    if bug:
        dens_prof_alt[bump_min_ind:bump_max_ind] *= bump_height  # correct?
    dens_prof_alt[bump_ind] *= bump_height

    fit_coeffs = np.polyfit(radius_arr[inds], dens_prof_alt[inds], 2)
    fit_func = np.poly1d(fit_coeffs)
    bump_dens = fit_func(radius_arr[bump_min_ind:bump_max_ind])

    dens_prof_alt[bump_min_ind:bump_max_ind] = bump_dens

    return dens_prof_alt


def density_add_timed_gaussian(radius_arr, dens_prof, n_points=100,
                               gauss_pos=1.19, gauss_integral=0.1,
                               gauss_width=0.02, gauss_t=0.,
                               oscillation_period=20.):
    """
    Add an oscillating gaussian bump to the density profile.

    Parameters
    ----------
    radius_arr : ndarray
        Radius array to be used.
    dens_prof : ndarray
        Density profile to be modified.
    gauss_pos : number, optional
        Central position of the gaussian.
    gauss_height : number, optional
        Height of the gaussian (relative to density of that part of profile).
    gauss_width : number, optional
        width of the gaussian
    gauss_t : number, optional
        current time
    oscillation_period: number, optional

    Returns
    -------
    ndarray
        Modified density profile.

    """

    dens_prof_alt = dens_prof.copy()
    gauss_multiplier = gauss_integral*np.sin(2.*gauss_t*np.pi/oscillation_period)

    dens_prof_alt += gaussian_function(radius_arr, gauss_multiplier, gauss_pos, gauss_width)

    dens_prof_alt[dens_prof_alt < 0] = 0

    return dens_prof_alt


def gaussian_function(x, gaussian_multiplier, gaussian_pos, gaussian_width):
    normalization_fac = gaussian_multiplier / (gaussian_width*np.sqrt(2*np.pi))
    shape_func = np.exp(-(x - gaussian_pos)**2 / (2. * gaussian_width**2))

    return normalization_fac * shape_func


def magnetic_field_profile(radius_arr, mag_field_ref, pos_ref):
    """
    Generate a typical magnetic field profile.

    Magnetic field profile is of shape :math:`B(r) = B_0 R_0 / r`.

    Parameters
    ----------
    radius_arr : ndarray
        Radius (value or array) to be used.
    mag_field_ref : number
        Magnetic field at reference point.
    pos_ref : number
        Radial position of reference point.

    Returns
    -------
    ndarray
        Magnetic field profile.
    """

    return mag_field_ref * pos_ref / radius_arr

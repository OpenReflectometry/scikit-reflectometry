import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.signal import spectrogram
from skreflectometry.mode_O import cutoff_freq_O
from skreflectometry.mode_X import cutoff_freq_X
from skreflectometry.physics import upper_hybrid_frequency
from skreflectometry.reflectometry_sim import beat_maximums


def plot_refractive_matrix(radius_arr, dens_prof, f_probe, refract,
                           mag_field=None, wave_mode='O', antenna_side='hfs',
                           norm=None, axis=None, title='',
                           legend_colors=None, legend_loc='best'):
    """
    TODO
    Parameters
    ----------
    radius_arr
    dens_prof
    f_probe
    refract
    mag_field
    norm
    axis
    title
    wave_mode
    legend_colors
    legend_loc

    Returns
    -------
    None
    """

    if axis is None:
        axis = plt.gca()

    if legend_colors is None:
        legend_colors = [
            (0.0, 0.9, 0.0),
            (0.0, 0.5, 1.0),
            (0.2, 0.8, 0.8),
            (0.8, 0.2, 0.8),
        ]

    if norm is None:
        norm = colors.SymLogNorm(linthresh=1e-3, clip=True, vmin=-1, vmax=1)

    plt.pcolormesh(radius_arr, f_probe / 1e9, refract,
                   cmap='RdGy', norm=norm)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('n$^2$')

    plt.xlabel('R (m)')
    plt.ylabel('f$_{probe}$ (GHz)')
    plt.title('Refractive Matrix of ' + title)

    ax_cutoff = axis.twiny()

    if antenna_side == 'hfs':
        reflect_pos = np.argmax(refract <= 0, axis=1)
        reflect_pos[reflect_pos == 0] = -1
    elif antenna_side == 'lfs':
        reflect_pos = refract.shape[1] - 1 - \
                      np.argmax(refract[:, ::-1] <= 0, axis=1)
        reflect_pos[reflect_pos == refract.shape[1] - 1] = 0

    ax_cutoff.plot(radius_arr[reflect_pos], f_probe * 1e-9, '-o',
                   label='Reflection Point', color=legend_colors[0],
                   linewidth=3, markersize=5, alpha=0.3)

    if wave_mode == 'O':
        cutoff_freqs = [
            (cutoff_freq_O(dens_prof), 'Cut-off')
        ]
    elif wave_mode == 'X':
        f_left, f_right = cutoff_freq_X(dens_prof, mag_field)
        cutoff_freqs = [
            (f_left, 'Left Cut-off'),
            (upper_hybrid_frequency(dens_prof, mag_field), 'Upper Hybrid'),
            (f_right, 'Right Cut-off')
        ]

    for n_cutoff, (cutoff_freq, label) in enumerate(cutoff_freqs, start=1):
        ax_cutoff.plot(radius_arr, cutoff_freq * 1e-9, '-',
                       color=legend_colors[n_cutoff], linewidth=2,
                       label=label + ' Frequency')

    ax_cutoff.set_xlim(radius_arr.min(), radius_arr.max())
    ax_cutoff.set_ylim(f_probe[0] * 1e-9, f_probe[-1] * 1e-9)
    ax_cutoff.xaxis.set_ticklabels('')

    leg = plt.legend(loc=legend_loc, framealpha=.5, facecolor='k')

    for text, color in zip(leg.texts, legend_colors):
        text.set_color(color)


def plot_signal_profile(radius_arr, dens_prof, f_samp, f_probe, beat_sig,
                        radius_calc, dens_calc,
                        radius_spect, dens_spect,
                        figsize=(8, 6), title='', filename='temp',
                        beat_ylims=None, dens_xlims=None, dens_ylims=None):
    """
    TODO
    Parameters
    ----------
    radius_arr
    dens_prof
    f_probe
    radius_calc
    dens_calc
    radius_spect
    dens_spect
    figsize
    title
    filename
    dens_xlims
    dens_ylims

    Returns
    -------

    """

    plt.figure(figsize=figsize)

    # Refractive Matrix

    ax_refract = plt.subplot(1, 2, 1)

    f_spectrum, t_spectrum, spectrum = \
        spectrogram(beat_sig, fs=f_samp,
                    nperseg=136, nfft=2048, noverlap=128)
    beat_max = beat_maximums(f_spectrum, spectrum)

    plt.pcolormesh(t_spectrum * 1e6, f_spectrum * 1e-6, spectrum, cmap='hot')
    plt.plot(t_spectrum * 1e6, beat_max * 1e-6, '--', label='Maximums',
             color=(0, 0, 1), lw=3)

    if beat_ylims is not None:
        plt.ylim(*beat_ylims)
    else:
        plt.ylim(0, f_samp / 2e6)

    plt.xlabel('t ($\mu$s)')
    plt.ylabel('f$_{beat}$ (MHz)')
    plt.title('Beat Signal of ' + title)

    plt.legend(loc='lower right')

    # Density Profiles

    plt.subplot(1, 2, 2)

    plt.plot(radius_arr, dens_prof, 'c-', label='Real profile', lw=20)
    plt.fill_between(radius_arr, dens_prof, 0.0, color='k', alpha=0.25)
    plt.plot(radius_calc, dens_calc, 'b-', lw=5,
             label='Profile from ideal group delay')
    plt.plot(radius_spect, dens_spect, 'r-', lw=2.5,
             label="Profile from spectrogram group delay")

    if dens_xlims is not None:
        plt.xlim(*dens_xlims)

    if dens_ylims is not None:
        plt.ylim(*dens_ylims)

    plt.xlabel('R (m)')
    plt.ylabel('n (m$^{-3}$)')
    plt.title('Density Profiles of ' + title)

    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig('images/' + filename + '.png', dpi=200)
    plt.show()

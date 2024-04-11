import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from helper import *


def plot_stft(time ,freq, power, det = "ic86"):

    hypos = ["signal", "null"]
    im, cb = [None,None], [None,None]
    
    fig, ax = plt.subplots(1,2, figsize = (10,4))

    for i in range(2):
        im[i] = ax[i].pcolormesh(time, freq, power[hypos[i]][det][0], cmap='viridis', shading = "nearest", vmin = 0, vmax = 3.5)
        
        cb = fig.colorbar(im[i])
        cb.ax.tick_params(labelsize=14)
        cb.set_label(label=r"$|H(f,t)|^2 - \langle |H(f,t)|^2 \rangle_t$",size=14)
        ax[i].set_xlabel(r'Time $t-t_{\rm bounce}$ [ms]', fontsize=14)
        ax[i].set_ylabel(f"Frequency $f$ in [Hz]", fontsize=14)
        ax[i].yaxis.get_offset_text().set_fontsize(14)
        ax[i].tick_params(labelsize=14)

    plt.tight_layout()

def plot_ts(ts, ts_stat, bkg_fit, det = "ic86"):

    # Plot TS distribution for null and signal hypothesis for gen2

    bins = 40

    # histogram TS distribution of null and signal hypothesis for gen2 
    y_signal, bins_signal = np.histogram(ts["signal"][det], bins = bins, density = True)
    y_null, bins_null = np.histogram(ts["null"][det], bins = bins, density = True)

    # get x values
    x_signal = (bins_signal[:-1]+bins_signal[1:])/2
    x_null = (bins_null[:-1]+bins_null[1:])/2

    # get fitted background distribution
    x_fit = np.linspace(np.minimum(x_null[0],x_signal[0]), np.maximum(x_null[-1],x_signal[-1]), 100)
    y_fit = bkg_fit[det].pdf(x_fit)

    # mask range of 16% and 84% quantiles
    mask_signal = np.logical_and(x_signal > ts_stat["signal"][det][1], x_signal < ts_stat["signal"][det][2])
    mask_null = np.logical_and(x_null > ts_stat["null"][det][1], x_null < ts_stat["null"][det][2])

    fig, ax = plt.subplots(1,1)

    ax.step(x_null, y_null, where = "mid", label = r"$H_0$")
    ax.step(x_signal, y_signal, where = "mid", label = r"$H_1$")
    ax.plot(x_fit, y_fit, "k--")

    ax.axvline(ts_stat["signal"][det][0], ymin = 0, ymax =np.max(y_signal), color = 'C1', ls = '-')
    ax.axvline(ts_stat["null"][det][0], ymin = 0, ymax =np.max(y_null), color = 'C0', ls = '-')
    ax.fill_between(x = x_signal[mask_signal], y1 = y_signal[mask_signal], color = 'C1', alpha = 0.5)
    ax.fill_between(x = x_null[mask_null], y1 = y_null[mask_null], color = 'C0', alpha = 0.5)


    ax.set_xlabel("TS value", fontsize = 14)
    ax.set_ylabel("Normalized Counts", fontsize = 14)
    ax.tick_params(labelsize = 14)
    ax.grid()
    ax.legend(fontsize = 14)

    plt.tight_layout()

def plot_parareco(fit_freq, fit_time, det = "ic86"):
    # Data
    fit_freq_null, fit_time_null = fit_freq["null"][det], fit_time["null"]["ic86"]
    fit_freq_signal, fit_time_signal = fit_freq["signal"][det], fit_time["signal"]["ic86"]

    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(4, 4)

    # Main scatter plot
    ax_main = fig.add_subplot(gs[1:, :-1])
    ax_main.scatter(fit_time_null, fit_freq_null, s=10, alpha=0.1, color="C0", label=r"$H_0$")
    ax_main.scatter(fit_time_signal, fit_freq_signal, s=10, alpha=0.1, color="C1", label=r"$H_1$")
    ax_main.set_xlabel("Time [ms]", fontsize=14)
    ax_main.set_ylabel("Frequency [Hz]", fontsize=14)
    ax_main.tick_params(labelsize=14)
    ax_main.grid()

    # Top histogram
    ax_top = fig.add_subplot(gs[0, :-1], sharex=ax_main)
    ax_top.hist(fit_time_null, bins=30, range = (0, 1000), alpha=0.5, color="C0", orientation='vertical')
    ax_top.hist(fit_time_signal, bins=30, range = (0, 1000), alpha=0.5, color="C1", orientation='vertical')
    ax_top.axvline(np.median(fit_time_null), color="C0")
    ax_top.axvline(np.median(fit_time_signal), color="C1")

    ax_top.set_ylabel("Counts", fontsize=14)
    ax_top.tick_params(labelsize=14)

    # Right histogram
    ax_right = fig.add_subplot(gs[1:, -1], sharey=ax_main)
    ax_right.hist(fit_freq_null, bins=30, range = (0, 500), alpha=0.5, color="C0", orientation='horizontal')
    ax_right.hist(fit_freq_signal, bins=30, range = (0, 500), alpha=0.5, color="C1", orientation='horizontal')
    ax_right.axhline(np.median(fit_freq_null), color="C0")
    ax_right.axhline(np.median(fit_freq_signal), color="C1")
    ax_right.set_xlabel("Counts", fontsize=14)
    ax_right.tick_params(labelsize=14)

    # Adjusting legend handles
    leg = ax_main.legend(fontsize=14)
    for h in leg.legend_handles:
        h.set_alpha(1)  # Set legend handle opacity to 1 (opaque)
        h.set_sizes([30])  # Set legend handle size to 30

    plt.tight_layout()

def plot_significance(dist_range, zscore, ts_stat):

    fig, ax = plt.subplots(1,2, figsize = (16,6))
    ax = ax.ravel()

    labels = [r'$IceCube$', r'$Gen2$', r'$Gen2+WLS$']
    colors = ['C0', 'C1', 'C2']

    for i, det in enumerate(["ic86", "gen2", "wls"]):

        ax[0].plot(dist_range, zscore[det][0], label=labels[i], color = colors[i])
        ax[0].fill_between(dist_range.value, zscore[det][2], zscore[det][1], color = colors[i], alpha = 0.15)

        ax[0].set_xlabel('Distance d [kpc]', fontsize = 12)
        ax[0].set_ylabel(r'SASI detection significance [$\sigma$]' , fontsize = 12)
        ax[0].set_xlim((1,20))
        ax[0].set_ylim((1,30))

        ax[0].tick_params(labelsize = 12)

        ax[0].set_yscale('log')
        ax[0].set_yticks([1,2,3,5,10,20,30])
        ax[0].set_yticklabels(['1','2','3','5','10','20','30'])

        ax[0].grid()
        ax[0].legend(loc='upper right', fontsize = 12)

        ax[0].axhline(3, color='k', ls = ':')
        ax[0].axhline(5, color='k', ls = '-.')
        ax[0].text(8, 3, r"3$\sigma$", size=12,
                ha="center", va="center",
                bbox=dict(boxstyle="square", ec='k', fc='white'))

        ax[0].text(8, 5, r"5$\sigma$", size=12,
                ha="center", va="center",
                bbox=dict(boxstyle="square", ec='k', fc='white'))

        rates = np.array([0.1,0.5,0.75,0.9,0.95,0.99,1])
        ax22 = ax[0].twiny()
        ax22.set_xlim(ax[0].get_xlim())
        ax22.set_xticks(inv_cdf(rates).flatten())
        ax22.set_xticklabels((rates*100).astype(dtype=int), rotation = 0, fontsize = 12)
        ax22.set_xlabel('Cumulative galactic CCSNe distribution \n from Adams et al. (2013) in [%]', fontsize = 12)

        #signal without errorbar
        ax[1].plot(dist_range, ts_stat["signal"][det][0], color = colors[i], label=r'TS$^{sig}_{IceCube}$')

        #background with errorbar
        ax[1].errorbar(x = dist_range, y = ts_stat["null"][det][0],
                    yerr = (ts_stat["null"][det][0]-ts_stat["null"][det][1],ts_stat["null"][det][2]-ts_stat["null"][det][0]), 
                    capsize=4, ls = ':', color = colors[i], label=r'TS$^{bkg}_{IceCube}$')

    #rearrange legend handels
    handles,labels = ax[1].get_legend_handles_labels()

    handles = [handles[0], handles[3], handles[1], handles[4], handles[2], handles[5]]
    labels = [labels[0], labels[3], labels[1], labels[4], labels[2], labels[5]]


    ax[1].set_xlabel('Distance d [kpc]', fontsize = 12)
    ax[1].set_ylabel('Median of test statistics (TS)', fontsize = 12)
    ax[1].set_xlim((1,10))
    ax[1].set_yscale('log')
    ax[1].tick_params(labelsize = 12)
    ax[1].legend(handles, labels, ncol = 3, fontsize = 12, bbox_to_anchor=(0.13, 1))
    ax[1].grid()

    plt.tight_layout()
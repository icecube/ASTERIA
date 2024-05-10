import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import skewnorm

from helper import *


def plot_hits(time, hits, det = "ic86"):

    time = time.value * 1000 # in ms

    fig, ax = plt.subplots(1,1)
    ax.step(time, hits["null"][det][0], color = 'C0', ls = '-', label=r'$H_{0}$', zorder = 0)
    ax.step(time, hits["signal"][det][0], color = 'C1', ls = '-', alpha = 0.5, label=r'$H_{1}$', zorder = 10)
    ax.set_xlabel('Time [ms]', fontsize = 14)
    ax.set_ylabel('Counts/bin', fontsize = 14)
    ax.set_xlim(0,1000)
    ax.tick_params(labelsize = 14)
    ax.legend(fontsize = 14)

    plt.tight_layout()

def plot_fft(freq, power, det = "ic86"):

    freq = freq.value

    fig, ax = plt.subplots(1,1)
    ax.step(freq, power["null"][det][0], color = 'C0', ls = '-', label=r'$H_{0}$', zorder = 0)
    ax.step(freq, power["signal"][det][0], color = 'C1', ls = '-', alpha = 0.5, label=r'$H_{1}$', zorder = 10)
    ax.set_xlabel('Frequency [Hz]', fontsize = 14)
    ax.set_ylabel('Power [au]', fontsize = 14)
    ax.set_xlim(0,500)
    ax.set_yscale("log")
    ax.tick_params(labelsize = 14)
    ax.legend(fontsize = 14)

    plt.tight_layout()


def plot_stft(time, freq, power, det = "ic86"):

    freq = freq.value
    time = time.value

    hypos = ["signal", "null"]
    im, cb = [None,None], [None,None]
    
    fig, ax = plt.subplots(1,2, figsize = (10,4))

    for i in range(2):
        im[i] = ax[i].pcolormesh(time, freq, np.log(power[hypos[i]][det][0]), cmap='plasma', shading = "nearest", vmin = 0, vmax = 8)
        
        cb = fig.colorbar(im[i])
        cb.ax.tick_params(labelsize=14)
        cb.set_label(label=r"$\log \{ S(f,t) \}$",size=14)
        ax[i].set_xlabel(r'Time $t-t_{\rm bounce}$ [ms]', fontsize=14)
        ax[i].set_ylabel(f"Frequency $f$ in [Hz]", fontsize=14)
        ax[i].yaxis.get_offset_text().set_fontsize(14)
        ax[i].tick_params(labelsize=14)

    plt.tight_layout()

def plot_stft_log(time, freq, log_power, det = "ic86"):

    freq = freq.value
    time = time.value

    hypos = ["signal", "null"]
    im, cb = [None,None], [None,None]
    
    fig, ax = plt.subplots(1,2, figsize = (10,4))

    for i in range(2):
        im[i] = ax[i].pcolormesh(time, freq, log_power[hypos[i]][det][0], cmap='viridis', shading = "nearest", vmin = 0, vmax = 3.5)
        
        cb = fig.colorbar(im[i])
        cb.ax.tick_params(labelsize=14)
        cb.set_label(label=r"$\max_0 \left( \log \{ S(f,t) \} - \langle \log \{  S(f,t) \} \rangle_t \right)$",size=14)
        ax[i].set_xlabel(r'Time $t-t_{\rm bounce}$ [ms]', fontsize=14)
        ax[i].set_ylabel(f"Frequency $f$ in [Hz]", fontsize=14)
        ax[i].yaxis.get_offset_text().set_fontsize(14)
        ax[i].tick_params(labelsize=14)

    plt.tight_layout()

def plot_stft_time_int(freq, log_int, det = "ic86"):

    freq = freq.value

    diff = log_int["signal"][det][0] - log_int["null"][det][0]
    
    fig, ax = plt.subplots(1,1)

    ax.step(freq, log_int["null"][det][0], color = "C0", ls = ":", label=r'$H_{0}$')
    ax.step(freq, log_int["signal"][det][0], color = "C1", ls = "--", label=r'$H_{1}$')
    ax.step(freq, diff, color = "k", label = r'$H_{1} - H_{0}$')

    ax.set_xlabel('Frequency [Hz]', fontsize = 14)
    ax.set_ylabel('summed log-power', fontsize = 14)
    ax.set_xlim(0,500)
    ax.tick_params(labelsize = 14)
    ax.legend(fontsize = 14)

    plt.tight_layout()

def plot_ts(ts, det = "ic86"):

    # Plot TS distribution for null and signal hypothesis for gen2

    bins = 40

    # get 16, 50, 86% of TS distribution
    ps_null = np.percentile(ts["null"][det], [16, 50, 84])
    ps_signal = np.percentile(ts["signal"][det], [16, 50, 84])

    fig, ax = plt.subplots(1,1)

    hist_null = plt.hist(ts["null"][det], histtype="step", density=True, bins = bins, color = 'C0', lw = 2, label = r"$H_0$")
    hist_signal = plt.hist(ts["signal"][det], histtype="step", density=True, bins = bins, color = 'C1', lw = 2, label = r"$H_1$")

    # get histogram bins and values
    bin_null, bin_signal = hist_null[1], hist_signal[1]
    y_null, y_signal = hist_null[0], hist_signal[0]
    # get x values
    x_null, x_signal = (bin_null[1:]+bin_null[:-1])/2, (bin_signal[1:]+bin_signal[:-1])/2


    # get fitted background distribution
    bkg_fit = skewnorm(*skewnorm.fit(ts["null"][det]))
    x_fit = np.linspace(np.minimum(x_null[0],x_signal[0]), np.maximum(x_null[-1],x_signal[-1]), 200)
    y_fit = bkg_fit.pdf(x_fit)

    # mask range of 16% and 84% quantiles
    mask_null = np.logical_and(x_null > ps_null[0], x_null < ps_null[2])
    mask_signal = np.logical_and(x_signal > ps_signal[0], x_signal < ps_signal[2])

    # Search for the heights of the bins in which the percentiles are
    hi_null = y_null[np.searchsorted(x_null, ps_null, side='left')-1]
    hi_signal = y_signal[np.searchsorted(x_signal, ps_signal, side='left')-1]

    ax.plot(x_fit, y_fit, "k--")

    ax.vlines(ps_null[1], ymin = 0, ymax = hi_null[1], color = 'C0', ls = '-')
    ax.vlines(ps_signal[1], ymin = 0, ymax = hi_signal[1], color = 'C1', ls = '-')

    ax.fill_between(x = x_signal[mask_signal], y1 = y_signal[mask_signal], color = 'C1', alpha = 0.5)
    ax.fill_between(x = x_null[mask_null], y1 = y_null[mask_null], color = 'C0', alpha = 0.5)

    ax.set_xlabel("TS value", fontsize = 14)
    ax.set_ylabel("Normalized Counts", fontsize = 14)
    ax.tick_params(labelsize = 14)
    ax.grid()
    ax.legend(fontsize = 14)

    plt.tight_layout()

def plot_fit_freq(fit_freq, det = "ic86"):
    # Data
    fit_freq_null, fit_freq_signal = fit_freq["null"][det], fit_freq["signal"][det]

    bins = 50

    fig, ax = plt.subplots()

    ax.hist(fit_freq_null, histtype="step", density=True, bins = bins, range = (0, 500), lw = 2, color="C0", label=r"$H_0$")
    ax.hist(fit_freq_signal, histtype="step", density=True, bins = bins, range = (0, 500), lw = 2, color="C1", label=r"$H_1$")
    ax.axvline(np.median(fit_freq_null), color="C0")
    ax.axvline(np.median(fit_freq_signal), color="C1")
    ax.set_xlabel("Frequency [Hz]", fontsize=14)
    ax.set_ylabel("Counts", fontsize=14)
    ax.set_yscale("log")
    ax.tick_params(labelsize=14)
    ax.legend(fontsize = 14)

    plt.tight_layout()

def plot_fit_time_freq(fit_freq, fit_time, det = "ic86"):
    # Data    
    fit_freq_null, fit_time_null = fit_freq["null"][det], fit_time["null"]["ic86"]
    fit_freq_signal, fit_time_signal = fit_freq["signal"][det], fit_time["signal"]["ic86"]

    bins = 50

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
    ax_top.hist(fit_time_null, bins=bins, range = (0, 1000), alpha=0.5, color="C0", orientation='vertical')
    ax_top.hist(fit_time_signal, bins=bins, range = (0, 1000), alpha=0.5, color="C1", orientation='vertical')
    ax_top.axvline(np.median(fit_time_null), color="C0")
    ax_top.axvline(np.median(fit_time_signal), color="C1")

    ax_top.set_ylabel("Counts", fontsize=14)
    ax_top.tick_params(labelsize=14)

    # Right histogram
    ax_right = fig.add_subplot(gs[1:, -1], sharey=ax_main)
    ax_right.hist(fit_freq_null, bins=bins, range = (0, 500), alpha=0.5, color="C0", orientation='horizontal')
    ax_right.hist(fit_freq_signal, bins=bins, range = (0, 500), alpha=0.5, color="C1", orientation='horizontal')
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

    mask = np.where(np.isinf(zscore["ic86"][2])==True, False, True)

    for i, det in enumerate(["ic86", "gen2", "wls"]):

        ax[0].plot(dist_range, zscore[det][0], label=labels[i], color = colors[i])
        ax[0].fill_between(dist_range.value, zscore[det][2], zscore[det][1], color = colors[i], alpha = 0.15)
    
        #signal without errorbar
        ax[1].plot(dist_range, ts_stat["signal"][det][0], color = colors[i], label=r'TS$^{sig}_{IceCube}$')

        #background with errorbar
        ax[1].errorbar(x = dist_range, y = ts_stat["null"][det][0],
                    yerr = (ts_stat["null"][det][0]-ts_stat["null"][det][1],ts_stat["null"][det][2]-ts_stat["null"][det][0]), 
                    capsize=4, ls = ':', color = colors[i], label=r'TS$^{bkg}_{IceCube}$')

    ax[0].set_xlabel('Distance d [kpc]', fontsize = 12)
    ax[0].set_ylabel(r'SASI detection significance [$\sigma$]' , fontsize = 12)
    #ax[0].set_xlim((1,20))
    ax[0].set_ylim((1,30))

    ax[0].tick_params(labelsize = 12)

    ax[0].set_yscale('log')
    ax[0].set_yticks([1,2,3,5,10,20,30])
    ax[0].set_yticklabels(['1','2','3','5','10','20','30'])

    ax[0].grid()
    ax[0].legend(loc='upper right', fontsize = 12)

    ax[0].axhline(3, color='k', ls = ':')
    ax[0].axhline(5, color='k', ls = '-.')
    ax[0].text(dist_range[mask][0].value, 3, r"3$\sigma$", size=12,
            ha="center", va="center",
            bbox=dict(boxstyle="square", ec='k', fc='white'))

    ax[0].text(dist_range[mask][0].value, 5, r"5$\sigma$", size=12,
            ha="center", va="center",
            bbox=dict(boxstyle="square", ec='k', fc='white'))

    # distance for CDF value of 0.1, 0.5, etc.
    cdf_val = np.array([0.1,0.5,0.75,0.9,0.95,0.99,1])
    stellar_dist = coverage_to_distance(cdf_val).flatten()

    ax22 = ax[0].twiny()
    ax22.set_xlim(ax[0].get_xlim())
    ax22.set_xticks(stellar_dist)
    ax22.set_xticklabels((cdf_val*100).astype(dtype=int), rotation = 0, fontsize = 12)
    ax22.set_xlabel('Cumulative galactic CCSNe distribution \n from Adams et al. (2013) in [%]', fontsize = 12)

    #rearrange legend handels
    handles,labels = ax[1].get_legend_handles_labels()
    handles = [handles[0], handles[3], handles[1], handles[4], handles[2], handles[5]]
    labels = [labels[0], labels[3], labels[1], labels[4], labels[2], labels[5]]

    ax[1].set_xlabel('Distance d [kpc]', fontsize = 12)
    ax[1].set_ylabel('Median of test statistics (TS)', fontsize = 12)
    #ax[1].set_xlim((1,20))
    ax[1].set_yscale('log')
    ax[1].tick_params(labelsize = 12)
    ax[1].legend(handles, labels, ncol = 3, fontsize = 12, bbox_to_anchor=(0.13, 1))
    ax[1].grid()
    plt.tight_layout()

def plot_para_scan(freq_range, ampl_range, data):

    fig, ax = plt.subplots(1,1, figsize = (10,4))

    im = ax.pcolormesh(freq_range.value, ampl_range*100, data, cmap="viridis", shading="nearest")
    cb = fig.colorbar(im)
    cb.ax.tick_params(labelsize=14)
    cb.set_label(r"3 $\sigma$ Distance horizon [kpc]", size=14)
    ax.set_xlabel(r'Frequency in [Hz]', fontsize=14)
    ax.set_ylabel(r'Amplitude [%]', fontsize=14)
    ax.yaxis.get_offset_text().set_fontsize(14)
    ax.tick_params(labelsize=14)

    plt.tight_layout()

def plot_bootstrap(zscore):

    fig, ax = plt.subplots(3, 1, figsize=(10, 5))

    for i, det in enumerate(["ic86", "gen2", "wls"]):

        rel_error = np.std(zscore[det], axis = 0)/np.mean(zscore[det], axis = 0)

        n1, _, _ = ax[i].hist(zscore[det][:, 1], bins=20, density=True, histtype="step", color="C1", label="16%")
        n0, _, _ = ax[i].hist(zscore[det][:, 0], bins=20, density=True, histtype="step", color="C0", label="50%")
        n2, _, _ = ax[i].hist(zscore[det][:, 2], bins=20, density=True, histtype="step", color="C2", label="84%")

        ax[i].text(zscore[det][:, 1].mean(), n1.max()/2, s = "{:1.1E}".format(rel_error[1]), color = "C1", fontsize = 10, weight = "bold", ha = "center")
        ax[i].text(zscore[det][:, 0].mean(), n0.max()/2, s = "{:1.1E}".format(rel_error[0]), color = "C0", fontsize = 10, weight = "bold", ha = "center")
        ax[i].text(zscore[det][:, 2].mean(), n2.max()/2, s = "{:1.1E}".format(rel_error[2]), color = "C2", fontsize = 10, weight = "bold", ha = "center")

        ax[i].text(0.15, 0.9, s = det, transform=ax[i].transAxes, fontsize=14, va='top', 
                   bbox= dict(boxstyle='round', facecolor='white', alpha=0.5))
        ax[i].tick_params(labelsize = 14)


    # Create a common legend for all subplots
    ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), fontsize=14, ncols = 3)

    # add a big axis, hide frame
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Mean Significance", fontsize = 14)
    plt.ylabel("Normalized Counts", fontsize = 14)

    plt.tight_layout()

    return fig, ax
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import norm, lognorm, skewnorm  

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
    ax.step(freq, power["null"][det][0], color = 'C0', ls = '-', label=r'$H_{0}$', zorder = 0, where = "mid")
    ax.step(freq, power["signal"][det][0], color = 'C1', ls = '-', alpha = 0.5, label=r'$H_{1}$', zorder = 10, where = "mid")
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

    vmin = np.min([power["null"][det][0], power["signal"][det][0]])
    vmax = np.max([power["null"][det][0], power["signal"][det][0]])

    hypos = ["signal", "null"]
    im, cb = [None,None], [None,None]
    
    fig, ax = plt.subplots(1,2, figsize = (10,4))

    for i in range(2):
        im[i] = ax[i].pcolormesh(time, freq, power[hypos[i]][det][0], cmap='plasma', shading = "nearest", vmin = vmin, vmax = vmax)
        
        cb = fig.colorbar(im[i])
        cb.ax.tick_params(labelsize=14)
        cb.set_label(label=r"$S(f,t)$",size=14)
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
    ax.set_ylabel('summed power', fontsize = 14)
    ax.set_xlim(0,500)
    ax.tick_params(labelsize = 14)
    ax.legend(fontsize = 14, ncol = 3)

    plt.tight_layout()

def plot_ts(ts, bkg_distr, det = "ic86"):

    # Plot TS distribution for null and signal hypothesis for gen2

    if bkg_distr == "lognorm":
            distr = lognorm
    elif bkg_distr == "skewnorm":
        distr = skewnorm
    else:
        raise ValueError('{} not supported. Choose from "lognorm" or "skewnorm"'.format(bkg_distr)) 

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
    bkg_fit = distr(*distr.fit(ts["null"][det]))
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

    bins = 499

    fig, ax = plt.subplots()

    ax.hist(fit_freq_null, histtype="step", density=True, bins = bins, range = (1, 499), lw = 2, color="C0", label=r"$H_0$", align='left')
    ax.hist(fit_freq_signal, histtype="step", density=True, bins = bins, range = (1, 499), lw = 2, color="C1", label=r"$H_1$", align='left')
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
    ax_top.hist(fit_time_null, bins=bins, range = (0, 1000), alpha=0.5, color="C0", orientation='vertical', align='left')
    ax_top.hist(fit_time_signal, bins=bins, range = (0, 1000), alpha=0.5, color="C1", orientation='vertical', align='left')
    ax_top.axvline(np.median(fit_time_null), color="C0")
    ax_top.axvline(np.median(fit_time_signal), color="C1")

    ax_top.set_ylabel("Counts", fontsize=14)
    ax_top.tick_params(labelsize=14)

    # Right histogram
    ax_right = fig.add_subplot(gs[1:, -1], sharey=ax_main)
    ax_right.hist(fit_freq_null, bins=bins, range = (0, 500), alpha=0.5, color="C0", orientation='horizontal', align='left')
    ax_right.hist(fit_freq_signal, bins=bins, range = (0, 500), alpha=0.5, color="C1", orientation='horizontal', align='left')
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
        for j in range(3):
            m = np.isinf(zscore[det][j])
            zscore[det][j][m] = zscore[det][j][np.isfinite(zscore[det][j])].max()
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
    ax[0].text(dist_range[mask][-1].value, 3, r"3$\sigma$", size=12,
            ha="center", va="center",
            bbox=dict(boxstyle="square", ec='k', fc='white'))

    ax[0].text(dist_range[mask][-1].value, 5, r"5$\sigma$", size=12,
            ha="center", va="center",
            bbox=dict(boxstyle="square", ec='k', fc='white'))

    # distance for CDF value of 0.1, 0.5, etc.
    cdf_val = np.array([0.1,0.5,0.75,0.9,0.95,0.99,1])
    stellar_dist = coverage_to_distance(cdf_val).flatten()
    # cut all stellar_dist and cdf entries that are smaller (larger) than the smallest (largest) entry in dist_range
    mask = np.logical_and(stellar_dist > dist_range.value.min(), stellar_dist < dist_range.value.max())

    ax22 = ax[0].twiny()
    ax22.set_xlim(ax[0].get_xlim())
    ax22.set_xticks(stellar_dist[mask])
    ax22.set_xticklabels((cdf_val[mask]*100).astype(dtype=int), rotation = 0, fontsize = 12)
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

def plot_resolution(dist_range, quant, zscore):
    
    fig, ax = plt.subplots(1,2, figsize = (14,4))
    ax = ax.ravel()

    labels = [r'$IceCube$', r'$Gen2$', r'$Gen2+WLS$']
    colors = ['C0', 'C1', 'C2']

    for i, det in enumerate(["ic86", "gen2", "wls"]):

        ax[0].plot(dist_range, 100 * quant[det][0], label=labels[i], color = colors[i])
        ax[0].fill_between(dist_range.value, 100 * quant[det][1], 100 * quant[det][2], color = colors[i], alpha = 0.2)

        ax[1].plot(zscore[det][0], 100 * quant[det][0], label=labels[i], color = colors[i])
        ax[1].fill_between(zscore[det][0], 100 * quant[det][1], 100 * quant[det][2], color = colors[i], alpha = 0.2)

    ax[0].set_xlabel('Distance d [kpc]', fontsize = 12)
    ax[0].set_ylabel(r'$H_1: (q_{reco} - q_{true})/q_{true}$ [%]' , fontsize = 12)
    ax[0].set_yscale("symlog")
    ax[0].tick_params(labelsize = 12)
    ax[0].grid()

    ax[1].set_xlabel(r'SASI detection significance [$\sigma$]' , fontsize = 12)
    ax[1].set_ylabel(r'$H_1: (q_{reco} - q_{true})/q_{true}$ [%]' , fontsize = 12)
    ax[1].legend()

    ax[1].set_yscale("symlog", linthresh = 1)
    ax[1].set_xticks([1,2,3,4,5])
    ax[1].get_xaxis().set_major_formatter(ScalarFormatter())
    ax[1].tick_params(labelsize = 12)

    ax[1].axvline(3, color = "k", ls = "--")
    ax[1].axvline(5, color = "k", ls = "--")

    plt.tight_layout()

def plot_para_scan(freq_range, ampl_range, data, det, sig, quant, type = "sign", scale = None, relative = False):

    fs = 14

    if sig == 3:
        isig = 0
    elif sig == 5:
        isig = 1
    
    if quant == 50:
        iquant = 0
    elif quant == 16:
        iquant = 1
    elif quant == 84:
        iquant = 2

    if scale == "log":
        norm = LogNorm()
    else:
        norm = None

    if relative:
        ddata = data[det][isig, :, :, iquant]-np.tile(np.nanmedian(data[det][isig, :, :, iquant], axis = 0), reps = freq_range.size).reshape(freq_range.size, ampl_range.size)
    else:
        ddata = data[det][isig, :, :, iquant]

    ddata = np.abs(ddata)
    if type == "fres" or type == "tres": ddata *= 100

    fig, ax = plt.subplots(1,1, figsize = (10,4))

    cmap = plt.get_cmap('viridis')  # viridis is the default colormap for imshow
    cmap.set_bad(color='grey')
    im = ax.pcolormesh(freq_range.value, ampl_range*100, np.transpose(ddata), cmap=cmap, shading="nearest", norm = norm)
    cb = fig.colorbar(im)
    cb.ax.tick_params(labelsize=fs)
        
    if type == "sign":
        if relative:
            cb.set_label(r"deviation {}$\sigma$ signifiance horizon across frequencies [kpc]".format(sig), size=fs)
        else:
            cb.set_label(r"{}$\sigma$ signifiance horizon [kpc]".format(sig), size=fs)
    elif type == "fres":
        cb.set_label(r"{}$\sigma$ rel. frequency resolution [%]".format(sig), size=fs)
    elif type == "tres":
        cb.set_label(r"{}$\sigma$ rel. time resolution [%]".format(sig), size=fs)

    ax.set_xlabel(r'Frequency in [Hz]', fontsize=fs)
    ax.set_ylabel(r'Amplitude [%]', fontsize=fs)
    ax.yaxis.get_offset_text().set_fontsize(fs)
    ax.tick_params(labelsize=fs)

    plt.tight_layout()

def plot_para_vs_amplitude(ampl_range, data, sig, type = "sign", scale = None):

    fs = 14

    if sig == 3:
        isig = 0
    elif sig == 5:
        isig = 1

    labels = [r'IceCube', r'Gen2', r'Gen2+WLS']
    colors = ['C0', 'C1', 'C2']

    fig, ax = plt.subplots(1,1)

    for i, det in enumerate(["ic86", "gen2", "wls"]):

        mean, std = np.nanmean(np.abs(data[det][isig,:,:,0]), axis = 0), np.nanstd(np.abs(data[det][isig,:,:,0]), axis = 0)
        q16, q84 = np.nanmean(np.abs(data[det][isig,:,:,1]), axis = 0), np.nanmean(np.abs(data[det][isig,:,:,2]), axis = 0)

        if type == "fres" or type == "tres": 
            mean *= 100
            std *= 100
            q16 *= 100
            q84 *= 100

        ax.plot(ampl_range * 100, mean, color = colors[i], label  = labels[i])
        ax.fill_between(ampl_range * 100, mean-std, mean+std, color = colors[i], alpha = 0.25)
        ax.fill_between(ampl_range * 100, q16, q84, color = colors[i], alpha = 0.1)

    ax.set_xlabel("Amplitude [%]", size=fs)

    if type == "sign":
        ax.set_ylabel(r"{}$\sigma$ significance horizon [kpc]".format(sig), size=fs)
    elif type == "fres":
        ax.set_ylabel(r"{}$\sigma$ rel. frequency resolution [%]".format(sig), size=fs)
    elif type == "tres":
        ax.set_ylabel(r"{}$\sigma$ rel. time resolution [%]".format(sig), size=fs)

    if scale == "log": ax.set_yscale("log")
    ax.tick_params(labelsize=fs)
    ax.legend()
    ax.grid()

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

def plot_hist_fit(ts_binned, fit_func, pvalue, bins, det = "ic86"):

    ts_xhist, ts_yhist = ts_binned[det]
    pvals = pvalue[det]

    zscore = norm.isf(pvals/2)
    diff_zscore = np.diff(zscore)

    # get fit params from npz file for given det and distance
    fig, ax = plt.subplots(2,2, figsize = (10,10))
    ax = ax.ravel()

    ax[0].step(ts_xhist, ts_yhist)
    ax[0].plot(ts_xhist, fit_func.pdf(ts_xhist), "k:")
    ax[0].set_ylabel("Normalized Counts")
    ax[0].set_yscale("log")
    ax[0].set_xticklabels([]) #Remove x-tic labels for the first frame

    # residual axis
    rax0 = ax[0].inset_axes([0, -0.25, 1, 0.2])
    rax0.plot(ts_xhist, (ts_yhist-fit_func.pdf(ts_xhist))/ts_yhist, color='k')
    rax0.set_xlabel("TS value")
    rax0.set_ylabel("rel. residuals")
    rax0.set_yscale("log")

    ax[1].step(ts_xhist, pvals, color = "C0")
    ax[1].plot(ts_xhist, fit_func.sf(ts_xhist), "k:")
    ax[1].set_ylabel("p-value")
    ax[1].set_xticklabels([]) #Remove x-tic labels for the first frame

    ax11 = ax[1].twinx()
    ax11.step(ts_xhist, zscore, color = "C1")
    ax11.plot(ts_xhist, norm.isf(fit_func.sf(ts_xhist)/2), "k:")
    ax11.set_ylabel(r"Z-score ($\sigma$)")

    # residual axis
    rax1 = ax[1].inset_axes([0, -0.25, 1, 0.2])
    rax1.plot(ts_xhist, pvals-fit_func.sf(ts_xhist), color='C0')
    rax1.plot(ts_xhist, (zscore-norm.isf(fit_func.sf(ts_xhist)/2))/zscore, color='C1', ls = "--")
    rax1.set_xlabel("TS value")
    rax1.set_ylabel("rel. residuals")
    rax1.set_yscale("log")

    ax[1].spines['left'].set_color('C0')
    ax[1].yaxis.label.set_color('C0')
    ax[1].tick_params(axis='y', colors='C0')
    ax11.spines['right'].set_color('C1')
    ax11.yaxis.label.set_color('C1')
    ax11.tick_params(axis='y', colors='C1')
    ax11.grid(axis="y")

    ax[2].step((ts_xhist[1:]+ts_xhist[:-1])/2, diff_zscore, color = "C1")
    ax[2].set_xlabel("TS value")
    ax[2].set_ylabel(r"$\Delta$Z-score ($\Delta\sigma$)")

    ax[3].hist(diff_zscore[~np.isinf(diff_zscore)], bins = int(np.sqrt(bins)))
    ax[3].set_xlabel(r"$\Delta$Z-score ($\Delta\sigma$)")
    ax[3].set_ylabel("Normalized Counts")
    ax[3].set_yscale("log")

    plt.tight_layout()
    plt.show()

def plot_summary_fft(self, relative = True, det = "ic86"):
                
    bkg_hist = np.load(self._file + "/files/background/hist/HIST_model_Sukhbold_2015_27_mode_{}_samples_{:.0e}_bins_{:.0e}_distance_{:.1f}kpc.npz".format(self.mode, self.bkg_trials, self.bkg_bins, self.distance.value))
    fs = 12 # fontsize
    fig, ax = plt.subplots(2,2, figsize = (10,10))
    ax = ax.ravel()

    # plot hit rate over time
    ax[0].plot(self._time_new, self._comb[det][0])
    ax[0].text(0.5, 0.9, s = r"$d$ = {:.1f} kpc".format(self.distance.value), transform = ax[0].transAxes, fontsize = fs)
    ax[0].set_xlabel("Time [ms]", fontsize = fs)                    
    ax[0].set_ylabel("Counts", fontsize = fs)
    ax[0].tick_params(labelsize = fs)
    ax[0].grid()  

    # plot fourier spectrum                  
    ax[1].plot(self._freq_new, self._fft[det][0], marker = "x")
    ax[1].plot(self._freq[self._freq <= 75 * u.Hz], self._fft0[det][0][self._freq <= 75 * u.Hz], marker="o")
    ax[1].axvspan(0, 75, color = "grey", alpha = 0.15)
    ax[1].set_xlabel("Frequency [Hz]", fontsize = fs)                    
    ax[1].set_ylabel("Power [a.u.]", fontsize = fs) 
    ax[1].set_yscale("log")
    ax[1].tick_params(labelsize = fs)
    ax[1].grid() 

    # plot inset around true frequency
    axins = ax[1].inset_axes([0.4, 0.7, 0.5, 0.2])
    axins.plot(self._freq_new, self._fft[det][0], color = "C0", marker="x")
    axins.set_xlim(self.temp_para["frequency"].value - 5, self.temp_para["frequency"].value + 5)
    axins.grid()
    axins.set_xticks(np.arange(self.temp_para["frequency"].value - 5, self.temp_para["frequency"].value + 6, 1))
    axins.tick_params(labelsize = fs)
    axins.tick_params(axis='x', labelrotation=90)

    # plot TS distribution for signal (H1) and null (H0) hypo
    ax[2].hist(self.ts[det], bins = 100, density = True, label = "H1")
    ax[2].step(bkg_hist[det][0], bkg_hist[det][1], label = "H0")
    ax[2].text(0.5, 0.9, s = r"$\sigma$ = {:.2f}".format(self.zscore[det][0]), transform = ax[2].transAxes, fontsize = fs)
    ax[2].set_xlabel("TS value", fontsize = fs)                    
    ax[2].set_ylabel("Counts", fontsize = fs) 
    ax[2].legend(loc = "upper right", fontsize = fs)
    ax[2].tick_params(labelsize = fs)
    ax[2].grid()              

    # plot TS value and relative best fit frequency with marginal distribitions to the side
    if relative: ffit = (self.ffit[det] - self.temp_para["frequency"].value)/self.temp_para["frequency"].value * 100
    else: ffit = self.ffit[det]
    ax[3].scatter(self.ts[det], ffit, alpha=0.2)
    ax[3].set_xlabel("TS value", fontsize = fs)
    ax[3].set_ylabel(r"$(f_{reco} - f_{true})/f_{true}$ [%]", fontsize = fs)
    ax[3].tick_params(labelsize = fs)
    ax[3].grid() 

    # add marginal plots to ax[3]
    divider = make_axes_locatable(ax[3])
    ax_top = divider.append_axes("top", 0.4, pad=0.3, sharex=ax[3])
    ax_right = divider.append_axes("right", 0.4, pad=0.3, sharey=ax[3])

    # histogram for TS above the scatter plot
    ax_top.hist(self.ts[det], bins=100, density=True, color='grey', orientation='vertical')
    ax_top.grid()
    ax_top.set_ylabel("%", fontsize = fs)
    ax_top.tick_params(labelsize=fs)

    # histogram for best fit frequency to the right of the scatter plot
    # for unrounded ffit
    #mask = np.logical_and(self._freq_new.value >= ffit_min, self._freq_new.value <= ffit_max)
    #ffit_range = np.round(self._freq_new.value)[mask]

    ffit_min, ffit_max, dffit = self.ffit[det].min(), self.ffit[det].max(), np.diff(self._freq_new.value)[0]
    ffit_range = np.arange(ffit_min, ffit_max + 2*dffit, dffit)
    
    f0 = self.temp_para["frequency"].value
    if relative:
        ffit_range = (ffit_range - self.temp_para["frequency"].value)/self.temp_para["frequency"].value * 100
        f0 = 0

    ax_right.hist(ffit, bins = ffit_range, density=True, color='grey', orientation='horizontal', align = "left")
    ax_right.axhline(np.median(ffit), color="C0", ls = "--", lw = 2, label = r"$\langle f_{reco} \rangle$")
    ax_right.axhline(f0, color="C1", lw = 2, label = r"$f_{true}$")
    ax_right.set_xlabel("%", fontsize = fs)
    ax_right.tick_params(labelsize = fs)
    ax_right.grid()
    ax_right.legend(loc = "upper center", fontsize = fs, bbox_to_anchor=(0.5, 1.3))
    
    rel_file = "/plots/scan/SUM_model_Sukhbold_2015_27_mode_{}_time_{:.0f}ms-{:.0f}ms_bkg_trials_{:.0e}_sig_trials_{:.0e}_ampl_{:.1f}%_freq_{:.0f}Hz_distance_{:.1f}kpc.pdf".format(self.mode, self.temp_para["time_start"].value, self.temp_para["time_end"].value, self.bkg_trials, self.sig_trials, self.temp_para["amplitude"] * 100, self.temp_para["frequency"].value, self.distance.value)
    abs_file = os.path.dirname(os.path.abspath(__file__)) + rel_file
    plt.savefig(abs_file, bbox_inches='tight')
    plt.close()

def plot_summary_stf(self, relative = True, det = "ic86"):
                
    bkg_hist = np.load(self._file + "/files/background/hist/HIST_model_Sukhbold_2015_27_mode_{}_samples_{:.0e}_bins_{:.0e}_distance_{:.1f}kpc.npz".format(self.mode, self.bkg_trials, self.bkg_bins, self.distance.value))
    fs = 12 # fontsize
    fig, ax = plt.subplots(2,2, figsize = (10,10))
    ax = ax.ravel()

    f0 = self.temp_para["frequency"].value
    t0 = (self.temp_para["time_start"].value + self.temp_para["time_end"].value)/2
    df, dt = np.diff(self._freq.value)[0], np.diff(self._time.value)[0]

    # plot hit rate over time
    ax[0].plot(self.sim.time.value * 1000, self._comb[det][0]) # time in ms
    ax[0].text(0.5, 0.9, s = r"$d$ = {:.1f} kpc".format(self.distance.value), transform = ax[0].transAxes, fontsize = fs)
    ax[0].set_xlabel("Time [ms]", fontsize = fs)                    
    ax[0].set_ylabel("Counts", fontsize = fs)
    ax[0].tick_params(labelsize = fs)
    ax[0].grid()  

    # plot fourier spectrum
    vmin = np.min(self._stf[det][0])
    vmax = np.max(self._stf[det][0])
    im = ax[1].pcolormesh(self._time.value, self._freq.value, self._stf0[det][0], cmap='plasma', shading = "nearest", vmin = vmin, vmax = vmax)    
    cb = fig.colorbar(im)
    cb.ax.tick_params(labelsize = fs)
    cb.set_label(label=r"Power [a.u.]",size = fs)
    ax[1].axhline(self._freq_new[0].value, color = "red", ls = "--")
    ax[1].set_xlabel(r'Time $t-t_{\rm bounce}$ [ms]', fontsize = fs)
    ax[1].set_ylabel(f"Frequency $f$ in [Hz]", fontsize = fs)
    ax[1].yaxis.get_offset_text().set_fontsize(fs)
    ax[1].tick_params(labelsize = fs)   
    
    # plot inset around true frequency
    axins = ax[1].inset_axes([0.7, 0.7, 0.2, 0.2])
    im = axins.pcolormesh(self._time.value, self._freq.value, self._stf0[det][0], cmap='plasma', shading = "nearest", vmin = vmin, vmax = vmax)    
    axins.set_xlim(t0 - 3*dt, t0 + 3*dt)
    axins.set_ylim(f0 - 3*df, f0 + 3*df)

    axins.set_xticks(np.arange(t0 - 3*dt, t0 + 4*dt, dt)[::3])
    axins.set_yticks(np.arange(f0 - 3*df, f0 + 4*df, df)[::3])
    axins.tick_params(labelsize = fs)
    
    # plot TS distribution for signal (H1) and null (H0) hypo
    ax[2].hist(self.ts[det], bins = 100, density = True, label = "H1")
    ax[2].step(bkg_hist[det][0], bkg_hist[det][1], label = "H0")
    ax[2].text(0.5, 0.9, s = r"$\sigma$ = {:.2f}".format(self.zscore[det][0]), transform = ax[2].transAxes, fontsize = fs)
    ax[2].set_xlabel("TS value", fontsize = fs)                    
    ax[2].set_ylabel("Counts", fontsize = fs) 
    ax[2].legend(loc = "upper right", fontsize = fs)
    ax[2].tick_params(labelsize = fs)
    ax[2].grid()              

    # plot TS value and relative best fit frequency with marginal distribitions to the side
    if relative: 
        ffit = (self.ffit[det] - f0)/f0 * 100
        tfit = (self.tfit[det] - t0)/t0 * 100
    else: 
        ffit = self.ffit[det]
        tfit = self.tfit[det]

    im = ax[3].scatter(x = tfit, y = ffit, c = self.ts[det], cmap = "viridis", alpha=0.1)
    ax[3].set_xlabel(r"$(t_{reco} - t_{true})/t_{true}$ [%]", fontsize = fs)
    ax[3].set_ylabel(r"$(f_{reco} - f_{true})/f_{true}$ [%]", fontsize = fs)
    ax[3].tick_params(labelsize = fs)
    ax[3].grid() 

    cb = fig.colorbar(im)
    cb.ax.tick_params(labelsize = fs)
    cb.solids.set(alpha=1)
    cb.set_label(label=r"TS value",size = fs)

    # add marginal plots to ax[3]
    divider = make_axes_locatable(ax[3])
    ax_top = divider.append_axes("top", 0.4, pad=0.3, sharex=ax[3])
    ax_right = divider.append_axes("right", 0.4, pad=0.3, sharey=ax[3])

    # histogram for best fit frequency to the right of the scatter plot
    ffit_min, ffit_max, dffit = self.ffit[det].min(), self.ffit[det].max(), np.diff(self._freq_new.value)[0]
    ffit_range = np.arange(ffit_min, ffit_max + 2*dffit, dffit)

    tfit_min, tfit_max, dtfit = self.tfit[det].min(), self.tfit[det].max(), np.diff(self._time_new.value)[0]
    tfit_range = np.arange(tfit_min, tfit_max + 2*dtfit, dtfit)

    f0 = self.temp_para["frequency"].value
    if relative:
        ffit_range = (ffit_range - f0)/f0 * 100
        tfit_range = (tfit_range - t0)/t0 * 100
        f0, t0 = 0, 0

    # histogram for reco time above the scatter plot
    ax_top.hist(tfit, bins = tfit_range, density=True, color='grey', orientation='vertical', align = "left")
    ax_top.axvline(np.median(tfit), color="C0", ls = "--", lw = 2, label = r"$\langle f_{reco} \rangle$")
    ax_top.axvline(t0, color="C1", lw = 2, label = r"$f_{true}$")
    ax_top.set_xlabel("%", fontsize = fs)
    ax_top.tick_params(labelsize = fs)
    ax_top.grid()

    # histogram for reco freq above the scatter plot
    ax_right.hist(ffit, bins = ffit_range, density=True, color='grey', orientation='horizontal', align = "left")
    ax_right.axhline(np.median(ffit), color="C0", ls = "--", lw = 2, label = r"$\langle f_{reco} \rangle$")
    ax_right.axhline(f0, color="C1", lw = 2, label = r"$f_{true}$")
    ax_right.set_xlabel("%", fontsize = fs)
    ax_right.tick_params(labelsize = fs)
    ax_right.grid()
    ax_right.legend(loc = "upper center", fontsize = fs, bbox_to_anchor=(0.5, 1.3))
    
    rel_file = "/plots/scan/SUM_model_Sukhbold_2015_27_mode_{}_time_{:.0f}ms-{:.0f}ms_bkg_trials_{:.0e}_sig_trials_{:.0e}_ampl_{:.1f}%_freq_{:.0f}Hz_distance_{:.1f}kpc.pdf".format(self.mode, self.temp_para["time_start"].value, self.temp_para["time_end"].value, self.bkg_trials, self.sig_trials, self.temp_para["amplitude"] * 100, self.temp_para["frequency"].value, self.distance.value)
    abs_file = os.path.dirname(os.path.abspath(__file__)) + rel_file
    plt.savefig(abs_file, bbox_inches='tight')
    plt.close()
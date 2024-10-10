import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import norm, lognorm, skewnorm  

import copy
from helper import *

plt.rcParams["font.family"] = "Times New Roman"

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

def plot_significance(self):

    dist_range = copy.deepcopy(self.dist_range)
    zscore = copy.deepcopy(self.Zscore)
    ts_stat = copy.deepcopy(self.Ts_stat)

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

    filename = self._file + "/plots/scan/{}/{}/SIG_model_{}_{:.0f}_mode_{}_ampl_{:.1f}%_freq_{:.0f}Hz_time_{:.0f}ms-{:.0f}ms_mix_{}_hier_{}_sig_var_{:+.0f}%_bkg_var_{:+.0f}%_sig_trials_{:1.0e}_bkg_trials_{:1.0e}_bins_{:1.0e}.pdf".format(
                self.ft_mode, self.scan_dir_name, 
                self.temp_para["model"]["name"], self.temp_para["model"]["param"]["progenitor_mass"].value, 
                self.ft_mode, self.temp_para["amplitude"] * 100, self.temp_para["frequency"].value,
                self.temp_para["time_start"].value, self.temp_para["time_end"].value, 
                self.mixing_scheme, self.hierarchy,
                self.sig_var * 100, self.bkg_var * 100,
                self.sig_trials, self.bkg_trials, self.bkg_bins)
    plt.savefig(filename)
    plt.close()

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

def plot_summary_fft(self, relative = True, det = "ic86"):
    
    filename_in = self._file + "/files/background/{}/{}/HIST_model_{}_{:.0f}_mode_{}_mix_{}_hier_{}_sig_var_{:+.0f}%_bkg_var_{:+.0f}%_bkg_trials_{:1.0e}_bins_{:1.0e}_distance_{:.1f}kpc.npz".format(
            self.mode, self.bkg_dir_name, self.temp_para["model"]["name"], self.temp_para["model"]["param"]["progenitor_mass"].value, 
            self.mode, self.mixing_scheme, self.hierarchy,
            self.sig_var * 100, self.bkg_var * 100,
            self.bkg_trials, self.bkg_bins, self.distance.value)
    bkg_hist = np.load(filename_in)

    fs = 12 # fontsize
    fig, ax = plt.subplots(1,3, figsize = (10,3))
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
    ax[1].plot(self._freq[self._freq <= 75 * u.Hz], self._fft0[det][self._freq <= 75 * u.Hz], marker="o")
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
    
    # filename of summary plot
    model_str = "ampl_{:.1f}%_freq_{:.0f}Hz".format(self.temp_para["amplitude"] * 100, self.temp_para["frequency"].value) # each model is saved in its own directory
    filename_out = self._file + "/plots/scan/{}/{}/{}/SUM_model_{}_{:.0f}_mode_{}_ampl_{:.1f}%_freq_{:.0f}Hz_time_{:.0f}ms-{:.0f}ms_mix_{}_hier_{}_sig_var_{:+.0f}%_bkg_var_{:+.0f}%_bkg_trials_{:1.0e}_bins_{:1.0e}_distance_{:.1f}kpc.pdf".format(
            self.mode, self.scan_dir_name, model_str, 
            self.temp_para["model"]["name"], self.temp_para["model"]["param"]["progenitor_mass"].value, 
            self.mode, self.temp_para["amplitude"] * 100, self.temp_para["frequency"].value,
            self.temp_para["time_start"].value, self.temp_para["time_end"].value, 
            self.mixing_scheme, self.hierarchy,
            self.sig_var * 100, self.bkg_var * 100,
            self.bkg_trials, self.bkg_bins, self.distance.value)
    
    if not os.path.exists(os.path.dirname(filename_out)):  # Check if directory already exists, if not, make directory
        os.mkdir(os.path.dirname(filename_out))

    plt.savefig(filename_out)
    plt.close()

def plot_summary_stf(self, relative = True, det = "ic86"):
                
    filename_in = self._file + "/files/background/{}/{}/HIST_model_{}_{:.0f}_mode_{}_mix_{}_hier_{}_sig_var_{:+.0f}%_bkg_var_{:+.0f}%_bkg_trials_{:1.0e}_bins_{:1.0e}_distance_{:.1f}kpc.npz".format(
            self.mode, self.bkg_dir_name, self.temp_para["model"]["name"], self.temp_para["model"]["param"]["progenitor_mass"].value, 
            self.mode, self.mixing_scheme, self.hierarchy,
            self.sig_var * 100, self.bkg_var * 100,
            self.bkg_trials, self.bkg_bins, self.distance.value)
    bkg_hist = np.load(filename_in)
    
    fs = 12 # fontsize
    fig, ax = plt.subplots(1,3, figsize = (10,3))
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
    im = ax[1].pcolormesh(self._time.value, self._freq.value, self._stf0[det], cmap='plasma', shading = "nearest", vmin = vmin, vmax = vmax)    
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
    im = axins.pcolormesh(self._time.value, self._freq.value, self._stf0[det], cmap='plasma', shading = "nearest", vmin = vmin, vmax = vmax)    
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
    
    # filename of summary plot
    model_str = "ampl_{:.1f}%_freq_{:.0f}Hz".format(self.temp_para["amplitude"] * 100, self.temp_para["frequency"].value) # each model is saved in its own directory
    filename_out = self._file + "/plots/scan/{}/{}/{}/SUM_model_{}_{:.0f}_mode_{}_ampl_{:.1f}%_freq_{:.0f}Hz_time_{:.0f}ms-{:.0f}ms_mix_{}_hier_{}_sig_var_{:+.0f}%_bkg_var_{:+.0f}%_bkg_trials_{:1.0e}_bins_{:1.0e}_distance_{:.1f}kpc.pdf".format(
            self.mode, self.scan_dir_name, model_str, 
            self.temp_para["model"]["name"], self.temp_para["model"]["param"]["progenitor_mass"].value, 
            self.mode, self.temp_para["amplitude"] * 100, self.temp_para["frequency"].value,
            self.temp_para["time_start"].value, self.temp_para["time_end"].value, 
            self.mixing_scheme, self.hierarchy,
            self.sig_var * 100, self.bkg_var * 100,
            self.bkg_trials, self.bkg_bins, self.distance.value)
    
    if not os.path.exists(os.path.dirname(filename_out)): # Check if directory already exists, if not, make directory
        os.mkdir(os.path.dirname(filename_out))

    plt.savefig(filename_out)
    plt.close()

def plot_reco(self, bins = 100, hypo = "signal", det = "ic86"):

    fs = 10
    
    fig, ax = plt.subplots(1,3, figsize = (10,4))

    ax[0].scatter(self.time_true, self.time_reco[hypo][det], s=10, alpha=0.1, color="C0", rasterized = True)
    ax[0].plot(np.linspace(0,1000,100), np.linspace(0,1000,100), "k--")
    ax[0].set_xlabel(r"$t_{true}$ [ms]", fontsize = fs)
    ax[0].set_ylabel(r"$t_{reco}$ [ms]", fontsize = fs)
    ax[0].tick_params(labelsize = fs)

    # add marginal plots to ax[0]
    divider0 = make_axes_locatable(ax[0])
    ax0_top = divider0.append_axes("top", 0.4, pad=0.3, sharex=ax[0])

    # histogram for reco time above the scatter plot
    ax0_top.hist(self.time_true, bins = bins, density=True, color='grey', orientation='vertical', align = "left")
    ax0_top.set_ylabel("%", fontsize = fs)

    ax[1].scatter(self.freq_true, self.freq_reco[hypo][det], s=10, alpha=0.1, color="C0", rasterized = True)
    ax[1].plot(np.linspace(0,500,100), np.linspace(0,500,100), "k--")
    ax[1].set_xlabel(r"$f_{true}$ [Hz]", fontsize = fs)
    ax[1].set_ylabel(r"$f_{reco}$ [Hz]", fontsize = fs)
    ax[1].tick_params(labelsize = fs)

    # add marginal plots to ax[1]
    divider1 = make_axes_locatable(ax[1])
    ax1_top = divider1.append_axes("top", 0.4, pad=0.3, sharex=ax[1])

    # histogram for reco time above the scatter plot
    ax1_top.hist(self.freq_true, bins = bins, density=True, color='grey', orientation='vertical', align = "left")
    ax1_top.tick_params(labelsize = fs)

    rel_time = self.time_true-self.time_reco[hypo][det]
    rel_freq = self.freq_true-self.freq_reco[hypo][det]

    rel_time_char = [np.median(rel_time).value, np.quantile(rel_time, 0.16).value, np.quantile(rel_time, 0.84).value]
    rel_freq_char = [np.median(rel_freq).value, np.quantile(rel_freq, 0.16).value, np.quantile(rel_freq, 0.84).value]

    tmin = np.maximum(rel_time_char[1]*5, -1000)
    tmax = np.minimum(rel_time_char[2]*5, 1000)

    fmin = np.maximum(rel_freq_char[1]*5, -500)
    fmax = np.minimum(rel_freq_char[2]*5, 500)

    ax[2].hist2d(rel_time.value, rel_freq.value, bins = bins, range = [[tmin, tmax],[fmin, fmax]])
    ax[2].axvline(0, color = "k", ls = "--")
    ax[2].axhline(0, color = "k", ls = "--")
    ax[2].set_xlabel(r"$t_{true} - t_{reco}$ [ms]", fontsize = fs)
    ax[2].set_ylabel(r"$f_{true} - f_{reco}$ [Hz]", fontsize = fs)
    ax[2].tick_params(labelsize = fs)

    # add marginal plots to ax[2]
    divider2 = make_axes_locatable(ax[2])
    ax2_top = divider2.append_axes("top", 0.4, pad=0.3, sharex=ax[2])
    ax2_right = divider2.append_axes("right", 0.4, pad=0.3, sharey=ax[2])

    # histogram for reco time above the scatter plot
    ax2_top.hist(rel_time.value, bins = bins, range = (tmin, tmax), density=True, color='grey', orientation='vertical', align = "left")
    ax2_top.axvline(rel_time_char[0], color="C0", ls = "--", lw = 2, label = r"$\langle t_{true} - t_{reco} \rangle$")
    ax2_top.axvspan(rel_time_char[1], rel_time_char[2], color="C0", alpha = 0.25)
    ax2_top.tick_params(labelsize = fs)

    # histogram for reco freq above the scatter plot
    ax2_right.hist(rel_freq.value, bins = bins, range = (fmin, fmax), density=True, color='grey', orientation='horizontal', align = "left")
    ax2_right.axhline(rel_freq_char[0], color="C0", ls = "--", lw = 2, label = r"$\langle f_{true} - f_{reco} \rangle$")
    ax2_right.axhspan(rel_freq_char[1], rel_freq_char[2], color="C0", alpha = 0.25)
    ax2_right.tick_params(labelsize = fs)

    plt.tight_layout()
    filename = "./plots/reco/{}/{}/RECO_model_{}_{:.0f}_mode_{}_duration_{:.0f}ms_ampl_{:.1f}%_mix_{}_hier_{}_trials_{:1.0e}_{}_{}_dist_{:.1f}kpc.pdf".format(
    self.ft_mode, self.reco_dir_name, self.model["name"], self.model["param"]["progenitor_mass"].value, 
    self.ft_mode, self.reco_para["duration"].value, self.reco_para["ampl"][0]*100, 
    self.mixing_scheme, self.hierarchy, self.trials, hypo, det, self.distance.value)
    plt.savefig(filename, dpi = 400)
    plt.close()

def plot_reco2(self, bins = 100, hypo = "signal", det = "ic86"):

    fs = 14
    
    fig, ax = plt.subplots(1,1)

    rel_time = self.time_true-self.time_reco[hypo][det]
    rel_freq = self.freq_true-self.freq_reco[hypo][det]

    rel_time_char = [np.median(rel_time).value, np.quantile(rel_time, 0.16).value, np.quantile(rel_time, 0.84).value]
    rel_freq_char = [np.median(rel_freq).value, np.quantile(rel_freq, 0.16).value, np.quantile(rel_freq, 0.84).value]

    tmin = np.maximum(rel_time_char[1]*5, -1000)
    tmax = np.minimum(rel_time_char[2]*5, 1000)

    fmin = np.maximum(rel_freq_char[1]*5, -500)
    fmax = np.minimum(rel_freq_char[2]*5, 500)

    ax.hist2d(rel_time.value, rel_freq.value, bins = bins, range = [[tmin, tmax],[fmin, fmax]])
    ax.axvline(0, color = "k", ls = "--")
    ax.axhline(0, color = "k", ls = "--")
    ax.set_xlabel(r"$t_{true} - t_{reco}$ [ms]", fontsize = fs)
    ax.set_ylabel(r"$f_{true} - f_{reco}$ [Hz]", fontsize = fs)
    ax.tick_params(labelsize = fs)

    # add marginal plots to ax[2]
    divider2 = make_axes_locatable(ax)
    ax2_top = divider2.append_axes("top", 0.4, pad=0.3, sharex=ax)
    ax2_right = divider2.append_axes("right", 0.4, pad=0.4, sharey=ax)

    # histogram for reco time above the scatter plot
    ax2_top.hist(rel_time.value, bins = bins, range = (tmin, tmax), density=True, color='grey', orientation='vertical', align = "left")
    ax2_top.axvline(rel_time_char[0], color="C0", ls = "--", lw = 2, label = r"$\langle t_{true} - t_{reco} \rangle$")
    ax2_top.axvspan(rel_time_char[1], rel_time_char[2], color="C0", alpha = 0.25)
    ax2_top.tick_params(labelsize = fs)

    # histogram for reco freq above the scatter plot
    ax2_right.hist(rel_freq.value, bins = bins, range = (fmin, fmax), density=True, color='grey', orientation='horizontal', align = "left")
    ax2_right.axhline(rel_freq_char[0], color="C0", ls = "--", lw = 2, label = r"$\langle f_{true} - f_{reco} \rangle$")
    ax2_right.axhspan(rel_freq_char[1], rel_freq_char[2], color="C0", alpha = 0.25)
    ax2_right.tick_params(labelsize = fs)

    plt.tight_layout()
    filename = "./plots/reco/{}/{}/RECO2_model_{}_{:.0f}_mode_{}_duration_{:.0f}ms_ampl_{:.1f}%_mix_{}_hier_{}_trials_{:1.0e}_{}_{}_dist_{:.1f}kpc.pdf".format(
    self.ft_mode, self.reco_dir_name, self.model["name"], self.model["param"]["progenitor_mass"].value, 
    self.ft_mode, self.reco_para["duration"].value, self.reco_para["ampl"][0]*100, 
    self.mixing_scheme, self.hierarchy, self.trials, hypo, det, self.distance.value)
    plt.savefig(filename, dpi = 400)
    plt.close()

def plot_reco_bias(self, hypo = "signal", det = "ic86"):

    fs = 10
    
    fig, ax = plt.subplots(1,2)

    rel_time = self.time_true-self.time_reco[hypo][det]
    rel_freq = self.freq_true-self.freq_reco[hypo][det]

    ax[0].scatter(self.time_true, rel_time, s=10, alpha=0.1, color="C0", rasterized = True)
    ax[0].set_xlabel(r"$t_{true}$ [ms]", fontsize = fs)
    ax[0].set_ylabel(r"$t_{true} - t_{reco}$ [ms]", fontsize = fs)
    ax[0].tick_params(labelsize = fs)

    ax[1].scatter(self.freq_true, rel_freq, s=10, alpha=0.1, color="C0", rasterized = True)
    ax[1].set_xlabel(r"$f_{true}$ [ms]", fontsize = fs)
    ax[1].set_ylabel(r"$f_{true} - f_{reco}$ [ms]", fontsize = fs)
    ax[1].tick_params(labelsize = fs)

    plt.tight_layout()
    filename = "./plots/reco/{}/{}/BIAS_model_{}_{:.0f}_mode_{}_duration_{:.0f}ms_ampl_{:.1f}%_mix_{}_hier_{}_trials_{:1.0e}_{}_{}_dist_{:.1f}kpc.pdf".format(
    self.ft_mode, self.reco_dir_name, self.model["name"], self.model["param"]["progenitor_mass"].value, 
    self.ft_mode, self.reco_para["duration"].value, self.reco_para["ampl"][0]*100, 
    self.mixing_scheme, self.hierarchy, self.trials, hypo, det, self.distance.value)
    plt.savefig(filename, dpi = 400)
    plt.close()


def plot_reco_horizon(self, ampl, hypo = "signal"):
    
    fs = 14

    fig, ax = plt.subplots(2,1, sharex = True)

    labels = [r'IceCube', r'Gen2', r'Gen2+WLS']
    colors = ["C0", "C1", "C2"]

    for i, det in enumerate(["ic86", "gen2", "wls"]): # loop over detector

        ax[0].plot(self.dist_range.value, self.freq_stat[hypo][det][ampl][:,0].value, color = colors[i], label = labels[i])
        ax[0].fill_between(self.dist_range.value, self.freq_stat[hypo][det][ampl][:,1].value, self.freq_stat[hypo][det][ampl][:,2].value, color = colors[i], alpha = 0.15)
        ax[1].plot(self.dist_range.value, self.time_stat[hypo][det][ampl][:,0].value, color = colors[i])
        ax[1].fill_between(self.dist_range.value, self.time_stat[hypo][det][ampl][:,1].value, self.time_stat[hypo][det][ampl][:,2].value, color = colors[i], alpha = 0.15)

    ax[1].set_xlabel("Distance [kpc]", fontsize = fs)
    ax[0].set_ylabel(r"$f_{reco}-f_{true}$ [Hz]", fontsize = fs)
    ax[1].set_ylabel(r"$t_{reco}-t_{true}$ [ms]", fontsize = fs)

    ax[0].set_xlim(0, 60)
    ax[1].set_xlim(0, 60)

    ax[0].tick_params(labelsize = fs)
    ax[1].tick_params(labelsize = fs)

    ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.4), fontsize=14, ncols = 3)

    plt.tight_layout()
    filename = "./plots/reco/{}/{}/HORI_model_{}_{:.0f}_mode_{}_duration_{:.0f}ms_ampl_{:.1f}%_mix_{}_hier_{}_trials_{:1.0e}_{}.pdf".format(
    self.ft_mode, self.reco_dir_name, self.model["name"], self.model["param"]["progenitor_mass"].value, 
    self.ft_mode, self.reco_para["duration"].value, self.reco_para["ampl"][0]*100,
    self.mixing_scheme, self.hierarchy, self.trials, hypo)
    plt.savefig(filename, dpi = 400)
    plt.close()

def plot_reco_horizon_diff(self, ampl, hypo = "signal"):
    
    fs = 14

    fig, ax = plt.subplots(2,1, sharex = True)

    labels = [r'IceCube', r'Gen2', r'Gen2+WLS']
    colors = ["C0", "C1", "C2"]
    lss = ["-", "--", ":"]

    for i, det in enumerate(["ic86", "gen2", "wls"]): # loop over detector

        ax[0].plot(self.dist_range.value, self.freq_diff[hypo][det][ampl,:].value, color = colors[i], ls = lss[i], label = labels[i])
        ax[1].plot(self.dist_range.value, self.time_diff[hypo][det][ampl,:].value, color = colors[i], ls = lss[i])

    ax[1].set_xlabel("Distance [kpc]", fontsize = fs)
    ax[0].set_ylabel(r"$f_{reco}-f_{true}$ [Hz]", fontsize = fs)
    ax[1].set_ylabel(r"$t_{reco}-t_{true}$ [ms]", fontsize = fs)

    for i, fthresh in enumerate(self.freq_thresh):
        ax[0].axhline(fthresh.value, color = "grey", ls = "--", lw = 2)
        ax[0].text(50, fthresh.value * 1.2, s = "{:.0f}".format(fthresh), fontsize = fs, color = "grey", weight = "bold")

    for i, tthresh in enumerate(self.time_thresh):
        ax[1].axhline(tthresh.value, color = "grey", ls = "--", lw = 2)
        ax[1].text(50, tthresh.value * 1.2, s = "{:.0f}".format(tthresh), fontsize = fs, color = "grey", weight = "bold")


    ax[0].set_xlim(0, 60)
    ax[1].set_xlim(0, 60)

    ax[0].set_ylim(5, 600)
    ax[1].set_ylim(10, 1000)

    ax[0].set_yscale("log")
    ax[1].set_yscale("log")

    ax[0].tick_params(labelsize = fs)
    ax[1].tick_params(labelsize = fs)

    ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.4), fontsize=14, ncols = 3)

    plt.tight_layout()
    filename = "./plots/reco/{}/{}/DIFF_model_{}_{:.0f}_mode_{}_duration_{:.0f}ms_ampl_{:.1f}%_mix_{}_hier_{}_trials_{:1.0e}_{}.pdf".format(
    self.ft_mode, self.reco_dir_name, self.model["name"], self.model["param"]["progenitor_mass"].value, 
    self.ft_mode, self.reco_para["duration"].value, self.ampl_range[ampl]*100,
    self.mixing_scheme, self.hierarchy, self.trials, hypo)
    plt.savefig(filename, dpi = 400)
    plt.close()

def plot_reco_horizon_amplitude(self, hypo = "signal"):
    
    fs = 16

    labels = [r'IceCube', r'Gen2', r'Gen2+WLS']
    colors = ['C0', 'C1', 'C2']

    fig, ax = plt.subplots(2,1, sharex = True)

    for j, det in enumerate(["ic86", "gen2", "wls"]):
        ax[0].plot(self.ampl_range * 100, self.freq_hori[hypo][det], color = colors[j], label  = labels[j])
        ax[1].plot(self.ampl_range * 100, self.time_hori[hypo][det], color = colors[j])

    ax[1].set_xlabel("Amplitude [%]", size=fs)
    ax[0].set_ylabel(r"$d_{20 \rm Hz}$ [kpc]", size=fs)
    ax[1].set_ylabel(r"$d_{50 \rm ms}$ [kpc]", size=fs)

    for i in range(2):
        ax[i].tick_params(labelsize=fs)
        ax[i].set_xlim(0,50)
        ax[i].set_ylim(0,50)
        ax[i].set_yticks([10,20,30,40,50])
        ax[i].grid()

    ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.4), fontsize=14, ncols = 3)

    plt.tight_layout()
    filename = "./plots/reco/{}/{}/AMPL_model_{}_{:.0f}_mode_{}_duration_{:.0f}ms_mix_{}_hier_{}_trials_{:1.0e}_{}.pdf".format(
    self.ft_mode, self.reco_dir_name, self.model["name"], self.model["param"]["progenitor_mass"].value, 
    self.ft_mode, self.reco_para["duration"].value, self.mixing_scheme, self.hierarchy, self.trials, hypo)
    plt.savefig(filename, dpi = 400)
    plt.close()

def plot_reco_boot(self, hypo = "signal"):

    fs = 14

    fig, ax = plt.subplots(3, 2, figsize=(12, 10), sharex = "col")
    ax = ax.T.ravel()

    dets = ["ic86", "gen2", "wls"]

    for i in range(6):

        j = i % 3
        det = dets[j]

        if i > 2: item = self.boot_time[hypo]
        else: item = self.boot_freq[hypo]

        error = np.std(item[det], axis = 1)

        n1, _, _ = ax[i].hist(item[det][1].value, bins=20, density=True, histtype="step", color="C1", label="16%")
        n0, _, _ = ax[i].hist(item[det][0].value, bins=20, density=True, histtype="step", color="C0", label="50%")
        n2, _, _ = ax[i].hist(item[det][2].value, bins=20, density=True, histtype="step", color="C2", label="84%")

        ax[i].text(item[det][1].mean().value, n1.max()/2, s = "{:1.1E}".format(error[1]), color = "C1", fontsize = fs, weight = "bold", ha = "center")
        ax[i].text(item[det][0].mean().value, n0.max()/2, s = "{:1.1E}".format(error[0]), color = "C0", fontsize = fs, weight = "bold", ha = "center")
        ax[i].text(item[det][2].mean().value, n2.max()/2, s = "{:1.1E}".format(error[2]), color = "C2", fontsize = fs, weight = "bold", ha = "center")

        ax[i].text(0.15, 0.9, s = det, transform=ax[i].transAxes, fontsize=fs, va='top', 
                bbox= dict(boxstyle='round', facecolor='white', alpha=0.5))
        ax[i].tick_params(labelsize = fs)


    # add xlabel for each row and one common ylabel
    ax[2].set_xlabel(r"$f_{reco}-f_{true}$ [Hz]", fontsize = fs)
    ax[5].set_xlabel(r"$t_{reco}-t_{true}$ [ms]", fontsize = fs)
    fig.supylabel("Normalized Counts", fontsize = fs)

    # Create a common legend for all subplots
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', fontsize = fs, ncol = 3, bbox_to_anchor=(0.5, 1.0))

    filename = "./plots/bootstrapping/reco/{}/BOOT_model_{}_{:.0f}_mode_{}_duration_{:.0f}ms_ampl_{:.1f}%_mix_{}_hier_{}_tot_trials_{:1.0e}_rep_trials_{:1.0e}_reps_{}_{}_dist_{:.1f}kpc.pdf".format(
    self.ft_mode, self.model["name"], self.model["param"]["progenitor_mass"].value, 
    self.ft_mode, self.reco_para["duration"].value, self.reco_para["ampl"][0]*100, 
    self.mixing_scheme, self.hierarchy, self.trials, self.rep_trials, self.repetitions, hypo, self.distance.value)
    plt.savefig(filename, bbox_inches='tight')

def plot_reco_at_distance(self, hypo = "signal"):
    
    fs = 16

    colors = ['C0', 'C1', 'C2']
    lss = ["-", ":"]

    legend1_handles = []
    legend2_handles = []
    legend1_labels = [r'IceCube', r'Gen2', r'Gen2+WLS']
    legend2_labels = ["MW Centre", "MW Edge"]

    fig, ax = plt.subplots(2,1, sharex = True, gridspec_kw={'hspace': 0.2})


    for j, det in enumerate(["ic86", "gen2", "wls"]):

        for i, d in enumerate(self.dist_range):
            dist_mask = self.dist_range == d
            if i == 0:
                handle, = ax[0].plot(self.ampl_range * 100, self.freq_diff[hypo][det][:,dist_mask], color = colors[j], ls = lss[i])
                legend1_handles.append(handle)
            ax[0].plot(self.ampl_range * 100, self.freq_diff[hypo][det][:,dist_mask], color = colors[j], ls = lss[i])
            ax[1].plot(self.ampl_range * 100, self.time_diff[hypo][det][:,dist_mask], color = colors[j], ls = lss[i])

    ax[0].axhline(6.88, color = "grey", ls = "--")
    ax[1].axhline(2*6.88, color = "grey", ls = "--")

    # Plot the grey lines outside of range, just for legend
    line1, = ax[0].plot(np.arange(-20, -10), np.ones_like(np.arange(-20, -10)), color="grey", ls="-")
    line2, = ax[0].plot(np.arange(-20, -10), np.ones_like(np.arange(-20, -10)), color="grey", ls=":")
    legend2_handles.extend([line1, line2])

    comb_handles = legend1_handles + legend2_handles
    comb_labels = legend1_labels + legend2_labels

    rearr = [0,3,1,4,2] # rearrange handels and labels
    rearr_handels, rearr_labels = [], []

    for i in rearr:
        rearr_handels.append(comb_handles[i])
        rearr_labels.append(comb_labels[i])


    # Create legends
    ax[0].legend(rearr_handels, rearr_labels, loc="upper center", bbox_to_anchor=(0.5, 1.55), fontsize=14, ncols=3)

    ax[1].set_xlabel("Amplitude [%]", size=fs)
    ax[0].set_ylabel(r"$ \langle f_{reco}-f_{true} \rangle_{1\sigma}$ [Hz]", size=fs)
    ax[1].set_ylabel(r"$ \langle t_{reco}-t_{true} \rangle_{1\sigma}$ [ms]", size=fs)

    for i in range(2):
        ax[i].tick_params(labelsize=fs)
        ax[i].set_xlim(0,50)
        ax[i].set_ylim(5, 1000)
        ax[i].set_yscale("log")
        ax[i].grid()

    filename = "./plots/reco/{}/{}/REFI_model_{}_{:.0f}_mode_{}_duration_{:.0f}ms_mix_{}_hier_{}_trials_{:1.0e}_{}.pdf".format(
    self.ft_mode, self.reco_dir_name, self.model["name"], self.model["param"]["progenitor_mass"].value, 
    self.ft_mode, self.reco_para["duration"].value, self.mixing_scheme, self.hierarchy, self.trials, hypo)
    plt.savefig(filename, dpi = 400, bbox_inches='tight')
    plt.close()
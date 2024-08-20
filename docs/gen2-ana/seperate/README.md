This directory contains all modules needed to perform the analysis explained in [cit!]. In particular:

- analysis.py : Analysis class using same number of trials for signal and null hypothesis. Used in signal_hypothesis.py to estimate the bounds of the distance range for which the significance falls between 5 and 3 sigma.
- background_trials.py : Wrapper class used to generate background trials using null_hypothesis.py. Results are histogrammed. Also contains methods to load and combine data.
- bootstrapping.py : Depricated class used to estimate the optimal number of signal and background trials. Not actively used.
- helper.py : Contains helper functions such as minimzer objective functions etc.
- null_hypothesis.py : Class used to generate background trials under the null hypothesis assumption. Generated seperatly as larger number of background trials is needed for sufficient statistics.
- plthelper.py : Plotting functions used in active files as well as test scripts in ./tests
- run_background_trials.py : Wrapper script used to feed the correct parameters to background_trials.py. Script takes input arguments that are supplied by run_background_trials.sh, the respective bash script executed in parallel with sbatch.
- run_scan.py : Wrapper script used to feed the correct parameters to scan.py. Script takes input arguments that are supplied by run_scan.sh, the respective bash script executed in parallel with sbatch.
- scan.py : Class used to perform generic model parameter scan. Main part is execution of signal_hyothesis, equivalent of null_hypothesis but for signal hypothesis, as the name suggests. The class returns significance horizon and parameter reconstruction resolution for each parameter combination. For parallelisation with run_scan.py, one amplitude is typically selected and the scan is performed for a range of frequencies.
- signal_hypothesis : The centre piece of the analysis computing the counts of the signal hypothesis on a discretized distance grid, loading the background distribution produced by background_hypothesis.py. Output is significance and reconstruction resolution as a function of the source distance in the prior defined distance interval.

Main methods and struction of analysis.py, null_hypothesis.py and signal_hypothesis.py:
- (averaged) background rate generation
- for analysis.py, signal_hypothesis.py: generate fast-time feature template
- sinal rate generation: baseline model (+ fast-time feature template)
- time and frequency filters
- FFT or STF method
- max TS value used as proxy for significance and best reconstructed parameter
- repeat for several trials to build a distribution
- for analysis.py and signal_hypothesis.py: compute significance of seperation between null hypothesis and signal hypothesis TS distribution
- repeat for range of distances 

We offer two analysis methods:
- FFT : Fast Fourier Transform, "standard" time-integrated Fourier analysis
- STF : Short Time Fourier Transform, time-depenent Fourier analysis

We define a default analysis as follows:
- fast-time feature duration: 150 ms - 300 ms
- signal trials = 10,000
- background trials = 10,000,000 binned into histogram of 20,000 bins
- no signal or background rate variation: sig_var = 0%, bkg_var = 0%
- no flavour transformation from source to Earth: mixing_scheme = "NoTransformation"

The following systematics are considered and saved in respective directories under ./files/background, ./files/scan and ./plots/scan:
- shorter fast-time feature duration: 150 ms - 200 ms (syst_time_150-200ms)
- longer fast-time feature duration: 150 ms - 400 ms (syst_time_150-400ms)
- 10% increase of the signal rate: sig_var = 10% (syst_det_sig_+10%)
- 10% reduction of the signal rate: sig_var = -10% (syst_det_sig_-10%)
- 10% increase of the background rate: bkg_var = 10% (syst_det_bkg_+10%)
- 10% reduction of the background rate: bkg_var = -10% (syst_det_bkg_-10%)
- complete flavour exchange from source to Earth: mixing_scheme = "CompleteExchange" (syst_mix_comp_exch)
- adiabatic MSW with normal hierarchy: mixing_scheme = "AdiabaticMSW", hierarchy = "normal" (syst_mix_MSW_NH)
- adiabatic MSW with inverted hierarchy: mixing_scheme = "AdiabaticMSW", hierarchy = "normal" (syst_mix_MSW_IH)
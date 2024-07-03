import os
import numpy as np
from asteria.stellardist import StellarDensity
from scipy.interpolate import PchipInterpolator, InterpolatedUnivariateSpline, interp1d
from scipy.optimize import brentq, minimize
import scipy.stats as stats

import astropy.units as u

def argmax_lastNaxes(A, N):
    # extension of argmax over several axis
    s = A.shape
    new_shp = s[:-N] + (np.prod(s[-N:]),)
    max_idx = np.nanargmax(A.reshape(new_shp), axis = -1)
    return np.unravel_index(max_idx, s[-N:])
    
def moving_average(a, n=3, zero_padding = False, const_padding = False):
    if zero_padding:
        a = np.insert(a, np.zeros(n-1,dtype=int), np.zeros(n-1), axis=-1)
        a = np.roll(a, -int((n-1)/2), axis=-1)
    if const_padding:
        l1 = int(n/2)
        if n%2 != 1: # n is even
            l2 = l1-1
        else: # n is odd
            l2 = l1
            ind2 = -np.arange(1,(n+1)/2).astype(int)
        a = np.insert(a, np.zeros(l1, dtype=int), np.ones(l1)*a[0])
        a = np.insert(a, -np.ones(l2, dtype=int), np.ones(l2)*a[-1])
    ret = np.cumsum(a, dtype=float, axis=-1)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def loss_stellar(dist, cdf_val):
    # for distances above 25 kpc the CDF = 1, but we want hit the 'edge'
    # we punish any value that is higher than 25 kpc + epsilon, epsilon << 1

    if dist > 25 + 1E-3:
        return 1E6
    else:
        loss = (stellar_inter(dist) - cdf_val)
        return loss

def distance_to_coverage(dist):
    # turns distances into coverages (CDF values)
    return stellar_inter(dist)
    
def coverage_to_distance(cdf_val):
    # turns coverages (CDF value) into distances

    # for single value
    if not isinstance(cdf_val, np.ndarray):
        root = brentq(loss_stellar, a = 1, b = 100, args = cdf_val, xtol = 1e-2)
        return root
    
    # for an array loop through the entries
    else:
        dist = [] # empty distance array
        for cv in cdf_val:
            root = brentq(loss_stellar, a = 1, b = 100, args = cv, xtol = 1e-2)
            dist.append(root)
        return np.array(dist)

def loss_dist_horizon(dist, quan, sig):
    return (quan(dist)-sig)

def significance_horizon(dist_range, Zscore, sigma = [3,5]):
                        
    dist = {"ic86": [], "gen2": [], "wls": []} # empty dictionary
    perc = {"ic86": [], "gen2": [], "wls": []}
                        

    for det in ["ic86", "gen2", "wls"]: # loop over detector

        dist_min = 0.5 * u.kpc

        # cutoff distance for interpolation, no infs, no nans?
        dm50 = np.logical_and(dist_range>dist_min, ~np.isinf(Zscore[det][0]))
        dm16 = np.logical_and(dist_range>dist_min, ~np.isinf(Zscore[det][1]))
        dm84 = np.logical_and(dist_range>dist_min, ~np.isinf(Zscore[det][2]))

        # cubic spline (k = 3) with constant values outside boundaries
        q50 = InterpolatedUnivariateSpline(x = dist_range[dm50], y = Zscore[det][0][dm50], k = 3, ext = 3)
        q16 = InterpolatedUnivariateSpline(x = dist_range[dm16], y = Zscore[det][1][dm16], k = 3, ext = 3)
        q84 = InterpolatedUnivariateSpline(x = dist_range[dm84], y = Zscore[det][2][dm84], k = 3, ext = 3) 
        quantiles = [q50, q16, q84]

        for sig in sigma: # loop over confidence level (e.g. 3, 5 sigma)
            di, pe = [], [] # temporary lists to store distance and percentage

            for quan in quantiles: # loop over quantiles
                root = brentq(loss_dist_horizon, a = 0.1, b = 100, args = (quan, sig), xtol = 1e-2)
                di.append(root)
                if root >= 25:
                    pe.append(1)
                else:
                    pe.append(stellar_inter(root))
            dist[det].append(np.array(di) * u.kpc)
            perc[det].append(np.array(pe)*100)
        
    return dist, perc

def resolution_at_horizon(dist_range, quant, horizon, sigma = [3,5]):

    reso =  {"ic86": [], "gen2": [], "wls": []} # empty dictionary

    for det in ["ic86", "gen2", "wls"]: # loop over detector

        # cubic spline (k = 3) with constant values outside boundaries
        q50 = InterpolatedUnivariateSpline(x = dist_range, y = quant[det][0], k = 3, ext = 3)
        q16 = InterpolatedUnivariateSpline(x = dist_range, y = quant[det][1], k = 3, ext = 3)
        q84 = InterpolatedUnivariateSpline(x = dist_range, y = quant[det][2], k = 3, ext = 3) 
        quantiles = [q50, q16, q84]

        for s, sig in enumerate(sigma): # loop over confidence level (e.g. 3, 5 sigma)
            re = [] # temporary lists to store resolution
            
            for q, quan in enumerate(quantiles): # loop over quantiles
                re.append(quan(horizon[det][s][q]))
            reso[det].append(np.array(re))

    return reso

def get_distribution_by_name(name):
    distribution = getattr(stats, name, None)
    if distribution is not None and callable(distribution):
        return distribution
    else:
        raise ValueError('{} not a supported method in scipy.stats.'.format(name))


def call_distribution(bkg_distr, para, x_hist, log_scale = None):
    distribution = bkg_distr
    distr = distribution(*para)
    if log_scale:
        return np.log10(distr.pdf(x_hist))
    else:
        return distr.pdf(x_hist)

def fit_distribution(bkg_distr, hist, log_scale = None):

    # bin centres and bin counts
    x_hist, y_hist = hist

    # remove zeros
    null_mask = y_hist == 0
    x_hist = x_hist[~null_mask]
    y_hist = y_hist[~null_mask]

    # fit only values larger than the max
    #ind_max = np.argmax(y_hist)
    #x_hist = x_hist[ind_max:]
    #y_hist = y_hist[ind_max:]

    # distribution number of parameters
    num_params = len(bkg_distr._param_info())

    # location and scale of distribtion from histogram as initial guess
    loc = np.sum(x_hist * y_hist) / np.sum(y_hist)
    std = np.sqrt(np.sum(y_hist * (x_hist - loc)**2) / np.sum(y_hist))

    if num_params == 2:     
        starting_params = (loc, std)
    elif num_params == 3:
        starting_params = (1, loc, std)

    # take logarithm of y values
    if log_scale: y_hist = np.log10(y_hist)  

    res = minimize(loss_lognorm, x0 = starting_params, args=[x_hist, y_hist, bkg_distr, log_scale], method="Nelder-Mead")
    return res

def loss_lognorm(para, args):
    x_hist, y_hist, bkg_distr, log_scale = args
    y_fit = call_distribution(bkg_distr, para, x_hist, log_scale)
    return np.sqrt(np.sum((y_fit-y_hist)**2))

def interpolate_bounds(bounds, distance):
    inter_bounds = {"ic86" : [], "gen2" : [], "wls": []}
    for det in ["ic86", "gen2", "wls"]:
        inter_bounds[det].append(InterpolatedUnivariateSpline(bounds["dist"], bounds[det][:,0], k = 3, ext = 0)(distance)) # lower bound interpolation
        inter_bounds[det].append(InterpolatedUnivariateSpline(bounds["dist"], bounds[det][:,1], k = 3, ext = 0)(distance)) # higher bound interpolation

    return inter_bounds

def quantiles_histogram(hist, perc):
    # get PDF and bin center, for pre-binned hist in files/background/hist
    bin_center, pdf = hist

    # calculate CDF
    cdf = np.cumsum(pdf)
    cdf = cdf / cdf[-1]

    # interpolate CDF function for approx quantile
    interp_cdf = interp1d(cdf, bin_center)

    quant = []
    for p in perc: quant.append(interp_cdf(p))

    return np.array(quant)

# stellar distribution file, Adams 2013 model, returns CDF
stellar_dist = StellarDensity(os.environ.get("ASTERIA") + '/data/stellar/sn_radial_distrib_adams.fits', add_LMC=False, add_SMC=False)
# interpolated CDF
stellar_inter = PchipInterpolator(stellar_dist.dist.value, stellar_dist.cdf)
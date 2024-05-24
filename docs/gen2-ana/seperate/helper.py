import os
import numpy as np
from asteria.stellardist import StellarDensity
from scipy.interpolate import PchipInterpolator, InterpolatedUnivariateSpline
from scipy.optimize import brentq

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

        # cutoff distance for interpolation, inf values need to be excluded, take 86% and first distance
        dist_cutoff = dist_range[np.logical_and(dist_range>dist_min, np.where(np.isinf(Zscore[det][2])==True, 0, 1))][0]
        
        # cubic spline (k = 3) with constant values outside boundaries
        q50 = InterpolatedUnivariateSpline(x = dist_range[dist_range>=dist_cutoff], y = Zscore[det][0][dist_range>=dist_cutoff], k = 3, ext = 3)
        q16 = InterpolatedUnivariateSpline(x = dist_range[dist_range>=dist_cutoff], y = Zscore[det][1][dist_range>=dist_cutoff], k = 3, ext = 3)
        q84 = InterpolatedUnivariateSpline(x = dist_range[dist_range>=dist_cutoff], y = Zscore[det][2][dist_range>=dist_cutoff], k = 3, ext = 3)
        
        quantiles = [q50, q16, q84]

        for sig in sigma: # loop over confidence level (e.g. 3, 5 sigma)
            di, pe = [], [] # temporary lists to store distance and percentage

            for quan in quantiles: # loop over quantiles

                root = brentq(loss_dist_horizon, a = 1, b = 100, args = (quan, sig), xtol = 1e-2)
                di.append(root)
                if root >= 25:
                    pe.append(1)
                else:
                    pe.append(stellar_inter(root))
                
            dist[det].append(np.array(di) * u.kpc)
            perc[det].append(np.array(pe)*100)
        
    return dist, perc


# stellar distribution file, Adams 2013 model, returns CDF
stellar_dist = StellarDensity(os.environ.get("ASTERIA") + '/data/stellar/sn_radial_distrib_adams.fits', add_LMC=False, add_SMC=False)
# interpolated CDF
stellar_inter = PchipInterpolator(stellar_dist.dist.value, stellar_dist.cdf)
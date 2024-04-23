import os
import numpy as np
from asteria.stellardist import StellarDensity
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize

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
        loss = np.abs(stellar_inter(dist) - cdf_val)
        return loss

def distance_to_coverage(dist):
    # turns distances into coverages (CDF values)
    return stellar_inter(dist)
    
def coverage_to_distance(cdf_val):
    # turns coverages (CDF value) into distances

    # for single value
    if not isinstance(cdf_val, np.ndarray):
        res = minimize(loss_stellar, x0 = 5, args=cdf_val)
        return res.x
    
    # for an array loop through the entries
    else:
        dist = [] # empty distance array
        for cv in cdf_val:
            loss, i = 1, 0 # initialize loss and counter
            x0 = np.array([5,10,15,20]) # list of initial guess
            while loss > 1E-3: # repeat as long as loss > 0.001
                res = minimize(loss_stellar, x0 = x0[i], args = cv)
                loss = res.fun
                i += 1
                if i > 3:
                    print('Fit failed to converge!')
                    break
            dist.append(res.x)
        dist = np.array(dist)
        return dist
    
def loss_dist_horizon(dist, quan, sig):
    return np.sqrt(np.sum((quan(dist)-sig)**2))

def significance_horizon(dist_range, Zscore, confidence_level = [3,5]):
                        
    sigma = [str(cl) + "sig" for cl in confidence_level] 

    dist = {key : {"ic86": None, "gen2": None, "wls": None} for key in sigma} # empty dictionary
    perc = {key : {"ic86": None, "gen2": None, "wls": None} for key in sigma}
                        

    for det in ["ic86", "gen2", "wls"]: # loop over detector

        dist_min = 0.5 * u.kpc

        # cutoff distance for interpolation, inf values need to be excluded, take 86% and first distance
        dist_cutoff = dist_range[np.logical_and(dist_range>dist_min, np.where(np.isinf(Zscore[det][2])==True, 0, 1))][0]

        # Interpolate Zscore data for 16%, 50% and 84% quantiles
        q50 = PchipInterpolator(dist_range[dist_range>dist_cutoff], Zscore[det][0][dist_range>dist_cutoff])
        q16 = PchipInterpolator(dist_range[dist_range>dist_cutoff], Zscore[det][1][dist_range>dist_cutoff])
        q84 = PchipInterpolator(dist_range[dist_range>dist_cutoff], Zscore[det][2][dist_range>dist_cutoff])
        
        quantiles = [q50, q16, q84]

        for i, cl in enumerate(confidence_level): # loop over confidence level (e.g. 3, 5 sigma)
            x0 = dist_cutoff.value #???
            di, pe = [], [] # temporary lists to store distance and percentage

            for quan in quantiles: # loop over quantiles
                loss = 1
                while loss > 1E-3:
                    res = minimize(loss_dist_horizon, x0=x0, args = (quan, cl))
                    loss = res.fun
                    x0 += 1
                    if x0 > 50:
                        print('Minimization did not converge')
                        break
                di.append(res.x[0])
                if res.x >= 25:
                    pe.append(1)
                else:
                    pe.append(stellar_inter(res.x[0]))
                
            dist[sigma[i]][det] = np.array(di) * u.kpc
            perc[sigma[i]][det] = np.array(pe)*100

        
    return dist, perc


# stellar distribution file, Adams 2013 model, returns CDF
stellar_dist = StellarDensity(os.environ.get("ASTERIA") + '/data/stellar/sn_radial_distrib_adams.fits', add_LMC=False, add_SMC=False)
# interpolated CDF
stellar_inter = PchipInterpolator(stellar_dist.dist.value, stellar_dist.cdf)
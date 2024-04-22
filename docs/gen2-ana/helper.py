import os
import numpy as np
from asteria.stellardist import StellarDensity
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize

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


def loss_function(dist, cdf_val):
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
        res = minimize(loss_function, x0 = 5, args=cdf_val)
        return res.x
    
    # for an array loop through the entries
    else:
        dist = [] # empty distance array
        for cv in cdf_val:
            loss, i = 1, 0 # initialize loss and counter
            x0 = np.array([5,10,15,20]) # list of initial guess
            while loss > 1E-3: # repeat as long as loss > 0.001
                res = minimize(loss_function, x0 = x0[i], args = cv)
                loss = res.fun
                i += 1
                if i > 3:
                    print('Fit failed to converge!')
                    break
            dist.append(res.x)
        dist = np.array(dist)
        return dist


# stellar distribution file, Adams 2013 model, returns CDF
stellar_dist = StellarDensity(os.environ.get("ASTERIA") + '/data/stellar/sn_radial_distrib_adams.fits', add_LMC=False, add_SMC=False)
# interpolated CDF
stellar_inter = PchipInterpolator(stellar_dist.dist.value, stellar_dist.cdf)
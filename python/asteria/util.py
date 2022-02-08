import numpy as np
from numbers import Number
from scipy.special import loggamma, gdtr

def _energy_pdf(a, Ea, E):
    return np.exp((1 + a) * np.log(1 + a) - loggamma(1 + a) +
                  a * np.log(E) - (1 + a) * np.log(Ea) - (1 + a) * (E / Ea))

def parts_by_index(x, n):
    """Returns a list of size-n numpy arrays containing indices for the
    elements of x, and one size-m array ( with m<n ) if there are remaining
    elements of x.

    Returns
    -------
    i_part : list
       List of index partitions (partitions are numpy array).
    """
    nParts = x.size//n
    i_part = [ np.arange( i*n, (i+1)*n ) for i in range(nParts) ]

    # Generate final partition of size <n if x.size is not multiple of n
    if len(i_part)*n != x.size:
        i_part += [ np.arange( len(i_part)*n, x.size ) ]

    # Ensure that last partition always has 2 or more elements
    if len(i_part[-1]) < 2:
        i_part[-2] = np.append(i_part[-2], i_part[-1])
        i_part = i_part[0:-1]

    return i_part



def energy_pdf(a, Ea, E, *, limit=1000):
    # TODO: Figure out how to reconcile this
    if all(isinstance(var, np.ndarray) for var in (a, Ea)):
        if isinstance(a, (list, tuple, np.ndarray)):
            # It is non-physical to have a<0 but some model files/interpolations still have this
            _vec_energy_pdf = np.vectorize(_energy_pdf, excluded=['E'], signature='(1,n),(1,n)->(m,n)')
            a[a<0] = 0
            cut = (a >= 0) & (Ea > 0)
            E_pdf = np.zeros( (E.size, a.size), dtype = float )
            E_pdf[:, cut] = _vec_energy_pdf( a[cut].reshape(1,-1), Ea[cut].reshape(1,-1), E=E.reshape(-1,1))
            cut = (a < 0) & (Ea > 0)
            E_pdf[:, cut] = _vec_energy_pdf(np.zeros_like(a[cut]).reshape(1, -1), Ea[cut].reshape(1, -1), E=E.reshape(-1, 1))
            return E_pdf

        # if a.size == Ea.size:
        #     # Vectorized function can lead to unregulated memory usage, better to define it only when needed
        #     _vec_energy_pdf = np.vectorize(_energy_pdf, excluded=['E'], signature='(1,n),(1,n)->(n,m)')
        #
        #     result = np.zeros(shape=(a.size, E.size), dtype=np.float64)
        #
        #     # Partition in time (same dimensionality as `a` and `Ea`), to regulate memory usage in vectorized function
        #     idx = 0
        #     if limit < E.size:
        #         idc_split = np.arange(E.size, step=limit)
        #         for idx in idc_split[:-1]:
        #             result[:, idx:idx+limit] = _vec_energy_pdf(a=a, Ea=Ea, E=E[idx:idx+limit])
        #     result[:, idx:] = _vec_energy_pdf(a=a, Ea=Ea, E=E[idx:idx+limit])
        #     return result
        # else:
        #     raise ValueError('Invalid input array size. Arguments `a` and `Ea` must have the same size.  '
        #                      f'Given sizes ({a.size}) and ({Ea.size}) respectively.')
    elif all(isinstance(var, Number) for var in (a, Ea)):
        return _energy_pdf(a, Ea, E)
    else:
        raise ValueError('Invalid argument types, arguments `a` and `Ea` must be numbers or np.ndarray.  '
                         f'Given types ({type(a)}) and ({type(Ea)}) respectively.')


def energy_cdf(a, Ea, E):
    return gdtr(1., a + 1., (a + 1.) * (E / Ea))
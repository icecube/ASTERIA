import numpy as np
import astropy.units as u


def get_sukhbold_2015_fname(eos, progenitor_mass, **kwargs):
    if eos not in ('LS220', 'SFHo'):
        raise ValueError(f'Invalid value for model argument `eos`, expected ("LS220", "SFHo") given "{eos}"')

    if progenitor_mass.value == 9.6:
        fname = f'sukhbold-{eos}-z{progenitor_mass.value:3.1f}.fits'
    elif progenitor_mass == 27.0:
        fname = f'sukhbold-{eos}-s{progenitor_mass.value:3.1f}.fits'
    else:
        raise ValueError('Invalid value for model argument `progenitor_mass`, expected (9.6, 27.0) Msun, '
                         f'given {progenitor_mass}')
    return fname


def get_tamborra_2014_fname(progenitor_mass, **kwargs):
    if progenitor_mass.value in (20.0, 27.0):
        fname = f's{progenitor_mass.value:3.1f}c_3D_dir1'
    else:
        raise ValueError('Invalid value for model argument `progenitor_mass`, expected (20.0, 27.0) Msun, '
                         f'given {progenitor_mass}')
    return fname


def get_bollig_2016_fname(progenitor_mass, **kwargs):
    if progenitor_mass.value in (11.2, 27.0):
        fname = f's{progenitor_mass.value:3.1f}c'
    else:
        raise ValueError('Invalid value for model argument `progenitor_mass`, expected (20.0, 27.0) Msun, '
                         f'given {progenitor_mass.value}')
    return fname


def get_walk_2018_fname(progenitor_mass=15. * u.Msun, **kwargs):
    if progenitor_mass.value != 15.0:
        raise ValueError('Invalid value for model argument `progenitor_mass`, expected (15.0) Msun, '
                         f'given {progenitor_mass}')
    return f's{progenitor_mass.value:3.1f}c_3D_nonrot_dir1'


def get_walk_2019_fname(progenitor_mass=40. * u.Msun, **kwargs):
    if progenitor_mass.value != 40.0:
        raise ValueError('Invalid value for model argument `progenitor_mass`, expected (40.0) Msun, '
                         f'given {progenitor_mass}')
    return f's{progenitor_mass.value:3.1f}c_3DBH_dir1'


def get_oconnor_2013_params(progenitor_mass, eos, **kwargs):
    if eos not in ('LS220', 'HShen'):
        raise ValueError(f'Invalid value for model argument `eos`, expected ("LS220", "SFHo") given "{eos}"')

    if progenitor_mass.value not in (12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                                     33, 35, 40, 45, 50, 55, 60, 70, 80, 100, 120):
        raise ValueError('Invalid value for model argument `progenitor_mass`, expected (12..33, 35, 40, 45, 50, 55, 60,'
                         f' 70, 80, 100, 120) Msun, given {progenitor_mass}')
    return int(progenitor_mass.value), eos


def get_oconnor_2015_fname(**kwargs):
    return 'M1_neutrinos.dat'


def get_zha_2021_fname(progenitor_mass, **kwargs):
    if progenitor_mass.value not in (16, 17, 18, 19.89, 19, 20, 21, 22.39, 22, 23, 24, 25, 26, 30, 33):
        raise ValueError('Invalid value for model argument `progenitor_mass`, expected (16, 17, 18, 19.89, 19, 20, 21, '
                         f'22.39, 22, 23, 24, 25, 26, 30, 33) Msun, given {progenitor_mass}')
    if progenitor_mass.value.is_integer():
        fname = f's{int(progenitor_mass.value):2d}.dat'
    else:
        fname = f's{progenitor_mass.value:4.2f}.dat'
    return fname


def get_warren_2020_fname(progenitor_mass, turb_mixing, **kwargs):
    if turb_mixing not in (1.23, 1.25, 1.27):
        raise ValueError('Invalid value for model argument `alpha_lambda`, expected (1.23, 1.25, 1.27) '
                         f'{turb_mixing}')
    if progenitor_mass.value in np.arange(9.25, 13., 0.25) or progenitor_mass.value in np.arange(13., 30.1, 0.1) or \
            progenitor_mass.value == 90.:
        fname = f'stir_a{turb_mixing:3.2f}/stir_multimessenger_a{turb_mixing:3.2f}_m{progenitor_mass.value:.2f}.h5'
    elif progenitor_mass in (31, 32, 33, 34, 35, 40, 45, 50, 55, 60, 70, 80, 100, 120):
        fname = f'stir_a{turb_mixing:3.2f}/stir_multimessenger_a{turb_mixing:3.2f}_m{progenitor_mass.value:d}.h5'
    else:

        raise ValueError(f'Invalid value for model argument `progenitor_mass`, given {progenitor_mass}, expected '
                         '9.25.. 0.25 ..13.0, 13.0.. 0.1 .. 30.0, 31..33, 35.. 5 ..60, 60.. 10 ..90, 80.. 20 ..120')
    return fname


def get_kuroda_2020_fname(rot_vel=0 * (u.rad / u.s), mag_field_exponent=0, **kwargs):
    if rot_vel.value not in (0, 1):
        raise ValueError(f'Invalid value for model argument `rot_vel`, expected (0, 1) rad / s, given {rot_vel}')
    if mag_field_exponent not in (0, 12, 13):
        raise ValueError('Invalid value for model argument `mag_field_exponent, expected (0, 12, 13), '
                         f'given {mag_field_exponent}')
    if (rot_vel.value == 0 and mag_field_exponent in (12, 13)) or (rot_vel.value == 1 and mag_field_exponent == 0):
        raise ValueError('Invalid combination of model arguments, Allowed combinations of model arguments `rot_val` and'
                         ' `mag_field_exponent` are (0 rad/s, 0), (1 rad/s, 12), and (1 rad/s, 13). Given '
                         f'{(rot_vel, mag_field_exponent)}')
    return f'LnuR{int(rot_vel.value):1d}0B{int(mag_field_exponent):02d}.dat'


def get_fornax_2019_fname(progenitor_mass, **kwargs):
    if progenitor_mass.value not in (9, 10, 12, 13, 14, 15, 16, 19, 25, 60):
        ValueError(f'Invalid value for model argument `progenitor_mass`, given {progenitor_mass}, expected '
                   '(9, 10, 12, 13, 14, 15, 16, 19, 25, 60)')
    if progenitor_mass.value == 16:
        return f'lum_spec_{int(progenitor_mass.value):2d}M_r250.h5'
    return f'lum_spec_{int(progenitor_mass.value):2d}M.h5'


lookup_dict = {'Sukhbold_2015': get_sukhbold_2015_fname,
               'Tamborra_2014': get_tamborra_2014_fname,
               'Bollig_2016': get_bollig_2016_fname,
               'Walk_2018': get_walk_2018_fname,
               'Walk_2019': get_walk_2019_fname,
               'OConnor_2013': get_oconnor_2013_params,  # UNIQUE INIT SIGNATURE
               'OConnor_2015': get_oconnor_2015_fname,
               'Zha_2021': get_zha_2021_fname,
               'Warren_2020': get_warren_2020_fname,
               'Kuroda_2020': get_kuroda_2020_fname,
               'Fornax_2019': get_fornax_2019_fname}

# import numpy as np
# from numbers import Number
# from scipy.special import loggamma, gdtr
#
# def _energy_pdf(a, Ea, E):
#     return np.exp((1 + a) * np.log(1 + a) - loggamma(1 + a) +
#                   a * np.log(E) - (1 + a) * np.log(Ea) - (1 + a) * (E / Ea))
#
# def parts_by_index(x, n):
#     """Returns a list of size-n numpy arrays containing indices for the
#     elements of x, and one size-m array ( with m<n ) if there are remaining
#     elements of x.
#
#     Returns
#     -------
#     i_part : list
#        List of index partitions (partitions are numpy array).
#     """
#     nParts = x.size//n
#     i_part = [ np.arange( i*n, (i+1)*n ) for i in range(nParts) ]
#
#     # Generate final partition of size <n if x.size is not multiple of n
#     if len(i_part)*n != x.size:
#         i_part += [ np.arange( len(i_part)*n, x.size ) ]
#
#     # Ensure that last partition always has 2 or more elements
#     if len(i_part[-1]) < 2:
#         i_part[-2] = np.append(i_part[-2], i_part[-1])
#         i_part = i_part[0:-1]
#
#     return i_part
#
#
#
# def energy_pdf(a, Ea, E, *, limit_size=True):
#     # TODO: Figure out how to reconcile this
#     if isinstance(E, np.ndarray):
#         if limit_size and E.size > 1e6:
#             raise ValueError('Input argument size exceeded. Argument`E` is a np.ndarray with size {E.size}, which may '
#                              'lead to large memory consumption while this function executes. To proceed, please reduce '
#                              'the size of `E` or use keyword argument `limit_size=False`')
#     if all(isinstance(var, np.ndarray) for var in (a, Ea)):
#         if a.size == Ea.size:
#             # Vectorized function can lead to unregulated memory usage, better to define it only when needed
#             _vec_energy_pdf = np.vectorize(_energy_pdf, excluded=['E'], signature='(1,n),(1,n)->(m,n)')
#             return _vec_energy_pdf(a=a, Ea=Ea, E=E)
#         else:
#             raise ValueError('Invalid input array size. Arguments `a` and `Ea` must have the same size.  '
#                              f'Given sizes ({a.size}) and ({Ea.size}) respectively.')
#     elif all(isinstance(var, Number) for var in (a, Ea)):
#         return _energy_pdf(a, Ea, E)
#     else:
#         raise ValueError('Invalid argument types, arguments `a` and `Ea` must be numbers or np.ndarray.  '
#                          f'Given types ({type(a)}) and ({type(Ea)}) respectively.')
#
# # def integrate_by_partitions(func, func_args, func_kwargs, axis=1, partition_size=1000):
# #     # Perform core calculation on partitions in E to regulate memory usage in vectorized function
# #     _size = func_args.values()[axis].size
# #     result = np.zeros(_size)
# #     idx = 0
# #     if limit < _size:
# #         idc_split = np.arange(E.size, step=limit)
# #         for idx in idc_split[:-1]:
# #             _E = Enu[idx:idx + limit]
# #             _phot = phot[idx:idx + limit]
# #             result[:, idx:idx + limit] = np.trapz(self.energy_spectrum(t=t, E=_E, flavor=flavor).value * _phot, _E,
# #                                                   axis=0)
# #
# #     _E = Enu[idx:]
# #     _phot = phot[idx:]
# #     result[:, idx:idx + limit] = np.trapz(self.energy_spectrum(t=t, E=_E, flavor=flavor).value * _phot, _E, axis=0)
# #     return result
#
# def energy_cdf(a, Ea, E):
#     return gdtr(1., a + 1., (a + 1.) * (E / Ea))

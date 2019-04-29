from os.path import isfile
from .neutrino import Flavor

import numpy as np
import tables

class _Binning(tables.IsDescription):
    start  = tables.Float64Col()   
    stop   = tables.Float64Col() 
    step   = tables.Float64Col()
    size   = tables.Int64Col()
    
class _Flavors(tables.IsDescription):
    nu_e     = tables.BoolCol(dflt=False)
    nu_e_bar = tables.BoolCol(dflt=False)
    nu_x     = tables.BoolCol(dflt=False)
    nu_x_bar = tables.BoolCol(dflt=False)

class _Interactions(tables.IsDescription):
    InvBetaTab      = tables.BoolCol(dflt=False)
    InvBetaPar      = tables.BoolCol(dflt=False)
    ElectronScatter = tables.BoolCol(dflt=False)
    Oxygen16CC      = tables.BoolCol(dflt=False)
    Oxygen16NC      = tables.BoolCol(dflt=False)
    Oxygen18        = tables.BoolCol(dflt=False)

def initialize(config):
    """Creates hdf5 file for storing processed simulations
    
    .. param :: config : asteria.config.Configuration
        - Loaded ASTERIA model Configuration object.
    """
    h5path = '/'.join([config.abs_base_path, config.IO.table.path])
    try:
        h5file = tables.open_file( filename=h5path, mode='w', title='Simulations of ASTERIA Source: '+ config.source.name)
    
        grp_options = h5file.create_group('/', 'options', 'Requested Simulation Options' )
        
        tab_tbins = h5file.create_table(grp_options, 'Time', _Binning, 'Signature time binning [s]' )
        tab_Ebins = h5file.create_table(grp_options, 'Enu', _Binning, 'Neutrino spectrum energy binning [MeV]' )  
        tab_Flavors = h5file.create_table(grp_options, 'Flavors', _Flavors, 'CCSN model neutrino Flavors' )
        tab_Interactions = h5file.create_table(grp_options, 'Interactions', _Interactions, 'Neutrino Interactions' )

        grp_data = h5file.create_group("/", 'data', 'ASTERIA output')
        for flavor in Flavor:
            vlarray_flavor = h5file.create_vlarray(grp_data, flavor.name,
                                                   tables.Float64Atom(shape=()),
                                                   'Flavor: '+flavor.name )
    except NameError as e:
        h5file.close()
        raise NameError(e)
        
    except ValueError as e:
        h5file.close()
        raise ValueError(e)

    h5file.close()
    
def WriteOption(table, option):
    row = table.row
    for key, val in option.requests.items():
        row[key] = val
    row.append()
    table.flush() 
    
# def FindOption(table, option):
    # statements = []
    # for key,val in Interactions.requests.items():
        # if val:
            # statements.append( '('+ key + ')' )
        # else:
            # statements.append(' ~( ' + key + ')')
            
    # condition = '&'.join( statements )
    # return set( row.nrow for row in table.where(condition) )
    
def WriteBinning(table, binning):
    bins = table.row
    bins['start'] = binning.min()
    bins['stop'] = binning.max()
    bins['size'] = binning.size
    bins['step'] = (binning.max() - binning.min())/(binning.size-1)
    bins.append()
    table.flush()
    
    
def find(group, Interactions, Flavors, Enu, time):
    """ Returns indices of simulations matching the provided options
    
    .. param :: group : tables.Group
        - Group of hdf5 file that stores simulation options
        
    .. param:: Interactions : asteria.interactions.Interactions
        - Enumeration of interactions used to create simulation
        
    .. param :: Flavors : asteria.neutrino.Flavor
        - Enumeration of CCSN Model neutrino types used to create simulation
        
    .. param :: Enu : ndarray
        - numpy array containing energy binning of simulation
        
    .. param :: time : ndarry
        - numpy array containing time binning of simulation.    
    """
    tab_interactions = group.Interactions
    tab_flavors = group.Flavors
    tab_time = group.Time
    tab_Enu  = group.Enu
    
    # Find Simulations that have the requested Interactions 
    statements = []
    for key,val in Interactions.requests.items():
        if val:
            statements.append( '('+ key + ')' )
        else:
            statements.append(' ~( ' + key + ')')
            
    condition = '&'.join( statements )
    pass_interactions = set( row.nrow for row in tab_interactions.where(condition) )

    # Find Simulations that have the requested flavors
    statements = []
    for key,val in Flavors.requests.items():
        if val:
            statements.append( '('+ key + ')' )
        else:
            statements.append(' ~( ' + key + ')')
            
    condition = '&'.join( statements )
    pass_flavors  = set( row.nrow for row in tab_flavors.where(condition) )

    # Find Simulations that have the requested neutrino Energy binning.
    statements = []
    statements.append( '(start == {0}) '.format(Enu.min()) )
    statements.append( ' (stop == {0}) '.format(Enu.max()) )
    statements.append( ' (step == {0}) '.format( (Enu.max() - Enu.min())/(Enu.size-1) ))
    condition = '&'.join( statements )
    pass_Enu = set( row.nrow for row in tab_Enu.where(condition) )

    # Find Simulations that have the requested time binning.
    statements = []
    statements.append( '(start <= {0}) '.format(time.min()) )
    statements.append( ' (stop >= {0}) '.format(time.max()) )
    statements.append( ' (step == {0}) '.format( (time.max() - time.min())/(time.size-1) ))
    condition = '&'.join( statements )
    pass_time = set( row.nrow for row in tab_time.where(condition) )

    pass_all = list(pass_time.intersection( pass_Enu, pass_flavors, pass_interactions))
    
    if not pass_all:
        simIndex = None
    elif len(pass_all) > 1:
        raise ValueError('Multiple matching simulations detected, aborting')
    else:
        simIndex = pass_all[0]
    return simIndex
    
    
def save(config, Interactions, Flavors, Enu, time, result, force=False):
    h5path = '/'.join([config.abs_base_path, config.IO.table.path])
    
    # Test file existence 
    if not isfile(h5path):
        print('Creating file: '+ h5path)
        initialize(config)        
        
    h5file = tables.open_file( filename=h5path, mode='a' )
    grp_options = h5file.root.options
    grp_data = h5file.root.data

    simIndex = find(grp_options, Interactions, Flavors, Enu, time) 
    print('Found ', simIndex )
    if simIndex is None:
        print('Writing new simulation to file')
        # Write simulation options
        WriteOption(grp_options.Interactions, Interactions)
        WriteOption(grp_options.Flavors, Flavors)
        WriteBinning(grp_options.Enu, Enu)
        WriteBinning(grp_options.Time, time)
        # Write requested flavors
        for nu, flavor in enumerate(Flavors):
            vlarray = grp_data[ flavor.name ]
            vlarray.append( result[nu] )
        # Write non-requested flavors
        for key, val in Flavors.requests.items():
            if not val:
                vlarray = grp_data[ key ]
                vlarray.append( np.empty( time.size ) )
    else:
        # If simulation already exists, but no force flag, throw error
        if not force:
            h5file.close()
            raise FileExistsError("""Simulation exists, Aborting. Use argument 'force = True' to force saving.""")
        # If simulation already exists and force flag is on, overwrite the data
        else:
            print('Deleted existing simulation, Rewriting.')                
            for nu, flavor in enumerate(Flavors):
                vlarray = grp_data[ flavor.name ]
                vlarray[simIndex] = result[nu]
            for key, val in Flavors.requests.items():
                if not val:
                    vlarray = grp_data[ key ]
                    vlarray.append( np.empty( time.size ) )     
                
    h5file.close()
    
    
def load(config, Interactions, Flavors, Enu, time):
    h5path = '/'.join([config.abs_base_path, config.IO.table.path])
    
    # Test file existence 
    if not isfile(h5path):
        raise FileNotFoundError('File {0} not found.'.format(h5path) )
        
    h5file = tables.open_file( filename=h5path, mode='r' )
    grp_options = h5file.root.options
    
    simIndex = find(grp_options, Interactions, Flavors, Enu, time)
    
    if simIndex is None:
        # If no matching simulations have been found, return none, or throw error?
        print('No matching Simulation found')
        h5file.close()
        return None
    
    else:
        t_min = grp_options.Time.read(simIndex)['start']
        dt = grp_options.Time.read(simIndex)['step']    
        t_max = grp_options.Time.read(simIndex)['stop']
        
        saved_time = np.arange(t_min, t_max + dt, dt)
        time_slice = (saved_time >= time.min()) & (saved_time <= time.max())
        
        result = np.zeros(shape=( len(Flavors), len(time))  )
        
        grp_data = h5file.root.data
        for nu, flavor in enumerate(Flavors):
            tab_data = h5file.get_node(grp_data, flavor.name)
            data = tab_data.read(simIndex)[0][time_slice]
            result[nu] = data
                
    h5file.close()
    return result

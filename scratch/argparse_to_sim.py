import argparse
import configparser
import logging

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='CCSN simulator',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--distance', dest='distance', type=float, default=10.,
                        help='Distance to progenitor [kpc]')
    parser.add_argument('-i', '--interaction', dest='interaction', type=str, default='all', 
                        choices=['default', 'all'],
                        help='Specify interaction type')
    parser.add_argument('-f', '--flavors', dest='flavors', type=str, default='all',
                        choices=['all', 'nu_e', 'nu_e_bar', 'nu_x', 'nu_x_bar'],
                        help='Flavors to simulate')
    parser.add_argument('-hrc', '--hierarchy', dest='hierarchy', type=str, default='default',
                        choices=['normal', 'inverted', 'None'],
                        help='Neutrino hierarchy')
    parser.add_argument('-c', '--config_file', dest='config_file', type=str,
                        help='Provide config file name')
    parser.add_argument('-m', '--model', dest='model', type=dict, 
                        default={'name': 'Nakazato_2013', 'progenitor_mass': 13, 
                                 'revival_time': 300, 'metallicity': 0.004, 'eos': 'shen'}, 
                        help='Dict of model parameters: {\'name\': , \'progenitor_mass\': , \'revival_time\': , \'metallicity\': , \'eos\': }')
    parser.add_argument('-s', '--scheme', dest='scheme', type=str, default='adiabatic-msw',
                        choices=['all', 'adiabatic-msw'],
                        help='Mixing scheme')
    parser.add_argument('-a', '--angle', dest='angle', type=float, default=33.2,
                        help='Provide mixing angle [deg]')
    parser.add_argument('-emin', dest='Emin', type=float, default=0.,
                        help='Minimum energy [MeV]')
    parser.add_argument('-emax', dest='Emax', type=float, default=100.,
                        help='Maximum energy [MeV]')
    parser.add_argument('-de', dest='dE', type=float, default=0.1,
                        help='Energy step [MeV]')
    parser.add_argument('-tmin', dest='tmin', type=float, default=-1.,
                        help='Minimum time [s]')
    parser.add_argument('-tmax', dest='tmax', type=float, default=1.,
                        help='Maximum time [s]')
    parser.add_argument('-dt', dest='dt', type=float, default=1.,
                        help='Time step [ms]')
    args = vars(parser.parse_args())
    print(f"Arguments from terminal:\n {args})")
    result = {}
    if args['config_file'] is None:
    # Example: python argparse_to_sim.py -d 10 -i all -f all
        config = configparser.ConfigParser()
        with open('config_to_sim.ini') as f:
            config.read_file(f)
            default = config['DEFAULT']
            model = dict(config['MODEL'])
            mixing = config['MIXING']
            energy = config['ENERGY']
            time = config['TIME']
            result.update(default)
            result.update(model)
            result.update(mixing)
            result.update(energy)
            result.update(time)
    else:
    # Example: python argparse_to_sim.py -c ./config_to_sim.ini
        config_file = args['config_file']
        logging.info(f'Reading configuration from {config_file}')
        
        config = configparser.ConfigParser()
        config.read(args['config_file'])
        default = config['DEFAULT']
        model = dict(config['MODEL'])
        mixing = config['MIXING']
        energy = config['ENERGY']
        time = config['TIME']
        result.update(default)
        result.update(model)
        result.update(mixing)
        result.update(energy)
        result.update(time)
        
    result.update({k: v for k, v in args.items() if v is not None})  # Update if v is not None
    print(f"Combined arguments from terminal + config:\n {result})")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
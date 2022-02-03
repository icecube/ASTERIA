import argparse
import configparser
import logging

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', dest='distance') # fixed
    parser.add_argument('-i', dest='interaction') # all
    parser.add_argument('-f', dest='flavors') # All
    parser.add_argument('-hrc', dest='hierarchy')
    parser.add_argument('-c', dest='config_file')
    args = vars(parser.parse_args())
    print(f"Arguments from terminal:\n {args})")
    if args['config_file'] is None:
    # Example: python test_arg_parse.py -d fixed -i all -f All
        config = configparser.ConfigParser()
        config.read('new_config.ini')
        conf = config['DEFAULT']
    else:
        #python test_arg_parse.py -c ./test_config.ini
        config_file = args['config_file']
        logging.info(f'Reading configuration from {config_file}')
        
        config = configparser.ConfigParser()
        config.read(args['config_file'])
        conf = config['DEFAULT']
        
    result = dict(conf)
    result.update({k: v for k, v in args.items() if v is not None})  # Update if v is not None
    print(f"Combined arguments from terminal + config:\n {result})")

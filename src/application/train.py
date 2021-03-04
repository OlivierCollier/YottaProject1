"""
    This class is one of the two entrypoints with predict.py.
    This class is used to train a ML model with some training date.
"""
import argparse
import ipdb


# Parse arguments
parser = argparse.ArgumentParser(description='Files containing the training datasets')
parser.add_argument('--client-data',
                '-c', 
                help='Provide the client dataset file',
                dest='client_file',
                required=True)
parser.add_argument('--eco-data',
                '-e',
                help='Provide the economic information dataset file',
                dest='eco_file',
                required=True)

args = parser.parse_args()




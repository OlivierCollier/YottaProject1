"""
    This class is one of the two entrypoints with predict.py.
    This class is used to train a ML model with some training date.
"""
import argparse
import os

import ipdb
import src.config.base as base
from src.infrastructure.build_dataset import DataBuilderFactory

# Parse arguments
parser = argparse.ArgumentParser(description='Files containing the training datasets', \
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--client-data',
                '-c', 
                help='Provide the client dataset file',
                default=os.path.join(base.RAW_DATA_DIR, base.CLIENT_DATA_FILE_NAME),
                dest='client_file')
parser.add_argument('--eco-data',
                '-e',
                help='Provide the economic information dataset file',
                default=os.path.join(base.RAW_DATA_DIR, base.ECO_DATA_FILE_NAME),
                dest='eco_file')

args = parser.parse_args()


# Build datasets
client_data = DataBuilderFactory(args.client_file, base.config_client_data, 'production', \
                                base.CLIENT_COLUMNS_TO_DROP, base.ALL_CLIENT_DATA_TRANSLATION)
eco_data = DataBuilderFactory(args.eco_file, base.config_eco_data, 'production', \
                            base.ECO_COLUMNS_TO_DROP)


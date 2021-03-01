import os

from src.config.config import read_yaml
from src.config.column_names import *

# Directories
THIS_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
RAW_DATA_DIR = os.path.join(REPO_DIR, 'data/raw/')
PROCESSED_DATA_DIR = os.path.join(REPO_DIR, 'data/processed/')
#PREDICTION_DATA_DIR = os.path.join(REPO_DIR, 'data/prediction/')
OUTPUTS_DIR = os.path.join(REPO_DIR, 'outputs')
LOGS_DIR = os.path.join(REPO_DIR, 'logs')
MODELS_DIR = os.path.join(REPO_DIR, 'models')
NOTEBOOKS_DIR = os.path.join(REPO_DIR, 'notebooks')
CONFIG_DIR = os.path.join(REPO_DIR, 'src/config')


# Config file
CONFIG_FILE_PATH = os.path.join(CONFIG_DIR, 'config.yml')

config_data = read_yaml(CONFIG_FILE_PATH)
DATA_PATH = os.path.join(RAW_DATA_DIR,config_data['subscription']['name'])
config_client_data = config_data.get('subscription')
DATA_DATE_FORMAT = config_data['subscription']['date_format']
DATA_SEP = config_data['subscription']['sep']
ECO_DATA_PATH = os.path.join(RAW_DATA_DIR,config_data['economic_info']['name'])
config_eco_data = config_data.get('economic_info')
ECO_DATA_DATE_FORMAT = config_data['economic_info']['date_format']
ECO_DATA_SEP = config_data['economic_info']['sep']

PROCESSED_DATA_NAME = 'processed_data.csv'
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR,PROCESSED_DATA_NAME)


CLIENT_COLUMNS_TO_DROP = [
    LAST_CONTACT_DURATION,
    COMMUNICATION_TYPE
]
ECO_COLUMNS_TO_DROP = [

]
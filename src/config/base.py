import os

from src.config.config import read_yaml
import src.config.column_names as col

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

# Datasets files
CLIENT_DATA_FILE_NAME = 'data.csv'
ECO_DATA_FILE_NAME = 'socio_eco.csv'


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


# Columns to drop from datasets in EDA mode
CLIENT_COLUMNS_TO_DROP = [
    col.LAST_CONTACT_DURATION,
    col.COMMUNICATION_TYPE
]
ECO_COLUMNS_TO_DROP = [
    col.END_MONTH
]


# Translation between french and english for column values
JOB_TYPE_TRANSLATION = {
    'Technicien': 'Technician',
    'Entrepreuneur': 'Entrepreneur',
    'Col bleu': 'Blue-collar worker',
    'Retraité': 'Retired',
    'Indépendant': 'Freelance',
    'Chomeur': 'Unemployed',
    'Employé de ménage': 'House keeper',
    'Etudiant': 'Student'
}
EDUCATION_TRANSLATION = {
    'Tertiaire': 'Graduate studies',
    'Secondaire': 'Secondary education',
    'Primaire': 'Primary education'
}
MARITAL_STATUS_TRANSLATION = {
    'Marié': 'Married',
    'Célibataire': 'Single',
    'Divorcé': 'Divorced'
}
RESULT_LAST_CAMPAIGN_TRANSLATION = {
    'Echec': 'Fail',
    'Autre': 'Other',
    'Succes': 'Success'
}
ALL_CLIENT_DATA_TRANSLATION = {
    col.JOB_TYPE: JOB_TYPE_TRANSLATION,
    col.EDUCATION: EDUCATION_TRANSLATION,
    col.MARITAL_STATUS: MARITAL_STATUS_TRANSLATION,
    col.RESULT_LAST_CAMPAIGN: RESULT_LAST_CAMPAIGN_TRANSLATION
}


# Initialize random seed
SEED=21


# Path to model saved
SAVED_MODEL_PATH = os.path.join(MODELS_DIR, 'ml_model.pkl')


# Models parameters grid
LOGISTIC_REGRESSION_PARAM = {
    'log_reg_clf__penalty': ['l1', 'l2'],
    'log_reg_clf__C': [0.01, 0.1, 1.0],
    'log_reg_clf__class_weight': ['balanced']

}
import os

from src.config.config import read_yaml
import src.config.column_names as col

# Directories
THIS_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
RAW_DATA_DIR = os.path.join(REPO_DIR, 'data/raw/')
PROCESSED_DATA_DIR = os.path.join(REPO_DIR, 'data/processed/')
PREDICTION_DATA_DIR = os.path.join(REPO_DIR, 'data/predictions/')
TRAIN_DATA_DIR = os.path.join(RAW_DATA_DIR, 'train/')
PREDICT_DATA_DIR = os.path.join(RAW_DATA_DIR, 'predict/')
OUTPUTS_DIR = os.path.join(REPO_DIR, 'outputs')
LOGS_DIR = os.path.join(REPO_DIR, 'logs')
MODELS_DIR = os.path.join(REPO_DIR, 'models')
NOTEBOOKS_DIR = os.path.join(REPO_DIR, 'notebooks')
CONFIG_DIR = os.path.join(REPO_DIR, 'src/config')

# Datasets files
CLIENT_DATA_FILE_NAME = 'data.csv'
ECO_DATA_FILE_NAME = 'socio_eco.csv'

# Train datasets full paths
TRAIN_CLIENT_DATA_PATH = os.path.join(TRAIN_DATA_DIR, CLIENT_DATA_FILE_NAME)
TRAIN_ECO_DATA_PATH = os.path.join(TRAIN_DATA_DIR, ECO_DATA_FILE_NAME)

# Predict datasets full paths
PREDICT_CLIENT_DATA_PATH = os.path.join(PREDICT_DATA_DIR, CLIENT_DATA_FILE_NAME)
PREDICT_ECO_DATA_PATH = os.path.join(PREDICT_DATA_DIR, ECO_DATA_FILE_NAME)

# Config file
CONFIG_FILE_PATH = os.path.join(CONFIG_DIR, 'config.yml')

config_data = read_yaml(CONFIG_FILE_PATH)
DATA_PATH = os.path.join(TRAIN_DATA_DIR,config_data['subscription']['name'])
config_client_data = config_data.get('subscription')
DATA_DATE_FORMAT = config_data['subscription']['date_format']
DATA_SEP = config_data['subscription']['sep']
ECO_DATA_PATH = os.path.join(TRAIN_DATA_DIR,config_data['economic_info']['name'])
config_eco_data = config_data.get('economic_info')
ECO_DATA_DATE_FORMAT = config_data['economic_info']['date_format']
ECO_DATA_SEP = config_data['economic_info']['sep']

PROCESSED_DATA_NAME = 'processed_data.csv'
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR,PROCESSED_DATA_NAME)

PREDICTIONS_FILE_PATH = os.path.join(PREDICTION_DATA_DIR, 'predictions.csv')

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
    'log_reg_clf__penalty': ['l2', 'l1'],
    'log_reg_clf__C': [.0001, .001, .01, .01],
    'log_reg_clf__class_weight': ['balanced']

}
RF_PARAM = {
    'rf_clf__n_estimators': [220, 240, 260, 280, 300],
    'rf_clf__max_depth': [6, 7, 8, 9, 10],
    'rf_clf__min_samples_leaf': [9, 10, 11, 12]
}


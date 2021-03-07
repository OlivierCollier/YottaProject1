
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from IPython.core.display import display
from scipy.stats import kstest, chi2_contingency
from statsmodels.regression.linear_model import OLS

def settings():
    warnings.filterwarnings('ignore')

def load_initial_data():
    data = pd.read_csv('../../data/processed/processed_data.csv')
    return data

def illustrate_target(data):
    sns.countplot(data.SUBSCRIPTION)
    plt.xlabel('HAS_SUBSCRIBED')
    plt.show()

def make_has_subscribed_binary(data):
    data['SUBSCRIPTION'] = data['SUBSCRIPTION'].eq('Yes').astype(int)

def correlation_target_balance(data, upper_clip=None, lower_clip=None):
    data_copy = data.copy()
    data_copy['BALANCE'] = data_copy['BALANCE'].clip(upper=upper_clip, lower=lower_clip)
    pearson = data_copy[['BALANCE', 'SUBSCRIPTION']].corr('pearson')['SUBSCRIPTION'][0]
    spearman = data_copy[['BALANCE', 'SUBSCRIPTION']].corr('spearman')['SUBSCRIPTION'][0]
    table = pd.DataFrame([pearson, spearman], columns=[''], index=['Corrélation de Pearson', 'Rho de Spearman'])
    display(table)

def clip_balance(data, upper_clip=None, lower_clip=None):
    data['BALANCE'] = data['BALANCE'].clip(upper=upper_clip, lower=lower_clip)

def plot_balance_knowing_target(data):
    sns.catplot(y='BALANCE', x='SUBSCRIPTION', kind='violin', data=data)
    plt.xlabel('HAS_SUBSCRIBED')
    plt.ylabel('clipped ACCOUNT_BALANCE')
    plt.show()

def test_independence_between_balance_and_target(data):
    test = kstest(data.loc[data['SUBSCRIPTION'] == 0, 'BALANCE'], data.loc[data['SUBSCRIPTION'] == 1, 'BALANCE'])
    print(f"La p-valeur du test de Kolmogorov-Smirnov est de {test.pvalue} : ACCOUNT_BALANCE n'a donc "
          f"pas la même distribution selon que la cible vaut 0 ou 1.")

def target_by_job_type(data):
    table = data[['JOB_TYPE', 'SUBSCRIPTION']].groupby('JOB_TYPE').agg(['sum', 'count', 'mean'])['SUBSCRIPTION']
    table.columns = ['Number of subscriptions', 'Number of clients', 'Proportion of subscription']
    table.sort_values(by='Proportion of subscription', inplace=True)
    display(table)

def test_independence_between_target_and_job_type(data):
    crosstab = pd.crosstab(data['SUBSCRIPTION'], data['JOB_TYPE'])
    test = chi2_contingency(crosstab)
    print(f"La p-valeur du test du chi2 d'indépendance est de {test[1]} : JOB_TYPE n'est donc"
          f" pas indépendant de la cible.")

def remove_missing_values_for_job_type(data):
    data = data[~data['JOB_TYPE'].isnull()]

def illustrate_status(data):
    sns.countplot(data.STATUS)
    plt.xlabel('MARITAL_STATUS')
    plt.show()

def target_by_marital_status(data):
    table = data[['STATUS', 'SUBSCRIPTION']].groupby('STATUS').agg(['sum', 'count', 'mean'])['SUBSCRIPTION']
    table.columns = ['Number of subscriptions', 'Number of clients', 'Proportion of subscription']
    table.sort_values(by='Proportion of subscription', inplace=True)
    display(table)

def test_independence_between_target_and_marital_status(data):
    crosstab = pd.crosstab(data['SUBSCRIPTION'], data['STATUS'])
    test = chi2_contingency(crosstab)
    print(f"La p-valeur du test du chi2 d'indépendance est de {test[1]} : MARITAL_STATUS n'est donc"
          f" pas indépendant de la cible.")

def marital_status_by_job_type(data):
    data_copy = data.copy()
    data_copy['MARITAL_STATUS'] = data_copy['STATUS']
    table = pd.crosstab(data_copy['JOB_TYPE'], data_copy['MARITAL_STATUS'])
    table['Subscription rate'] = data[['JOB_TYPE', 'SUBSCRIPTION']].groupby('JOB_TYPE').agg('mean')
    display(table)

def show_predictions_subscription_rate(data):
    data_copy = data.copy()
    data_copy['MARITAL_STATUS'] = data_copy['STATUS']
    table = pd.crosstab(data_copy['JOB_TYPE'], data_copy['MARITAL_STATUS'])
    table['Proportion Single'] = table['Single'] / (table['Single'] + table['Married'] + table['Divorced'])
    table['Proportion Married'] = table['Married'] / (table['Single'] + table['Married'] + table['Divorced'])
    table['Subscription rate'] = data_copy[['JOB_TYPE', 'SUBSCRIPTION']].groupby('JOB_TYPE').agg('mean')
    table = table.drop(columns=['Single', 'Married', 'Divorced'])
    y = table['Subscription rate']
    X = table.drop(columns='Subscription rate')
    ols = OLS(y, X).fit()
    table['Predictions'] = ols.fittedvalues
    display(table)

def show_results_linear_regression_subscription_rate(data):
    data_copy = data.copy()
    data_copy['MARITAL_STATUS'] = data_copy['STATUS']
    table = pd.crosstab(data_copy['JOB_TYPE'], data_copy['MARITAL_STATUS'])
    table['Proportion Single'] = table['Single'] / (table['Single'] + table['Married'] + table['Divorced'])
    table['Proportion Married'] = table['Married'] / (table['Single'] + table['Married'] + table['Divorced'])
    table['Subscription rate'] = data_copy[['JOB_TYPE', 'SUBSCRIPTION']].groupby('JOB_TYPE').agg('mean')
    table = table.drop(columns=['Single', 'Married', 'Divorced'])
    y = table['Subscription rate']
    X = table.drop(columns='Subscription rate')
    ols = OLS(y, X).fit()
    print(ols.summary2())

def bonferroni_outlier_test_retired(data):
    data_copy = data.copy()
    data_copy['MARITAL_STATUS'] = data_copy['STATUS']
    table = pd.crosstab(data_copy['JOB_TYPE'], data_copy['MARITAL_STATUS'])
    table['Proportion Single'] = table['Single'] / (table['Single'] + table['Married'] + table['Divorced'])
    table['Proportion Married'] = table['Married'] / (table['Single'] + table['Married'] + table['Divorced'])
    table['Subscription rate'] = data_copy[['JOB_TYPE', 'SUBSCRIPTION']].groupby('JOB_TYPE').agg('mean')
    table = table.drop(columns=['Single', 'Married', 'Divorced'])
    y = table['Subscription rate']
    X = table.drop(columns='Subscription rate')
    ols = OLS(y, X).fit()
    print(f"Bonferroni outlier test p-value: {ols.outlier_test()['bonf(p)']['Retired']}")

def show_results_linear_regression_subscription_rate_without_retired(data):
    data_copy = data.copy()
    data_copy['MARITAL_STATUS'] = data_copy['STATUS']
    data_copy = data_copy[~data_copy['JOB_TYPE'].eq('Retired')]
    table = pd.crosstab(data_copy['JOB_TYPE'], data_copy['MARITAL_STATUS'])
    table['Proportion Single'] = table['Single'] / (table['Single'] + table['Married'] + table['Divorced'])
    table['Proportion Married'] = table['Married'] / (table['Single'] + table['Married'] + table['Divorced'])
    table['Subscription rate'] = data_copy[['JOB_TYPE', 'SUBSCRIPTION']].groupby('JOB_TYPE').agg('mean')
    table = table.drop(columns=['Single', 'Married', 'Divorced'])
    y = table['Subscription rate']
    X = table.drop(columns='Subscription rate')
    ols = OLS(y, X).fit()
    print(ols.summary2())

def show_predictions_subscription_rate_without_retired(data):
    data_copy = data.copy()
    data_copy = data_copy[~data_copy['JOB_TYPE'].eq('Retired')]
    data_copy['MARITAL_STATUS'] = data_copy['STATUS']
    table = pd.crosstab(data_copy['JOB_TYPE'], data_copy['MARITAL_STATUS'])
    table['Proportion Single'] = table['Single'] / (table['Single'] + table['Married'] + table['Divorced'])
    table['Proportion Married'] = table['Married'] / (table['Single'] + table['Married'] + table['Divorced'])
    table['Subscription rate'] = data_copy[['JOB_TYPE', 'SUBSCRIPTION']].groupby('JOB_TYPE').agg('mean')
    table = table.drop(columns=['Single', 'Married', 'Divorced'])
    y = table['Subscription rate']
    X = table.drop(columns='Subscription rate')
    ols = OLS(y, X).fit()
    table['Predictions'] = ols.fittedvalues
    display(table)

def show_mean_age_by_job_type(data):
    table = data.loc[~data.AGE.eq(123), ['AGE', 'JOB_TYPE']].groupby('JOB_TYPE').agg('mean')
    display(table)

def replace_age123_by_mean_in_job_type(data):
    replacements = data.loc[~data.AGE.eq(123), ['AGE', 'JOB_TYPE']].groupby('JOB_TYPE').agg('mean')
    to_replace = data[data.AGE.eq(123)]



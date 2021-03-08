
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from IPython.core.display import display
from scipy.stats import kstest, chi2_contingency
from statsmodels.regression.linear_model import OLS
import datetime as dt

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
    replacements = data.loc[~data.AGE.eq(123), ['AGE', 'JOB_TYPE']].groupby('JOB_TYPE').agg('mean').to_dict()
    relevant_observations = data.AGE.eq(123)
    corresponding_job_types = data.loc[relevant_observations, 'JOB_TYPE']
    data.loc[relevant_observations, 'AGE'] = \
        corresponding_job_types.replace(replacements['AGE'])

def plot_age_histogram(data):
    sns.displot(data,
                x='AGE',
                hue='SUBSCRIPTION',
                stat='density',
                common_norm=False,
                element='step',
                palette='bright')
    plt.show()

def plot_date_by_year(data):
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    data['year-month'] = data['DATE_x'].apply(lambda x: x[:7])
    data['date'] = data['DATE_x'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))
    data['year'] = data['date'].apply(lambda x: x.year)
    data['MONTH'] = data['date'].apply(lambda x: months[x.month-1])
    data['WEEKDAY'] = data['date'].apply(lambda x: weekdays[x.weekday()])
    sns.countplot(data.year)
    plt.xlabel('YEAR')
    plt.show()

def plot_date_year_and_month(data):
    sns.countplot(data['year-month'])
    plt.xlabel('YEAR-MONTH')
    plt.xticks([0, 7, 13, 19, 25], ('2008-05', '2009-01', '2009-07', '2010-01', '2010-07'))
    plt.show()

def show_table_subscription_by_weekday(data):
    table = data[['WEEKDAY', 'SUBSCRIPTION']].groupby('WEEKDAY').agg('mean')['SUBSCRIPTION'].sort_values()
    table = pd.DataFrame(table)
    table.columns = ['Subscription rate']
    display(table)

def test_independence_between_target_and_weekday(data):
    crosstab = pd.crosstab(data['SUBSCRIPTION'], data['year-month'])
    test = chi2_contingency(crosstab)
    print(f"La p-valeur du test du chi2 d'indépendance est de {test[1]} : le jour de la semaine n'est donc"
          f" pas indépendant de la cible.")

def show_table_subscription_by_month(data):
    table = data[['MONTH', 'SUBSCRIPTION']].groupby('MONTH').agg('mean')['SUBSCRIPTION'].sort_values()
    table = pd.DataFrame(table)
    table.columns = ['Subscription rate']
    display(table)

def plot_education(data):
    sns.countplot(data['EDUCATION'])
    plt.show()

def show_table_subscription_by_education(data):
    table = data[['EDUCATION', 'SUBSCRIPTION']].groupby('EDUCATION').agg('mean')['SUBSCRIPTION'].sort_values()
    table = pd.DataFrame(table)
    table.columns = ['Subscription rate']
    display(table)

def plot_has_housing_and_perso_loan(data):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.countplot(ax=ax[0], x='HAS_HOUSING_LOAN', data=data)
    sns.countplot(ax=ax[1], x='HAS_PERSO_LOAN', data=data)
    plt.show()

def show_correlation_loan_variables_with_target(data):
    data['HAS_HOUSING_LOAN'] = data['HAS_HOUSING_LOAN'].eq('Yes').astype(int)
    data['HAS_PERSO_LOAN'] = data['HAS_PERSO_LOAN'].eq('Yes').astype(int)
    data['HAS_LOAN'] = data['HAS_HOUSING_LOAN'] + data['HAS_PERSO_LOAN'] \
                               - data['HAS_HOUSING_LOAN'] * data['HAS_PERSO_LOAN']
    correlations = data[['HAS_PERSO_LOAN', 'HAS_HOUSING_LOAN', 'HAS_LOAN', 'SUBSCRIPTION']].corr()['SUBSCRIPTION']
    correlations = pd.DataFrame(correlations)
    correlations.drop(labels=['SUBSCRIPTION'], axis=0, inplace=True)
    correlations.columns = ['Correlation with target']
    display(correlations)

def plot_has_default(data):
    sns.countplot(data['HAS_DEFAULT'])
    plt.show()

def compute_correlation_has_default_target(data):
    data['HAS_DEFAULT'] = data['HAS_DEFAULT'].eq('Yes').astype(int)
    correlation = data[['HAS_DEFAULT', 'SUBSCRIPTION']].corr()['SUBSCRIPTION'][0]
    print(f'La corrélation avec la cible est de {correlation}.')

def missing_values_percentage_of_result_by_month(data):
    table = data[['RESULT_LAST_CAMPAIGN', 'year-month']]\
        .groupby('year-month')\
        .agg(lambda x: 100 * x.isnull().sum() / len(x))
    plt.plot(table)
    plt.ylabel('Percentage')
    plt.title('Percentage of missing values by month')
    plt.xticks([0, 7, 13, 19, 25], ('2008-05', '2009-01', '2009-07', '2010-01', '2010-07'))
    plt.show()

def fill_missing_values_result_last_campaign(data):
    data.loc[(data['NB_DAY_LAST_CONTACT'].ne(-1))&(data['RESULT_LAST_CAMPAIGN'].isnull()), 'RESULT_LAST_CAMPAIGN'] = "Other"
    data.loc[data['RESULT_LAST_CAMPAIGN'].isnull(), 'RESULT_LAST_CAMPAIGN'] = "First contact"

def show_table_subscription_by_result_last_campaign(data):
    table = data[['RESULT_LAST_CAMPAIGN', 'SUBSCRIPTION']].groupby('RESULT_LAST_CAMPAIGN').agg('mean')
    table = pd.DataFrame(table)
    table.columns = ['Subscription rate']
    display(table)

def plot_nb_days_last_contact(data):
    data['NB_DAYS_LAST_CONTACT'] = data['NB_DAY_LAST_CONTACT']
    sns.histplot(data.loc[data.NB_DAYS_LAST_CONTACT.ne(-1), 'NB_DAYS_LAST_CONTACT'], kde=True)
    plt.title('Distribution of NB_DAYS_LAST_CONTACT for nonnegative values')
    plt.show()

def compute_binned_nb_days_last_contact(data):
    data_to_bin = data.loc[data['NB_DAY_LAST_CONTACT'].ne(-1), 'NB_DAY_LAST_CONTACT']
    groups = pd.qcut(data_to_bin, 4)
    data_to_bin = data_to_bin.groupby(groups).transform('mean')
    data['BINNED_NB_DAYS_LAST_CONTACT'] = data['NB_DAY_LAST_CONTACT']
    data.loc[data['NB_DAY_LAST_CONTACT'].ne(-1), 'BINNED_NB_DAYS_LAST_CONTACT'] = data_to_bin
    data.loc[data['NB_DAY_LAST_CONTACT'].eq(-1), 'BINNED_NB_DAYS_LAST_CONTACT'] = data_to_bin.max()

def compute_correlation_binned_nb_days_last_contact_target(data):
    correlation = data[['SUBSCRIPTION', 'BINNED_NB_DAYS_LAST_CONTACT']].corr()['SUBSCRIPTION'][1]
    print(f'La corrélation entre le nouveau feature et la cible est de {correlation}.')

def histogram_nb_contacts(data):
    plt.hist(data['NB_CONTACT'])
    plt.ylabel('Count')
    plt.xlabel('NB_CONTACTS')
    plt.title('Distribution of NB_CONTACTS')
    plt.show()

def clip_nb_contacts(data):
    data['CLIPPED_NB_CONTACT'] = np.clip(data['NB_CONTACT'], a_min=0, a_max=15)

def histogram_clipped_nb_contacts(data):
    plt.hist(data['CLIPPED_NB_CONTACT'])
    plt.ylabel('Count')
    plt.xlabel('clipped NB_CONTACTS')
    plt.title('Distribution of clipped NB_CONTACTS')
    plt.show()

def clip_nb_contacts_last_campaign(data):
    data['CLIPPED_NB_CONTACT_LAST_CAMPAIGN'] = np.clip(data['NB_CONTACT_LAST_CAMPAIGN'], a_min=0, a_max=15)

def histogram_clipped_nb_contacts_last_campaign(data):
    plt.hist(data['CLIPPED_NB_CONTACT_LAST_CAMPAIGN'])
    plt.ylabel('Count')
    plt.xlabel('clipped NB_CONTACTS_LAST_CAMPAIGN')
    plt.title('Distribution of clipped NB_CONTACTS_LAST_CAMPAIGN')
    plt.show()
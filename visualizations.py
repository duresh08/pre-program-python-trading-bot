import pandas as pd
import numpy as np
from tqdm import tqdm
from pandas.tseries.offsets import DateOffset
import lightgbm as lgb
from hyperopt import hp, STATUS_OK, fmin, Trials, tpe

pd.set_option('display.max_columns', None)


def pull_data_from_db(start_date, end_date, db_path):
    print("pulling db data from start_date: {}, end_date: {}".format(start_date, end_date))
    db_path = "sqlite:///{}".format(db_path)
    df = pd.read_sql("SELECT * FROM signals WHERE date >= '{}' AND date <= '{}'".format(start_date, end_date), db_path)
    return df


def data_descroption_pipeline(df):
    print("columns are: {}".format(df.columns))
    print("\n\n")
    print("dtypes: {}".format(df.info()))
    print("\n\n")
    print("symbol_list: {}".format(list(df['symbol'].unique())))
    print("number of unique symbols: {}".format(len(df['symbol'].unique())))
    print(df.describe())


def pre_process_data(df):
    cols_to_drop = ['id', 'reported_currency']
    df = df.drop(columns=cols_to_drop)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df['date'] = pd.to_datetime(df['date'], format='mixed')
    return df


def finding_3_mo_returns(df):
    df['target_date_initial'] = df['date'] + pd.DateOffset(months=3)
    result_list = []
    for symbol in tqdm(sorted(df['symbol'].unique())):
        df_symbol = df[df['symbol'] == symbol].sort_values(by='date').reset_index(drop=True)
        df_symbol['target_date'] = pd.merge_asof(
            df_symbol[['target_date_initial']].sort_values(by='target_date_initial'),
            df_symbol[['date']].sort_values(by='date'),
            left_on='target_date_initial',
            right_on='date',
            direction='forward'
        )['date']
        result_list.append(df_symbol)
    df_final = pd.concat(result_list).sort_values(['date']).reset_index(drop=True)
    target_dates = sorted(df_final['target_date'].unique())
    future_df = df_final[df_final['date'].isin(target_dates)].loc[:, ['date', 'symbol', 'adj_close']].rename(
        columns={'date': 'target_date',
                 'adj_close': 'adj_close_3_mo'})
    df_final = pd.merge(df_final, future_df, on=['target_date', 'symbol'], how='left').reset_index(drop=True)
    df_final['returns_3_mo'] = ((df_final['adj_close_3_mo'] - df_final['adj_close']) / df_final['adj_close'])
    df_final = df_final.drop(columns=['target_date_initial', 'target_date', 'adj_close_3_mo'])
    df_final = df_final[~pd.isna(df_final['returns_3_mo'])].reset_index(drop=True)
    return df_final


def lightgbm_categorical_training_pipeline(X_train, y_train, train_weights, X_cv=None, y_cv=None, cv_weights=None,
                                           space=None, early_stopping_rounds=50, verbose=10, categorical_features=None):
    if categorical_features is None:
        categorical_features = []
    else:
        pass
    params = {
        'max_depth': int(space['max_depth']),
        'num_leaves': int(space['num_leaves']),
        'subsample': space['subsample'],
        'colsample_bytree': space['colsample_bytree'],
        'min_child_weight': int(space['min_child_weight']),
        'reg_lambda': int(space['reg_lambda']),
        'reg_alpha': int(space['reg_alpha']),
        'learning_rate': space['learning_rate'],
        'metric': space['metric'],
        'random_state': space['random_state'],
        'objective': space['objective'],
        'verbosity': space['verbosity'],
        'max_bin': 1000
    }
    if X_cv is not None:
        train_data = lgb.Dataset(X_train, label=y_train, weight=train_weights.squeeze(),
                                 categorical_feature=categorical_features)
        test_data = lgb.Dataset(X_cv, label=y_cv, weight=cv_weights.squeeze(),
                                categorical_feature=categorical_features,
                                reference=train_data)

        model = lgb.train(params, train_data, num_boost_round=int(space['num_boost_round']),
                          valid_sets=[train_data, test_data],
                          callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                                     lgb.log_evaluation(period=verbose)],
                          categorical_feature=categorical_features)
    else:
        train_data = lgb.Dataset(X_train, label=y_train, weight=train_weights.squeeze(),
                                 categorical_feature=categorical_features)

        model = lgb.train(params, train_data, num_boost_round=int(space['num_boost_round']),
                          categorical_feature=categorical_features)
    return model


def lightgbm_model_training(train_df, hyperparameters, cv_df=None, categorical_features=None, early_stopping_rounds=50,
                            verbose=10):
    params = {
        'max_depth': int(hyperparameters['max_depth']),
        'num_leaves': int(hyperparameters['num_leaves']),
        'subsample': hyperparameters['subsample'],
        'colsample_bytree': hyperparameters['colsample_bytree'],
        'min_child_weight': int(hyperparameters['min_child_weight']),
        'reg_lambda': int(hyperparameters['reg_lambda']),
        'reg_alpha': int(hyperparameters['reg_alpha']),
        'learning_rate': hyperparameters['learning_rate'],
        'max_bin': 1000,
        'metric': 'rmse',
        'random_state': 0,
        'objective': 'regression',
        'verbosity': verbose,
        'early_stopping_rounds': early_stopping_rounds,
        'device': 'gpu'
    }
    if categorical_features is not None:
        train_df.loc[:, categorical_features] = train_df.loc[:, categorical_features].astype('category')
        categorical_indices = [train_df.columns.get_loc(col) for col in categorical_features if col in train_df.columns]
    else:
        categorical_indices = []
    X_train = train_df.loc[:, list(train_df.drop(columns=['returns_3_mo']).columns)]
    y_train = train_df.loc[:, ['returns_3_mo']]

    if cv_df is not None:
        if categorical_features is not None:
            cv_df.loc[:, categorical_features] = cv_df.loc[:, categorical_features].astype('category')
            categorical_indices = [cv_df.columns.get_loc(col) for col in categorical_features if
                                   col in cv_df.columns]
        else:
            categorical_indices = []
        X_cv = cv_df.loc[:, list(cv_df.drop(columns=['returns_3_mo']).columns)]
        y_cv = cv_df.loc[:, ['returns_3_mo']]
    else:
        X_cv = None
        y_cv = None

    if cv_df is not None:
        train_data = lgb.Dataset(X_train, label=y_train,
                                 categorical_feature=categorical_indices)
        test_data = lgb.Dataset(X_cv, label=y_cv,
                                categorical_feature=categorical_indices,
                                reference=train_data)
        model = lgb.train(params, train_data, num_boost_round=int(hyperparameters['num_boost_round']),
                          valid_sets=[train_data, test_data],
                          callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                                     lgb.log_evaluation(period=verbose)])
    else:
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_indices)
        model = lgb.train(params, train_data, num_boost_round=int(hyperparameters['num_boost_round']))
    return model


def k_fold_cv(df, n_folds=3, train_n_months=15, cv_n_months=3):
    unique_dates = pd.to_datetime(df['date']).sort_values().unique()
    for fold in range(n_folds):
        val_end_date = unique_dates[-1] - pd.DateOffset(months=fold * cv_n_months)
        val_start_date = unique_dates[-1] - pd.DateOffset(months=(fold + 1) * cv_n_months)

        # Get the start of the training window
        train_start_date = val_start_date - pd.DateOffset(months=train_n_months)

        # Filter train and validation sets
        train_set = df[(df['date'] > train_start_date) & (df['date'] <= val_start_date)]
        val_set = df[(df['date'] > val_start_date) & (df['date'] <= val_end_date)]

        print("\n\n")
        print("train_set end_date: {}, start_date: {}, num_dates: {}".format(train_set['date'].astype(str).max(),
                                                                             train_set['date'].astype(str).min(),
                                                                             len(train_set['date'].unique())))
        print("val_set end_date: {}, start_date: {}, num_dates: {}".format(val_set['date'].astype(str).max(),
                                                                           val_set['date'].astype(str).min(),
                                                                           len(val_set['date'].unique())))


if __name__ == '__main__':
    lgb_hyperparameter_space = {
        'max_depth': 13,
        'num_leaves': 1024,
        'subsample': 0.609,
        'colsample_bytree': 0.632,
        'min_child_weight': 4,
        'reg_lambda': 101,
        'reg_alpha': 88,
        'learning_rate': 0.013,
        'num_boost_round': 325,
        'metric': 'rmse',
        'random_state': 0,
        'objective': 'regression',
        'verbosity': 3,
        'early_stopping_rounds': 50,
        'device': 'gpu'
    }
    hyperparameter_space = {
        'max_depth': hp.quniform('max_depth', 3, 15 + 1, 1),
        'num_leaves': hp.choice('num_leaves', [2 ** i for i in range(1, 12)]),
        'subsample': hp.uniform('subsample', 0.5, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
        'min_child_weight': hp.quniform('min_child_weight', 0, 10 + 1, 1),
        'reg_lambda': hp.quniform('reg_lambda', 0, 100 + 1, 1),
        'reg_alpha': hp.quniform('reg_alpha', 0, 50 + 1, 1),
        'learning_rate': hp.uniform('learning_rate', 0.005, 0.3),
        'num_boost_round': hp.quniform('num_boost_round', 100, 5000, 50),
    }
    db_path = "D:\\pre-program-python-trading-bot\\data.db\\data.db"
    df = pull_data_from_db("2000-01-01", "2002-01-01", db_path)
    df = pre_process_data(df)
    df = finding_3_mo_returns(df)
    k_fold_cv(df, n_folds=2, train_n_months=15, cv_n_months=3)

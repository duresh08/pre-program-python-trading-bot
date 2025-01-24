import os
import datetime as dt
import time
import matplotlib.pyplot as plt
import warnings
from functools import wraps
import gc
import lightgbm as lgb
import numpy as np
import pandas as pd
from hyperopt import hp, STATUS_OK, fmin, Trials, tpe
from tqdm import tqdm
import math

warnings.filterwarnings('ignore')
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
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
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
    }
    eval_results = {}
    if X_cv is not None:
        train_data = lgb.Dataset(X_train, label=y_train, weight=train_weights.squeeze(),
                                 categorical_feature=categorical_features)
        test_data = lgb.Dataset(X_cv, label=y_cv, weight=cv_weights.squeeze(),
                                categorical_feature=categorical_features,
                                reference=train_data)

        model = lgb.train(params, train_data, num_boost_round=int(space['num_boost_round']),
                          valid_sets=[train_data, test_data],
                          callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                                     lgb.log_evaluation(period=verbose),
                                     lgb.record_evaluation(eval_results)],
                          categorical_feature=categorical_features)
        plot_lgb_performance(eval_results)
    else:
        train_data = lgb.Dataset(X_train, label=y_train, weight=train_weights.squeeze(),
                                 categorical_feature=categorical_features)

        model = lgb.train(params, train_data, num_boost_round=int(space['num_boost_round']),
                          categorical_feature=categorical_features)
    return model


def plot_lgb_performance(eval_results):
    plt.figure(figsize=(10, 5))
    for dataset, metrics in eval_results.items():
        for metric_name, values in metrics.items():
            plt.plot(values, label=f'{dataset} {metric_name}')
    plt.xlabel('Iteration')
    plt.ylabel('Metric Value')
    plt.title('Train vs. CV Performance')
    plt.legend()
    plt.grid()
    plt.show()


def lightgbm_model_training(train_df, hyperparameters, cv_df=None, categorical_features=None, early_stopping_rounds=50,
                            verbose=10):
    params = {
        'max_depth': int(hyperparameters['max_depth']),
        'num_leaves': 2 ** (int(hyperparameters['max_depth']) - 1),
        'subsample': hyperparameters['subsample'],
        'colsample_bytree': hyperparameters['colsample_bytree'],
        'reg_lambda': int(hyperparameters['reg_lambda']),
        'reg_alpha': int(hyperparameters['reg_alpha']),
        'learning_rate': hyperparameters['learning_rate'],
        'metric': 'rmse',
        'random_state': 0,
        'objective': 'regression',
        'verbosity': verbose,
        'early_stopping_rounds': early_stopping_rounds,
        'device': 'cpu'
    }
    X_train, categorical_indices, y_train = categorify_and_return(categorical_features, train_df)

    if cv_df is not None:
        X_cv, categorical_indices, y_cv = categorify_and_return(categorical_features, cv_df)
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
        del params['early_stopping_rounds']
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_indices)
        model = lgb.train(params, train_data, num_boost_round=int(hyperparameters['num_boost_round']))
    return model


def categorify_and_return(categorical_features, train_df):
    if categorical_features is not None:
        for col in categorical_features:
            train_df[col] = train_df[col].astype('category')
        categorical_indices = [train_df.columns.get_loc(col) for col in categorical_features if col in train_df.columns]
    else:
        categorical_indices = []
    X_train = train_df.loc[:, list(train_df.drop(columns=['returns_3_mo', 'date']).columns)]
    y_train = train_df.loc[:, ['returns_3_mo']]
    return X_train, categorical_indices, y_train


def evaluate_model(model, val_set, categorical_features=None, top_n_stock_picks=100):
    X_test, categorical_indices, y_test = categorify_and_return(categorical_features, val_set)
    y_pred = model.predict(X_test).reshape(X_test.shape[0], 1)
    val_set['y_pred'] = y_pred
    loss = performance_evaluation(val_set, y_pred_col_name='y_pred', y_actual_col_name='returns_3_mo')
    return loss


def performance_evaluation(df, y_pred_col_name='y_pred', y_actual_col_name='returns_3_mo'):
    df['y_actual_rank'] = df.groupby('date')[y_actual_col_name].rank(ascending=False, method='dense')
    df['y_pred_rank'] = df.groupby('date')[y_pred_col_name].rank(ascending=False, method='dense')
    df = df.sort_values(by=['date', 'y_actual_rank'], ascending=[True, True]).reset_index(drop=True)
    df['loss_metric'] = abs(df['y_pred_rank'] - df['y_actual_rank'])
    df.loc[((df[y_actual_col_name] > 0) &
            (df[y_pred_col_name] < 0)), 'loss_metric'] = (df.loc[((df[y_actual_col_name] > 0) &
                                                                  (df[y_pred_col_name] < 0)), 'loss_metric']) * 2
    df.loc[((df[y_actual_col_name] < 0) &
            (df[y_pred_col_name] > 0)), 'loss_metric'] = (df.loc[((df[y_actual_col_name] < 0) &
                                                                  (df[y_pred_col_name] > 0)), 'loss_metric']) * 4
    return df['loss_metric'].mean()


def saving_to_csv(output_path, file_name):
    def saving_to_csv_decorator(func):
        @wraps(func)
        def saving_to_csv_wrapper(*args, **kwargs):
            df = func(*args, **kwargs)
            if not isinstance(df, pd.DataFrame):
                raise ValueError("return object is not a dataframe, cannot save")
            os.makedirs(output_path, exist_ok=True)
            if os.path.exists(output_path + "\\{}.csv".format(file_name)):
                df = pd.concat([df, pd.read_csv(output_path + "\\{}.csv".format(file_name))]).reset_index(drop=True)
            df.to_csv(output_path + "\\{}.csv".format(file_name), index=False,
                      mode='w', header=True)
            try:
                loss = df['cv_loss'].mean()
            except:
                loss = math.nan
            return {'status': STATUS_OK, 'loss': loss}

        return saving_to_csv_wrapper

    return saving_to_csv_decorator


def time_taken_to_run_function():
    def time_taken_to_run_function_decorator(func):
        @wraps(func)
        def time_taken_to_run_function_wrapper(*args, **kwargs):
            start_time = time.time()
            func(*args, **kwargs)
            print(f"\n\nfunction: {func.__name__} took: {round(time.time() - start_time, 2)} seconds to run")

        return time_taken_to_run_function_wrapper

    return time_taken_to_run_function_decorator


@saving_to_csv(os.getcwd(), 'results')
def k_fold_cv(df, hyperparameters, n_folds=3, train_n_months=15, cv_n_months=3, categorical_features=None,
              hyperopt_return=False):
    unique_dates = pd.to_datetime(df['date']).sort_values().unique()
    performance_df = pd.DataFrame()
    for fold in range(n_folds):
        val_end_date = unique_dates[-1] - pd.DateOffset(months=fold * cv_n_months)
        val_start_date = unique_dates[-1] - pd.DateOffset(months=(fold + 1) * cv_n_months)
        train_start_date = val_start_date - pd.DateOffset(months=train_n_months)
        train_set = df[(df['date'] > train_start_date) & (df['date'] <= val_start_date)]
        val_set = df[(df['date'] > val_start_date) & (df['date'] <= val_end_date)]
        print("\n\n")
        print("train_set end_date: {}, start_date: {}, num_dates: {}".format(train_set['date'].astype(str).max(),
                                                                             train_set['date'].astype(str).min(),
                                                                             len(train_set['date'].unique())))
        print("val_set end_date: {}, start_date: {}, num_dates: {}".format(val_set['date'].astype(str).max(),
                                                                           val_set['date'].astype(str).min(),
                                                                           len(val_set['date'].unique())))
        model = lightgbm_model_training(train_set, hyperparameters, cv_df=val_set,
                                        categorical_features=categorical_features)

        train_loss = evaluate_model(model, train_set, categorical_features=categorical_features)
        cv_loss = evaluate_model(model, val_set, categorical_features=categorical_features)
        print("\n\n")
        print("fold_no: {}, train_loss: {}, cv_loss: {}".format(fold, train_loss, cv_loss))

        feature_importance_df = pd.DataFrame({
            'feature': model.feature_name(),
            'importance': model.feature_importance(importance_type='split')
        }).set_index('feature').T.reset_index(drop=True)
        feature_importance_df = (feature_importance_df / feature_importance_df.sum(axis=1)[0])

        params_df = pd.DataFrame.from_dict(hyperparameters, orient='index').T

        params_df['train_start_date'] = train_set['date'].astype(str).min()
        params_df['train_end_date'] = train_set['date'].astype(str).max()
        params_df['cv_start_date'] = val_set['date'].astype(str).min()
        params_df['cv_end_date'] = val_set['date'].astype(str).max()

        params_df['fold_no'] = fold + 1
        params_df['train_loss'] = train_loss
        params_df['cv_loss'] = cv_loss
        params_df['model'] = 'lightgbm'
        if hyperopt_return:
            params_df['mode'] = 'hyperopt'
        params_df = pd.concat([params_df, feature_importance_df], axis=1).reset_index(drop=True)
        performance_df = pd.concat([performance_df, params_df]).reset_index(drop=True)

        gc.enable()
        gc.collect()
    return performance_df


def hyperparameter_optimization_pipeline(df, optimization_space, max_evals=100, categorical_features=None):
    trials = Trials()
    best_hyper_params = fmin(
        fn=lambda space: k_fold_cv(df, space, n_folds=3, train_n_months=15, cv_n_months=3,
                                   categorical_features=categorical_features,
                                   hyperopt_return=True),
        space=optimization_space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials
    )
    return best_hyper_params


def calculate_pnl():
    return


def walk_forward_simulation(hyperparameters, train_n_months=15, categorical_features=None, start_date="2006-04-01",
                            final_end_date="2024-12-31", delta_months=3, top_n_picks=100,
                            stock_allocation=10000):
    master_pnl_df = pd.DataFrame()
    while tqdm(start_date < final_end_date, 'parsing through date windows one at a time'):
        training_start_date = (pd.to_datetime(start_date, format='%Y-%m-%d') -
                               pd.DateOffset(months=train_n_months + delta_months)).strftime("%Y-%m-%d")
        training_end_date = (dt.datetime.strptime(start_date, "%Y-%m-%d") +
                             dt.timedelta(days=10)).strftime('%Y-%m-%d')

        db_path = "D:\\pre-program-python-trading-bot\\data.db\\data.db"

        df = pull_data_from_db(training_start_date, training_end_date, db_path)
        df = pre_process_data(df)
        df = finding_3_mo_returns(df)
        df = df[df['date'].astype(str) < (pd.to_datetime(start_date, format='%Y-%m-%d') -
                                          pd.DateOffset(months=delta_months)).strftime("%Y-%m-%d")].reset_index(
            drop=True)

        model = lightgbm_model_training(df, hyperparameters, cv_df=None, categorical_features=categorical_features)

        X_test, categorical_indices, y_test = categorify_and_return(categorical_features, df)
        y_pred = model.predict(X_test).reshape(X_test.shape[0], 1)
        df['y_pred'] = y_pred
        df = df[df['date'] == df['date'].max()].reset_index(drop=True)
        df = df.sort_values(['y_pred'], ascending=False).head(top_n_picks).reset_index(drop=True)
        pnl = (((1 + df['returns_3_mo']) * stock_allocation).sum() -
               top_n_picks * stock_allocation)
        master_pnl_df = pd.concat([master_pnl_df, pd.DataFrame({'date': [start_date],
                                                                'pnl': [pnl]})]).reset_index(drop = True)

        start_date = (pd.to_datetime(start_date, format='%Y-%m-%d') +
                      pd.DateOffset(months=delta_months)).strftime("%Y-%m-%d")
        master_pnl_df.to_csv(os.getcwd() + '\\pnl_walk_1.csv', index = False)

    return


if __name__ == '__main__':
    lgb_hyperparameters = {
        'max_depth': 12,
        'num_leaves': 2 ** (12 - 1),
        'subsample': 0.790005627,
        'colsample_bytree': 0.875694597,
        'reg_lambda': 58,
        'reg_alpha': 9,
        'learning_rate': 0.226591168,
        'num_boost_round': 4100,
        'metric': 'rmse',
        'random_state': 0,
        'objective': 'regression',
        'device': 'gpu'
    }
    hyperparameter_space = {
        'max_depth': hp.quniform('max_depth', 3, 15 + 1, 1),
        'subsample': hp.uniform('subsample', 0.5, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
        'reg_lambda': hp.quniform('reg_lambda', 0, 100 + 1, 1),
        'reg_alpha': hp.quniform('reg_alpha', 0, 50 + 1, 1),
        'learning_rate': hp.uniform('learning_rate', 0.005, 0.3),
        'num_boost_round': hp.quniform('num_boost_round', 100, 5000, 50),
    }
    categorical_features = ['symbol']
    # db_path = "C:\\Users\\dhruv.suresh\\Downloads\\data.db\\data.db"
    # df = pull_data_from_db("2000-01-01", "2005-12-31", db_path)
    # df = pre_process_data(df)
    # df = finding_3_mo_returns(df)
    # ------------------------------------------------------------------------------
    # k_fold_cv(df, lgb_hyperparameters, n_folds=3, train_n_months=15, cv_n_months=3,
    #           categorical_features=categorical_features)
    # ------------------------------------------------------------------------------
    # hyperparameter_optimization_pipeline(df, hyperparameter_space, max_evals=100,
    #                                      categorical_features=categorical_features)
    # ------------------------------------------------------------------------------
    walk_forward_simulation(lgb_hyperparameters, train_n_months=15, categorical_features=categorical_features,
                            start_date="2006-04-01", final_end_date="2024-12-31", delta_months=3)

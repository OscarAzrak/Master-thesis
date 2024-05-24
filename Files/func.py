import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.metrics import MeanSquaredError, Accuracy
from keras.callbacks import EarlyStopping
from scikeras.wrappers  import KerasClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import balanced_accuracy_score, make_scorer
import numpy as np
import xgboost as xgb
import re
import pickle
import lightgbm as lgb
from scipy.stats import skew, kurtosis

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def load_and_preprocess_data(filename, date_col, start_date, seperator, fill, lim):
    df_read = pd.read_csv(filename, sep=seperator)
    
    # Convert the date column to datetime and set it as the index
    df_read[date_col] = pd.to_datetime(df_read[date_col])
    df_read.set_index(date_col, inplace=True)
    
    # Data cleaning: replace commas with periods and convert to float
    for column in df_read.columns:
        if df_read[column].dtype == 'object':
            df_read[column] = df_read[column].str.replace(',', '.').astype(float)
    
    # Start dataset from start date
    df_filtered = df_read[start_date:]

    # Fill missing values
    df_filtered.fillna(fill, limit=lim)
    
    return df_filtered

def add_features(df, windows):

    feature_cols = [col for col in df.columns if 'macro' not in col.lower()]
    macro_cols = [col for col in df.columns if 'MACRO' in col.upper()]
    
    # Initialize a dictionary to hold all the new feature data
    features_dict = {}


    # Perform rolling calculations for each window size
    for w in windows:
        for col in feature_cols:
            # Create unique feature names for each statistic and window size
            features_dict[f'{col}_VaR_{w}']         = df[col].rolling(window=w, min_periods=int(w//2)).quantile(0.05)
            features_dict[f'{col}_momentum_{w}']    = df[col].rolling(window=w, min_periods=int(w//2)).sum()
            features_dict[f'{col}_avgreturn_{w}']   = df[col].rolling(window=w, min_periods=int(w//2)).mean()
            features_dict[f'{col}_skew_{w}']        = df[col].rolling(window=w, min_periods=int(w//2)).skew()
            features_dict[f'{col}_volatility_{w}']  = df[col].rolling(window=w, min_periods=int(w//2)).std()

    # Convert the dictionary of Series to a DataFrame
    features_df = pd.DataFrame(features_dict, index=df.index)

    # Concatenate 'MACRO' columns to the features DataFrame
    if macro_cols:
        macro_df = df[macro_cols]
        features_df = pd.concat([features_df, macro_df], axis=1)

    return features_df

def add_target(df, windows):

    feature_cols = [col for col in df.columns if 'macro' not in col.lower()]
    
    # Initialize a dictionary to hold all the new feature data
    features_dict = {}


    # Perform rolling calculations for each window size
    for w in windows:
        for col in feature_cols:
            # Create unique feature names for each statistic and window size
            features_dict[f'{col}_avgreturn_{w}']   = df[col].rolling(window=w, min_periods=int(w//2)).mean()
            features_dict[f'{col}_volatility_{w}']  = df[col].rolling(window=w, min_periods=int(w//2)).std()
            #räkna sharpe direkt här
            features_dict[f'{col}_sharpe_ratio_{w}'] = features_dict[f'{col}_avgreturn_{w}']/features_dict[f'{col}_volatility_{w}']

    # Convert the dictionary of Series to a DataFrame
    features_df = pd.DataFrame(features_dict, index=df.index)
    
    return features_df

def transform_and_pivot_df(df, date_col):
    # Reset the index to make the date a regular column
    df_reset = df.reset_index()
    
    # Melt the DataFrame to long format
    long_df = df_reset.melt(id_vars=date_col, var_name='metric', value_name='value')
    
    # Split the 'metric' column to extract components
    split_metrics = long_df['metric'].str.split('_', expand=True)
    
    # Identify 'MACRO' rows
    macro_mask = split_metrics[0] == 'MACRO'
    
    # For non-'MACRO' metrics, define 'asset' and 'metric_type'
    long_df['asset'] = split_metrics[0] + '_' + split_metrics[1]
    long_df['metric_type'] = split_metrics[2] + '_' + split_metrics[3]

    # Reset index on the left-hand side DataFrame slice to ensure alignment
    lhs = long_df.loc[macro_mask, 'metric_type'].reset_index(drop=True)

    # Reset index on the right-hand side Series to ensure alignment
    rhs = (split_metrics.loc[macro_mask, 0] + '_' + split_metrics.loc[macro_mask, 1]).reset_index(drop=True)

    # Assign the values after ensuring both sides have the same length
    lhs = rhs

    # Assign the modified Series back to the original DataFrame (if needed)
    long_df.loc[macro_mask, 'metric_type'] = lhs.values
    
    # For 'MACRO' metrics, adjust 'metric_type' and 'asset'
    long_df.loc[macro_mask, 'metric_type'] = split_metrics.loc[macro_mask, 0] + '_' + split_metrics.loc[macro_mask, 1]
    long_df.loc[macro_mask, 'asset'] = 'MACRO'
    
    # Remove 'MACRO' placeholder rows
    long_df = long_df[long_df['asset'] != 'MACRO']
    
    # Pivot the DataFrame back to wide format
    final_df = long_df.pivot_table(index=[date_col, 'asset'], columns='metric_type', values='value').reset_index()
    
    # Handle 'MACRO' metrics separately
    macro_df = df.filter(regex='^MACRO').copy()
    macro_df[date_col] = df_reset[date_col]
    
    # Merge 'MACRO' metrics back into the final DataFrame
    if date_col in macro_df.columns:
        macro_df = macro_df.drop(columns=date_col)
    
    macro_df = macro_df.reset_index()
    final_df = pd.merge(final_df, macro_df, on=date_col, how='left')
    
    return final_df

def add_y_col(df, df_read, date_col, target_days, return_col, volatility_col, cross):
    df_target = add_target(df_read, [target_days])
    df_combined = pd.concat([df, df_target], axis=1)
    df = transform_and_pivot_df(df_combined, date_col)
    
    return_col = return_col + '_' + str(target_days)
    volatility_col = volatility_col + '_' + str(target_days)

    if cross == True:
        # cross time series
        sharpe_ratio_mean = df.groupby(date_col)['sharpe_ratio'].mean().rename('sharpe_ratio_mean')
        df = df.merge(sharpe_ratio_mean, on=date_col)

        df['sharpe_ratio'] = df['sharpe_ratio'].shift(-target_days)
        df['sharpe_ratio_mean'] = df['sharpe_ratio_mean'].shift(-target_days)
        df = df.dropna()

        df['Y'] = np.where(df['sharpe_ratio'] > df['sharpe_ratio_mean'], 1, 0)
        df = df.drop(columns=['sharpe_ratio', 'sharpe_ratio_mean', return_col, volatility_col])

    elif cross == False:
        # time series
        df['sharpe_ratio'] = df['sharpe_ratio'].shift(-target_days)

        df = df.dropna()

        df['Y'] = np.where(df['sharpe_ratio'] > 0, 1, 0)
        df = df.drop(columns=['sharpe_ratio', return_col, volatility_col])


    
    return df


def prepare_training_dataset(df, date_col, shuffle=False, train_split=0.25, eval_split=0.25):

    # Separate features and target variable
    X = df.drop(columns=['Y', 'asset'])

    y = df['Y']

    # Convert date column to datetime if not already done
    X[date_col] = pd.to_datetime(X[date_col])

    if shuffle:
        # Split the data randomly
        train_size = 1 - train_split
        X_temp, X_train, y_temp, y_train = train_test_split(X, y, train_size=train_size)
        X_eval, X_test, y_eval, y_test = train_test_split(X_temp, y_temp, test_size=(2/3))
    else:
        # Split the data sequentially
        train_end_idx = int(len(X) * train_split)
        eval_end_idx = train_end_idx + int(len(X) * eval_split)

        X_train = X.iloc[:train_end_idx]
        y_train = y.iloc[:train_end_idx]
        X_eval = X.iloc[train_end_idx:eval_end_idx]
        y_eval = y.iloc[train_end_idx:eval_end_idx]
        X_test = X.iloc[eval_end_idx:]
        y_test = y.iloc[eval_end_idx:]

    # Drop the date column
    X_train = X_train.drop(date_col, axis=1)
    X_eval = X_eval.drop(date_col, axis=1)
    X_test = X_test.drop(date_col, axis=1)

    # Combine training and evaluation sets for the final model training
    X_train_eval = pd.concat([X_train, X_eval])
    y_train_eval = pd.concat([y_train, y_eval])


    return X_train, X_eval, X_test, y_train, y_eval, y_test, X_train_eval, y_train_eval



def optimize_and_train_ridge(X_train, y_train, X_eval, y_eval, param_grid, cross, cv=5):

    model = RidgeClassifier()
    if cross:
        scoring = 'accuracy'
    else:
        scoring = make_scorer(balanced_accuracy_score)
    # Initialize GridSearchCV with the provided model and parameter grid
    grid_search = GridSearchCV(model, param_grid, scoring=scoring, cv=cv)
    
    # Fit GridSearchCV on the training set
    grid_search.fit(X_train, y_train)
    
    # Print the best parameters and the accuracy on the evaluation set
    print("Best parameters:", grid_search.best_params_)
    print("Best accuracy on evaluation set:", grid_search.best_score_)
    #best_params = {'alpha': 10.0}
    # Retrain the model with the best parameters on the combined training and evaluation sets
    model_best = model.__class__(**grid_search.best_params_)
    #model_best = model.__class__(**best_params)
    model_best.fit(pd.concat([X_train, X_eval]), pd.concat([y_train, y_eval]))
    
    # ändra från klassificierig till sannolikhet


    return model_best, grid_search.best_params_

def sigmoid(x):
    return 1 / (1 + np.exp(-x))



def evaluate_model_performance(y_true, y_pred):
    



    conf_matrix = confusion_matrix(y_true, y_pred)


    precision = precision_score(y_true, y_pred)*100
    recall = recall_score(y_true, y_pred)*100
    f1 = f1_score(y_true, y_pred)*100


    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    # Print the performance metrics
    print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"MSE: {mse*100}")
    print(f"RMSE: {rmse*100}")

    return conf_matrix, precision, recall, f1, mse, rmse



def optimize_and_train_xgb(X_train, y_train, X_eval, y_eval, param_grid, cross, cv=5, n_jobs=-1, early_stopping_rounds=10):
    if cross:
        scoring = 'accuracy'
    else:
        scoring = make_scorer(balanced_accuracy_score)
    # Initialize the XGBoost model
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    # Perform grid search
    grid_search = GridSearchCV(xgb_model, param_grid, scoring=scoring, cv=cv, n_jobs=n_jobs)
    grid_search.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], early_stopping_rounds=early_stopping_rounds, verbose=False)
    # Extract best hyperparameters
    best_params = grid_search.best_params_

    ##best_params = {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50}
    
    print("Best hyperparameters:", best_params)

    # Retrain the model with the best parameters on the combined training and evaluation set
    xgb_best = xgb.XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss')
    xgb_best.fit(pd.concat([X_train, X_eval]), pd.concat([y_train, y_eval]))

    return xgb_best, best_params



def optimize_and_train_lgb(X_train, y_train, X_eval, y_eval, param_grid, cross, cv=5, n_jobs=-1):

    # Initialize the LightGBM model
    lgb_model = lgb.LGBMClassifier()
    if cross:
        scoring = 'accuracy'
    else:
        scoring = make_scorer(balanced_accuracy_score)
    # Perform grid search
    grid_search = GridSearchCV(lgb_model, param_grid, scoring=scoring, cv=cv, n_jobs=n_jobs)
    grid_search.fit(X_train, y_train)

    # Extract best hyperparameters
    best_params = grid_search.best_params_
    ##best_params =  {'learning_rate': 0.01, 'max_depth': 10, 'n_estimators': 200, 'num_leaves': 31}

    print("Best hyperparameters:", best_params)

    # Retrain the model with the best parameters on the combined training and evaluation set
    lgb_best = lgb.LGBMClassifier(**best_params)
    lgb_best.fit(pd.concat([X_train, X_eval]), pd.concat([y_train, y_eval]), eval_set=[(X_eval, y_eval)])

    return lgb_best, best_params

def create_model(input_dim, optimizer='adam', init='glorot_uniform'):
    model = Sequential()
    model.add(Dense(64, activation='relu', kernel_initializer=init, input_shape=(input_dim,)))
    model.add(Dense(32, activation='relu', kernel_initializer=init))
    model.add(Dense(16, activation='relu', kernel_initializer=init))
    model.add(Dense(8, activation='relu', kernel_initializer=init))
    model.add(Dense(1, activation='sigmoid', kernel_initializer=init))  # Output layer for binary classification

    # Compile the model
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
    return model

def optimize_and_train_NN(X_train, y_train, X_eval, y_eval, X_test, param_grid, cross, cv=5, n_jobs=-1):
    # Initialize the scaler and scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_eval_scaled = scaler.transform(X_eval)
    X_test_scaled = scaler.transform(X_test)

    # Get the input dimension for the model
    input_dim = X_train_scaled.shape[1]

    # Wrap the Keras model for use with scikit-learn
    model = KerasClassifier(build_fn=lambda: create_model(input_dim=input_dim), verbose=0)

    if cross:
        scoring = 'accuracy'
    else:
        scoring = make_scorer(balanced_accuracy_score)

    # Perform grid search
    grid_search = GridSearchCV(
        estimator=model, 
        param_grid=param_grid,
        scoring=scoring, 
        cv=cv, 
        n_jobs=n_jobs
    )
    grid_search.fit(X_train_scaled, y_train)

    # Extract best hyperparameters
    best_params = grid_search.best_params_

    print("Best hyperparameters:", best_params)

    # Retrain the model with the best parameters
    best_model = create_model(input_dim=input_dim, optimizer=best_params['optimizer'], init='glorot_uniform')
    best_model.fit(
        X_train_scaled, y_train,
        epochs=best_params['epochs'],
        batch_size=best_params['batch_size'],
        validation_data=(X_eval_scaled, y_eval),
        callbacks=[
            EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, verbose=1, restore_best_weights=True)
        ]
    )

    return best_model, best_params, X_test_scaled




def predict_and_analyze_ext(model, X_test, df, name, df_read, date_col, cross, top_percentile=90, bottom_percentile=10):
    
    if name == 'benchmark':
                
        df_ranks = add_features(df_read, [63, 126, 252])

        #only keep columns with "momentum"
        X_ranked = df_ranks.filter(regex='momentum')

        X_ranked_trans = transform_and_pivot_df(X_ranked, date_col)
        columns_to_rank = X_ranked_trans.filter(regex='momentum').columns

        for col in columns_to_rank:
            X_ranked_trans[col + '_rank'] = X_ranked_trans[col].rank(method='average')
        
        rank_columns = [col + '_rank' for col in columns_to_rank]
        X_ranked_trans[name] = X_ranked_trans[rank_columns].mean(axis=1)
        if cross:
            ranked_top_10 = X_ranked_trans.groupby('todate')[name].apply(lambda x: np.percentile(x, top_percentile))
            ranked_bottom_10 = X_ranked_trans.groupby('todate')[name].apply(lambda x: np.percentile(x, bottom_percentile))
        
            ranked_top_10_df = ranked_top_10.reset_index()
            ranked_top_10_df.columns = ['todate', 'top_threshold']
            ranked_bottom_10_df = ranked_bottom_10.reset_index()
            ranked_bottom_10_df.columns = ['todate', 'bottom_threshold']
            ranked_merged = X_ranked_trans.merge(ranked_top_10_df, on='todate').merge(ranked_bottom_10_df, on='todate')
            top_assets = ranked_merged[ranked_merged[name] >= ranked_merged['top_threshold']]
            bottom_assets = ranked_merged[ranked_merged[name] <= ranked_merged['bottom_threshold']]
        else:
            ranked_top_10 = X_ranked_trans.groupby('asset')[name].apply(lambda x: np.percentile(x, top_percentile))
            ranked_bottom_10 = X_ranked_trans.groupby('asset')[name].apply(lambda x: np.percentile(x, bottom_percentile))
            ranked_top_10_df = ranked_top_10.reset_index()
            ranked_top_10_df.columns = ['asset', 'top_threshold']
            ranked_bottom_10_df = ranked_bottom_10.reset_index()
            ranked_bottom_10_df.columns = ['asset', 'bottom_threshold']
            ranked_merged = X_ranked_trans.merge(ranked_top_10_df, on='asset').merge(ranked_bottom_10_df, on='asset')
            top_assets = ranked_merged[ranked_merged[name] >= ranked_merged['top_threshold']]
            bottom_assets = ranked_merged[ranked_merged[name] <= ranked_merged['bottom_threshold']]
 
        return top_assets, bottom_assets
    
    
    X_predict = X_test.copy()

    # Prediction handling based on model type
    if name == 'ridge':
        y_scores = model.decision_function(X_predict)
        y_pred_prob = sigmoid(y_scores)  
    elif name == 'NN':
        y_pred_prob = model.predict(X_predict).flatten()  
    else:
        y_pred_prob = model.predict_proba(X_predict)[:, 1]

    # Store predictions in X_predict for further analysis
    y_pred_prob = pd.Series(y_pred_prob, index=X_test.index)

    X_predict[name] = y_pred_prob  

    # Merge predictions with the additional data
    a = X_predict.index
    b = df.index.intersection(a)
    c = df.loc[b, ['asset', 'todate']]
    d = X_predict[[name]].join(c)

    # Calculate top and bottom percentiles
    e_top_10 = d.groupby('todate')[name].apply(lambda x: np.percentile(x, top_percentile))
    e_bottom_10 = d.groupby('todate')[name].apply(lambda x: np.percentile(x, bottom_percentile))

    # Convert to DataFrame and merge
    e_top_10_df = e_top_10.reset_index()
    e_top_10_df.columns = ['todate', 'top_threshold']
    e_bottom_10_df = e_bottom_10.reset_index()
    e_bottom_10_df.columns = ['todate', 'bottom_threshold']

    d_merged = d.merge(e_top_10_df, on='todate').merge(e_bottom_10_df, on='todate')
    
    # Select top and bottom assets
    top_assets = d_merged[d_merged[name] >= d_merged['top_threshold']]
    bottom_assets = d_merged[d_merged[name] <= d_merged['bottom_threshold']]

    return top_assets, bottom_assets


def get_indices_by_date(df, date, date_column=None):
    df[date_column] = pd.to_datetime(df[date_column])
    # Filter and return the DataFrame
    return df[df[date_column] == pd.to_datetime(date)]



def calculate_trade_volume(df):
    # Calculate the absolute difference between consecutive days
    position_changes = df.diff().abs().fillna(0)

    # Any non-zero value in position_changes indicates a trade
    trades = position_changes != 0

    # Calculate the sum of trades for each day
    trade_volume_per_day = trades.sum(axis=1)

    return trade_volume_per_day




def financial_metrics(daily_returns, weights, transaction_cost_rate=0.01):
    # Handle edge cases
    if daily_returns.empty:
        return "Input series is empty"

    daily_returns = pd.to_numeric(daily_returns, errors='coerce')
    daily_returns.dropna(inplace=True)  # Drop any entries that couldn't be converted

    # Find the first non-zero index in a more robust way
    non_zero_start = daily_returns[daily_returns != 0].index.min()
    if non_zero_start is None:
        return "No non-zero entries found in the series"
    
    daily_returns = daily_returns.loc[non_zero_start:]  # Start from the first non-zero

    # Calculate metrics
    yearly_returns = daily_returns.mean() * 252
    yearly_std_dev = daily_returns.std() * np.sqrt(252)
    sharpe_ratio = yearly_returns / yearly_std_dev if yearly_std_dev != 0 else np.nan

    # Calculate cumulative returns for max drawdown calculation
    cumulative_returns = np.exp(daily_returns.cumsum())
    rolling_max = cumulative_returns.cummax()
    daily_drawdown = cumulative_returns / rolling_max - 1
    max_drawdown = daily_drawdown.min()

    volatility = daily_returns.std() * np.sqrt(252)
    calmar_ratio = yearly_returns / -max_drawdown if max_drawdown != 0 else np.nan
    return_skewness = skew(daily_returns)
    return_kurtosis = kurtosis(daily_returns)

    weights_yearly = weights.copy()

    weights_yearly['Year'] = weights.index.year
    yearly_metrics = weights_yearly.groupby('Year').apply(lambda x: pd.Series({
        'Yearly Trades': calculate_trade_volume(x).sum(),
        'Yearly Turnover': x.diff().abs().sum().sum(),
        'Yearly Transaction Costs': x.diff().abs().sum().sum() * transaction_cost_rate * transaction_cost_rate
    }))

    # Return a dictionary of results
    return {
        "Average Yearly Return": yearly_returns,
        "Average Yearly Standard Deviation": yearly_std_dev,
        "Yearly Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
        "Yearly Volatility": volatility,
        "Calmar Ratio": calmar_ratio,
        "Skewness": return_skewness,
        "Kurtosis": return_kurtosis,
        "Yearly Trades": yearly_metrics['Yearly Trades'].mean(),
        "Yearly Turnover": yearly_metrics['Yearly Turnover'].mean(),
        "Yearly Transaction Costs": yearly_metrics['Yearly Transaction Costs'].mean(),
    }







def calculate_portfolio_volatility(weights, returns):
    """ Calculate portfolio volatility as the square root of (W.T * Cov * W) """
    cov_matrix = returns.cov()
    if weights.shape[0] != cov_matrix.T.shape[0]:
        raise ValueError("Dimension mismatch")
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))


def determine_leverage_factors(portfolio_volatility, target_volatility=0.10):
    """ Determine leverage factors based on target volatility """
    if portfolio_volatility == 0:
        return 0
    return target_volatility / portfolio_volatility

def apply_leverage(weights, leverage_factor):
    # Apply the calculated leverage factor to adjust weights
    return weights * leverage_factor

def calculate_annualized_volatility(df, window=252):
    """ Calculate the annualized volatility for each asset in the dataframe. """
    return df.rolling(window=window, min_periods = int(window//2)).std() * np.sqrt(window)




def update_df_with_asset_performance(signals_df, portfolio_df, target_days, returns_df, target_volatility=0.10):
    # Calculate volatilities using the existing function for annualized volatility
    volatilities = calculate_annualized_volatility(returns_df.fillna(0))
    
    # Ensure portfolio DataFrame columns are of float type to avoid dtype issues when updating
    portfolio_df = portfolio_df.astype(float)
    
    # Convert signals DataFrame index to datetime if it's not already
    signals_df.index = pd.to_datetime(signals_df.index)
    
    # Start processing from the first date in signals_df
    current_date = signals_df.index.min()

    while current_date <= signals_df.index.max():
        if current_date in signals_df.index:
            current_index = signals_df.index.get_loc(current_date)
            row = signals_df.loc[current_date]
            assets = row[row != 0].index.tolist()
            asset_signals = row[row != 0].values
            
            if assets:
                start_index = current_index + 2  # Start two trading days after the current date
                if start_index < len(signals_df):
                    start_date = signals_df.index[start_index]
                    end_index = start_index + target_days - 1
                    end_date = signals_df.index[min(end_index, len(signals_df)-1)]

                    # Your existing logic for volatilities and weights
                    vol_ = returns_df.loc[start_date-pd.DateOffset(days=252):start_date, assets].std()
                    weights = asset_signals / vol_
                    adjusted_weights = weights / np.abs(weights).sum()
                    #adjusted_weights = normalized_weights * asset_signals

                    past_returns = returns_df.loc[start_date - pd.DateOffset(days=target_days):start_date, assets]
                    port_vol = calculate_portfolio_volatility(adjusted_weights, past_returns) * np.sqrt(252)
                    leverage = determine_leverage_factors(port_vol, target_volatility)
                    adjusted_weights *= leverage
                    
                    # Update portfolio
                    portfolio_df.loc[start_date:end_date, assets] = adjusted_weights.values

                    # Set current_date to one day before end_date
                    if (end_index - 1) < len(signals_df):
                        current_date = signals_df.index[end_index - 1]
                    else:
                        break  # If end_index - 1 is out of bounds
                else:
                    break  # start_index is out of range
            else:
                current_date += pd.DateOffset(days=1)  # No assets to process, move to next day
        else:
            current_date += pd.DateOffset(days=1)  # Current date not in signals_df, move to next day

    return portfolio_df










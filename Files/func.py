import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.metrics import MeanSquaredError, Accuracy
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error
import numpy as np
import xgboost as xgb
import re
import pickle
import lightgbm as lgb
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

def add_y_col(df, df_read, date_col, target_days, return_col, volatility_col):
    df_target = add_target(df_read, [target_days])
    df_combined = pd.concat([df, df_target], axis=1)
    df = transform_and_pivot_df(df_combined, date_col)
    
    return_col = return_col + '_' + str(target_days)
    volatility_col = volatility_col + '_' + str(target_days)
    
    #create sharpe mean column
    #Nedan gjorde så när vi dropna blev df tom
    #df['sharpe_ratio_mean'] = df.groupby(date_col)['sharpe_ratio'].mean()
    
    sharpe_ratio_mean = df.groupby(date_col)['sharpe_ratio'].mean().rename('sharpe_ratio_mean')
    df = df.merge(sharpe_ratio_mean, on=date_col)

    #shift the sharpe ratio by target_days
    df['sharpe_ratio'] = df['sharpe_ratio'].shift(-target_days)
    df['sharpe_ratio_mean'] = df['sharpe_ratio_mean'].shift(-target_days)
    
    #drop na values
    #df = df.dropna(subset=['sharpe_ratio', 'sharpe_ratio_mean'])
    df = df.dropna()

    df['Y'] = np.where(df['sharpe_ratio'] > df['sharpe_ratio_mean'], 1, 0)
    

    df = df.drop(columns=['sharpe_ratio', 'sharpe_ratio_mean', return_col, volatility_col])

    
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





def optimize_and_train_ridge(X_train, y_train, X_train_eval, y_train_eval, param_grid, scoring='accuracy', cv=5):

    model = RidgeClassifier()

    # Initialize GridSearchCV with the provided model and parameter grid
    grid_search = GridSearchCV(model, param_grid, scoring=scoring, cv=cv)
    
    # Fit GridSearchCV on the training set
    grid_search.fit(X_train, y_train)
    
    # Print the best parameters and the accuracy on the evaluation set
    print("Best parameters:", grid_search.best_params_)
    print("Best accuracy on evaluation set:", grid_search.best_score_)
    
    # Retrain the model with the best parameters on the combined training and evaluation sets
    model_best = model.__class__(**grid_search.best_params_)
    model_best.fit(X_train_eval, y_train_eval)
    
    # ändra från klassificierig till sannolikhet


    return model_best, grid_search






def evaluate_model_performance(y_true, y_pred):
 

    conf_matrix = confusion_matrix(y_true, y_pred)


    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)


    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    # Print the performance metrics
    print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")

    return conf_matrix, precision, recall, f1, mse, rmse




def optimize_and_train_xgb(X_train, y_train, X_eval, y_eval, param_grid, scoring='accuracy', cv=5, n_jobs=-1, early_stopping_rounds=10):

    # Initialize the XGBoost model
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    # Perform grid search
    grid_search = GridSearchCV(xgb_model, param_grid, scoring=scoring, cv=cv, n_jobs=n_jobs)
    grid_search.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], early_stopping_rounds=early_stopping_rounds, verbose=False)

    # Extract best hyperparameters
    best_params = grid_search.best_params_
    print("Best hyperparameters:", best_params)

    # Retrain the model with the best parameters on the combined training and evaluation set
    xgb_best = xgb.XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss')
    xgb_best.fit(pd.concat([X_train, X_eval]), pd.concat([y_train, y_eval]))

    return xgb_best, best_params




def optimize_and_train_lgb(X_train, y_train, X_eval, y_eval, param_grid, scoring='accuracy', cv=5, n_jobs=-1):

    # Initialize the LightGBM model
    lgb_model = lgb.LGBMClassifier()

    # Perform grid search
    grid_search = GridSearchCV(lgb_model, param_grid, scoring=scoring, cv=cv, n_jobs=n_jobs)
    grid_search.fit(X_train, y_train)

    # Extract best hyperparameters
    best_params = grid_search.best_params_
    print("Best hyperparameters:", best_params)

    # Retrain the model with the best parameters on the combined training and evaluation set
    lgb_best = lgb.LGBMClassifier(**best_params)
    lgb_best.fit(pd.concat([X_train, X_eval]), pd.concat([y_train, y_eval]), eval_set=[(X_eval, y_eval)])

    return lgb_best, best_params




def train_and_evaluate_NN(X_train_eval, y_train_eval, X_eval, y_eval, X_test, y_test, epochs=50, batch_size=32):

    # Initialize the scaler and scale the data
    scaler = StandardScaler()
    # undersök data leakage här
    X_train_eval_scaled = scaler.fit_transform(X_train_eval)
    X_eval_scaled = scaler.transform(X_eval)
    X_test_scaled = scaler.transform(X_test)

    # Define the model architecture
    #undersök relu
    model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_eval_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'), 
    Dense(8, activation='relu'),   
    Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    # Train the model
    history = model.fit(
        X_train_eval_scaled, y_train_eval,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_eval_scaled, y_eval)
    )
    return model, history, X_test_scaled



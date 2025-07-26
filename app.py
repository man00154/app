import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import warnings
warnings.filterwarnings("ignore")

print("Libraries imported successfully!")

def load_itsm_data(file_path="ITSM_data.csv"):
    """
    Loads and initially processes the ITSM incident data from a CSV file.

    Args:
        file_path (str): Path to the ITSM CSV file.

    Returns:
        pd.DataFrame: A DataFrame with 'Date', 'Category', and 'Ticket_Count'
                      aggregated daily for each category.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data from {file_path}. Shape: {df.shape}")
        print("Columns:", df.columns.tolist())
        print(df.head())

        df['Close_Time'] = pd.to_datetime(df['Close_Time'], errors='coerce', dayfirst=True)

        df.dropna(subset=['Close_Time'], inplace=True)
        print(f"After dropping rows with invalid 'Close_Time', shape: {df.shape}")

        df['Date'] = df['Close_Time'].dt.date

        relevant_categories = df['Category'].unique()
        print(f"Identified categories: {relevant_categories}")

        daily_incident_counts = df.groupby(['Date', 'Category']).size().reset_index(name='Ticket_Count')
        print("\nDaily Incident Counts Head:")
        print(daily_incident_counts.head())
        print("\nDaily Incident Counts Tail:")
        print(daily_incident_counts.tail())

        return daily_incident_counts

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please ensure it's in the correct directory.")
    except Exception as e:
        print(f"An error occurred during data loading or initial processing: {e}")
        return pd.DataFrame()


def preprocess_data(df, time_granularity='quarterly'):
    """
    Preprocesses the data by aggregating ticket counts based on time granularity.

    Args:
        df (pd.DataFrame): Input DataFrame with 'Date', 'Category', 'Ticket_Count' (daily aggregated).
        time_granularity (str): 'quarterly' or 'annually'.

    Returns:
        dict: A dictionary where keys are categories and values are DataFrames
              aggregated by the specified time granularity.
    """
    processed_data = {}
    unique_categories = df['Category'].unique()

    for category in unique_categories:
        category_df = df[df['Category'] == category].copy()
        #category_df.set_index('Date', inplace=True) #set index before resampling

        if time_granularity == 'quarterly':
            category_df['Date'] = pd.to_datetime(category_df['Date'])
            category_df.set_index('Date', inplace=True)
            agg_df = category_df['Ticket_Count'].resample('QS').sum().reset_index()
            agg_df.rename(columns={'Date': 'ds', 'Ticket_Count': 'y'}, inplace=True)
            print(f"Aggregated {category} data to quarterly.")
        elif time_granularity == 'annually':
            category_df['Date'] = pd.to_datetime(category_df['Date'])
            category_df.set_index('Date', inplace=True)
            agg_df = category_df['Ticket_Count'].resample('AS').sum().reset_index()
            agg_df.rename(columns={'Date': 'ds', 'Ticket_Count': 'y'}, inplace=True)
            print(f"Aggregated {category} data to annually.")
        else:
            raise ValueError("time_granularity must be 'quarterly' or 'annually'.")

        agg_df['ds'] = pd.to_datetime(agg_df['ds']).dt.normalize()
        processed_data[category] = agg_df
        print(agg_df.head())
    return processed_data

def evaluate_model(y_true, y_pred, model_name):
    """
    Evaluates a time series model using MAE, RMSE, and MAPE.

    Args:
        y_true (pd.Series or np.array): Actual values.
        y_pred (pd.Series or np.array): Predicted values.
        model_name (str): Name of the model for printing.

    Returns:
        dict: A dictionary containing MAE, RMSE, and MAPE.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Avoid division by zero in MAPE calculation
    y_true_no_zero = np.where(y_true != 0, y_true, np.nan)  # Replace zeros with NaN
    mape = np.nanmean(np.abs((y_true - y_pred) / y_true_no_zero)) * 100

    print(f"--- {model_name} Evaluation ---")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

def train_and_forecast_arima(train_df, test_df, order=(1,1,1), seasonal_order=(1,1,0,4)):
    """
    Trains and forecasts using SARIMA model.

    Args:
        train_df (pd.DataFrame): Training data with 'ds' and 'y'.
        test_df (pd.DataFrame): Test data with 'ds' and 'y'.
        order (tuple): (p,d,q) order of the ARIMA model.
        seasonal_order (tuple): (P,D,Q,s) seasonal order of the SARIMA model.

    Returns:
        tuple: (pd.Series, dict) - Predicted values and evaluation metrics.
    """
    try:
        train_series = train_df.set_index('ds')['y']
        
        # Infer frequency if possible, otherwise set to None
        try:
            freq = pd.infer_freq(train_series.index)
        except ValueError:
            freq = None
        
        # If frequency is not inferred, try to determine it based on time differences
        if freq is None and len(train_series) > 1:
            time_diff = train_series.index.to_series().diff().dropna()
            if (time_diff.dt.days == 91).all():
                freq = 'QS'
            elif (time_diff.dt.days == 365).all():
                freq = 'AS'
        
        # Set frequency if determined
        if freq:
            train_series.index.freq = freq
        else:
            print("SARIMA: Could not infer frequency.  Skipping model.")
            return pd.Series([np.nan] * len(test_df), index=test_df['ds']), {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}

        model = ARIMA(train_series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit()
        forecast_steps = len(test_df)
        predictions = model_fit.predict(start=len(train_df), end=len(train_df) + forecast_steps - 1)
        metrics = evaluate_model(test_df['y'], predictions, "SARIMA")
        return predictions, metrics
    except Exception as e:
        print(f"SARIMA training failed: {e}")
        return pd.Series([np.nan] * len(test_df), index=test_df['ds']), {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}


#def train_and_forecast_prophet(train_df, test_df, time_granularity):
    #"""
   # Trains and forecasts using Facebook Prophet.

    #Args:
        #train_df (pd.DataFrame): Training data with 'ds' and 'y'.
        #test_df (pd.DataFrame): Test data with 'ds' and 'y'.
        #time_granularity (str): 'quarterly' or 'annually'.

    #Returns:
       # tuple: (pd.Series, dict) - Predicted values and evaluation metrics.
    #"""
   # try:
       #from prophet import Prophet
        #from prophet import Prophet
        #model = Prophet(
         #   yearly_seasonality=True,
       # )
        #if time_granularity == 'quarterly':
          #  model.add_seasonality(name='quarterly', period=365.25/4, fourier_order=5)

       # model.fit(train_df)
        #future = model.make_future_dataframe(periods=len(test_df), freq='QS' if time_granularity == 'quarterly' else 'AS')
       # forecast = model.predict(future)
        #predictions = forecast['yhat'].tail(len(test_df)).values
       # metrics = evaluate_model(test_df['y'], predictions, "Prophet")
        #return pd.Series(predictions, index=test_df['ds']), metrics
   # except Exception as e:
       # print(f"Prophet training failed: {e}")
        #return pd.Series([np.nan] * len(test_df), index=test_df['ds']), {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}

def train_and_forecast_ets(train_df, test_df, time_granularity):
    """
    Trains and forecasts using Exponential Smoothing (Holt-Winters).

    Args:
        train_df (pd.DataFrame): Training data with 'ds' and 'y'.
        test_df (pd.DataFrame): Test data with 'ds' and 'y'.
        time_granularity (str): 'quarterly' or 'annually'.

    Returns:
        tuple: (pd.Series, dict) - Predicted values and evaluation metrics.
    """
    try:
        train_series = train_df.set_index('ds')['y']
        
        # Infer frequency if possible, otherwise set to None
        try:
            freq = pd.infer_freq(train_series.index)
        except ValueError:
            freq = None
        
        # If frequency is not inferred, try to determine it based on time differences
        if freq is None and len(train_series) > 1:
            time_diff = train_series.index.to_series().diff().dropna()
            if (time_diff.dt.days == 91).all():
                freq = 'QS'
            elif (time_diff.dt.days == 365).all():
                freq = 'AS'
        
        # Set frequency if determined
        if freq:
            train_series.index.freq = freq
        else:
            print("ETS: Could not infer frequency.  Skipping model.")
            return pd.Series([np.nan] * len(test_df), index=test_df['ds']), {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}

        seasonal_periods = 4 if time_granularity == 'quarterly' else 1

        model = ExponentialSmoothing(
            train_series,
            seasonal_periods=seasonal_periods,
            initialization_method="estimated"
        )
        model_fit = model.fit()
        predictions = model_fit.forecast(len(test_df))
        metrics = evaluate_model(test_df['y'], predictions, "ETS")
        return predictions, metrics
    except Exception as e:
        print(f"ETS training failed: {e}")
        return pd.Series([np.nan] * len(test_df), index=test_df['ds']), {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}


def create_time_features(df):
    """
    Creates time-based features for ML models.

    Args:
        df (pd.DataFrame): DataFrame with a 'ds' (datetime) column.

    Returns:
        pd.DataFrame: DataFrame with added time features.
    """
    df['year'] = df['ds'].dt.year
    df['quarter'] = df['ds'].dt.quarter
    df['month'] = df['ds'].dt.month
    df['dayofyear'] = df['ds'].dt.dayofyear
    df['weekofyear'] = df['ds'].dt.isocalendar().week.astype(int)
    return df

def train_and_forecast_ml_model(train_df, test_df, model_type='RandomForest'):
    """
    Trains and forecasts using a machine learning regressor (Random Forest or XGBoost).

    Args:
        train_df (pd.DataFrame): Training data with 'ds' and 'y'.
        test_df (pd.DataFrame): Test data with 'ds' and 'y'.
        model_type (str): 'RandomForest'.

    Returns:
        tuple: (pd.Series, dict) - Predicted values and evaluation metrics.
    """
    try:
        train_features = create_time_features(train_df.copy())
        test_features = create_time_features(test_df.copy())

        features = ['year', 'quarter', 'month', 'dayofyear', 'weekofyear']
        existing_features = [f for f in features if f in train_features.columns and f in test_features.columns]
        if not existing_features:
            print(f"ML Model: No relevant time features found for training. Skipping {model_type}.")
            return pd.Series([np.nan] * len(test_df), index=test_df['ds']), {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}

        X_train = train_features[existing_features]
        y_train = train_features['y']
        X_test = test_features[existing_features]

            model_type == 'RandomForest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model_name = "RandomForest"
       # elif model_type == 'XGBoost':
        #    model = XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror')
         #   model_name = "XGBoost"
        #else:
            #raise ValueError("model_type must be 'RandomForest' or 'XGBoost'.")

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        predictions = pd.Series(predictions, index=test_df['ds'])
        metrics = evaluate_model(test_df['y'], predictions, model_name)
        return predictions, metrics
    except Exception as e:
        print(f"{model_type} training failed: {e}")
        return pd.Series([np.nan] * len(test_df), index=test_df['ds']), {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}

def train_and_forecast_naive(train_df, test_df):
    """
    Trains and forecasts using a Naive (last value) model.

    Args:
        train_df (pd.DataFrame): Training data with 'ds' and 'y'.
        test_df (pd.DataFrame): Test data with 'ds' and 'y'.

    Returns:
        tuple: (pd.Series, dict) - Predicted values and evaluation metrics.
    """
    if not train_df.empty:
        last_value = train_df['y'].iloc[-1]
        predictions = pd.Series([last_value] * len(test_df), index=test_df['ds'])
    else:
        predictions = pd.Series([np.nan] * len(test_df), index=test_df['ds'])
    metrics = evaluate_model(test_df['y'], predictions, "Naive")
    return predictions, metrics

def plot_forecast(actual, train, predictions, title, forecast_horizon_label, filename):
    """
    Plots the actual, training, and predicted values.

    Args:
        actual (pd.Series): Actual values (including train and test).
        train (pd.Series): Training data values.
        predictions (pd.Series): Predicted values.
        title (str): Title of the plot.
        forecast_horizon_label (str): Label for the forecast horizon (e.g., 'next 4 quarters').
        filename (str): Name to save the plot.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(train.index, train, label='Training Data', color='blue')
    plt.plot(actual.index, actual, label='Actual Data', color='green', linestyle='--')
    plt.plot(predictions.index, predictions, label=f'Forecasted Data ({forecast_horizon_label})', color='red')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Ticket Count')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    """
    Main function to orchestrate data loading, preprocessing, model training,
    evaluation, selection, and forecasting using ITSM data.
    """
    print("--- Loading and Processing ITSM Data ---")
    daily_incident_data = load_itsm_data(file_path="ITSM_data.csv")

    if daily_incident_data is None or daily_incident_data.empty:
        print("No data loaded or processed from ITSM_data.csv. Exiting.")
        return

    forecast_horizon_quarterly = 4
    forecast_horizon_annually = 2

    for granularity in ['quarterly', 'annually']:
        print(f"\n--- Processing for {granularity.capitalize()} Granularity ---")
        processed_data_by_category = preprocess_data(daily_incident_data, time_granularity=granularity)

        for category, df_agg in processed_data_by_category.items():
            print(f"\n--- Forecasting for Category: {category} ({granularity.capitalize()}) ---")

            test_size = forecast_horizon_quarterly if granularity == 'quarterly' else forecast_horizon_annually
            if len(df_agg) <= test_size + 2:
                print(f"Not enough data for {granularity} split for {category}. Skipping. (Need > {test_size+2} points, have {len(df_agg)})")
                continue

            train_df = df_agg.iloc[:-test_size].copy()
            test_df = df_agg.iloc[-test_size:].copy()

            print(f"Train data size: {len(train_df)}")
            print(f"Test data size: {len(test_df)}")

            all_model_results = {}
            all_model_predictions = {}

            # Define seasonal_order outside the loop
            sarima_seasonal_order = (1, 1, 0, 4)

            naive_predictions, naive_metrics = train_and_forecast_naive(train_df, test_df)
            all_model_results['Naive'] = naive_metrics
            all_model_predictions['Naive'] = naive_predictions

            sarima_order = (1,1,1)
            sarima_predictions, sarima_metrics = train_and_forecast_arima(train_df, test_df, order=sarima_order, seasonal_order=sarima_seasonal_order)
            all_model_results['SARIMA'] = sarima_metrics
            all_model_predictions['SARIMA'] = sarima_predictions

            #prophet_predictions, prophet_metrics = train_and_forecast_prophet(train_df, test_df, granularity)
            #all_model_results['Prophet'] = prophet_metrics
            #all_model_predictions['Prophet'] = prophet_predictions

            ets_predictions, ets_metrics = train_and_forecast_ets(train_df, test_df, granularity)
            all_model_results['ETS'] = ets_metrics
            all_model_predictions['ETS'] = ets_predictions

            rf_predictions, rf_metrics = train_and_forecast_ml_model(train_df, test_df, 'RandomForest')
            all_model_results['RandomForest'] = rf_metrics
            all_model_predictions['RandomForest'] = rf_predictions

            #xgb_predictions, xgb_metrics = train_and_forecast_ml_model(train_df, test_df, 'XGBoost')
            #all_model_results['XGBoost'] = xgb_metrics
            #all_model_predictions['XGBoost'] = xgb_predictions

            best_model_name = None
            min_rmse = float('inf')

            print("\n--- Model Comparison ---")
            for model_name, metrics in all_model_results.items():
                if not np.isnan(metrics['RMSE']) and metrics['RMSE'] < min_rmse:
                    min_rmse = metrics['RMSE']
                    best_model_name = model_name
                print(f"{model_name}: RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}, MAPE={metrics['MAPE']:.2f}%")

            if best_model_name:
                print(f"\nBest model for {category} ({granularity}): {best_model_name} (RMSE: {min_rmse:.2f})")
            else:
                print(f"\nCould not determine best model for {category} ({granularity}). All models failed or produced NaN RMSE.")
                continue

            print(f"\n--- Forecasting Future Volumes with {best_model_name} ---")

            full_df = df_agg.copy()

            future_periods = forecast_horizon_quarterly if granularity == 'quarterly' else forecast_horizon_annually
            
            last_date = full_df['ds'].max()
            if granularity == 'quarterly':
                future_dates = pd.date_range(start=last_date + pd.DateOffset(months=3), periods=future_periods, freq='QS')
            else:
                future_dates = pd.date_range(start=last_date + pd.DateOffset(years=1), periods=future_periods, freq='AS')
            
            future_df = pd.DataFrame({'ds': future_dates})


            if best_model_name == 'Naive':
                if not full_df.empty:
                    final_forecast_predictions = pd.Series([full_df['y'].iloc[-1]] * future_periods, index=future_df['ds'])
                else:
                    final_forecast_predictions = pd.Series([np.nan] * future_periods, index=future_df['ds'])
            elif best_model_name == 'SARIMA':
                full_series = full_df.set_index('ds')['y']
                
                # Infer frequency if possible, otherwise set to None
                try:
                    freq = pd.infer_freq(full_series.index)
                except ValueError:
                    freq = None
                
                # If frequency is not inferred, try to determine it based on time differences
                if freq is None and len(full_series) > 1:
                    time_diff = full_series.index.to_series().diff().dropna()
                    if (time_diff.dt.days == 91).all():
                        freq = 'QS'
                    elif (time_diff.dt.days == 365).all():
                        freq = 'AS'
                
                # Set frequency if determined
                if freq:
                    full_series.index.freq = freq
                else:
                    print("SARIMA: Could not infer frequency for final forecast.  Skipping model.")
                    final_forecast_predictions = pd.Series([np.nan] * future_periods, index=future_df['ds'])
                    continue
                
                try:
                    model = ARIMA(full_series, order=sarima_order, seasonal_order=sarima_seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
                    model_fit = model.fit()
                    final_forecast_predictions = model_fit.predict(start=len(full_df), end=len(full_df) + future_periods - 1)
                    final_forecast_predictions.index = future_df['ds']
                except Exception as e:
                    print(f"SARIMA final forecast failed: {e}")
                    final_forecast_predictions = pd.Series([np.nan] * future_periods, index=future_df['ds'])
            #elif best_model_name == 'Prophet':
                #from prophet import Prophet
                #from prophet import Prophet
                #model = Prophet(
                    #yearly_seasonality=True,
                #)
               # if granularity == 'quarterly':
                   # model.add_seasonality(name='quarterly', period=365.25/4, fourier_order=5)
                #model.fit(full_df)
                #future = model.make_future_dataframe(periods=future_periods, freq='QS' if granularity == 'quarterly' else 'AS')
                #forecast = model.predict(future)
                #final_forecast_predictions = forecast['yhat'].tail(future_periods)
                #final_forecast_predictions = pd.Series(final_forecast_predictions.values, index=future_df['ds'])
            elif best_model_name == 'ETS':
                full_series = full_df.set_index('ds')['y']
                
                # Infer frequency if possible, otherwise set to None
                try:
                    freq = pd.infer_freq(full_series.index)
                except ValueError:
                    freq = None
                
                # If frequency is not inferred, try to determine it based on time differences
                if freq is None and len(full_series) > 1:
                    time_diff = full_series.index.to_series().diff().dropna()
                    if (time_diff.dt.days == 91).all():
                        freq = 'QS'
                    elif (time_diff.dt.days == 365).all():
                        freq = 'AS'
                
                # Set frequency if determined
                if freq:
                    full_series.index.freq = freq
                else:
                    print("ETS: Could not infer frequency for final forecast.  Skipping model.")
                    final_forecast_predictions = pd.Series([np.nan] * future_periods, index=future_df['ds'])
                    continue

                seasonal_periods = 4 if granularity == 'quarterly' else 1
                try:
                    model = ExponentialSmoothing(
                        full_series,
                        seasonal_periods=seasonal_periods,
                        initialization_method="estimated"
                    )
                    model_fit = model.fit()
                    final_forecast_predictions = model_fit.forecast(future_periods)
                    final_forecast_predictions.index = future_df['ds']
                except Exception as e:
                    print(f"ETS final forecast failed: {e}")
                    final_forecast_predictions = pd.Series([np.nan] * future_periods, index=future_df['ds'])
            elif best_model_name in ['RandomForest', 'XGBoost']:
                full_features = create_time_features(full_df.copy())
                future_features = create_time_features(future_df.copy())
                features = ['year', 'quarter', 'month', 'dayofyear', 'weekofyear']
                
                existing_features_full = [f for f in features if f in full_features.columns]
                existing_features_future = [f for f in features if f in future_features.columns]

                if not existing_features_full or not existing_features_future:
                    print(f"ML Model final forecast: Missing features for training or prediction. Skipping {best_model_name}.")
                    final_forecast_predictions = pd.Series([np.nan] * future_periods, index=future_df['ds'])
                else:
                    X_full = full_features[existing_features_full]
                    y_full = full_features['y']
                    X_future = future_features[existing_features_future]

                    if best_model_name == 'RandomForest':
                        model = RandomForestRegressor(n_estimators=100, random_state=42)
                   # else:
                    #    model = XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror')
                    
                    #model.fit(X_full, y_full)
                    #final_forecast_predictions = pd.Series(model.predict(X_future), index=future_df['ds'])
            else:
                final_forecast_predictions = pd.Series([np.nan] * future_periods, index=future_df['ds'])

            print(f"Forecasted {future_periods} {granularity} periods for {category}:")
            print(final_forecast_predictions)

            actual_data_for_plot = df_agg.set_index('ds')['y']
            train_data_for_plot = train_df.set_index('ds')['y']
            
            combined_series = pd.concat([actual_data_for_plot, final_forecast_predictions])
            
            plot_forecast(
                actual=combined_series,
                train=train_data_for_plot,
                predictions=final_forecast_predictions,
                title=f'{category} Ticket Volume Forecast ({granularity.capitalize()}) - Best Model: {best_model_name}',
                forecast_horizon_label=f'next {future_periods} {granularity}',
                filename=f'forecast_{category}_{granularity}.png'
            )

if __name__ == "__main__":
    main()

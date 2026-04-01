# Data Preparation - Task 1

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_energy_data(filepath):
    """Load the AEP hourly energy consumption dataset."""
    df = pd.read_csv(filepath, parse_dates=['Datetime'])
    return df


def load_air_pollution_data(filepath, station_code=101):
    """Load air pollution data and filter for station 101."""
    df = pd.read_csv(filepath, parse_dates=['Measurement date'])
    df = df[df['Station code'] == station_code].copy()
    return df


def aggregate_to_daily(df, datetime_col, value_col):
    """Convert hourly data to daily averages."""
    df = df.copy()
    df['Date'] = df[datetime_col].dt.date
    
    daily_df = df.groupby('Date')[value_col].mean().reset_index()
    daily_df.columns = ['Date', value_col]
    daily_df['Date'] = pd.to_datetime(daily_df['Date'])
    daily_df = daily_df.sort_values('Date').reset_index(drop=True)
    
    return daily_df


def create_energy_daily_data(filepath):
    """Process energy data into daily averages."""
    df = load_energy_data(filepath)
    daily_df = aggregate_to_daily(df, 'Datetime', 'AEP_MW')
    return daily_df


def create_air_pollution_daily_data(filepath, station_code=101):
    """Process air pollution data into daily PM2.5 averages."""
    df = load_air_pollution_data(filepath, station_code)
    daily_df = aggregate_to_daily(df, 'Measurement date', 'PM2.5')
    return daily_df


def create_binary_labels_energy(daily_values):
    """Create labels: 1 if value > mean, else 0."""
    mean_value = np.mean(daily_values)
    labels = (daily_values > mean_value).astype(int)
    return labels, mean_value


def create_binary_labels_air_pollution(daily_values, threshold=35):
    """Create labels: 1 if PM2.5 > 35, else 0."""
    labels = (daily_values > threshold).astype(int)
    return labels, threshold


def create_sliding_window_features(values, window_size=7):
    """Create 7-day sliding window features for prediction."""
    n = len(values)
    X = []
    valid_indices = []
    
    for i in range(window_size - 1, n - 1):
        window = values[i - window_size + 1:i + 1]
        X.append(window)
        valid_indices.append(i + 1)
    
    return np.array(X), np.array(valid_indices)


def handle_missing_values(df, value_col, method='interpolate'):
    """Handle missing values by interpolation or dropping."""
    df = df.copy()
    
    if method == 'interpolate':
        df[value_col] = df[value_col].interpolate(method='linear')
        df[value_col] = df[value_col].bfill().ffill()
    elif method == 'drop':
        df = df.dropna(subset=[value_col])
    
    return df


def normalize_features(X_train, X_test=None, method='standard'):
    """Normalize features using StandardScaler."""
    if method == 'standard':
        scaler = StandardScaler()
    else:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler
    
    return X_train_scaled, scaler


def validate_data(X, y):
    """Check that data is valid."""
    assert X.shape[0] == y.shape[0], "X and y must have same number of samples"
    assert not np.isnan(X).any(), "X contains NaN values"
    assert not np.isnan(y).any(), "y contains NaN values"
    return True


def prepare_dataset(daily_df, value_col, label_func, window_size=7):
    """Full pipeline to prepare a dataset for SVM."""
    daily_df = handle_missing_values(daily_df, value_col)
    
    values = daily_df[value_col].values
    dates = daily_df['Date'].values
    
    labels, threshold = label_func(values)
    X, valid_indices = create_sliding_window_features(values, window_size)
    y = labels[valid_indices]
    target_dates = dates[valid_indices]
    
    validate_data(X, y)
    
    return {
        'X': X,
        'y': y,
        'dates': target_dates,
        'threshold': threshold,
        'value_col': value_col,
        'daily_df': daily_df
    }


def prepare_energy_dataset(filepath, window_size=7):
    """Prepare energy dataset for SVM."""
    daily_df = create_energy_daily_data(filepath)
    result = prepare_dataset(daily_df, 'AEP_MW', create_binary_labels_energy, window_size)
    result['dataset_name'] = 'Energy Consumption'
    return result


def prepare_air_pollution_dataset(filepath, window_size=7, station_code=101):
    """Prepare air pollution dataset for SVM."""
    daily_df = create_air_pollution_daily_data(filepath, station_code)
    result = prepare_dataset(daily_df, 'PM2.5', create_binary_labels_air_pollution, window_size)
    result['dataset_name'] = 'Air Pollution (PM2.5)'
    return result


def temporal_train_test_split(X, y, dates=None, train_ratio=0.8):
    """Split data chronologically (80% train, 20% test)."""
    n_samples = X.shape[0]
    split_idx = int(n_samples * train_ratio)
    
    result = {
        'X_train': X[:split_idx],
        'X_test': X[split_idx:],
        'y_train': y[:split_idx],
        'y_test': y[split_idx:],
        'split_idx': split_idx
    }
    
    if dates is not None:
        result['dates_train'] = dates[:split_idx]
        result['dates_test'] = dates[split_idx:]
    
    return result


if __name__ == "__main__":
    print("Testing data preparation...")
    
    energy_data = prepare_energy_dataset('Hourly_Energy _Consumption_AEP_hourly.csv')
    print(f"\nEnergy Dataset:")
    print(f"  Samples: {energy_data['X'].shape[0]}")
    print(f"  Features: {energy_data['X'].shape[1]}")
    print(f"  Class distribution: {np.bincount(energy_data['y'])}")
    
    air_data = prepare_air_pollution_dataset('Air_Pollution_in_Seoul_Measurement_summary.csv')
    print(f"\nAir Pollution Dataset:")
    print(f"  Samples: {air_data['X'].shape[0]}")
    print(f"  Features: {air_data['X'].shape[1]}")
    print(f"  Class distribution: {np.bincount(air_data['y'])}")

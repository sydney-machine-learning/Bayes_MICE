import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge
from statsmodels.tsa.seasonal import STL


# ==================== IMPUTATION METHODS ====================

def create_lagged_features_df(series, max_past_lags=3, max_future_lags=3):
    """
    Create lagged features DataFrame from a univariate time series.
    
    Parameters:
    -----------
    series : pd.Series
        Time series
    max_past_lags : int
        Number of past lags
    max_future_lags : int
        Number of future lags
    
    Returns:
    --------
    pd.DataFrame : DataFrame with target and lagged features
    """
    data = pd.DataFrame({'target': series.values}, index=series.index)
    
    # Add past lags
    for lag in range(1, max_past_lags + 1):
        data[f'past_lag_{lag}'] = series.shift(lag).values
    
    # Add future lags
    for lag in range(1, max_future_lags + 1):
        data[f'future_lag_{lag}'] = series.shift(-lag).values
    
    return data


def mice_imputation_with_lags(series, max_past_lags=3, max_future_lags=3, 
                               max_iter=5, random_state=42):
    """
    MICE imputation using time-lagged features.
    
    Uses sklearn's IterativeImputer with lagged features as predictors.
    
    Parameters:
    -----------
    series : pd.Series
        Time series with missing values
    max_past_lags : int
        Number of past lags to use as features
    max_future_lags : int
        Number of future lags to use as features
    max_iter : int
        Maximum number of MICE iterations
    random_state : int
        Random seed
    
    Returns:
    --------
    pd.Series : Imputed time series
    """
    # Create lagged features
    lagged_data = create_lagged_features_df(series, max_past_lags, max_future_lags)
    
    # Apply IterativeImputer
    imputer = IterativeImputer(max_iter=max_iter, random_state=random_state)
    imputed_array = imputer.fit_transform(lagged_data.values)
    
    # Return the target column (first column)
    imputed_series = pd.Series(imputed_array[:, 0], index=series.index, name=series.name)
    
    return imputed_series


def interpolation_imputation(series, method='linear'):
    """
    Simple interpolation imputation.
    
    Parameters:
    -----------
    series : pd.Series
        Time series with missing values
    method : str
        Interpolation method: 'linear', 'quadratic', 'cubic', 'spline'
    
    Returns:
    --------
    pd.Series : Imputed time series
    """
    series_filled = series.copy()
    
    if method in ['linear', 'quadratic', 'cubic']:
        series_filled = series_filled.interpolate(method=method)
    elif method == 'spline':
        series_filled = series_filled.interpolate(method='spline', order=3)
    
    # Handle edges 
    series_filled = series_filled.ffill()
    series_filled = series_filled.bfill()

    return series_filled


def mean_imputation(series):
    """
    Simple mean imputation.
    
    Parameters:
    -----------
    series : pd.Series
        Time series with missing values
    
    Returns:
    --------
    pd.Series : Imputed time series
    """
    return series.fillna(series.mean())


def median_imputation(series):
    """
    Simple median imputation.
    """
    return series.fillna(series.median())


def locf_imputation(series):
    """
    Last Observation Carried Forward (LOCF).
    """
    series_filled = series.fillna(method='ffill')
    series_filled.fillna(method='bfill', inplace=True)  # Handle leading NAs
    return series_filled


def knn_imputation(series, n_neighbors=5):
    """
    KNN imputation for univariate time series.
    
    Simple approach: uses raw series values without creating lagged features.
    KNN finds k nearest neighbors based on temporal proximity and value similarity.
    
    Parameters:
    -----------
    series : pd.Series
        Time series with missing values
    n_neighbors : int
        Number of neighbors to use
    
    Returns:
    --------
    pd.Series : Imputed time series
    """
    # Reshape to 2D array (required by KNNImputer)
    values = series.values.reshape(-1, 1)
    
    # Apply KNN imputation
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_array = imputer.fit_transform(values)
    
    # Return as series with original index
    imputed_series = pd.Series(imputed_array.flatten(), index=series.index, name=series.name)
    
    return imputed_series


def seasonal_decomposition_imputation(series, period=7):
    """
    Imputation using seasonal decomposition.
    
    Parameters:
    -----------
    series : pd.Series
        Time series with missing values
    period : int
        Seasonal period
    
    Returns:
    --------
    pd.Series : Imputed time series
    """
    # First do a simple interpolation to handle missing values for decomposition
    series_temp = series.interpolate(method='linear')
    series_temp.fillna(method='ffill', inplace=True)
    series_temp.fillna(method='bfill', inplace=True)
    
    if len(series_temp) < 2 * period:
        # Not enough data for decomposition, fall back to interpolation
        return interpolation_imputation(series, method='linear')
    
    try:
        # Decompose
        stl = STL(series_temp, period=period, robust=True)
        result = stl.fit()
        
        # Get components
        trend = result.trend
        seasonal = result.seasonal
        
        # For original missing values, use trend + seasonal
        series_filled = series.copy()
        missing_mask = series.isnull()
        
        for idx in series[missing_mask].index:
            loc = series.index.get_loc(idx)
            series_filled.loc[idx] = trend.iloc[loc] + seasonal.iloc[loc]
        
        return series_filled
    
    except Exception as e:
        warnings.warn(f"Seasonal decomposition failed: {e}. Using linear interpolation.")
        return interpolation_imputation(series, method='linear')


def calculate_metrics(true_values, imputed_values):
    """
    Calculate imputation performance metrics.
    
    Parameters:
    -----------
    true_values : array
        True values at missing positions
    imputed_values : array
        Imputed values at missing positions
    
    Returns:
    --------
    dict : Dictionary of metrics
    """
    mae = np.mean(np.abs(true_values - imputed_values))
    rmse = np.sqrt(np.mean((true_values - imputed_values)**2))
    
    # Normalized metrics
    if np.std(true_values) > 0:
        nmae = mae / np.std(true_values)
        nrmse = rmse / np.std(true_values)
    else:
        nmae = mae
        nrmse = rmse
    
    # Mean relative error
    if np.mean(np.abs(true_values)) > 0:
        mre = np.mean(np.abs((true_values - imputed_values) / true_values))
    else:
        mre = np.inf
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'NMAE': nmae,
        'NRMSE': nrmse,
        'MRE': mre
    }


# ==================== PLOTTING FUNCTIONS ====================

def plot_imputation_comparison_focused(complete_data, missing_data, imputed_datasets_dict, 
                                       target_col, time_window=None, missing_indices=None, 
                                       dataset_name=None, fname=None):
    """
    Plot focused comparison showing actual vs imputed values for each method.
    Only plots the time series with missing points highlighted.
    
    Parameters:
    -----------
    complete_data : pd.Series or pd.DataFrame
        Original complete dataset
    missing_data : pd.Series or pd.DataFrame
        Dataset with missing values
    imputed_datasets_dict : dict
        Dictionary with method names as keys and imputed Series as values
    target_col : str
        Column to visualize (can be None if Series)
    time_window : tuple, optional
        (start_idx, end_idx) to zoom into specific time period
    missing_indices : list, optional
        Indices of missing values to highlight
    dataset_name : str, optional
        Name for the plot title
    fname : str, optional
        Filename to save the plot
    """
    
    COLORS = {
        "Original": "black",
        "MICE (Time-Lagged)": "#E69F00",
        "KNN (k=5)": "#56B4E9",
        "Linear Interpolation": "#009E73",
        "Cubic Interpolation": "#F0E442",
        "Mean": "#CC79A7",
        "Median": "#D55E00",
        "LOCF": "#0072B2",
        "Seasonal Decomposition": "#7B61FF"
    }
    
    # Handle Series vs DataFrame
    if isinstance(complete_data, pd.Series):
        complete_slice_data = complete_data
        missing_slice_data = missing_data
    else:
        complete_slice_data = complete_data[target_col]
        missing_slice_data = missing_data[target_col]
    
    # Determine number of methods
    n_methods = len(imputed_datasets_dict)
    
    # Create subplots: 1 row per method + 1 for original
    fig, axes = plt.subplots(n_methods + 1, 1, figsize=(16, 3 * (n_methods + 1)))
    
    # Make axes always a list
    if n_methods == 0:
        axes = [axes]
    
    # Determine time window
    if time_window is None:
        start_idx, end_idx = 0, len(complete_slice_data)
    else:
        start_idx, end_idx = time_window
    
    # Create time axis
    x_axis = np.arange(start_idx, end_idx)
    x_label = 'Time Index'
    
    # Slice data for window
    complete_slice = complete_slice_data.iloc[start_idx:end_idx]
    
    # Filter missing indices to window
    if missing_indices is not None:
        window_missing = [idx for idx in missing_indices if start_idx <= idx < end_idx]
    else:
        window_missing = []
    
    # ==========================================
    # Plot 1: Original Complete Data
    # ==========================================
    ax = axes[0]
    ax.plot(x_axis, complete_slice, color=COLORS["Original"], 
            linewidth=1.5, alpha=0.8, label="Original")
    
    # Highlight missing positions with vertical lines
    if window_missing:
        for idx in window_missing:
            ax.axvline(x=idx, color='red', alpha=0.2, linewidth=0.5, linestyle='--')
    
    ylabel = target_col if target_col else 'Value'
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, frameon=True, loc='upper right')
    ax.set_title('Original Complete Data', fontsize=14, fontweight='bold', pad=10)
    
    # ==========================================
    # Plot each method
    # ==========================================
    for idx, (method_name, imputed_data) in enumerate(imputed_datasets_dict.items(), 1):
        ax = axes[idx]
        
        # Get color for this method
        color = COLORS.get(method_name, "#333333")  # default gray if not in dict
        
        # Plot original data in light gray
        ax.plot(x_axis, complete_slice, color='lightgray', 
                linewidth=1.5, alpha=0.5, label="Original", zorder=1)
        
        # Handle Series vs DataFrame
        if isinstance(imputed_data, pd.Series):
            imputed_slice = imputed_data.iloc[start_idx:end_idx]
        else:
            imputed_slice = imputed_data[target_col].iloc[start_idx:end_idx]
        
        # Plot imputed data
        ax.plot(x_axis, imputed_slice, color=color, 
                linewidth=1.5, alpha=0.8, label=method_name, linestyle='--', zorder=2)
        
        # Highlight imputed points at missing positions
        if window_missing:
            missing_x = [idx for idx in window_missing]
            true_y = [complete_slice.iloc[idx - start_idx] for idx in window_missing]
            imputed_y = [imputed_slice.iloc[idx - start_idx] for idx in window_missing]
            
            # Plot true values as circles
            ax.scatter(missing_x, true_y, c='black', s=50, alpha=0.7,
                      edgecolors='white', linewidths=1.5, zorder=4, label='True values')
            
            # Plot imputed values as crosses
            ax.scatter(missing_x, imputed_y, c=color, s=50, alpha=0.9,
                      marker='x', linewidths=2, zorder=5, label=f'{method_name} imputed')
            
            # Draw connecting lines between true and imputed
            for mx, ty, iy in zip(missing_x, true_y, imputed_y):
                ax.plot([mx, mx], [ty, iy], color=color, alpha=0.3, linewidth=1, zorder=3)
        
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, frameon=True, loc='upper right', ncol=2)
        ax.set_title(f'Original vs {method_name}', fontsize=14, fontweight='bold', pad=10)
    
    # Set x-label only on bottom plot
    axes[-1].set_xlabel(x_label, fontsize=14, fontweight='bold')
    
    fig.tight_layout()
    
    if fname is not None:
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved to: {fname}")
        plt.close()
    else:
        plt.show()


def plot_imputation_comparison_grid(complete_data, missing_data, imputed_datasets_dict, 
                                    target_col, time_window=None, missing_indices=None, 
                                    dataset_name=None, fname=None):
    """
    Plot grid comparison (2x2 or dynamic grid) showing actual vs imputed values.
    """
    
    COLORS = {
        "Original": "black",
        "MICE (Time-Lagged)": "#E69F00",
        "KNN (k=5)": "#56B4E9",
        "Linear Interpolation": "#009E73",
        "Cubic Interpolation": "#F0E442",
        "Mean": "#CC79A7",
        "Median": "#D55E00",
        "LOCF": "#0072B2",
        "Seasonal Decomposition": "#7B61FF"
    }
    
    # Handle Series vs DataFrame
    if isinstance(complete_data, pd.Series):
        complete_slice_data = complete_data
    else:
        complete_slice_data = complete_data[target_col]
    
    # Determine grid size
    n_methods = len(imputed_datasets_dict)
    n_cols = 2
    n_rows = int(np.ceil(n_methods / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    
    # Flatten axes for easier iteration
    if n_methods == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Determine time window
    if time_window is None:
        start_idx, end_idx = 0, len(complete_slice_data)
    else:
        start_idx, end_idx = time_window
    
    # Create time axis
    x_axis = np.arange(start_idx, end_idx)
    x_label = 'Time Index'
    
    # Slice data for window
    complete_slice = complete_slice_data.iloc[start_idx:end_idx]
    
    # Filter missing indices to window
    if missing_indices is not None:
        window_missing = [idx for idx in missing_indices if start_idx <= idx < end_idx]
    else:
        window_missing = []
    
    # ==========================================
    # Plot each method in grid
    # ==========================================
    for idx, (method_name, imputed_data) in enumerate(imputed_datasets_dict.items()):
        ax = axes[idx]
        
        # Get color for this method
        color = COLORS.get(method_name, "#333333")
        
        # Plot original data
        ax.plot(x_axis, complete_slice, color='black', 
                linewidth=2, alpha=0.6, label="Original", zorder=1)
        
        # Handle Series vs DataFrame
        if isinstance(imputed_data, pd.Series):
            imputed_slice = imputed_data.iloc[start_idx:end_idx]
        else:
            imputed_slice = imputed_data[target_col].iloc[start_idx:end_idx]
        
        # Plot imputed data
        ax.plot(x_axis, imputed_slice, color=color, 
                linewidth=2, alpha=0.8, label=method_name, linestyle='--', zorder=2)
        
        # Highlight imputed points at missing positions
        if window_missing:
            missing_x = [idx for idx in window_missing]
            true_y = [complete_slice.iloc[idx - start_idx] for idx in window_missing]
            imputed_y = [imputed_slice.iloc[idx - start_idx] for idx in window_missing]
            
            # Plot true values
            ax.scatter(missing_x, true_y, c='black', s=60, alpha=0.8,
                      edgecolors='white', linewidths=2, zorder=4, label='True')
            
            # Plot imputed values
            ax.scatter(missing_x, imputed_y, c=color, s=60, alpha=0.9,
                      marker='X', linewidths=2, zorder=5, label='Imputed')
        
        ylabel = target_col if target_col else 'Value'
        ax.set_xlabel(x_label, fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.tick_params(axis='both', labelsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, frameon=True, loc='upper right', ncol=2)
        ax.set_title(f'{method_name}', fontsize=14, fontweight='bold', pad=10)
    
    # Hide extra subplots
    for idx in range(n_methods, len(axes)):
        axes[idx].axis('off')
    
    fig.tight_layout()
    
    if fname is not None:
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved to: {fname}")
        plt.close()
    else:
        plt.show()


def plot_error_comparison(complete_data, imputed_datasets_dict, target_col, 
                          missing_indices, fname=None):
    """
    Plot error bars showing prediction errors for each method at missing positions.
    """
    
    COLORS = {
        "MICE (Time-Lagged)": "#E69F00",
        "KNN (k=5)": "#56B4E9",
        "Linear Interpolation": "#009E73",
        "Cubic Interpolation": "#F0E442",
        "Mean": "#CC79A7",
        "Median": "#D55E00",
        "LOCF": "#0072B2",
        "Seasonal Decomposition": "#7B61FF"
    }
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    
    # Handle Series vs DataFrame
    if isinstance(complete_data, pd.Series):
        true_values = complete_data.iloc[missing_indices].values
    else:
        true_values = complete_data[target_col].iloc[missing_indices].values
    
    # Calculate errors for each method
    errors_dict = {}
    for method_name, imputed_data in imputed_datasets_dict.items():
        if isinstance(imputed_data, pd.Series):
            imputed_values = imputed_data.iloc[missing_indices].values
        else:
            imputed_values = imputed_data[target_col].iloc[missing_indices].values
        errors = imputed_values - true_values
        errors_dict[method_name] = errors
    
    # Plot errors
    x_positions = np.arange(len(missing_indices))
    bar_width = 0.8 / len(imputed_datasets_dict)
    
    for idx, (method_name, errors) in enumerate(errors_dict.items()):
        color = COLORS.get(method_name, "#333333")
        offset = (idx - len(imputed_datasets_dict)/2 + 0.5) * bar_width
        
        ax.bar(x_positions + offset, errors, bar_width, 
               label=method_name, color=color, alpha=0.7)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
    ax.set_xlabel('Missing Value Index', fontsize=14, fontweight='bold')
    ax.set_ylabel('Prediction Error (Imputed - True)', fontsize=14, fontweight='bold')
    ax.set_title('Prediction Errors at Missing Positions', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, frameon=True, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='both', labelsize=12)
    
    fig.tight_layout()
    
    if fname is not None:
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved to: {fname}")
        plt.close()
    else:
        plt.show()


# ==================== EVALUATION FUNCTIONS ====================

def evaluate_all_methods(series_with_missing, complete_data, verbose=True):
    """
    Evaluate all imputation methods on a SINGLE missing pattern.
    Each method runs ONCE.
    
    Parameters:
    -----------
    series_with_missing : pd.Series
        Your time series with missing values
    complete_data : pd.Series
        Complete version (for ground truth)
    verbose : bool
        Print progress
    
    Returns:
    --------
    tuple : (results_dict, imputed_datasets_dict)
        - results_dict: Dictionary with metrics for each method
        - imputed_datasets_dict: Dictionary with imputed series for each method
    """
    # Identify missing values
    missing_mask = series_with_missing.isnull()
    missing_indices = series_with_missing[missing_mask].index
    true_values = complete_data.loc[missing_indices].values
    
    if verbose:
        print("="*80)
        print("IMPUTATION METHODS EVALUATION")
        print("="*80)
        print(f"Total observations: {len(series_with_missing)}")
        print(f"Missing values: {len(missing_indices)} ({100*len(missing_indices)/len(series_with_missing):.1f}%)")
        print("="*80)
    
    results = {}
    imputed_datasets = {}
    
    # 1. MICE with Time-Lagged Features
    if verbose:
        print(f"\n1. MICE (Time-Lagged)...")
    
    imputed = mice_imputation_with_lags(
        series_with_missing,
        max_past_lags=1,
        max_future_lags=1,
        max_iter=10,
        random_state=42
    )
    metrics = calculate_metrics(true_values, imputed.loc[missing_indices].values)
    results['MICE (Time-Lagged)'] = metrics
    imputed_datasets['MICE (Time-Lagged)'] = imputed
    
    if verbose:
        print(f"   RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}")
    
    # 2-8. All other methods
    methods_list = [
        ('KNN (k=5)', lambda s: knn_imputation(s, n_neighbors=5)),
        ('Linear Interpolation', lambda s: interpolation_imputation(s, method='linear')),
        ('Cubic Interpolation', lambda s: interpolation_imputation(s, method='cubic')),
        ('Mean', mean_imputation),
        ('Median', median_imputation),
        #('LOCF', locf_imputation),
        #('Seasonal Decomposition', lambda s: seasonal_decomposition_imputation(s, period=12))
    ]
    
    for idx, (name, impute_func) in enumerate(methods_list, 2):
        if verbose:
            print(f"\n{idx}. {name}...")
        
        try:
            imputed = impute_func(series_with_missing)
            metrics = calculate_metrics(true_values, imputed.loc[missing_indices].values)
            results[name] = metrics
            imputed_datasets[name] = imputed
            
            if verbose:
                print(f"   RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}")
        except Exception as e:
            if verbose:
                print(f"   ERROR: {str(e)}")
            results[name] = {
                'RMSE': np.nan,
                'MAE': np.nan,
                'NMAE': np.nan,
                'NRMSE': np.nan,
                'MRE': np.nan
            }
            imputed_datasets[name] = None
    
    return results, imputed_datasets


def create_results_table(results):
    """
    Create summary table with single values (no mean/std).
    
    Parameters:
    -----------
    results : dict
        Dictionary with method names as keys and metrics dict as values
    
    Returns:
    --------
    pd.DataFrame : Results table
    """
    rows = []
    
    for method_name, metrics in results.items():
        rows.append({
            'Method': method_name,
            'RMSE': f"{metrics['RMSE']:.4f}",
            'MAE': f"{metrics['MAE']:.4f}",
            'NMAE': f"{metrics['NMAE']:.4f}",
            'NRMSE': f"{metrics['NRMSE']:.4f}",
            'MRE': f"{metrics['MRE']:.4f}",
            'RMSE_sort': metrics['RMSE']  # For sorting
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values('RMSE_sort')
    df = df.drop('RMSE_sort', axis=1)
    
    return df


def print_interpretation(results_df):
    """Print interpretation of results."""
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    
    best_method = results_df.iloc[0]['Method']
    best_rmse = results_df.iloc[0]['RMSE']
    
    print(f"\n🏆 Best performing method: {best_method}")
    print(f"   RMSE: {best_rmse}")
    
    print("\nMetrics explanation:")
    print("  - RMSE: Root Mean Square Error (lower is better)")
    print("  - MAE: Mean Absolute Error (lower is better)")
    print("  - NMAE: Normalized MAE (relative to std of true values)")
    print("  - NRMSE: Normalized RMSE (relative to std of true values)")
    print("  - MRE: Mean Relative Error (percentage error)")


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    print("="*80)
    print("SINGLE EVALUATION ON FIXED MISSING PATTERN")
    print("="*80)
    
    # Load your data
    data_with_missing = pd.read_csv('physio_with_missing_v1.csv')
    data_complete = pd.read_csv('physio_subdata.csv')
    
    series_masked = data_with_missing['WBC']
    series_complete = data_complete['WBC']
    
    # Evaluate all methods (single run each)
    results, imputed_datasets = evaluate_all_methods(
        series_with_missing=series_masked,
        complete_data=series_complete,
        verbose=True
    )
    
    # Create and display results table
    print("\n" + "="*80)
    print("RESULTS TABLE")
    print("="*80)
    results_df = create_results_table(results)
    print("\n" + results_df.to_string(index=False))
    
    # Print interpretation
    print_interpretation(results_df)
    
    # Save results
    results_df.to_csv('imputation_results.csv', index=False)
    print(f"\n💾 Results saved to: imputation_results.csv")
    
    # ==================== GENERATE PLOTS ====================
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # Get missing indices
    missing_mask = series_masked.isnull()
    missing_indices = list(series_masked[missing_mask].index)
    
    # Remove None values from imputed_datasets (failed methods)
    imputed_datasets_clean = {k: v for k, v in imputed_datasets.items() if v is not None}
    
    # 1. Focused comparison plot (full series)
    print("\n📊 Creating focused comparison plot (full series)...")
    plot_imputation_comparison_focused(
        complete_data=series_complete,
        missing_data=series_masked,
        imputed_datasets_dict=imputed_datasets_clean,
        target_col=None,  # It's a Series, not DataFrame
        time_window=None,  # Full series
        missing_indices=missing_indices,
        fname='imputation_focused_full.png'
    )
    
    # 2. Focused comparison plot (zoomed window)
    print("\n📊 Creating focused comparison plot (zoomed window)...")
    # Find a window with missing values
    if len(missing_indices) > 0:
        # Center around first missing value
        center = missing_indices[0]
        window_size = 200
        start = max(0, center - window_size // 2)
        end = min(len(series_complete), start + window_size)
        
        plot_imputation_comparison_focused(
            complete_data=series_complete,
            missing_data=series_masked,
            imputed_datasets_dict=imputed_datasets_clean,
            target_col=None,
            time_window=(start, end),
            missing_indices=missing_indices,
            fname='imputation_focused_zoom.png'
        )
    
    # 3. Grid comparison plot
    print("\n📊 Creating grid comparison plot...")
    plot_imputation_comparison_grid(
        complete_data=series_complete,
        missing_data=series_masked,
        imputed_datasets_dict=imputed_datasets_clean,
        target_col=None,
        time_window=(start, end) if len(missing_indices) > 0 else None,
        missing_indices=missing_indices,
        fname='imputation_grid.png'
    )
    
    # 4. Error comparison plot (first 30 missing values for clarity)
    print("\n📊 Creating error comparison plot...")
    n_errors_to_show = min(30, len(missing_indices))
    plot_error_comparison(
        complete_data=series_complete,
        imputed_datasets_dict=imputed_datasets_clean,
        target_col=None,
        missing_indices=missing_indices[:n_errors_to_show],
        fname='imputation_errors.png'
    )
    
    print("\n" + "="*80)
    print("✅ All visualizations generated!")
    print("="*80)
    print("\nGenerated files:")
    print("  - imputation_results.csv")
    print("  - imputation_focused_full.png")
    print("  - imputation_focused_zoom.png")
    print("  - imputation_grid.png")
    print("  - imputation_errors.png")
    
    print("\n" + "="*80)
    print("COMPLETED!")
    print("="*80)
    print(f"\n💾 Results saved to: imputation_results.csv")
    
    print("\n" + "="*80)
    print("COMPLETED!")
    print("="*80)
import os
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor

def load_data(directory, dates):
    data = {}
    for date in dates:
        file_path = os.path.join(directory, str(date))
        data[date] = pd.read_csv(file_path, index_col=0, header=None)
    return data

def check_dirs_vectorized(values, val_1030, prediction_val):
    values = np.array(values)
    conditions = [
        values == 0,
        (values < val_1030) & (prediction_val == -1),
        (values > val_1030) & (prediction_val == 1)
    ]
    return np.select(conditions, [0, 1, 1], default=-1)

def generate_dates(directory):
    return set(os.listdir(directory))

def calculate_cumulative_return(values, val_1030):
    returns = (values / val_1030) - 1
    return np.cumprod(1 + returns) - 1

def process_date(date, predictions, data, usable_stocks):
    indices = predictions.index.intersection(data.index).intersection(usable_stocks.index)
    predictions = predictions.loc[indices].squeeze()

    top_500 = predictions.nlargest(500)
    metrics = {
        'p_gt_0': np.zeros(17),
        'p_gt_avg': np.zeros(17),
        'expected_diff': np.zeros(17),
        'expected_diff_std': np.zeros(17)
    }

    returns_matrix = []
    for stock in indices:
        vals = data.loc[stock].values
        if len(vals) < 36 or np.isnan(vals[3]):
            continue
        val_1030 = vals[3]
        future_vals = vals[16:33]
        returns = calculate_cumulative_return(future_vals, val_1030)
        returns_matrix.append(returns)

    if not returns_matrix:
        return date, metrics

    returns_matrix = np.array(returns_matrix)
    mean_returns = np.mean(returns_matrix, axis=0)
    std_returns = np.std(returns_matrix, axis=0)

    top_returns = []
    for stock in top_500.index:
        vals = data.loc[stock].values
        if len(vals) < 32 or np.isnan(vals[3]):
            continue
        val_1030 = vals[3]
        future_vals = vals[16:33]
        returns = calculate_cumulative_return(future_vals, val_1030)
        top_returns.append(returns)

    top_returns = np.array(top_returns)
    if len(top_returns) == 0:
        return date, metrics

    metrics['p_gt_0'] = np.mean(top_returns > 0, axis=0)
    metrics['p_gt_avg'] = np.mean(top_returns > mean_returns, axis=0)
    metrics['expected_diff'] = np.mean(top_returns - mean_returns, axis=0)
    metrics['expected_diff_std'] = np.mean((top_returns - mean_returns) / (std_returns + 1e-8), axis=0)

    return date, metrics

def main():
    predictions_directory = 'a'
    data_directory = 'b'
    stock_directory = 'c'
    grouping_directory = 'd'

    all_dates = sorted(list(generate_dates(grouping_directory) & generate_dates(predictions_directory)))

    all_predictions = load_data(predictions_directory, all_dates)
    all_data = load_data(data_directory, all_dates)
    all_usable_stocks = load_data(stock_directory, all_dates)

    result_dict = {
        'p_gt_0': {},
        'p_gt_avg': {},
        'expected_diff': {},
        'expected_diff_std': {},
    }

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_date,
                date,
                all_predictions[date],
                all_data[date],
                all_usable_stocks[date],
            )
            for date in all_dates
        ]

        for future in futures:
            date, metrics = future.result()
            for key in result_dict:
                result_dict[key][date] = metrics[key]

    for key in result_dict:
        df = pd.DataFrame.from_dict(result_dict[key], orient='index')
        df.columns = [f'T+{i}' for i in range(16, 33)]
        df.index.name = 'Date'
        df.to_csv(f'{key}.csv')

if __name__ == '__main__':
    main()
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.ad import KMeansScorer, ThresholdDetector
from loguru import logger
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express
plt_colors = plotly.express.colors.qualitative.D3


# df = pd.read_csv('data/2023-04-18.csv', parse_dates=['time'])

def anomaly_filtering(var_id:str, df:pd.DataFrame = None, threshold:float=None, 
                      visualize_result:bool= False, 
                      df_train:pd.DataFrame = None, df_validation:pd.DataFrame = None) -> np.ndarray:

    """
    Anomaly detection and filtering using darts library

    Parameters
    ----------
    df
    var_id
    threshold
    visualize_result

    Returns
    -------

    """

    if df is None and (df_train is None or df_validation is None):
        raise ValueError('Either df or df_train and df_validation must be provided')
    
    if df is not None:
        
        original_length = len(df)
        logger.debug(f'Pre-processing signal {var_id} with {original_length} samples')

        # Resample to exactly one second
        try:
            df = df.resample('1S', on='time').mean()
        except KeyError:
            df = df.resample('1S').mean()

        # Filter nan with ffill
        df = df.ffill()

        logger.debug(f'Pre-processing signal {var_id} with {len(df)} samples after resampling and ffilling nan values')


        series = TimeSeries.from_dataframe(df, value_cols=var_id, fill_missing_dates=True, freq='1000L')

        # Split the series into training and validation parts
        train, val = series.split_before(0.05)
        
    else:
        
        logger.debug(f'Processing signal {var_id} with {len(df_train)} samples for training and {len(df_validation)} samples for validation')
        
        # Resample to exactly one second
        try:
            df_train = df_train.resample('1S', on='time').mean()
            df_validation = df_validation.resample('1S', on='time').mean()
        except KeyError:
            df_train = df_train.resample('1S').mean()
            df_validation = df_validation.resample('1S').mean()
            
        # Filter nan with ffill
        df_train = df_train.ffill()
        df_validation = df_validation.ffill()
        
        logger.debug(f'Pre-processing signal {var_id} with {len(df_train)} samples for training and {len(df_validation)} samples for validation after resampling and ffilling nan values')
        
        train = TimeSeries.from_dataframe(df_train, value_cols=var_id, fill_missing_dates=True, freq='1000L')
        val = TimeSeries.from_dataframe(df_validation, value_cols=var_id, fill_missing_dates=True, freq='1000L')

    # Compute anomaly score
    scorer = KMeansScorer(k=2, window=5)
    scorer.fit(train)
    anom_score = scorer.score(val)

    # Detect anomalies
    # binary_anom = anom_score > anom_score..pd_series().mean()*1.5
    if not threshold:
        threshold = anom_score.pd_series().mean() * 10
    detector = ThresholdDetector(high_threshold=threshold)
    # detector.fit(scorer.score(train)) # if QuantileDetector is used
    binary_anom = detector.detect(anom_score)
    # Prepend zeros to the binary anomaly series to make it the same length as the original series
    binary_anom = binary_anom.prepend_values(np.zeros(len(series) - len(binary_anom)))

    # Create a new column with the filtered values, copying the original values
    # making them NaN where the anomaly is detected, and then filling the NaNs
    filtered_signal = df[var_id].copy()
    detected_anom = binary_anom.pd_series().values
    # Convert to boolean
    detected_anom = detected_anom.astype(bool)

    filtered_signal[detected_anom] = np.nan
    filtered_signal.ffill(inplace=True)

    if visualize_result:
        # Create a plotly figure with two subplots
        # In the first subplot, plot the original signal and the filtered signal
        # In the second subplot, plot the anomaly score and the anomaly detection threshold
        # and highlight the detected anomalies using detected_anom with a vertical area

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)

        # First subplot
        fig.add_trace(
            go.Scatter(x=df.index, y=df[var_id], name='Original signal', line=dict(color=plt_colors[7])),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=filtered_signal, name='Filtered signal', line=dict(color=plt_colors[2])),
            row=1, col=1
        )

        # Second subplot
        fig.add_trace(
            go.Scatter(
                x=anom_score.pd_series().index, y=anom_score.pd_series().values,
                name='Anomaly score',
                line=dict(color=plt_colors[0])
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=anom_score.pd_series().index,
                y=np.ones(len(anom_score.pd_series())) * threshold,
                name='Anomaly threshold',
                line=dict(color=plt_colors[3], width=4)
            ),
            row=2, col=1
        )

        fig.update_layout(
            title=f'Anomaly detection for {var_id}',
            xaxis2_title='Time',
            yaxis_title='Signal values',
            yaxis2_title='Anomaly values'
        )
        fig.show()

    logger.info(f'Pre-processed signal {var_id}, threshold {threshold:.2f} and {detected_anom.sum()} anomalies detected')

    df[var_id] = filtered_signal.to_numpy()

    return df

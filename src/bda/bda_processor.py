import numpy as np
import traceback

from pyspark.sql.types import StructType, StructField, DoubleType, ArrayType, IntegerType
from pyspark.sql.functions import pandas_udf, PandasUDFType

from .bda_core import calculate_numerical_derivates, calculate_phase_rate


def assign_temporal_window(df, x):
    """
    Assign temporal windows based on accumulated x value.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Input DataFrame with necessary columns.
    x : float
        Threshold x value for window assignment.

    Returns
    -------
    pyspark.sql.DataFrame
        DataFrame with assigned window IDs and accumulated x values.
    """
    try:    
        schema = StructType(df.schema.fields + [
            StructField('window_id', DoubleType(), True),
            StructField('x_accumulated', DoubleType(), True)
        ])

        @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
        def assign_windows(pdf):
            pdf = pdf.sort_values(by='time').reset_index(drop=True)

            n = len(pdf)
            window_ids = np.zeros(n, dtype=np.float64)
            x_accumulated = np.zeros(n, dtype=np.float64)

            current_window = 0
            x_acc = 0.0

            for i in range(n):
                # Calculate effective integration time
                dt_eff = pdf.loc[i, 'exposure'] if pdf.loc[i, 'exposure'] > 0 else pdf.loc[i, 'interval']

                if dt_eff <= 0:
                    dt_eff = 0.0

                phi_dot = pdf.loc[i, 'phi_dot']
                x_inc = abs(phi_dot) * dt_eff / 2.0

                # Check if should close current window
                if (x_acc + x_inc) > x: 
                    current_window += 1
                    x_acc = x_inc # Reset accumulated x for new window
                else: 
                    x_acc += x_inc # Continue accumulating in current window

                window_ids[i] = current_window
                x_accumulated[i] = x_acc
            
            pdf['window_id'] = window_ids
            pdf['x_accumulated'] = x_accumulated

            return pdf

        df_windowed = df.groupBy('baseline_key','scan_number').apply(assign_windows)

        return df_windowed

    except Exception as e:
        print(f"Error assigning temporal window: {e}")
        traceback.print_exc()
        raise


def average_by_window(df):
    """
    Average the DataFrame by window ID.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Input DataFrame with necessary columns.

    Returns
    -------
    pyspark.sql.DataFrame
        DataFrame with averaged values by window ID.
    """
    try:
        schema =  StructType([
            StructField('subms_id', DoubleType(), True),
            StructField('field_id', DoubleType(), True),
            StructField('spw_id', DoubleType(), True),
            StructField('polarization_id', DoubleType(), True),
            StructField('n_channels', DoubleType(), True),
            StructField('n_correlations', DoubleType(), True),
            StructField('antenna1', DoubleType(), True),
            StructField('antenna2', DoubleType(), True),
            StructField('baseline_key', DoubleType(), True),
            StructField('scan_number', DoubleType(), True),
            StructField('window_id', DoubleType(), True),
            StructField('time', DoubleType(), True),
            StructField('interval', DoubleType(), True),
            StructField('exposure', DoubleType(), True),
            StructField('u', DoubleType(), True),
            StructField('v', DoubleType(), True),
            StructField('visibilities', ArrayType(ArrayType(ArrayType(DoubleType()))), True),
            StructField('weight', ArrayType(DoubleType()), True),
            StructField('flag', ArrayType(ArrayType(IntegerType())), True)
        ])

        @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
        def average_visibilities(pdf):
            try:
                bda_avg = {
                    'subms_id': pdf['subms_id'].iloc[0],
                    'field_id': pdf['field_id'].iloc[0],
                    'spw_id': pdf['spw_id'].iloc[0],
                    'polarization_id': pdf['polarization_id'].iloc[0],
                    'n_channels': pdf['n_channels'].iloc[0],
                    'n_correlations': pdf['n_correlations'].iloc[0],
                    'antenna1': pdf['antenna1'].iloc[0],
                    'antenna2': pdf['antenna2'].iloc[0],
                    'baseline_key': pdf['baseline_key'].iloc[0],
                    'scan_number': pdf['scan_number'].iloc[0],
                    'window_id': pdf['window_id'].iloc[0],
                    'time': pdf['time'].iloc[0],
                    'interval': pdf['interval'].sum(),
                    'exposure': pdf['exposure'].sum(),
                }

                bda_avg['u'] = (pdf['u'] * pdf['exposure']).sum() / bda_avg['exposure']
                bda_avg['v'] = (pdf['v'] * pdf['exposure']).sum() / bda_avg['exposure']

                n = len(pdf)
                vs_list, ws_list, fs_list = [], [], []

                for i in range(n):
                    vs_data = pdf.iloc[i]['visibilities']
                    ws_data = pdf.loc[i, 'weight']
                    fs_data = pdf.loc[i, 'flag']

                    visibilities = np.array([[corr for corr in chan] for chan in vs_data], dtype=np.float64)
                    weights = np.array([w for w in ws_data], dtype=np.float64)
                    flags = np.array([[f for f in chan] for chan in fs_data], dtype=np.bool_)

                    """ if bda_avg['baseline_key'] == "0-1":
                        print(f"[DEBUG][VISIBILITIES SHAPE]: {visibilities.shape}")
                        print(f"[DEBUG][VISIBILITIES]: {visibilities}")

                        print(f"[DEBUG][WEIGHTS SHAPE]: {weights.shape}")
                        print(f"[DEBUG][WEIGHTS]: {weights}")

                        print(f"[DEBUG][FLAGS SHAPE]: {flags.shape}")
                        print(f"[DEBUG][FLAGS]: {flags}") """

                    vs_list.append(visibilities)
                    ws_list.append(weights)
                    fs_list.append(flags)
                
                vs = np.stack(vs_list, axis=0)
                ws = np.stack(ws_list, axis=0)
                fs = np.stack(fs_list, axis=0)

                N, C, P, _ = vs.shape
                valid_mask = ~fs.astype(bool)

                ws_broadcast = ws[:, None, :]
                ws_broadcast = np.broadcast_to(ws_broadcast, (N, C, P))
                ws_valid = np.where(valid_mask, ws_broadcast, 0.0)
                ws_sum = np.sum(ws_valid, axis=0)

                vs_real = (vs[..., 0] * ws_valid).sum(axis=0)
                vs_imag = (vs[..., 1] * ws_valid).sum(axis=0)

                with np.errstate(divide='ignore', invalid='ignore'):
                    vs_avg_real = np.where(ws_sum > 0, vs_real / ws_sum, 0.0)
                    vs_avg_imag = np.where(ws_sum > 0, vs_imag / ws_sum, 0.0)
                
                valid_count = (ws_valid > 0).sum(axis=0)

                vs_avg = np.stack([vs_avg_real, vs_avg_imag], axis=-1)
                ws_avg = np.where(valid_count > 0, ws_sum / valid_count, 0.0).mean()
                fs_avg = (ws_sum == 0).astype(np.int32)

                bda_avg['visibilities'] = vs_avg.tolist()
                bda_avg['weight'] = ws_avg.tolist()
                bda_avg['flag'] = fs_avg.tolist()

                return bda_avg

            except Exception as e:
                print(f"Error averaging visibilities: {e}")
                traceback.print_exc()
                raise

        # Apply average to each group
        df_avg = df.groupBy('baseline_key', 'scan_number', 'window_id').apply(average_visibilities)

        return df_avg

    except Exception as e:
        print(f"Error averaging by window: {e}")
        traceback.print_exc()
        raise


def process_rows(df, bda_config):
    """
    Process rows for baseline-dependent averaging (BDA).

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Input DataFrame with necessary columns.
    bda_config : dict
        Configuration dictionary with BDA parameters.
    
    Returns
    -------
    pyspark.sql.DataFrame
        Processed DataFrame after applying BDA.
    """
    try:
        fov = bda_config.get('fov', 1.0)
        x = bda_config.get('x', 0.0)

        print(f"[DEBUG] Initial df count: {df.count()}")
        df = calculate_numerical_derivates(df)

        print(f"[DEBUG] df count after derivates: {df.count()}")
        df = calculate_phase_rate(df, fov)

        print(f"[DEBUG] df count after phase rate: {df.count()}")
        df_windowed = assign_temporal_window(df, x)

        print(f"[DEBUG] df count after window assignment: {df_windowed.count()}")
        df_avg = average_by_window(df_windowed)

        print(f"[DEBUG] df count after averaging: {df_avg.count()}")

        return df_avg

    except Exception as e:
        print(f"Error processing rows for bda: {e}")
        traceback.print_exc()
        raise

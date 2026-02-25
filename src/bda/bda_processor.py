import numpy as np
import traceback
import pandas as pd

from pyspark.sql.types import StructType, StructField, DoubleType, ArrayType, IntegerType, StringType
from pyspark.sql.functions import pandas_udf, PandasUDFType, col

from .bda_core import calculate_phase_difference, calculate_uv_distance, sinc


def assign_temporal_window(df, decorr_factor, fov, lambda_ref):
    """
    Assign temporal windows based on accumulated x value.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Input DataFrame with necessary columns.
    decorr_factor : float
        Decorrelation factor threshold.
    fov : float
        Field of view parameter.

    Returns
    -------
    pyspark.sql.DataFrame
        DataFrame with assigned window IDs and accumulated x values.
    """
    try:    
        schema = StructType(df.schema.fields + [
            StructField('window_id', IntegerType(), True),
            StructField('d_uv', DoubleType(), True),
            StructField('phi_dot', DoubleType(), True),
            StructField('sinc_value', DoubleType(), True),
        ])

        @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
        def assign_windows(pdf):
            pdf = pdf.sort_values(by='time').reset_index(drop=True)
            n = len(pdf)

            window_ids = np.zeros(n, dtype=np.int32)
            d_uv_arr = np.zeros(n, dtype=np.float64)
            phi_dot_arr = np.zeros(n, dtype=np.float64)
            sinc_arr = np.zeros(n, dtype=np.float64)

            current_window = 1
            window_ids[0] = current_window

            u_ref = pdf.loc[0, 'u']
            v_ref = pdf.loc[0, 'v']

            d_uv_arr[0] = np.nan
            phi_dot_arr[0] = np.nan
            sinc_arr[0] = np.nan

            for i in range(1, n):
                u = pdf.loc[i, 'u']
                v = pdf.loc[i, 'v']

                d_uv = calculate_uv_distance(u, v, u_ref, v_ref, lambda_ref)
                x_inc = calculate_phase_difference(d_uv, fov)
                
                # Calculate decorrelation term
                # sinc(ΔΦ/2)
                sinc_val = sinc(x_inc / 2.0)

                # Window assignment based on decorrelation factor
                if sinc_val >= decorr_factor:
                    window_ids[i] = current_window
                else:
                    current_window += 1
                    window_ids[i] = current_window

                    u_ref, v_ref = u, v

                d_uv_arr[i] = d_uv
                phi_dot_arr[i] = x_inc
                sinc_arr[i] = sinc_val

            pdf['window_id'] = window_ids
            pdf['d_uv'] = d_uv_arr
            pdf['phi_dot'] = phi_dot_arr
            pdf['sinc_value'] = sinc_arr

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
            StructField('subms_id', IntegerType(), True),
            StructField('field_id', IntegerType(), True),
            StructField('spw_id', IntegerType(), True),
            StructField('polarization_id', IntegerType(), True),

            StructField('n_channels', IntegerType(), True),
            StructField('n_correlations', IntegerType(), True),

            StructField('antenna1', IntegerType(), True),
            StructField('antenna2', IntegerType(), True),
            StructField('baseline_key', StringType(), True),
            StructField('scan_number', IntegerType(), True),
            StructField('window_id', IntegerType(), True),

            StructField('exposure', DoubleType(), True),
            StructField('interval', DoubleType(), True),
            StructField('time', DoubleType(), True),

            StructField('u', DoubleType(), True),
            StructField('v', DoubleType(), True),
            
            StructField('visibility', ArrayType(ArrayType(ArrayType(DoubleType()))), True),
            StructField('weight', (ArrayType(ArrayType(DoubleType()))), True),
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
                    vs_data = pdf.iloc[i]['visibility']
                    ws_data = pdf.iloc[i]['weight']
                    fs_data = pdf.iloc[i]['flag']

                    visibilities = np.array([[corr for corr in chan] for chan in vs_data], dtype=np.float64)
                    weights = np.array([[w for w in chan] for chan in ws_data], dtype=np.float64)
                    flags = np.array([[f for f in chan] for chan in fs_data], dtype=np.bool_)

                    vs_list.append(visibilities)
                    ws_list.append(weights)
                    fs_list.append(flags)
                
                vs = np.stack(vs_list, axis=0) # Shape: (N, C, P, 2)
                ws = np.stack(ws_list, axis=0) # Shape: (N, C, P)
                fs = np.stack(fs_list, axis=0) # Shape: (N, C, P)

                N, C, P, _ = vs.shape
                valid_mask = ~fs

                ws_valid = np.where(valid_mask, ws, 0.0)
                ws_sum = ws_valid.sum(axis=0)

                vs_real = (vs[..., 0] * ws_valid).sum(axis=0)
                vs_imag = (vs[..., 1] * ws_valid).sum(axis=0)

                with np.errstate(divide='ignore', invalid='ignore'):
                    real_avg = np.where(ws_sum > 0, vs_real / ws_sum, 0.0)
                    imag_avg = np.where(ws_sum > 0, vs_imag / ws_sum, 0.0)

                vs_avg = np.stack([real_avg, imag_avg], axis=-1)
                ws_avg = ws_sum / N
                fs_avg = (ws_sum == 0).astype(np.int32)

                bda_avg['visibility'] = vs_avg.tolist()
                bda_avg['weight'] = ws_avg.tolist()
                bda_avg['flag'] = fs_avg.tolist()

                return pd.DataFrame([bda_avg])

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


def process_rows(df_scientific, bda_config):
    """
    Process rows for baseline-dependent averaging (BDA).

    Parameters
    ----------
    df_scientific : pyspark.sql.DataFrame
        Input DataFrame with necessary columns.
    bda_config : dict
        Configuration dictionary with BDA parameters.
    
    Returns
    -------
    tuple of pyspark.sql.DataFrame
        Processed DataFrames after applying BDA.
    """
    try:
        decorr_factor = bda_config.get('decorr_factor', 0.95)
        lambda_ref = bda_config.get('lambda_ref', 0.1)
        fov = bda_config.get('fov', 0.01)

        df_windowed = assign_temporal_window(df_scientific, decorr_factor, fov, lambda_ref)
        
        df_averaged = average_by_window(df_windowed)

        return df_averaged, df_windowed

    except Exception as e:
        print(f"Error processing rows for bda: {e}")
        traceback.print_exc()
        raise

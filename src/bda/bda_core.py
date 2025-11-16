import numpy as np
import traceback

from pyspark.sql.functions import lit, sqrt, col, lag, lead, row_number, count, when
from pyspark.sql.window import Window


def calculate_amplitude_loss(x):
    try:
        x = float(np.float64(x))
        sinc_value = np.sinc(x / np.pi)

        return float(1.0 - sinc_value)

    except Exception as e:
        print(f"Error calculating amplitude loss: {e}")
        traceback.print_exc()
        raise


def calculate_loss_exact(decorr_limit):
    try:
        x = float(np.float64(decorr_limit))

        x_min, x_max = 0.0, np.pi
        f_min = calculate_amplitude_loss(x_min) - x
        f_max = calculate_amplitude_loss(x_max) - x

        if not (f_min * f_max < 0):
            return None
        
        while True:
            x_mid = (x_min + x_max) / 2.0
            f_mid = calculate_amplitude_loss(x_mid) - x

            if f_mid == 0.0:
                return x_mid
            
            ulp_space = np.spacing(x_mid)
            if (x_max - x_min) <= ulp_space:
                return x_mid

            if f_min * f_mid < 0:
                x_max = x_mid
                f_max = f_mid
            else:
                x_min = x_mid
                f_min = f_mid

    except Exception as e:
        print(f"Error calculating loss exact: {e}")
        traceback.print_exc()
        raise


def calculate_threshold_loss(x_exact):
    try:
        x_arr = np.array(x_exact, dtype=np.float64)
        
        return np.nextafter(x_arr, np.inf)

    except Exception as e:
        print(f"Error calculating threshold loss: {e}")
        traceback.print_exc()
        raise


def calculate_numerical_derivates(df):
    """
    Calculate numerical derivatives for the input DataFrame.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Input DataFrame with necessary columns.

    Returns
    -------
    pyspark.sql.DataFrame
        DataFrame with calculated numerical derivatives.
    """
    MJD_TO_SECONDS = 86400.0

    try:
        w = Window.partitionBy('baseline_key', 'scan_number').orderBy('time')

        # Prepare first lag/lead columns
        df = df \
            .withColumn('u_prev', lag('u', 1).over(w)) \
            .withColumn('u_next', lead('u', 1).over(w)) \
            .withColumn('v_prev', lag('v', 1).over(w)) \
            .withColumn('v_next', lead('v', 1).over(w)) \
            .withColumn('time_prev', lag('time', 1).over(w)) \
            .withColumn('time_next', lead('time', 1).over(w)
        )

        # Prepare second lag/lead columns
        df = df \
            .withColumn('u_prev_2', lag('u', 2).over(w)) \
            .withColumn('u_next_2', lead('u', 2).over(w)) \
            .withColumn('v_prev_2', lag('v', 2).over(w)) \
            .withColumn('v_next_2', lead('v', 2).over(w)) \
            .withColumn('time_prev_2', lag('time', 2).over(w)) \
            .withColumn('time_next_2', lead('time', 2).over(w)
        )
        
        # Add row number and total rows for boundary conditions
        df = df \
            .withColumn('row_num', row_number().over(w)) \
            .withColumn('total_rows', count('*').over(w)
        )

        # Centered difference
        centered_du = (col('u_next') - col('u_prev')) / ((col('time_next') - col('time_prev')) * lit(MJD_TO_SECONDS))
        centered_dv = (col('v_next') - col('v_prev')) / ((col('time_next') - col('time_prev')) * lit(MJD_TO_SECONDS))

        # Forward difference
        left_du = (-3.0 * col('u') + 4.0 * col('u_next') - col('u_next_2')) / ((col('time_next_2') - col('time')) * lit(MJD_TO_SECONDS))
        left_dv = (-3.0 * col('v') + 4.0 * col('v_next') - col('v_next_2')) / ((col('time_next_2') - col('time')) * lit(MJD_TO_SECONDS))

        # Backward difference
        right_du = (3.0 * col('u') - 4.0 * col('u_prev') + col('u_prev_2')) / ((col('time') - col('time_prev_2')) * lit(MJD_TO_SECONDS))
        right_dv = (3.0 * col('v') - 4.0 * col('v_prev') + col('v_prev_2')) / ((col('time') - col('time_prev_2')) * lit(MJD_TO_SECONDS))

        # Assign derivatives based on row position
        # First row uses forward difference, last row uses backward difference, others use centered difference
        df = df.withColumn('du_dt',
            when((col('row_num') == 1), left_du)
            .when((col('row_num') == col('total_rows')), right_du)
            .otherwise(centered_du)
        )
        
        df = df.withColumn('dv_dt',
            when((col('row_num') == 1), left_dv)
            .when((col('row_num') == col('total_rows')), right_dv)
            .otherwise(centered_dv)
        )

        drop_cols = [
            'u_prev', 'u_next', 'v_prev', 'v_next', 'time_prev', 'time_next',
            'u_prev_2', 'u_next_2', 'v_prev_2', 'v_next_2', 'time_prev_2', 'time_next_2',
            'row_num', 'total_rows'
        ]

        df = df.drop(*drop_cols)

        return df

    except Exception as e:
        print(f"Error calculating numerical derivatives: {e}")
        traceback.print_exc()
        raise


def calculate_phase_rate(df, fov):
    """
    Calculate phase rate for the input DataFrame.
    
    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Input DataFrame with necessary columns.
    fov : float
        Field of view parameter.
    
    Returns
    -------
    pyspark.sql.DataFrame
        DataFrame with calculated phase rate.
    """
    try:
        df = df.withColumn('phi_dot',
            lit(2.0 * np.pi) * sqrt(col('du_dt') ** 2 + col('dv_dt') ** 2) * lit(fov)
        )

        return df

    except Exception as e:
        print(f"Error calculating phase rate: {e}")
        traceback.print_exc()
        raise

import traceback
import numpy as np
import matplotlib.pyplot as plt

from pyspark.sql import functions as F

def coverage_uv(df_scientific, df_averaging):
    try:
        df_uv_scientific = df_scientific.select(
            F.col('u').alias('u_scientific'),
            F.col('v').alias('v_scientific'),
            F.lit(None).cast('double').alias('u_averaged'),
            F.lit(None).cast('double').alias('v_averaged')
        )

        df_uv_averaging = df_averaging.select(
            F.lit(None).cast('double').alias('u_scientific'),
            F.lit(None).cast('double').alias('v_scientific'),
            F.col('u').alias('u_averaged'),
            F.col('v').alias('v_averaged')
        )

        df_coverage_uv = df_uv_scientific.unionByName(df_uv_averaging)

        return df_coverage_uv

    except Exception as e:
        print(f"Error calculating coverage UV: {e}")
        traceback.print_exc()
        raise


def calculate_coverage_uv(df_coverage_uv, max_points=50000):
    try:
        total_count = df_coverage_uv.count()
        print(f"Total UV points: {total_count:,}")
        
        sample_fraction = min(1.0, max_points / total_count) if total_count > 0 else 1.0
        
        if sample_fraction < 1.0:
            print(f"Sampling {sample_fraction:.1%} of data ({max_points:,} points) for visualization...")
            df_sample = df_coverage_uv.sample(False, sample_fraction, seed=42)
        else:
            print("Using all data points for visualization...")
            df_sample = df_coverage_uv
            print("Using all data points for visualization...")
            df_sample = df_coverage_uv
        
        print("Preparing UV coordinates from scientific data...")
        uv_scientific = df_sample.select(
            (F.col('u_scientific') / 1000).alias('u_scientific'),
            (F.col('v_scientific') / 1000).alias('v_scientific')
        ).filter(
            F.col('u_scientific').isNotNull() & F.col('v_scientific').isNotNull()
        ).collect()
        
        uv_averaging = df_sample.select(
            (F.col('u_averaged') / 1000).alias('u_averaged'),
            (F.col('v_averaged') / 1000).alias('v_averaged')
        ).filter(
            F.col('u_averaged').isNotNull() & F.col('v_averaged').isNotNull()
        ).collect()

        u_scientific = np.array([row['u_scientific'] for row in uv_scientific if row['u_scientific'] is not None])
        v_scientific = np.array([row['v_scientific'] for row in uv_scientific if row['v_scientific'] is not None])

        u_averaging = np.array([row['u_averaged'] for row in uv_averaging if row['u_averaged'] is not None])
        v_averaging = np.array([row['v_averaged'] for row in uv_averaging if row['v_averaged'] is not None])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Original
        ax1.scatter(u_scientific, v_scientific, c='#1f77b4', s=0.5, alpha=0.6)
        ax1.scatter(-u_scientific, -v_scientific, c='#1f77b4', s=0.5, alpha=0.6)
        ax1.set_xlabel('u (km)')
        ax1.set_ylabel('v (km)')
        ax1.set_title('Original')
        ax1.grid(True, alpha=0.2)
        ax1.set_aspect('equal')
        
        # BDA
        ax2.scatter(u_averaging, v_averaging, c='#ff7f0e', s=0.5, alpha=0.6)
        ax2.scatter(-u_averaging, -v_averaging, c='#ff7f0e', s=0.5, alpha=0.6)
        ax2.set_xlabel('u (km)')
        ax2.set_ylabel('v (km)')
        ax2.set_title('BDA')
        ax2.grid(True, alpha=0.2)
        ax2.set_aspect('equal')
        
        plt.tight_layout()
        
        plt.savefig('uv_coverage.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.scatter(u_scientific, v_scientific, c='#1f77b4', s=0.5, alpha=0.3, label='Original')
        ax.scatter(-u_scientific, -v_scientific, c='#1f77b4', s=0.5, alpha=0.3)
        
        ax.scatter(u_averaging, v_averaging, c='#ff7f0e', s=0.5, alpha=0.5, label='BDA')
        ax.scatter(-u_averaging, -v_averaging, c='#ff7f0e', s=0.5, alpha=0.5)
        
        ax.set_xlabel('u (km)')
        ax.set_ylabel('v (km)')
        ax.grid(True, alpha=0.2)
        ax.set_aspect('equal')
        ax.legend(loc='upper right', markerscale=5)
        
        plt.tight_layout()
        
        plt.savefig('uv_coverage_overlay.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    except Exception as e:
        print(f"Error calculating coverage UV values: {e}")
        traceback.print_exc()
        raise
import traceback
import csv
import os
import numpy as np
import matplotlib.pyplot as plt

from pyspark.sql import functions as F

def coverage_uv(df_scientific, df_averaging):
    try:
        df_uv_scientific = df_scientific.select(
            (F.col('u') / 1000).alias('u_scientific'),
            (F.col('v') / 1000).alias('v_scientific')
        ).filter(
            F.col('u_scientific').isNotNull() & F.col('v_scientific').isNotNull()
        ).collect()

        df_uv_averaging = df_averaging.select(
            (F.col('u') / 1000).alias('u_averaged'),
            (F.col('v') / 1000).alias('v_averaged')
        ).filter(
            F.col('u_averaged').isNotNull() & F.col('v_averaged').isNotNull()
        ).collect()

        return df_uv_scientific, df_uv_averaging

    except Exception as e:
        print(f"Error calculating coverage UV: {e}")
        traceback.print_exc()
        raise


def calculate_coverage_uv(df_scientific, df_averaging, output_coverage):
    try:
        df_uv_scientific, df_uv_averaging = coverage_uv(df_scientific, df_averaging)

        u_scientific = np.array([row['u_scientific'] for row in df_uv_scientific if row['u_scientific'] is not None])
        v_scientific = np.array([row['v_scientific'] for row in df_uv_scientific if row['v_scientific'] is not None])

        u_averaging = np.array([row['u_averaged'] for row in df_uv_averaging if row['u_averaged'] is not None])
        v_averaging = np.array([row['v_averaged'] for row in df_uv_averaging if row['v_averaged'] is not None])

        print(f"[Evaluation] Length of scientific UV:  ({len(u_scientific), len(v_scientific)})")
        print(f"[Evaluation] Length of averaging UV:   ({len(u_averaging), len(v_averaging)})")

        output_coordinates_csv = os.path.splitext(output_coverage)[0] + '_coordinates.csv'
        with open(output_coordinates_csv, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['dataset', 'u_km', 'v_km'])
            writer.writerows(zip(['scientific'] * len(u_scientific), u_scientific, v_scientific))
            writer.writerows(zip(['averaged'] * len(u_averaging), u_averaging, v_averaging))

        print(f"[Evaluation] Coordinates exported to: {output_coordinates_csv}")

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
        
        plt.savefig(output_coverage, dpi=150, bbox_inches='tight')
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
        
        plt.savefig(output_coverage.replace('.png', '_overlay.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Zoom view: dynamic center region based on percentiles
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.scatter(u_scientific, v_scientific, c='#1f77b4', s=0.5, alpha=0.3, label='Original')
        ax.scatter(-u_scientific, -v_scientific, c='#1f77b4', s=0.5, alpha=0.3)
        
        ax.scatter(u_averaging, v_averaging, c='#ff7f0e', s=0.5, alpha=0.5, label='BDA')
        ax.scatter(-u_averaging, -v_averaging, c='#ff7f0e', s=0.5, alpha=0.5)
        
        u_min, u_max = np.percentile(np.concatenate([u_averaging, -u_averaging]), [25, 75])
        v_min, v_max = np.percentile(np.concatenate([v_averaging, -v_averaging]), [25, 75])
        
        ax.set_xlim(u_min, u_max)
        ax.set_ylim(v_min, v_max)
        ax.set_xlabel('u (km)')
        ax.set_ylabel('v (km)')
        ax.set_title('BDA - Zoom Center Region')
        ax.grid(True, alpha=0.2)
        ax.set_aspect('equal')
        ax.legend(loc='upper right', markerscale=5)
        
        plt.tight_layout()
        
        plt.savefig(output_coverage.replace('.png', '_zoom.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

        plt.close('all')

        print("[Evaluation] ✓ UV Coverage visualization completed successfully.")

    except Exception as e:
        print(f"Error calculating coverage UV values: {e}")
        traceback.print_exc()
        raise
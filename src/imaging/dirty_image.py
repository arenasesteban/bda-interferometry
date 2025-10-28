import numpy as np
import matplotlib.pyplot as plt
import traceback

from pyspark.sql.functions import col


def build_corr_map(corrs_string):
    if isinstance(corrs_string, list):
        if len(corrs_string) > 0 and isinstance(corrs_string[0], list):
            corrs = corrs_string[0]
        else:
            corrs = corrs_string
    else:
        corrs = [c.strip() for c in corrs_string.split(',')]
    return {idx: corr for idx, corr in enumerate(corrs)}


def dataframe_to_grid(gridded_df, grid_config):
    img_size = grid_config["img_size"]
    padding_factor = grid_config["padding_factor"]
    
    imsize = [img_size, img_size]
    u_size, v_size = int(imsize[0] * padding_factor), int(imsize[1] * padding_factor)

    corrs_names = build_corr_map(grid_config['corrs_string'])

    grids = np.zeros((u_size, v_size), dtype=np.complex128)

    try:
        for idx, name in corrs_names.items():
            if name == 'XX' or name == 'YY' or name == 'RR' or name == 'LL':
                df_corr = gridded_df.filter(col("corr") == idx)
                rows = df_corr.collect()

                for row in rows:
                    u = row.u_pix
                    v = row.v_pix
                    grids[u, v] += row.vs_real + 1j * row.vs_imag

        corrs_set = set(corrs_names.values())
        if corrs_set >= {'XX', 'YY'} or corrs_set >= {'RR', 'LL'}:
            return grids * 0.5
        
        return grids

    except Exception as e:
        print(f"Error during dataframe to grid conversion: {e}")
        traceback.print_exc()
        raise


def generate_dirty_image(gridded_df, grid_config):
    print("Dataframe to grid conversion...")
    grids = dataframe_to_grid(gridded_df, grid_config)

    print("Generating dirty image via IFFT...")
    fourier = np.fft.ifftshift(grids)
    dirty_image = np.fft.ifft2(fourier)
    dirty_image = np.fft.fftshift(np.abs(dirty_image))

    save_dirty_image(dirty_image)


def save_dirty_image(dirty_image):
    plt.figure(figsize=(8, 8))
    plt.imshow(dirty_image, cmap='hot', origin='lower')
    plt.colorbar(label='Intensity')
    plt.title('Dirty Image')
    plt.xlabel('X [pixels]')
    plt.ylabel('Y [pixels]')
    plt.tight_layout()
    plt.savefig('dirty_image.png')
    plt.close()

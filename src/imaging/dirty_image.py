import numpy as np
import matplotlib.pyplot as plt
import traceback


def build_corr_map(corr_string):
    corr_map = {}
    for idx, corr in enumerate(corr_string.split(',')):
        corr_map[idx] = corr.strip()
    return corr_map


def dataframe_to_grid(gridded_df, grid_config):
    img_size = grid_config["img_size"]
    padding_factor = grid_config["padding_factor"]
    
    imsize = [img_size, img_size]
    u_size, v_size = int(imsize[0] * padding_factor), int(imsize[1] * padding_factor)

    grids = {}
    weights = {}

    corr_names = build_corr_map(grid_config['corr_string'])

    for name in corr_names.values():
        grids[name] = np.zeros((u_size, v_size), dtype=np.complex128)
        weights[name] = np.zeros((u_size, v_size), dtype=np.float64)

    try:
        for idx, name in corr_names.items():
            df_corr = gridded_df.filter(gridded_df.corr == idx)
            rows = df_corr.collect()

            for row in rows:
                u = row.u_pix
                v = row.v_pix
                grids[name][u, v] += row.vs_real + 1j * row.vs_imag
                weights[name][u, v] += row.weights
        
        return grids

    except Exception as e:
        print(f"Error during dataframe to grid conversion: {e}")
        traceback.print_exc()
        raise


def compute_stokes_I(grids):
    if 'XX' in grids and 'YY' in grids:
        return (grids['XX'] + grids['YY']) * 0.5
    if 'RR' in grids and 'LL' in grids:
        return (grids['RR'] + grids['LL']) * 0.5

    for pref in ('XX', 'RR'):
        if pref in grids:
            return grids[pref]


def generate_dirty_image(gridded_df, grid_config):    
    grids, normalized_ws = dataframe_to_grid(gridded_df, grid_config)
    vs_grid = compute_stokes_I(grids)

    fourier = np.fft.ifftshift(vs_grid * normalized_ws)
    dirty_image = np.fft.ifft2(fourier)
    dirty_image = np.fft.fftshift(np.abs(dirty_image))

    save_dirty_image(dirty_image)


def save_dirty_image(dirty_image):
    plt.figure(figsize=(8, 8))
    plt.imshow(dirty_image, cmap='hot', origin='lower')
    plt.colorbar(label='Intensity')
    plt.title('Dirty Image - BDA')
    plt.xlabel('X [pixels]')
    plt.ylabel('Y [pixels]')
    plt.tight_layout()
    plt.savefig('dirty_image_bda.png')
    plt.close()

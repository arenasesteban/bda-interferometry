import numpy as np
import matplotlib.pyplot as plt


def dataframe_to_grid(gridded_df, grid_config):
    u_size = grid_config["u_size"]
    v_size = grid_config["v_size"]

    vs_grid = np.zeros((u_size, v_size), dtype=np.complex128)
    ws_grid = np.zeros((u_size, v_size), dtype=np.float64)

    rows = gridded_df.collect()
    for row in rows:
        u = row.u_idx
        v = row.v_idx
        vs_grid[u, v] += row.vs_real + 1j * row.vs_imag
        ws_grid[u, v] += row.weights

    return vs_grid, ws_grid


def generate_dirty_image(gridded_df, grid_config):    
    vs_grid, ws_grid = dataframe_to_grid(gridded_df, grid_config)
    
    weighted_vs = np.fft.ifftshift(vs_grid * ws_grid)

    dirty_image = np.fft.ifft2(weighted_vs)
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

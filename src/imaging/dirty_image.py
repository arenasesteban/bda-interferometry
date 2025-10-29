import numpy as np
import matplotlib.pyplot as plt
import traceback

from pyspark.sql.functions import col


def dataframe_to_grid(gridded_df, grid_config):
    img_size = grid_config["img_size"]
    padding_factor = grid_config["padding_factor"]
    
    imsize = [img_size, img_size]
    u_size, v_size = int(imsize[0] * padding_factor), int(imsize[1] * padding_factor)

    grids = np.zeros((u_size, v_size), dtype=np.complex128)
    weights = np.zeros((u_size, v_size), dtype=np.float64)

    rows = gridded_df.collect()

    for row in rows:
        u = row.u_pix
        v = row.v_pix
        grids[u, v] += row.vs_real + 1j * row.vs_imag
        weights[u, v] += row.weights

    ws_sum = np.sum(weights)
    normalized_weights = weights / ws_sum if ws_sum > 0 else weights

    return grids * 0.5, normalized_weights


def generate_dirty_image(gridded_df, grid_config):
    print("Dataframe to grid conversion...")
    grids, normalized_weights = dataframe_to_grid(gridded_df, grid_config)

    img_size = grid_config["img_size"]
    padding_factor = grid_config["padding_factor"]
    imsize = [img_size, img_size]
    grid_size = [int(imsize[0] * padding_factor), int(imsize[1] * padding_factor)]

    print("Generating dirty image via IFFT...")
    fourier = np.fft.ifftshift(grids * normalized_weights)
    dirty_image = np.fft.ifft2(fourier, norm= 'forward')
    dirty_image = np.fft.fftshift(np.abs(dirty_image))

    dirty_image = dirty_image * grid_size[0] * grid_size[1]

    if padding_factor > 1.0:
        start_x = (grid_size[0] - img_size) // 2
        start_y = (grid_size[1] - img_size) // 2
        dirty_image = dirty_image[start_x:start_x + img_size, start_y:start_y + img_size]

    save_dirty_image(dirty_image)
    
    print("Generating PSF image via IFFT...")

    fourier_psf = np.fft.ifftshift(normalized_weights)
    psf_image = np.fft.ifft2(fourier_psf, norm= 'forward')
    psf_image = np.fft.fftshift(np.abs(psf_image))

    psf_image = psf_image * grid_size[0] * grid_size[1]

    save_psf_image(psf_image)


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


def save_psf_image(psf_image):
    plt.figure(figsize=(8, 8))
    plt.imshow(psf_image, cmap='hot', origin='lower')
    plt.colorbar(label='Intensity')
    plt.title('Point Spread Function (PSF)')
    plt.xlabel('X [pixels]')
    plt.ylabel('Y [pixels]')
    plt.tight_layout()
    plt.savefig('psf_image.png')
    plt.close()
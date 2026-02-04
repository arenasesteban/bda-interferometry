import numpy as np
import matplotlib.pyplot as plt
import cmcrameri


def dataframe_to_grid(df_gridded, grid_config):
    img_size = grid_config["img_size"]
    padding_factor = grid_config["padding_factor"]
    
    u_size, v_size = int(img_size * padding_factor), int(img_size * padding_factor)

    pdf = df_gridded.toPandas()

    u_coords, v_coords = pdf['u_pix'], pdf['v_pix']
    vs_real, vs_imag = pdf['vs_real'], pdf['vs_imag']
    weight = pdf['weight']

    grids = np.zeros((v_size, u_size), dtype=np.complex128)
    weights = np.zeros((v_size, u_size), dtype=np.float64)

    grids[v_coords, u_coords] = vs_real + 1j * vs_imag
    weights[v_coords, u_coords] = weight

    grids *= 0.5

    return grids, weights


def generate_dirty_image(df_gridded, grid_config, dirty_image_output, psf_output):
    grids, weights = dataframe_to_grid(df_gridded, grid_config)

    img_size = grid_config["img_size"]
    padding_factor = grid_config["padding_factor"]
    imsize = [img_size, img_size]
    grid_size = [int(imsize[0] * padding_factor), int(imsize[1] * padding_factor)]

    fourier = np.fft.ifftshift(grids * weights)
    dirty_image = np.fft.fft2(fourier, norm= 'forward')
    dirty_image = np.fft.fftshift(dirty_image).real

    dirty_image = dirty_image * grid_size[0] * grid_size[1]

    if padding_factor > 1.0:
        start_x = (grid_size[0] - img_size) // 2
        start_y = (grid_size[1] - img_size) // 2
        dirty_image = dirty_image[start_x:start_x + img_size, start_y:start_y + img_size]

    fourier_psf = np.fft.ifftshift(weights)
    psf_image = np.fft.fft2(fourier_psf, norm= 'forward')
    psf_image = np.fft.fftshift(np.abs(psf_image))

    psf_image = psf_image * grid_size[0] * grid_size[1]

    save_dirty_image(dirty_image, dirty_image_output)
    save_psf_image(psf_image, psf_output)


def save_dirty_image(dirty_image, output_file):
    plt.figure(figsize=(8, 8))
    plt.imshow(dirty_image, cmap='cmc.acton', origin='lower')
    plt.title('Dirty Image - BDA', fontdict={'fontsize': 16, 'fontweight': 'bold'})
    plt.xlabel('X [pixels]')
    plt.ylabel('Y [pixels]')
    plt.savefig(output_file)
    plt.close()


def save_psf_image(psf_image, output_file):
    plt.figure(figsize=(8, 8))
    plt.imshow(psf_image, cmap='cmc.acton', origin='lower')
    plt.colorbar(label='Intensity')
    plt.title('Point Spread Function (PSF) - BDA', fontdict={'fontsize': 16, 'fontweight': 'bold'})
    plt.xlabel('X [pixels]')
    plt.ylabel('Y [pixels]')
    plt.savefig(output_file)
    plt.close()
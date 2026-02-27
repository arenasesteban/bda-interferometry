import numpy as np
import matplotlib.pyplot as plt
import cmcrameri


def fft(grids, weights, grid_config, type="dirty"):
    img_size = grid_config["img_size"]
    padding_factor = grid_config["padding_factor"]
    imsize = [img_size, img_size]
    grid_size = [int(imsize[0] * padding_factor), int(imsize[1] * padding_factor)]

    if type == "dirty":
        fourier = np.fft.ifftshift(grids * weights)
        image = np.fft.fft2(fourier, norm= 'forward')
        image = np.fft.fftshift(image).real

    else:
        fourier = np.fft.ifftshift(weights)
        image = np.fft.fft2(fourier, norm= 'forward')
        image = np.fft.fftshift(np.abs(image))

    if type == "dirty" and padding_factor > 1.0:
        start_x = (grid_size[0] - img_size) // 2
        start_y = (grid_size[1] - img_size) // 2
        image = image[start_x:start_x + img_size, start_y:start_y + img_size]

    return image


def save_dirty_image(dirty_image, output_file):
    plt.figure(figsize=(8, 8))
    plt.imshow(dirty_image, cmap="cmc.acton", origin="lower")
    plt.title("Dirty Image", fontdict={"fontsize": 16, "fontweight": "bold"})
    plt.xlabel("X [pixels]")
    plt.ylabel("Y [pixels]")
    plt.savefig(output_file)
    plt.close()


def save_psf_image(psf_image, output_file):
    plt.figure(figsize=(8, 8))
    plt.imshow(psf_image, cmap="cmc.acton", origin="lower")
    plt.colorbar(label="Intensity")
    plt.title("Point Spread Function (PSF)", fontdict={"fontsize": 16, "fontweight": "bold"})
    plt.xlabel("X [pixels]")
    plt.ylabel("Y [pixels]")
    plt.savefig(output_file)
    plt.close()


def generate_dirty_image(grids, weights, grid_config, slurm_job_id):
    dirty_image = fft(grids, weights, grid_config, type="dirty")
    psf_image = fft(grids, weights, grid_config, type="psf")

    output_dirty_image = f"./output/dirtyimage_{slurm_job_id}.png"
    output_psf_image = f"./output/psf_{slurm_job_id}.png"

    save_dirty_image(dirty_image, output_dirty_image)
    save_psf_image(psf_image, output_psf_image)

    print(f"[Imaging] ✓ Dirty image saved to: {output_dirty_image}")
    print(f"[Imaging] ✓ PSF image saved to: {output_psf_image}")

    plt.close('all')

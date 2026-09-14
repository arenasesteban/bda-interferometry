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


def save_dirty_image(dirty_image, output_file, pixel_scale_arcsec=None, unit="Jy/beam"):
    dirty_image = np.asarray(dirty_image)
    ny, nx = dirty_image.shape

    fig, ax = plt.subplots(figsize=(6, 5))

    if pixel_scale_arcsec is None:
        # Ejes en píxeles
        im = ax.imshow(
            dirty_image,
            cmap="cmc.acton",
            origin="lower",
            aspect="equal"
        )
        ax.set_xlabel("X [pixels]")
        ax.set_ylabel("Y [pixels]")

    else:
        # Ejes en coordenadas angulares
        x_half = (nx * pixel_scale_arcsec) / 2
        y_half = (ny * pixel_scale_arcsec) / 2
        extent = [-x_half, x_half, -y_half, y_half]

        im = ax.imshow(
            dirty_image,
            cmap="cmc.acton",
            origin="lower",
            extent=extent,
            aspect="equal"
        )
        ax.set_xlabel(r"$\Delta \alpha$ [arcsec]")
        ax.set_ylabel(r"$\Delta \delta$ [arcsec]")

    cbar = fig.colorbar(im, ax=ax, pad=0.04)
    cbar.set_label(f"Flux Density [{unit}]")

    fig.tight_layout()
    fig.savefig(output_file, dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_psf_image(psf_image, output_file):
    plt.imshow(psf_image, cmap="cmc.acton", origin="lower", vmax=0.3)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_file, dpi=100)
    plt.close()


def generate_dirty_image(grids, weights, grid_config, slurm_job_id):
    dirty_image = fft(grids, weights, grid_config, type="dirty")
    psf_image = fft(grids, weights, grid_config, type="psf")

    output_dirty_image = f"./output/{slurm_job_id}/dirtyimage_{slurm_job_id}.png"
    output_psf_image = f"./output/{slurm_job_id}/psf_{slurm_job_id}.png"

    save_dirty_image(dirty_image, output_dirty_image)
    save_psf_image(psf_image, output_psf_image)

    print(f"[Imaging] ✓ Dirty image saved to: {output_dirty_image}")
    print(f"[Imaging] ✓ PSF image saved to: {output_psf_image}")

    plt.close('all')

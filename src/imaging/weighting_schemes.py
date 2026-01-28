import traceback
import numpy as np

def natural_weighting():
    def apply_natural_weighting(weight_original, weight_grid):
        return weight_original
    
    return apply_natural_weighting


def uniform_weighting():

    def apply_uniform_weighting(weight_original, weight_grid):
        epsilon = 1e-12
        return weight_original / max(weight_grid, epsilon)

    return apply_uniform_weighting


def apply_weighting(accumulated_grid, grid_config):
    try:
        weight_scheme = grid_config.get("weight_scheme", "NATURAL")

        if weight_scheme == "NATURAL":
            weight_fn = natural_weighting()
        elif weight_scheme == "UNIFORM":
            weight_fn = uniform_weighting()
        else:
            raise ValueError(f"Unknown weighting scheme: {weight_scheme}")

        weighted_grid = {}
        for key, values in accumulated_grid.items():
            weight_grid = np.sum(values['weights'])  

            for visibility, weight_original in zip(values['visibilities'], values['weights']):
                weighted = weight_fn(weight_original, weight_grid)

                if key not in weighted_grid:
                    weighted_grid[key] = {
                        'real': visibility.real * weighted,
                        'imag': visibility.imag * weighted,
                        'weight': weighted
                    }
                else:
                    weighted_grid[key]['real'] += visibility.real * weighted
                    weighted_grid[key]['imag'] += visibility.imag * weighted
                    weighted_grid[key]['weight'] += weighted

        return weighted_grid

    except Exception as e:
        print(f"Error applying weighting: {e}")
        traceback.print_exc()
        raise


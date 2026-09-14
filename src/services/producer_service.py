import sys
import traceback
import json
import argparse
import traceback
import astropy.units as u
from pathlib import Path

from data.simulation import generate_dataset
from data.extraction import stream_dataset


# Configuration constants
DEFAULT_TOPIC = 'visibility-stream'
STRATEGY = ["FIXED", "DERIVED"]


def load_simulation_config(config_path):
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load config file {config_path}: {e}")
            print("Using default configuration.")
    
    return {}


def update_bda_config(config_path, lambda_ref, min_diameter, offset):
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        fov = float(1.02 * lambda_ref / min_diameter)
        theta_fov = fov / 2
        theta_max = theta_fov * offset
        threshold = lambda_ref / (theta_fov * offset)
        
        config["lambda_ref"] = lambda_ref
        config["fov"] = fov
        config["theta_max"] = theta_max
        config["threshold"] = threshold

        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
            
    except Exception as e:
        print(f"Error updating BDA config: {e}")
        traceback.print_exc()
        raise


def update_grid_config(config_path, theo_resolution, corrs_string, chan_freq):
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        if config["cellsize_strategy"] == "DERIVED":
            config["cellsize"] = theo_resolution / 7

        elif config["cellsize_strategy"] == "FIXED":
            if config["cellsize_flag"] == True:
                config["cellsize"] = (config["cellsize"] * u.arcsec).to(u.rad).value
                config["cellsize_flag"] = False
        else:
            raise ValueError(f"Invalid cellsize_strategy. Supported strategies: {STRATEGY}")

        config["corrs_string"] = corrs_string
        config["chan_freq"] = chan_freq

        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
            
    except Exception as e:
        print(f"Error updating grid config: {e}")
        traceback.print_exc()
        raise


def stream_kafka(dataset, topic, bootstrap_servers, run_id):
    try:
        print(f"[Producer] Run ID: {run_id}")
        print(f"[Producer] Kafka: {bootstrap_servers}")
        print(f"[Producer] Topic: {topic}")
        print(f"[Producer] Starting streaming...")


        if not hasattr(dataset, "ms_list") or dataset.ms_list is None:
            raise ValueError("Dataset does not contain 'ms_list' or it is None.")

        for subms in dataset.ms_list:
            if subms is None or subms.visibilities is None:
                continue

            stream_dataset(subms.visibilities, subms, dataset.antenna, topic)

        print(f"[Producer] Finished streaming.")
    
    except Exception as e:
        print(f"[Producer] Unexpected error during streaming: {e}")
        traceback.print_exc()
        raise


def run_producer(
    topic, bootstrap_servers, run_id, 
    antenna_config_path, simulation_config_path, 
    bda_config_path, grid_config_path, offset
):
    try:
        sim_config = load_simulation_config(simulation_config_path)
        print("✓ Loaded simulation configuration.", flush=True)

        dataset = generate_dataset(antenna_config_path, sim_config)
        print("✓ Dataset generation complete.", flush=True)

        update_bda_config(
            config_path=bda_config_path,
            lambda_ref=dataset.spws.lambda_ref,
            min_diameter=dataset.antenna.min_diameter,
            offset=offset
        )
        print("✓ BDA configuration updated.", flush=True)

        update_grid_config(
            config_path=grid_config_path,
            theo_resolution=dataset.theo_resolution,
            corrs_string=dataset.polarization.corrs_string,
            chan_freq=dataset.spws.dataset[0].CHAN_FREQ.compute().values[0].tolist()
        )
        print("✓ Grid configuration updated.", flush=True)

        streaming_results = stream_kafka(dataset, topic, bootstrap_servers, run_id)
        print("✓ Streaming complete.", flush=True)

        return streaming_results

    except Exception as e:
        print(f"Error in producer service: {e}")
        traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser(
        description="BDA Interferometry Producer Service",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter    
    )
    
    parser.add_argument(
        "--topic",
        required=False,
        default=DEFAULT_TOPIC,
        help="Kafka topic to stream the dataset to."
    )
    parser.add_argument(
        "--bootstrap-servers",
        required=False,
        help="Kafka bootstrap server address."
    )
    parser.add_argument(
        "--run-id",
        required=False,
        help="Unique identifier for this run."
    )

    parser.add_argument(
        "--antenna-config",
        required=True, 
        help="Path to antenna configuration file"
    )
    parser.add_argument(
        "--simulation-config",
        required=True, 
        help="Path to simulation configuration file"
    )
    parser.add_argument(
        "--bda-config",
        required=True,
        help="Path to BDA configuration file"
    )
    parser.add_argument(
        "--grid-config",
        required=True,
        help="Path to grid configuration file"
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=0.01,
        help="Offset factor for Field of View objective size."
    )

    args = parser.parse_args()

    try:
        run_producer(
            topic=args.topic,
            bootstrap_servers=args.bootstrap_servers,
            run_id=args.run_id,
            antenna_config_path=args.antenna_config,
            simulation_config_path=args.simulation_config,
            bda_config_path=args.bda_config,
            grid_config_path=args.grid_config,
            offset=args.offset
        )
    
    except Exception as e:
        print(f"Fatal error in main: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
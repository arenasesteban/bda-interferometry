import sys
import traceback
import json
from pathlib import Path
import argparse
import traceback
from astropy.constants import c
import astropy.units as u

# Kafka imports
from kafka import KafkaProducer
from kafka.errors import KafkaError

# Configuration constants
DEFAULT_KAFKA_SERVERS = ['localhost:9092']
DEFAULT_TOPIC = 'visibility-stream'

# Add src directory to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.append(str(src_path))

from data.simulation import generate_dataset
from data.extraction import stream_dataset

# Supported padding strategies
PADDING_STRATEGY = ["FIXED", "DERIVED"]


def load_simulation_config(config_path):
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load config file {config_path}: {e}")
            print("Using default configuration.")
    
    return {}


def update_bda_config(config_path, ref_nu, min_diameter):
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    
        lambda_ = c.value / ref_nu
        config['fov'] = float(1.02 * lambda_ / min_diameter)

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
            
    except Exception as e:
        print(f"Error updating BDA config: {e}")
        traceback.print_exc()
        raise


def update_grid_config(config_path, theo_resolution, corrs_string, chan_freq):
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        if config.get('cellsize_strategy') not in PADDING_STRATEGY:
            raise ValueError(f"Invalid cellsize_strategy. Supported strategies: {PADDING_STRATEGY}")
        
        if config['cellsize_strategy'] == "DERIVED":
            config['cellsize'] = theo_resolution / 7
        elif config['cellsize_strategy'] == "FIXED" and config.get('cellsize_flag', True):
            config['cellsize'] = (config['cellsize'] * u.arcsec).to(u.rad).value
            config['cellsize_flag'] = False

        config['corrs_string'] = corrs_string
        config['chan_freq'] = chan_freq

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
            
    except Exception as e:
        print(f"Error updating grid config: {e}")
        traceback.print_exc()
        raise


def stream_kafka(dataset, topic):
    try:
        print(f"[Producer] Starting to stream dataset to Kafka topic '{topic}'...")
        if not hasattr(dataset, 'ms_list') or dataset.ms_list is None:
            raise ValueError("Dataset does not contain 'ms_list' or it is None.")

        for subms in dataset.ms_list:
            if subms is None or subms.visibilities is None:
                continue

            stream_dataset(subms.visibilities, subms, topic)

        print(f"[Producer] Finished streaming dataset to Kafka topic '{topic}'.")
    
    except Exception as e:
        print(f"[Producer] Unexpected error during streaming: {e}")
        traceback.print_exc()
        raise


def run_producer(antenna_config_path, simulation_config_path, topic):
    if topic is None:
        topic = DEFAULT_TOPIC
    
    try:
        sim_config = load_simulation_config(simulation_config_path)
        print("✓ Loaded simulation configuration.", flush=True)

        if "source_path" in sim_config:
            dataset = generate_dataset(
                antenna_config_path=antenna_config_path,
                freq_min=sim_config["freq_min"],
                freq_max=sim_config["freq_max"],
                n_chans=sim_config["n_chans"],
                observation_time=sim_config["observation_time"],
                declination=sim_config["declination"],
                integration_time=sim_config["integration_time"],
                spectral_index=sim_config["spectral_index"],
                source_path=sim_config["source_path"]
            )
        else:
            dataset = generate_dataset(
                antenna_config_path=antenna_config_path,
                freq_min=sim_config["freq_min"],
                freq_max=sim_config["freq_max"],
                n_chans=sim_config["n_chans"],
                observation_time=sim_config["observation_time"],
                declination=sim_config["declination"],
                integration_time=sim_config["integration_time"],
                date_string=sim_config["date_string"],
                flux_density=sim_config["flux_density"],
                spectral_index=sim_config["spectral_index"]
            )
        print("✓ Dataset generation complete.", flush=True)

        update_bda_config(
            config_path="./configs/bda_config.json",
            ref_nu=dataset.spws.ref_nu,
            min_diameter=dataset.antenna.min_diameter,
        )
        print("✓ BDA configuration updated.", flush=True)

        update_grid_config(
            config_path="./configs/grid_config.json",
            theo_resolution=dataset.theo_resolution,
            corrs_string=dataset.polarization.corrs_string,
            chan_freq=dataset.spws.dataset[0].CHAN_FREQ.compute().values[0].tolist()
        )
        print("✓ Grid configuration updated.", flush=True)

        streaming_results = stream_kafka(dataset, topic)
        print("✓ Streaming complete.", flush=True)

        return streaming_results

    except Exception as e:
        print(f"Error in producer service: {e}")
        traceback.print_exc()
        raise


def main():
    """
    Main entry point for the producer service.
    """
    parser = argparse.ArgumentParser(description="BDA Interferometry Producer Service")
    
    parser.add_argument("--antenna-config", help="Path to antenna configuration file")
    parser.add_argument("--simulation-config", help="Path to simulation configuration file")
    parser.add_argument("--topic", help=f"Kafka topic name")

    args = parser.parse_args()

    try:
        run_producer(
            antenna_config_path=args.antenna_config,
            simulation_config_path=args.simulation_config,
            topic=args.topic
        )
    
    except Exception as e:
        print(f"Fatal error in main: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
import logging
import sys
import traceback
import msgpack
import numpy as np
import zlib
import json
import time
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
from data.extraction import stream_dataset, setup_client

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


def serialize_chunk(chunk):
    try:
        msgpack_chunk = {}
        
        for key, value in chunk.items():
            if isinstance(value, np.ndarray):
                array_bytes = value.tobytes()
                
                if len(array_bytes) > 10240:
                    msgpack_chunk[key] = {
                        'type': 'ndarray_compressed',
                        'data': zlib.compress(array_bytes, level=6),
                        'shape': value.shape,
                        'dtype': str(value.dtype)
                    }
                else:
                    msgpack_chunk[key] = {
                        'type': 'ndarray',
                        'data': array_bytes,
                        'shape': value.shape,
                        'dtype': str(value.dtype)
                    }
            else:
                msgpack_chunk[key] = value
        
        return msgpack.packb(msgpack_chunk, use_bin_type=True, strict_types=False)
    
    except Exception as e:
        print(f"Error serializing chunk: {e}")
        traceback.print_exc()
        raise


def create_kafka_producer(kafka_servers=None, **kwargs):
    kafka_servers = kafka_servers or DEFAULT_KAFKA_SERVERS

    config = {
        'bootstrap_servers': kafka_servers,
        'value_serializer': serialize_chunk,
        'key_serializer': lambda x: str(x).encode('utf-8') if x else None,
        'compression_type': 'lz4',
        'batch_size': 104857600,        # 100 MB
        'linger_ms': 500,               # 500 ms
        'max_request_size': 104857600,  # 100 MB
        'buffer_memory': 4294967296,    # 4 GB
        'enable_idempotence': True,
        'acks': "all",
        'retries': 10,
        'max_in_flight_requests_per_connection': 1,
        'request_timeout_ms': 120000,   # 120s
        'delivery_timeout_ms': 600000,  # 600s = 10 min
        'metadata_max_age_ms': 300000,
    }

    # Apply any overrides
    config.update(kwargs)
    
    try:
        producer = KafkaProducer(**config)
        return producer
    
    except KafkaError as e:
        print(f"✗ Kafka error creating producer: {e}")
        raise
    except Exception as e:
        print(f"✗ Unexpected error creating producer: {e}")
        traceback.print_exc()
        raise


def stream_kafka(dataset, producer, topic, base_delay=0.01, enable_warmup=True):
    MAX_PENDING = 5000
    HIGH_LATENCY = 500

    pending_futures = []
    send_times = []
    total_msgs = 0
    failed_msgs = 0

    current_delay = 0.5 if enable_warmup else base_delay
    warmup_remaining = 100 if enable_warmup else 0
    
    last_flush = time.time()
    start_time = time.time()

    try:
        for chunk, subms_id in stream_dataset(dataset):
            chunk_start = time.time()
            key = None
            msg_id = f"{subms_id}-{chunk['chunk_id']}"

            total_msgs += 1
            
            future = producer.send(topic, value=chunk, key=key)
            pending_futures.append({'future': future, 'start_time': chunk_start, 'msg_id': msg_id})
            
            completed = []
            for pending in pending_futures:
                if pending['future'].is_done:
                    completed.append(pending)
                    try:
                        pending['future'].get(timeout = 0.1)

                        send_times.append((time.time() - pending['start_time']) * 1000)
                        
                        print(f"[Producer] Sent message ID {pending['msg_id']}", flush=True)

                        if len(send_times) > 100:
                            send_times.pop(0)
                    except KafkaError:
                        failed_msgs += 1
                        print(f"[Producer] Error sending message ID {pending['msg_id']}", flush=True)

            pending_futures = [p for p in pending_futures if p not in completed]
            
            avg_time = sum(send_times) / len(send_times) if send_times else 0

            if len(pending_futures) > MAX_PENDING or avg_time > HIGH_LATENCY:
                current_delay = min(current_delay * 1.2, 1.0)
            elif len(pending_futures) < 100 and avg_time < HIGH_LATENCY * 0.7:
                current_delay = max(current_delay * 0.95, base_delay)
            
            if warmup_remaining > 0:
                warmup_remaining -= 1
                if warmup_remaining == 0:
                    current_delay = base_delay
                    print("[Producer] Warmup period complete", flush=True)

            if time.time() - last_flush >= 1.0:
                producer.flush(timeout = 1)
                last_flush = time.time()
            
            if current_delay > 0:
                time.sleep(current_delay)

    except Exception as e:
        print(f"[Producer] Error during streaming: {e}")
        traceback.print_exc()
        raise

    finally:
        for pending in pending_futures:
            try:
                pending['future'].get(timeout = 10)
                print(f"[Producer] Sent message ID {pending['msg_id']}", flush=True)
            except KafkaError:
                failed_msgs += 1
                print(f"[Producer] Error sending message ID {pending['msg_id']}", flush=True)
        
        producer.flush(timeout = 5)

        print("[Producer] Sending END signal...")

        control_message = {'message_type': 'END', 'timestamp': time.time()}
        producer.send(topic, value=control_message, key="__CONTROL__").get(timeout = 10)

        producer.flush(timeout = 5)

        print("[Producer] ✓ END_OF_STREAM signal sent")
        
        total_time = time.time() - start_time
        sent = total_msgs - failed_msgs
        throughput = total_msgs / total_time if total_time > 0 else 0
        
        print(f"\n{'=' * 60}")
        print(f"SUMMARY: {total_msgs:,} total | {sent:,} sent | {failed_msgs:,} failed")
        print(f"Time: {total_time:.2f}s | Throughput: {throughput:,.0f} msg/s")
        print(f"{'=' * 60}\n")
        
    return {
        'total_messages': total_msgs, 
        'sent': sent, 
        'failed': failed_msgs, 
        'time': total_time, 
        'throughput': throughput
    }


def run_producer(antenna_config_path, simulation_config_path, topic):
    """
    Run the producer service to stream data to Kafka.

    Parameters
    ----------
    antenna_config_path : str
        Path to the antenna configuration file.
    simulation_config_path : str
        Path to the simulation configuration file.
    topic : str
        Kafka topic to send data to.
    
    Returns
    -------
    dict
        Streaming results and metrics.
    """
    producer = None
    client = None
    
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

        producer = create_kafka_producer()
        print("✓ Kafka producer created.", flush=True)

        client = setup_client()
        print("✓ Dask client setup complete.", flush=True)

        streaming_results = stream_kafka(dataset, producer, topic)
        print("✓ Streaming complete.", flush=True)

        return streaming_results

    except Exception as e:
        print(f"Error in producer service: {e}")
        traceback.print_exc()
        raise
        
    finally:
        if producer:
            producer.flush()
            producer.close()

        if client:
            client.shutdown()

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
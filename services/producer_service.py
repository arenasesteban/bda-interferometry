"""
Producer Service - Interferometry Data Streaming Microservice

Independent microservice that generates Pyralysis datasets and transmits
chunks via Kafka using functional programming and lossless compression.

This service creates simulated interferometry datasets and streams them
as compressed chunks to Kafka topics for real-time data transmission.
"""

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
from data.extraction import stream_subms_chunks


def load_simulation_config(config_path):
    """
    Load simulation configuration from a JSON file.
    
    Parameters
    ----------
    config_path : str
        Path to the configuration JSON file.
    
    Returns
    -------
    dict
        Configuration dictionary.
    """
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load config file {config_path}: {e}")
            print("Using default configuration.")
    
    return {}


def update_bda_config(config_path, ref_nu, min_diameter):
    """
    Update the BDA configuration file with new values.

    Parameters
    ----------
    config_path : str
        Path to the configuration JSON file.
    ref_nu : float
        Reference frequency.
    min_diameter : float
        Minimum diameter.

    Returns
    -------
    None
    """
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
    """
    Update the grid configuration file with new values.
    
    Parameters
    ----------
    config_path : str
        Path to the configuration JSON file.
    theo_resolution : float
        Theoretical resolution.
    corrs_string : str
        Correlations string.
    chan_freq : list
        List of channel frequencies.
    
    Returns
    -------
    None
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        config['cellsize'] = (theo_resolution / 7)
        config['corrs_string'] = corrs_string
        config['chan_freq'] = chan_freq

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
            
    except Exception as e:
        print(f"Error updating grid config: {e}")
        traceback.print_exc()
        raise


def serialize_chunk(chunk):
    """
    Serialize a chunk of data for Kafka.
    
    Parameters
    ----------
    chunk : dict
        Chunk of data to serialize.
    
    Returns
    -------
    bytes
        Serialized chunk.
    """
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
    """
    Create a Kafka producer instance with optimized settings.

    Parameters
    ----------
    kafka_servers : list, optional
        List of Kafka server addresses. Defaults to DEFAULT_KAFKA_SERVERS.
    **kwargs : dict
        Additional producer configuration overrides.
    
    Returns
    -------
    KafkaProducer
        Configured Kafka producer instance.
    
    Raises
    ------
    KafkaError
        If producer creation fails.
    """
    kafka_servers = kafka_servers or DEFAULT_KAFKA_SERVERS
    
    producer_config = {
        'bootstrap_servers': kafka_servers,
        'value_serializer': serialize_chunk,
        'key_serializer': lambda x: str(x).encode('utf-8') if x is not None else None,

        # Compression and batching
        'compression_type': 'lz4',
        'batch_size': 65536,  # 64 KB
        'linger_ms': 150,

        # Buffer sizes
        'max_request_size': 10485760,  # 10 MB
        'buffer_memory': 33554432,  # 32 MB

        # Reliability settings
        'enable_idempotence': True,
        'acks': 'all',
        'retries': 5,

        # Throughput optimizations
        'max_in_flight_requests_per_connection': 1,

        # Timeouts
        'request_timeout_ms': 30000,
        'delivery_timeout_ms': 120000,

        # Metadata refresh
        'metadata_max_age_ms': 300000, # 5 minutes
    }

    # Apply any overrides
    producer_config.update(kwargs)
    
    try:
        producer = KafkaProducer(**producer_config)
        
        return producer
    
    except KafkaError as e:
        print(f"✗ Kafka error creating producer: {e}")
        raise
    except Exception as e:
        print(f"✗ Unexpected error creating producer: {e}")
        traceback.print_exc()
        raise


def stream_chunks_to_kafka(dataset, producer, topic, base_streaming_delay, enable_warmup):
    """
    Stream chunks of data to Kafka.

    Parameters
    ----------
    dataset : Dataset
        The Pyralysis dataset.
    producer : KafkaProducer
        The Kafka producer instance.
    topic : str
        Kafka topic to send data to.
    base_streaming_delay : float
        Base delay between sending chunks (in seconds).
    enable_warmup : bool
        Whether to enable warmup period for initial chunks.

    Returns
    -------
    dict
        Streaming statistics and metrics.
    """
    # Initialize metrics and state
    pending_futures = []
    current_delay = 0.8 if enable_warmup else base_streaming_delay
    warmup_chunks_remaining = 20 if enable_warmup else 0
    
    # Timing for periodic flush
    last_flush_time = time.time()
    flush_interval = 2.0  # 2 seconds
    
    # Backpressure thresholds
    MAX_PENDING_FUTURES = 30
    HIGH_LATENCY_THRESHOLD = 3000.0  # ms

    # Simple metrics tracking
    total_chunks = 0
    failed_chunks = 0
    send_times = []  # Store recent send times for averaging
    max_send_times_window = 50  # Keep last 50 send times

    print("Starting streaming to Kafka...")

    try:
        for chunk in stream_subms_chunks(dataset):
            chunk_start_time = time.time()
            key = f"{chunk['subms_id']}_{chunk['chunk_id']}"
            total_chunks += 1
            
            try:                
                # Non-blocking send
                future = producer.send(topic, value=chunk, key=key)
                pending_futures.append({
                    'future': future,
                    'start_time': chunk_start_time,
                    'key': key
                })
                
                # Clean up completed futures and measure send times
                completed_futures = []
                for pending in pending_futures:
                    if pending['future'].is_done:
                        completed_futures.append(pending)
                        try:
                            pending['future'].get(timeout=0.1)
                            
                            # Record successful send time
                            send_time_ms = (time.time() - pending['start_time']) * 1000
                            send_times.append(send_time_ms)
                            print(f"✓ Sending chunk: {key} - {send_time_ms:.2f} ms")

                            # Keep only recent send times
                            if len(send_times) > max_send_times_window:
                                send_times.pop(0)
                        except KafkaError as e:
                            failed_chunks += 1
                            print(f"✗ Error sending chunk {pending['key']}: {e}")
                
                # Remove completed futures
                pending_futures = [p for p in pending_futures if p not in completed_futures]
                
                # Calculate average send time from recent samples
                avg_send_time = sum(send_times) / len(send_times) if send_times else 0
                
                # Sophisticated backpressure logic
                backpressure_triggered = False
                
                # 1. Too many pending futures
                if len(pending_futures) > MAX_PENDING_FUTURES:
                    current_delay = min(current_delay * 1.2, 1.0)  # Max 1s delay
                    backpressure_triggered = True
                    print(f"Backpressure: Too many pending futures ({len(pending_futures)})")
                
                # 2. High average send latency
                if avg_send_time > HIGH_LATENCY_THRESHOLD:
                    current_delay = min(current_delay * 1.1, 1.0)
                    backpressure_triggered = True
                    print(f"Backpressure: High latency detected ({avg_send_time:.2f}ms)")
                
                # 3. Recovery: reduce delay if conditions improve
                if not backpressure_triggered:
                    if len(pending_futures) < 10 and avg_send_time < HIGH_LATENCY_THRESHOLD * 0.7:
                        target_delay = 0.8 if warmup_chunks_remaining > 0 else base_streaming_delay
                        current_delay = max(current_delay * 0.95, target_delay)
                
                # Warm-up period management
                if warmup_chunks_remaining > 0:
                    warmup_chunks_remaining -= 1
                    if warmup_chunks_remaining == 0:
                        current_delay = base_streaming_delay
                        print("Warm-up period completed")
                
                # Time-based flushing
                current_time = time.time()
                if current_time - last_flush_time >= flush_interval:
                    producer.flush(timeout=1)
                    last_flush_time = current_time
                
                # Apply current delay
                time.sleep(current_delay)
                
            except Exception as e:
                failed_chunks += 1
                print(f"Error processing chunk {key}: {e}")
                continue

    except Exception as e:
        print(f"Fatal error during streaming: {e}")
        traceback.print_exc()
        raise
        
    finally:
        # Wait for remaining futures
        if pending_futures:
            print(f"Waiting for {len(pending_futures)} pending futures...")
            
            for pending in pending_futures:
                try:
                    pending['future'].get(timeout=10)
                    send_time_ms = (time.time() - pending['start_time']) * 1000
                    send_times.append(send_time_ms)
                    print(f"✓ Final chunk: {pending['key']} sent")
                except KafkaError as e:
                    failed_chunks += 1
                    print(f"✗ Final chunk {pending['key']}: FAILED - {e}")
        
        # Final flush
        producer.flush(timeout=5)
        
        # Calculate final statistics
        sent_chunks = total_chunks - failed_chunks
        avg_send_time_final = sum(send_times) / len(send_times) if send_times else 0
        
        print(f"\n=== Streaming Summary ===")
        print(f"Total chunks: {total_chunks}")
        print(f"Sent successfully: {sent_chunks}")
        print(f"Failed: {failed_chunks}")
        print(f"Average send time: {avg_send_time_final:.2f}ms")
        
    return {
        'total_chunks': total_chunks,
        'failed_chunks': failed_chunks,
        'sent_chunks': sent_chunks,
        'average_send_time_ms': avg_send_time_final,
        'send_times': send_times
    }


def run_producer_service(antenna_config_path, simulation_config_path, topic):
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
    
    if topic is None:
        topic = DEFAULT_TOPIC
    
    try:
        sim_config = load_simulation_config(simulation_config_path)
        print("✓ Loaded simulation configuration.")

        if "source_path" in sim_config:
            dataset = generate_dataset(
                antenna_config_path=antenna_config_path,
                freq_min=sim_config["freq_min"],
                freq_max=sim_config["freq_max"],
                n_chans=sim_config["n_chans"],
                observation_time=sim_config["observation_time"],
                declination=sim_config["declination"],
                integration_time=sim_config["integration_time"],
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
        print("✓ Dataset generation complete.")

        update_bda_config(
            config_path="./configs/bda_config.json",
            ref_nu=dataset.spws.ref_nu,
            min_diameter=dataset.antenna.min_diameter,
        )
        print("✓ BDA configuration updated.")

        update_grid_config(
            config_path="./configs/grid_config.json",
            theo_resolution=dataset.theo_resolution,
            corrs_string=dataset.polarization.corrs_string,
            chan_freq=dataset.spws.dataset[0].CHAN_FREQ.compute().values[0].tolist()
        )
        print("✓ Grid configuration updated.")

        producer = create_kafka_producer()
        print("✓ Kafka producer created.")

        streaming_results = stream_chunks_to_kafka(
            dataset, 
            producer, 
            topic, 
            sim_config.get('base_streaming_delay', 0.1),
            sim_config.get('enable_warmup', True)
        )
        print("✓ Streaming complete.")

        return streaming_results

    except Exception as e:
        print(f"Error in producer service: {e}")
        traceback.print_exc()
        raise
        
    finally:
        if producer:
            producer.flush()
            producer.close()


def main():
    """
    Main entry point for the producer service.
    """
    parser = argparse.ArgumentParser(description="BDA Interferometry Producer Service")
    
    parser.add_argument("antenna_config", help="Path to antenna configuration file")
    parser.add_argument("--simulation-config", help="Path to simulation configuration JSON file")
    parser.add_argument("--topic", help=f"Kafka topic (default: {DEFAULT_TOPIC})")
    
    args = parser.parse_args()

    try:
        run_producer_service(
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
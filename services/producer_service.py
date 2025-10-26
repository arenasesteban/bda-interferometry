"""
Producer Service - Interferometry Data Streaming Microservice

Independent microservice that generates Pyralysis datasets and transmits
chunks via Kafka using functional programming and lossless compression.

This service creates simulated interferometry datasets and streams them
as compressed chunks to Kafka topics for real-time data transmission.
"""

import sys
import msgpack
import numpy as np
import zlib
import json
import time
import logging
import statistics
from collections import deque
from pathlib import Path
import argparse
from typing import Dict, Any, List

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


class StreamingMetrics:
    """
    Advanced telemetry for streaming performance monitoring.
    
    Tracks send times, chunk sizes, backpressure events, and provides
    real-time percentile calculations for empirical tuning.
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.send_times = deque(maxlen=window_size)
        self.chunk_sizes = deque(maxlen=window_size)
        self.backpressure_events = []
        self.total_chunks = 0
        self.failed_chunks = 0
        
        # Setup clean logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s'  # Simple format, just the message
        )
        self.logger = logging.getLogger("producer")
    
    def record_send_time(self, send_time_ms: float):
        """Record time taken for a successful send operation."""
        self.send_times.append(send_time_ms)
        
    def record_chunk_size(self, size_bytes: int):
        """Record serialized chunk size in bytes."""
        self.chunk_sizes.append(size_bytes)
        
    def record_backpressure_event(self, delay: float, pending_futures: int, trigger: str):
        """Record backpressure activation event."""
        event = {
            'timestamp': time.time(),
            'delay': delay,
            'pending_futures': pending_futures,
            'trigger': trigger
        }
        self.backpressure_events.append(event)
        # Only log backpressure if it's significant (not every minor adjustment)
        if delay >= 1.0 or len(self.backpressure_events) % 5 == 0:
            self.logger.info(f"⚡ Backpressure: {trigger} -> delay={delay:.2f}s")
    
    def get_send_time_percentiles(self) -> Dict[str, float]:
        """Calculate send time percentiles from recent measurements."""
        if not self.send_times:
            return {'p50': 0, 'p90': 0, 'p99': 0}
        
        times = list(self.send_times)
        return {
            'p50': statistics.median(times),
            'p90': statistics.quantiles(times, n=10)[8] if len(times) >= 10 else max(times),
            'p99': statistics.quantiles(times, n=100)[98] if len(times) >= 100 else max(times)
        }
    
    def get_chunk_size_percentiles(self) -> Dict[str, float]:
        """Calculate chunk size percentiles from recent measurements."""
        if not self.chunk_sizes:
            return {'p50': 0, 'p90': 0, 'p99': 0}
        
        sizes = list(self.chunk_sizes)
        return {
            'p50': statistics.median(sizes),
            'p90': statistics.quantiles(sizes, n=10)[8] if len(sizes) >= 10 else max(sizes),
            'p99': statistics.quantiles(sizes, n=100)[98] if len(sizes) >= 100 else max(sizes)
        }
    
    def get_average_send_time(self) -> float:
        """Get average send time from recent window."""
        return statistics.mean(self.send_times) if self.send_times else 0.0
    
    def get_backpressure_rate(self) -> float:
        """Calculate backpressure activation rate."""
        if self.total_chunks == 0:
            return 0.0
        return len(self.backpressure_events) / self.total_chunks * 100
    
    def log_final_report(self):
        """Log clean final summary."""
        send_percentiles = self.get_send_time_percentiles()
        size_percentiles = self.get_chunk_size_percentiles()
        backpressure_rate = self.get_backpressure_rate()
        
        self.logger.info("\n📊 === STREAMING SUMMARY ===")
        success_rate = ((self.total_chunks - self.failed_chunks) / max(self.total_chunks, 1)) * 100
        self.logger.info(f"✅ Sent: {self.total_chunks - self.failed_chunks}/{self.total_chunks} chunks ({success_rate:.1f}%)")
        
        if self.failed_chunks > 0:
            self.logger.info(f"❌ Failed: {self.failed_chunks} chunks")
        
        # Only show key metrics
        if self.send_times:
            avg_latency = statistics.mean(self.send_times)
            self.logger.info(f"⏱️  Avg latency: {avg_latency:.0f}ms")
        
        if self.chunk_sizes:
            avg_size = statistics.mean(self.chunk_sizes)
            self.logger.info(f"📦 Avg chunk size: {avg_size/1024:.0f}KB")
        
        return {
            'send_times': send_percentiles,
            'chunk_sizes': size_percentiles,
            'backpressure_rate': backpressure_rate,
            'total_chunks': self.total_chunks,
            'failed_chunks': self.failed_chunks
        }


def load_simulation_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load simulation configuration from JSON file or return defaults.
    
    Parameters
    ----------
    config_path : str, optional
        Path to JSON configuration file. If None, returns default configuration.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing simulation parameters
    """    
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load config file {config_path}: {e}")
            print("Using default configuration.")
    
    return user_config


def serialize_chunk(chunk: Dict[str, Any]) -> bytes:
    """
    Serialize visibility chunk using optimized MessagePack with lossless compression.
    
    Applies zlib compression to large arrays while preserving complete
    scientific data integrity of the original data.
    
    Parameters
    ----------
    chunk : Dict[str, Any]
        Dictionary containing visibility chunk data
        
    Returns
    -------
    bytes
        Serialized and compressed data with MessagePack
    """
    
    msgpack_chunk = {}
    
    for key, value in chunk.items():
        if isinstance(value, np.ndarray):
            # Serialize array without modifying data types
            array_bytes = value.tobytes()
            
            # Apply zlib compression for large arrays (>10KB)
            if len(array_bytes) > 10240:
                compressed_data = zlib.compress(array_bytes, level=6)
                
                msgpack_chunk[key] = {
                    'type': 'ndarray_compressed',
                    'data': compressed_data,
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


def create_kafka_producer(kafka_servers=None):
    """
    Create Kafka producer with optimized configuration for large messages.
    
    Configured to handle large scientific chunks with multiple compression layers:
    - zlib compression at array level
    - gzip compression at Kafka level
    - Extended limits for large messages
    
    Parameters
    ----------
    kafka_servers : list of str, optional
        List of Kafka server addresses. If None, uses default servers.
        
    Returns
    -------
    KafkaProducer
        Configured producer for large message transmission
    """
    
    if kafka_servers is None:
        kafka_servers = DEFAULT_KAFKA_SERVERS
    
    return KafkaProducer(
        bootstrap_servers=kafka_servers,
        compression_type='lz4',                    # Faster compression for better throughput
        
        # Real batching configuration
        batch_size=65536,                          # 64KB for real batching
        linger_ms=150,                             # 150ms to properly fill batches
        
        # Memory limits for 8GB systems
        max_request_size=2097152,                  # 2MB maximum per message
        buffer_memory=33554432,                    # 32MB buffer (reduced from 64MB)
        
        # Reliability configuration - no payload changes
        enable_idempotence=True,                   # Prevent duplicates/losses
        acks='all',                                # Confirm to leader + ISR
        retries=5,                                 # More retries for robustness
        max_in_flight_requests_per_connection=1,   # Balance: 1=strict order, 5=max throughput
        
        value_serializer=serialize_chunk,
        key_serializer=lambda x: str(x).encode('utf-8') if x is not None else None
    )


def stream_chunks_to_kafka(dataset, producer, topic: str, 
                          base_streaming_delay: float = 0.5,
                          enable_warmup: bool = True, longitude: float = -67.755, latitude: float = -23.029) -> Dict[str, Any]:
    """
    Advanced streaming with sophisticated backpressure, telemetry, and warm-up.
    
    Features:
    - Warm-up period with higher delay for first 20 chunks
    - Sophisticated backpressure based on pending futures and send latency
    - Real-time telemetry with percentile tracking
    - Time-based flushing (every 2 seconds) instead of message count
    - Moving window for send time analysis
    
    Parameters
    ----------
    dataset : object
        Pyralysis dataset object containing visibility data
    producer : KafkaProducer
        Configured Kafka producer instance
    topic : str
        Kafka topic name for streaming
    base_streaming_delay : float, optional
        Base delay between chunk transmissions (default: 0.3s)
    enable_warmup : bool, optional
        Enable warm-up period with higher initial delays
        
    Returns
    -------
    Dict[str, Any]
        Comprehensive streaming statistics and telemetry
        
    Raises
    ------
    Exception
        If streaming encounters critical errors during transmission
    """
    # Initialize metrics and state
    metrics = StreamingMetrics(window_size=100)
    pending_futures = []
    current_delay = 0.8 if enable_warmup else base_streaming_delay
    warmup_chunks_remaining = 20 if enable_warmup else 0
    
    # Timing for periodic flush
    last_flush_time = time.time()
    flush_interval = 2.0  # 2 seconds
    
    # Backpressure thresholds (adjusted for current system performance)
    MAX_PENDING_FUTURES = 30        # Reduced for max_in_flight=3
    HIGH_LATENCY_THRESHOLD = 3000.0  # ms (more realistic for acks='all')

    try:
        for chunk in stream_subms_chunks(dataset, longitude, latitude):
            chunk_start_time = time.time()
            metrics.total_chunks += 1
            key = f"{chunk['subms_id']}_{chunk['chunk_id']}"
            
            try:
                # Serialize and measure chunk size for telemetry
                serialized_size = len(serialize_chunk(chunk))
                metrics.record_chunk_size(serialized_size)
                
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
                            pending['future'].get(timeout=0.1)  # Non-blocking check
                            # Record successful send time
                            send_time_ms = (time.time() - pending['start_time']) * 1000
                            metrics.record_send_time(send_time_ms)
                            # Log successful chunk send
                            sent_count = metrics.total_chunks - metrics.failed_chunks - len(pending_futures) + len(completed_futures)
                            metrics.logger.info(f"✓ Chunk {sent_count}: {pending['key']} ({send_time_ms:.0f}ms)")
                        except KafkaError as e:
                            metrics.failed_chunks += 1
                            metrics.logger.info(f"✗ Chunk {pending['key']}: FAILED - {e}")
                
                # Remove completed futures
                pending_futures = [p for p in pending_futures if p not in completed_futures]
                
                # Sophisticated backpressure logic
                backpressure_triggered = False
                
                # 1. Too many pending futures
                if len(pending_futures) > MAX_PENDING_FUTURES:
                    current_delay = min(current_delay * 1.2, 1.0)  # Max 1s delay
                    metrics.record_backpressure_event(current_delay, len(pending_futures), "high_pending_futures")
                    backpressure_triggered = True
                
                # 2. High average send latency (moving window)
                avg_send_time = metrics.get_average_send_time()
                if avg_send_time > HIGH_LATENCY_THRESHOLD:
                    current_delay = min(current_delay * 1.1, 1.0)
                    metrics.record_backpressure_event(current_delay, len(pending_futures), "high_send_latency")
                    backpressure_triggered = True
                
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
                        metrics.logger.info("🚀 Warm-up completed, switching to normal streaming")
                
                # Time-based flushing (every 2 seconds) - silent
                current_time = time.time()
                if current_time - last_flush_time >= flush_interval:
                    producer.flush(timeout=1)
                    last_flush_time = current_time
                
                # Apply current delay
                time.sleep(current_delay)
                
            except Exception as e:
                metrics.failed_chunks += 1
                metrics.logger.info(f"✗ Error processing chunk {key}: {e}")
                continue
                
    except Exception as e:
        metrics.logger.info(f"❌ Critical streaming error: {e}")
        raise
        
    finally:
        # Wait for remaining futures silently
        if pending_futures:
            metrics.logger.info(f"⏳ Finalizing {len(pending_futures)} remaining chunks...")
        
        for pending in pending_futures:
            try:
                pending['future'].get(timeout=10)
                send_time_ms = (time.time() - pending['start_time']) * 1000
                metrics.record_send_time(send_time_ms)
                metrics.logger.info(f"✓ Final chunk: {pending['key']} sent")
            except KafkaError as e:
                metrics.failed_chunks += 1
                metrics.logger.info(f"✗ Final chunk {pending['key']}: FAILED")
        
        # Final flush
        producer.flush(timeout=5)
        
        # Generate comprehensive report
        final_report = metrics.log_final_report()
        
    return {
        'total_chunks': metrics.total_chunks,
        'failed_chunks': metrics.failed_chunks,
        'sent_chunks': metrics.total_chunks - metrics.failed_chunks,
        'telemetry': final_report
    }


def run_producer_service(antenna_config_path: str, 
                        simulation_config_path: str = None,
                        topic: str = None) -> Dict[str, Any]:
    """
    Execute complete production service.
    
    Generates simulated visibility data and transmits it via Kafka
    with comprehensive error handling and result reporting.
    
    Parameters
    ----------
    antenna_config_path : str
        Path to antenna configuration file
    simulation_config_path : str, optional
        Path to simulation configuration JSON file. If None, uses defaults.
    topic : str, optional
        Kafka topic for visibility streaming. If None, uses default topic.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing transmission statistics and success status
    """
    
    producer = None
    
    if topic is None:
        topic = DEFAULT_TOPIC
    
    try:
        # Load simulation configuration
        sim_config = load_simulation_config(simulation_config_path)
        
        # Generate dataset with loaded configuration
        dataset = generate_dataset(
            antenna_config_path=antenna_config_path,
            freq_start=sim_config["freq_start"],
            freq_end=sim_config["freq_end"],
            n_frequencies=sim_config["n_frequencies"],
            date_string=sim_config["date_string"],
            observation_time=sim_config["observation_time"],
            declination=sim_config["declination"],
            integration_time=sim_config["integration_time"],
            n_point_sources=sim_config["n_point_sources"],
            point_flux_density=sim_config["point_flux_density"],
            point_spectral_index=sim_config["point_spectral_index"],
            include_gaussian=sim_config["include_gaussian"],
            gaussian_flux_density=sim_config["gaussian_flux_density"],
            gaussian_position=tuple(sim_config["gaussian_position"]),
            gaussian_minor_radius=sim_config["gaussian_minor_radius"],
            gaussian_major_radius=sim_config["gaussian_major_radius"],
            gaussian_theta_angle=sim_config["gaussian_theta_angle"]
        )

        # Create producer and stream chunks with advanced telemetry
        producer = create_kafka_producer()
        streaming_results = stream_chunks_to_kafka(dataset, producer, topic, sim_config["longitude"], sim_config["latitude"])
        
        return {
            'success': True,
            'chunks_sent': streaming_results['sent_chunks'],
            'chunks_failed': streaming_results['failed_chunks'],
            'chunks_total': streaming_results['total_chunks'],
            'telemetry': streaming_results['telemetry']
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'chunks_sent': 0,
            'chunks_failed': 0,
            'chunks_total': 0,
            'telemetry': {}
        }
        
    finally:
        if producer:
            producer.flush()
            producer.close()


def main():
    """Main entry point for producer microservice."""
    
    parser = argparse.ArgumentParser(description="BDA Interferometry Producer Service")
    
    parser.add_argument(
        "antenna_config", 
        help="Path to antenna configuration file (required)"
    )
    parser.add_argument(
        "--simulation-config", 
        help="Path to simulation configuration JSON file (optional, uses defaults if not provided)"
    )
    parser.add_argument(
        "--topic", 
        help=f"Kafka topic for visibility streaming (default: {DEFAULT_TOPIC})"
    )
    
    args = parser.parse_args()
    
    # Execute service
    result = run_producer_service(
        antenna_config_path=args.antenna_config,
        simulation_config_path=args.simulation_config,
        topic=args.topic
    )
    
    # Print clean final summary
    if result['success']:
        print(f"\n🎉 STREAMING COMPLETED!")
        print(f"� {result['chunks_sent']}/{result['chunks_total']} chunks sent successfully")
        
        if result['chunks_failed'] > 0:
            print(f"⚠️  {result['chunks_failed']} chunks failed")
        
        # Show success rate
        if result['chunks_total'] > 0:
            success_rate = (result['chunks_sent'] / result['chunks_total']) * 100
            status = "✅" if success_rate >= 90 else "⚠️"
            print(f"{status} Success rate: {success_rate:.1f}%")
                
    else:
        print(f"\n❌ STREAMING FAILED: {result['error']}")
    
    # Exit based on result (consider partial success)
    if result['success']:
        failure_rate = result.get('chunks_failed', 0) / max(result.get('chunks_total', 1), 1)
        success = failure_rate < 0.1  # Accept up to 10% failure rate
    else:
        success = False
        
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

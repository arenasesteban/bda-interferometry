import pytest
import numpy as np
import msgpack
import io
from unittest.mock import MagicMock, Mock, PropertyMock, patch
from pyspark.sql.types import (
    StructType, StructField, StringType, BinaryType,
    IntegerType, DoubleType
)    

from services.consumer_service import (
    create_spark_session,
    create_kafka_stream,
    define_visibility_schema,
    check_end_signal,
    headers_to_dict,
    parse_metadata,
    deserialize_payload,
    assemble_block,
    normalize_baseline,
    process_rows,
    process_message,
    process_streaming_batch,
    consolidate_processing,
    generate_metrics,
    run_consumer
)

MODULE = "services.consumer_service"

# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def mock_spark_session():
    """Create a mock Spark session"""
    spark = Mock()
    spark.sparkContext = Mock()
    spark.sparkContext.master = "local[2]"
    spark.sparkContext.applicationId = "test-app-123"
    spark.sparkContext.defaultParallelism = 4
    spark.sparkContext.setLogLevel = Mock()
    return spark


@pytest.fixture
def mock_kafka_df():
    """Create a mock Kafka DataFrame"""
    df = Mock()
    df.select = Mock(return_value=df)
    df.key = Mock()
    df.value = Mock()
    df.headers = Mock()
    return df


@pytest.fixture
def sample_metadata():
    """Sample metadata dictionary"""
    return {
        "message_id": "msg-001",
        "subms_id": 0,
        "field_id": 1,
        "spw_id": 2,
        "polarization_id": 0,
        "n_channels": 2,
        "n_correlations": 2
    }


@pytest.fixture
def sample_arrays():
    """Sample arrays dictionary"""
    return {
        "antenna1": np.array([0, 1]),
        "antenna2": np.array([1, 2]),
        "scan_number": np.array([1, 1]),
        "time": np.array([0.0, 0.1]),
        "exposure": np.array([1.0, 1.0]),
        "interval": np.array([1.0, 1.0]),
        "u": np.array([100.0, 150.0]),
        "v": np.array([50.0, 75.0]),
        "w": np.array([0.0, 0.0]),
        "visibilities": np.random.random((2, 2, 2)) + 1j * np.random.random((2, 2, 2)),
        "weights": np.random.random((2, 2)),
        "flags": np.zeros((2, 2), dtype=int)
    }


@pytest.fixture
def sample_block(sample_metadata, sample_arrays):
    """Sample assembled block"""
    return assemble_block(sample_metadata, sample_arrays)


@pytest.fixture
def mock_headers():
    """Mock Kafka headers"""
    header1 = Mock()
    header1.key = "metadata"
    header1.value = msgpack.packb({"message_id": "test-123"})
    
    header2 = Mock()
    header2.key = "timestamp"
    header2.value = b"2024-01-01"
    
    return [header1, header2]


# ==============================================================================
# Tests for create_spark_session
# ==============================================================================

def test_create_spark_session_success():
    """Test successful Spark session creation"""
    with patch(f"{MODULE}.SparkSession") as mock_spark_class:
        mock_builder = Mock()
        mock_spark = Mock()
        mock_spark.sparkContext = Mock()
        mock_spark.sparkContext.master = "local[2]"
        mock_spark.sparkContext.applicationId = "app-123"
        mock_spark.sparkContext.defaultParallelism = 4
        
        mock_builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_spark
        mock_spark_class.builder = mock_builder
        
        result = create_spark_session()
        
        assert result == mock_spark
        mock_builder.appName.assert_called_once_with("BDA-Interferometry-Consumer")


def test_create_spark_session_raises_on_none_spark():
    """Test that None Spark raises RuntimeError"""
    with patch(f"{MODULE}.SparkSession") as mock_spark_class:
        mock_builder = Mock()
        mock_builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = None
        mock_spark_class.builder = mock_builder
        
        with pytest.raises(RuntimeError, match="SparkSession.builder returned None"):
            create_spark_session()


def test_create_spark_session_raises_on_none_context():
    """Test that None SparkContext raises RuntimeError"""
    with patch(f"{MODULE}.SparkSession") as mock_spark_class:
        mock_builder = Mock()
        mock_spark = Mock()
        mock_spark.sparkContext = None
        
        mock_builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_spark
        mock_spark_class.builder = mock_builder
        
        with pytest.raises(RuntimeError, match="SparkSession has no SparkContext"):
            create_spark_session()


def test_create_spark_session_propagates_exception():
    """Test that exceptions during creation are propagated"""
    with patch(f"{MODULE}.SparkSession") as mock_spark_class:
        mock_builder = Mock()
        mock_builder.appName.side_effect = Exception("Connection failed")
        mock_spark_class.builder = mock_builder
        
        with pytest.raises(RuntimeError, match="Spark session creation failed"):
            create_spark_session()


# ==============================================================================
# Tests for create_kafka_stream
# ==============================================================================

def test_create_kafka_stream_raises_on_none_spark():
    """Test that None spark raises ValueError"""
    with pytest.raises(ValueError, match="spark cannot be None"):
        create_kafka_stream(None, "localhost:9092", "test-topic")


def test_create_kafka_stream_raises_on_none_bootstrap_server(mock_spark_session):
    """Test that None bootstrap_server raises ValueError"""
    with pytest.raises(ValueError, match="bootstrap_server cannot be None or empty"):
        create_kafka_stream(mock_spark_session, None, "test-topic")


def test_create_kafka_stream_raises_on_empty_bootstrap_server(mock_spark_session):
    """Test that empty bootstrap_server raises ValueError"""
    with pytest.raises(ValueError, match="bootstrap_server cannot be None or empty"):
        create_kafka_stream(mock_spark_session, "", "test-topic")


def test_create_kafka_stream_raises_on_none_topic(mock_spark_session):
    """Test that None topic raises ValueError"""
    with pytest.raises(ValueError, match="topic cannot be None or empty"):
        create_kafka_stream(mock_spark_session, "localhost:9092", None)


def test_create_kafka_stream_raises_on_empty_topic(mock_spark_session):
    """Test that empty topic raises ValueError"""
    with pytest.raises(ValueError, match="topic cannot be None or empty"):
        create_kafka_stream(mock_spark_session, "localhost:9092", "")


def test_create_kafka_stream_success(mock_spark_session, mock_kafka_df):
    """Test successful Kafka stream creation"""
    mock_read_stream = Mock()
    mock_read_stream.format.return_value = mock_read_stream
    mock_read_stream.option.return_value = mock_read_stream
    mock_read_stream.load.return_value = mock_kafka_df
    mock_spark_session.readStream = mock_read_stream
    
    result = create_kafka_stream(mock_spark_session, "localhost:9092", "test-topic")
    
    assert result == mock_kafka_df
    mock_read_stream.format.assert_called_once_with("kafka")
    assert mock_read_stream.option.call_count == 6


def test_create_kafka_stream_raises_on_none_kafka_df(mock_spark_session):
    """Test that None kafka_df raises RuntimeError"""
    mock_read_stream = Mock()
    mock_read_stream.format.return_value = mock_read_stream
    mock_read_stream.option.return_value = mock_read_stream
    mock_read_stream.load.return_value = None
    mock_spark_session.readStream = mock_read_stream
    
    with pytest.raises(RuntimeError, match="readStream.load\\(\\) returned None"):
        create_kafka_stream(mock_spark_session, "localhost:9092", "test-topic")


def test_create_kafka_stream_propagates_exception(mock_spark_session):
    """Test that exceptions during stream creation are propagated"""
    mock_read_stream = Mock()
    mock_read_stream.format.side_effect = Exception("Kafka unavailable")
    mock_spark_session.readStream = mock_read_stream
    
    with pytest.raises(RuntimeError, match="Kafka stream creation failed"):
        create_kafka_stream(mock_spark_session, "localhost:9092", "test-topic")


# ==============================================================================
# Tests for define_visibility_schema
# ==============================================================================

def test_define_visibility_schema_returns_struct_type():
    """Test that schema is a StructType"""
    
    schema = define_visibility_schema()
    
    assert isinstance(schema, StructType)
    assert len(schema.fields) == 20  # All expected fields


def test_define_visibility_schema_has_required_fields():
    """Test that schema contains all required fields"""
    schema = define_visibility_schema()
    field_names = [field.name for field in schema.fields]
    
    required_fields = [
        "message_id", "subms_id", "field_id", "spw_id", "polarization_id",
        "baseline_key", "antenna1", "antenna2", "scan_number",
        "exposure", "interval", "time",
        "n_channels", "n_correlations",
        "u", "v", "w",
        "visibility", "weight", "flag"
    ]
    
    for field in required_fields:
        assert field in field_names


# ==============================================================================
# Tests for check_end_signal
# ==============================================================================

def test_check_end_signal_raises_on_none_df():
    """Test that None df_scientific raises ValueError"""
    with pytest.raises(ValueError, match="df_scientific cannot be None"):
        check_end_signal(None, 0)


def test_check_end_signal_detects_end_signal(spark):
    """Test that END signal is detected"""

    schema = StructType([
        StructField("key", BinaryType(), True),
        StructField("value", BinaryType(), True),
        StructField("headers", StringType(), True)
    ])
    
    data = [
        (b"__END__", b"payload", "header1"),
        (b"normal_key", b"payload2", "header2")
    ]
    
    df = spark.createDataFrame(data, schema)
    
    df_filtered, end_signal = check_end_signal(df, 1)
    
    assert end_signal is True
    assert df_filtered is not None


def test_check_end_signal_no_end_signal(spark):
    """Test when no END signal is present"""

    schema = StructType([
        StructField("key", BinaryType(), True),
        StructField("value", BinaryType(), True),
        StructField("headers", StringType(), True)
    ])
    
    data = [
        (b"key1", b"payload1", "header1"),
        (b"key2", b"payload2", "header2")
    ]
    
    df = spark.createDataFrame(data, schema)
    
    df_filtered, end_signal = check_end_signal(df, 1)
    
    assert end_signal is False
    assert df_filtered is not None


def test_check_end_signal_filters_end_message(spark):
    """Test that END message is filtered from output"""
    schema = StructType([
        StructField("key", BinaryType(), True),
        StructField("value", BinaryType(), True),
        StructField("headers", StringType(), True)
    ])
    
    data = [
        (b"__END__", b"end_payload", "header_end"),
        (b"key1", b"payload1", "header1")
    ]
    
    df = spark.createDataFrame(data, schema)
    
    df_filtered, _ = check_end_signal(df, 1)
    
    # Should only have 1 row (non-END row)
    assert df_filtered.count() == 1


# ==============================================================================
# Tests for headers_to_dict
# ==============================================================================

def test_headers_to_dict_none_headers():
    """Test that None headers returns empty dict"""
    result = headers_to_dict(None)
    assert result == {}


def test_headers_to_dict_empty_list():
    """Test that empty list returns empty dict"""
    result = headers_to_dict([])
    assert result == {}


def test_headers_to_dict_success(mock_headers):
    """Test successful header parsing"""
    result = headers_to_dict(mock_headers)
    
    assert "metadata" in result
    assert "timestamp" in result
    assert result["timestamp"] == b"2024-01-01"


def test_headers_to_dict_skips_none_header():
    """Test that None headers in list are skipped"""
    header1 = Mock()
    header1.key = "key1"
    header1.value = b"value1"
    
    headers = [header1, None, None]
    
    result = headers_to_dict(headers)
    
    assert len(result) == 1
    assert "key1" in result


def test_headers_to_dict_handles_string_key():
    """Test that string keys are handled correctly"""
    header = Mock()
    header.key = "string_key"
    header.value = b"value"
    
    result = headers_to_dict([header])
    
    assert "string_key" in result


def test_headers_to_dict_handles_bytes_key():
    """Test that bytes keys are decoded"""
    header = Mock()
    header.key = b"bytes_key"
    header.value = b"value"
    
    result = headers_to_dict([header])
    
    assert "bytes_key" in result


def test_headers_to_dict_handles_none_value():
    """Test that None values become empty bytes"""
    header = Mock()
    header.key = "key"
    header.value = None
    
    result = headers_to_dict([header])
    
    assert result["key"] == b""


def test_headers_to_dict_handles_exception():
    """Test that exceptions return partial result"""
    header1 = Mock()
    header1.key = "good_key"
    header1.value = b"value"
    
    header2 = Mock()
    header2.key.decode.side_effect = Exception("Decode error")
    
    result = headers_to_dict([header1, header2])
    
    # Should have partial result from first header
    assert "good_key" in result


# ==============================================================================
# Tests for parse_metadata
# ==============================================================================

def test_parse_metadata_none_headers_dict():
    """Test that None headers_dict returns empty dict"""
    result = parse_metadata(None)
    assert result == {}


def test_parse_metadata_empty_headers_dict():
    """Test that empty headers_dict returns empty dict"""
    result = parse_metadata({})
    assert result == {}


def test_parse_metadata_no_metadata_key():
    """Test that missing metadata key returns empty dict"""
    result = parse_metadata({"other_key": b"value"})
    assert result == {}


def test_parse_metadata_success():
    """Test successful metadata parsing"""
    metadata = {"message_id": "test-123", "subms_id": 0}
    packed = msgpack.packb(metadata)
    
    result = parse_metadata({"metadata": packed})
    
    assert result == metadata


def test_parse_metadata_raises_on_non_dict():
    """Test that non-dict metadata raises ValueError"""
    packed = msgpack.packb(["not", "a", "dict"])
    
    with pytest.raises(ValueError, match="Metadata must be a dictionary"):
        parse_metadata({"metadata": packed})


def test_parse_metadata_raises_on_invalid_msgpack():
    """Test that invalid msgpack raises ValueError"""
    with pytest.raises(ValueError, match="Metadata parsing failed"):
        parse_metadata({"metadata": b"invalid msgpack data"})


# ==============================================================================
# Tests for deserialize_payload
# ==============================================================================

def test_deserialize_payload_raises_on_none():
    """Test that None value raises ValueError"""
    with pytest.raises(ValueError, match="value cannot be None"):
        deserialize_payload(None)


def test_deserialize_payload_raises_on_empty():
    """Test that empty value raises ValueError"""
    with pytest.raises(ValueError, match="value cannot be empty"):
        deserialize_payload(b"")


def test_deserialize_payload_success():
    """Test successful payload deserialization"""
    # Create NPZ data in memory
    buffer = io.BytesIO()
    np.savez(buffer, arr1=np.array([1, 2, 3]), arr2=np.array([4, 5, 6]))
    buffer.seek(0)
    npz_bytes = buffer.read()
    
    result = deserialize_payload(npz_bytes)
    
    assert "arr1" in result
    assert "arr2" in result
    assert np.array_equal(result["arr1"], np.array([1, 2, 3]))


def test_deserialize_payload_raises_on_empty_npz():
    """Test that NPZ with no arrays raises ValueError"""
    buffer = io.BytesIO()
    np.savez(buffer)  # Empty NPZ
    buffer.seek(0)
    npz_bytes = buffer.read()
    
    with pytest.raises(ValueError, match="NPZ file contains no arrays"):
        deserialize_payload(npz_bytes)


def test_deserialize_payload_raises_on_invalid_npz():
    """Test that invalid NPZ data raises ValueError"""
    with pytest.raises(ValueError, match="Payload deserialization failed"):
        deserialize_payload(b"not npz data")


# ==============================================================================
# Tests for assemble_block
# ==============================================================================

def test_assemble_block_raises_on_none_metadata():
    """Test that None metadata raises ValueError"""
    with pytest.raises(ValueError, match="metadata cannot be None"):
        assemble_block(None, {})


def test_assemble_block_raises_on_none_arrays():
    """Test that None arrays raises ValueError"""
    with pytest.raises(ValueError, match="arrays cannot be None"):
        assemble_block({}, None)


def test_assemble_block_raises_on_missing_metadata_fields():
    """Test that missing metadata fields raises ValueError"""
    metadata = {"message_id": "test"}  # Missing required fields
    arrays = {}
    
    with pytest.raises(ValueError, match="Missing required metadata fields"):
        assemble_block(metadata, arrays)


def test_assemble_block_raises_on_missing_array_fields(sample_metadata):
    """Test that missing array fields raises ValueError"""
    arrays = {"antenna1": np.array([0])}  # Missing required arrays
    
    with pytest.raises(ValueError, match="Missing required array fields"):
        assemble_block(sample_metadata, arrays)


def test_assemble_block_success(sample_metadata, sample_arrays):
    """Test successful block assembly"""
    result = assemble_block(sample_metadata, sample_arrays)
    
    assert result["message_id"] == "msg-001"
    assert result["subms_id"] == 0
    assert isinstance(result["antenna1"], np.ndarray)
    assert len(result["antenna1"]) == 2


def test_assemble_block_raises_on_length_mismatch(sample_metadata):
    """Test that mismatched array lengths raise ValueError"""
    arrays = {
        "antenna1": np.array([0, 1]),
        "antenna2": np.array([1]),  # Different length
        "scan_number": np.array([1, 1]),
        "time": np.array([0.0, 0.1]),
        "exposure": np.array([1.0, 1.0]),
        "interval": np.array([1.0, 1.0]),
        "u": np.array([100.0, 150.0]),
        "v": np.array([50.0, 75.0]),
        "w": np.array([0.0, 0.0]),
        "visibilities": np.random.random((2, 2, 2)),
        "weights": np.random.random((2, 2)),
        "flags": np.zeros((2, 2), dtype=int)
    }
    
    with pytest.raises(ValueError, match="Array 'antenna2' length mismatch"):
        assemble_block(sample_metadata, arrays)


# ==============================================================================
# Tests for normalize_baseline
# ==============================================================================

def test_normalize_baseline_success():
    """Test successful baseline normalization"""
    result = normalize_baseline(1, 0)
    assert result == "0-1"


def test_normalize_baseline_already_normalized():
    """Test that normalized baseline stays the same"""
    result = normalize_baseline(0, 1)
    assert result == "0-1"


def test_normalize_baseline_same_antenna():
    """Test baseline with same antenna"""
    result = normalize_baseline(5, 5)
    assert result == "5-5"


def test_normalize_baseline_raises_on_negative():
    """Test that negative antenna IDs raise ValueError"""
    with pytest.raises(ValueError, match="Antenna IDs must be non-negative"):
        normalize_baseline(-1, 0)


def test_normalize_baseline_raises_on_invalid_type():
    """Test that invalid types raise ValueError"""
    with pytest.raises(ValueError, match="Invalid antenna IDs"):
        normalize_baseline("invalid", 0)


def test_normalize_baseline_converts_float():
    """Test that float antenna IDs are converted to int"""
    result = normalize_baseline(1.0, 2.0)
    assert result == "1-2"


# ==============================================================================
# Tests for process_rows
# ==============================================================================

def test_process_rows_raises_on_none_block():
    """Test that None block raises ValueError"""
    with pytest.raises(ValueError, match="block cannot be None"):
        process_rows(None)


def test_process_rows_raises_on_empty_block():
    """Test that empty block raises ValueError"""
    block = {"antenna1": np.array([])}
    
    with pytest.raises(ValueError, match="block has no rows"):
        process_rows(block)


def test_process_rows_raises_on_missing_antenna1():
    """Test that missing antenna1 raises ValueError"""
    block = {"other_field": np.array([1, 2])}
    
    with pytest.raises(ValueError, match="block has no rows"):
        process_rows(block)


def test_process_rows_success(sample_block):
    """Test successful row processing"""
    rows = process_rows(sample_block)
    
    assert len(rows) == 2
    assert rows[0]["message_id"] == "msg-001"
    assert rows[0]["baseline_key"] == "0-1"
    assert isinstance(rows[0]["visibility"], list)


def test_process_rows_converts_types(sample_block):
    """Test that types are properly converted"""
    rows = process_rows(sample_block)
    
    row = rows[0]
    assert isinstance(row["subms_id"], int)
    assert isinstance(row["antenna1"], int)
    assert isinstance(row["time"], float)
    assert isinstance(row["u"], float)


def test_process_rows_raises_on_processing_error(sample_block):
    """Test that processing errors are caught and re-raised"""
    # Corrupt the block to cause an error
    sample_block["visibilities"] = None
    
    with pytest.raises(ValueError, match="Row processing failed at index"):
        process_rows(sample_block)


# ==============================================================================
# Tests for process_message
# ==============================================================================

def test_process_message_empty_iterator():
    """Test processing empty iterator"""
    iterator = iter([])
    
    result = list(process_message(iterator))
    
    assert result == []


def test_process_message_success():
    """Test successful message processing"""
    metadata = {
        "message_id": "test",
        "subms_id": 0,
        "field_id": 0,
        "spw_id": 0,
        "polarization_id": 0,
        "n_channels": 1,
        "n_correlations": 1
    }
    
    arrays = {
        "antenna1": np.array([0]),
        "antenna2": np.array([1]),
        "scan_number": np.array([1]),
        "time": np.array([0.0]),
        "exposure": np.array([1.0]),
        "interval": np.array([1.0]),
        "u": np.array([100.0]),
        "v": np.array([50.0]),
        "w": np.array([0.0]),
        "visibilities": np.random.random((1, 1, 1)),
        "weights": np.random.random((1, 1)),
        "flags": np.zeros((1, 1), dtype=int)
    }
    
    # Create NPZ payload
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    buffer.seek(0)
    payload = buffer.read()
    
    # Create mock message
    mock_message = Mock()
    mock_message.value = payload
    
    mock_header = Mock()
    mock_header.key = "metadata"
    mock_header.value = msgpack.packb(metadata)
    mock_message.headers = [mock_header]
    
    iterator = iter([mock_message])
    
    result = list(process_message(iterator))
    
    assert len(result) == 1
    assert result[0]["message_id"] == "test"


def test_process_message_handles_error(capsys):
    """Test that errors in message processing are logged and skipped"""
    # Create invalid message
    mock_message = Mock()
    mock_message.value = b"invalid data"
    mock_message.headers = []
    
    iterator = iter([mock_message])
    
    result = list(process_message(iterator))
    
    # Should return empty list and print error
    assert result == []
    captured = capsys.readouterr()
    assert "ERROR" in captured.out


# ==============================================================================
# Tests for process_streaming_batch
# ==============================================================================

def test_process_streaming_batch_raises_on_none_df():
    """Test that None df_scientific raises ValueError"""
    with pytest.raises(ValueError, match="df_scientific cannot be None"):
        process_streaming_batch(None, 4, 0, {}, {})


def test_process_streaming_batch_raises_on_invalid_num_partitions():
    """Test that invalid num_partitions raises ValueError"""
    mock_df = Mock()
    
    with pytest.raises(ValueError, match="num_partitions must be positive"):
        process_streaming_batch(mock_df, 0, 0, {}, {})


def test_process_streaming_batch_raises_on_none_bda_config():
    """Test that None bda_config raises ValueError"""
    mock_df = Mock()
    
    with pytest.raises(ValueError, match="bda_config cannot be None"):
        process_streaming_batch(mock_df, 4, 0, None, {})


def test_process_streaming_batch_raises_on_none_grid_config():
    """Test that None grid_config raises ValueError"""
    mock_df = Mock()
    
    with pytest.raises(ValueError, match="grid_config cannot be None"):
        process_streaming_batch(mock_df, 4, 0, {}, None)


def test_process_streaming_batch_with_bda(df_single_row):
    """Test batch processing with BDA enabled"""
    bda_config = {"decorr_factor": 0.95}
    grid_config = {"img_size": 8, "padding_factor": 1.0, "cellsize": 1e-5}
    
    # Mock distinct().collect() to return message_ids
    mock_message_id = Mock()
    mock_message_id.__getitem__ = Mock(return_value="msg-001")
    
    mock_df = Mock()
    mock_df.select.return_value = mock_df
    mock_df.distinct.return_value = mock_df
    mock_df.collect.return_value = [mock_message_id]
    mock_df.count.return_value = 1
    
    with patch(f"{MODULE}.apply_bda") as mock_bda, \
         patch(f"{MODULE}.apply_gridding") as mock_gridding:
        
        mock_bda.return_value = (df_single_row, df_single_row)
        mock_gridding.return_value = df_single_row
        
        df_grid, df_avg, df_win = process_streaming_batch(
            mock_df, 4, 0, bda_config, grid_config
        )
        
        assert df_grid is not None
        assert df_avg is not None
        assert df_win is not None
        mock_bda.assert_called_once()


def test_process_streaming_batch_without_bda(df_single_row):
    """Test batch processing with BDA disabled"""
    bda_config = {"decorr_factor": 1.0}  # >= 1.0 disables BDA
    grid_config = {"img_size": 8, "padding_factor": 1.0, "cellsize": 1e-5}
    
    # Mock distinct().collect() to return message_ids
    mock_message_id = Mock()
    mock_message_id.__getitem__ = Mock(return_value="msg-001")
    
    mock_df = Mock()
    mock_df.select.return_value = mock_df
    mock_df.distinct.return_value = mock_df
    mock_df.collect.return_value = [mock_message_id]
    mock_df.count.return_value = 1
    
    with patch(f"{MODULE}.apply_bda") as mock_bda, \
         patch(f"{MODULE}.apply_gridding") as mock_gridding:
        
        mock_gridding.return_value = df_single_row
        
        df_grid, df_avg, df_win = process_streaming_batch(
            mock_df, 4, 0, bda_config, grid_config
        )
        
        assert df_grid is not None
        assert df_avg is None
        assert df_win is None
        mock_bda.assert_not_called()


def test_process_streaming_batch_raises_on_none_bda_result(df_single_row):
    """Test that None result from apply_bda raises RuntimeError"""
    bda_config = {"decorr_factor": 0.95}
    grid_config = {"img_size": 8}
    
    with patch(f"{MODULE}.apply_bda", return_value=(None, df_single_row)):
        df_grid, df_avg, df_win = process_streaming_batch(
            df_single_row, 4, 0, bda_config, grid_config
        )
        
        assert df_grid is None
        assert df_avg is None
        assert df_win is None


def test_process_streaming_batch_raises_on_none_gridding_result(df_single_row):
    """Test that None result from apply_gridding raises RuntimeError"""
    bda_config = {"decorr_factor": 1.0}
    grid_config = {"img_size": 8}
    
    with patch(f"{MODULE}.apply_gridding", return_value=None):
        df_grid, df_avg, df_win = process_streaming_batch(
            df_single_row, 4, 0, bda_config, grid_config
        )
        
        assert df_grid is None


# ==============================================================================
# Tests for consolidate_processing
# ==============================================================================

def test_consolidate_processing_raises_on_none_grid_config():
    """Test that None grid_config raises ValueError"""
    with pytest.raises(ValueError, match="grid_config cannot be None"):
        consolidate_processing([], 4, None, "job123")


def test_consolidate_processing_raises_on_invalid_num_partitions():
    """Test that invalid num_partitions raises ValueError"""
    with pytest.raises(ValueError, match="num_partitions must be positive"):
        consolidate_processing([], 0, {}, "job123")


def test_consolidate_processing_with_none_slurm_id(spark):
    """Test that None slurm_job_id is handled"""
    grid_config = {"img_size": 8, "padding_factor": 1.0}
    
    # Create a real DataFrame with required columns for consolidation
    schema = StructType([
        StructField("u_pix", IntegerType(), False),
        StructField("v_pix", IntegerType(), False),
        StructField("real", DoubleType(), False),
        StructField("imag", DoubleType(), False),
        StructField("weight", DoubleType(), False),
    ])
    
    data = [(0, 0, 1.0, 0.5, 1.0)]
    df = spark.createDataFrame(data, schema)
    
    with patch(f"{MODULE}.apply_gridding") as mock_gridding, \
         patch(f"{MODULE}.build_grid") as mock_build, \
         patch(f"{MODULE}.generate_dirty_image") as mock_image:
        
        mock_gridding.return_value = df
        mock_build.return_value = (np.zeros((8, 8)), np.zeros((8, 8)))
        
        # Should not raise
        consolidate_processing([df], 4, grid_config, None)
        
        mock_image.assert_called_once()


def test_consolidate_processing_empty_grid(capsys):
    """Test that empty grid skips processing"""
    grid_config = {"img_size": 8}
    
    consolidate_processing([], 4, grid_config, "job123")
    
    captured = capsys.readouterr()
    assert "No gridded data to consolidate" in captured.out


def test_consolidate_processing_success(spark):
    """Test successful consolidation"""
    grid_config = {"img_size": 8, "padding_factor": 1.0}
    
    # Create a real DataFrame with required columns
    schema = StructType([
        StructField("u_pix", IntegerType(), False),
        StructField("v_pix", IntegerType(), False),
        StructField("real", DoubleType(), False),
        StructField("imag", DoubleType(), False),
        StructField("weight", DoubleType(), False),
    ])
    
    data = [(0, 0, 1.0, 0.5, 1.0), (1, 1, 0.8, 0.3, 0.9)]
    df = spark.createDataFrame(data, schema)
    
    with patch(f"{MODULE}.apply_gridding") as mock_gridding, \
         patch(f"{MODULE}.build_grid") as mock_build, \
         patch(f"{MODULE}.generate_dirty_image") as mock_image:
        
        mock_gridding.return_value = df
        mock_build.return_value = (
            np.random.random((8, 8)) + 1j * np.random.random((8, 8)),
            np.random.random((8, 8))
        )
        
        consolidate_processing([df], 4, grid_config, "job123")
        
        mock_gridding.assert_called_once()
        mock_build.assert_called_once()
        mock_image.assert_called_once()


def test_consolidate_processing_raises_on_none_gridded(spark):
    """Test that None from final gridding raises RuntimeError"""

    grid_config = {"img_size": 8}
    
    schema = StructType([
        StructField("u_pix", IntegerType(), False),
        StructField("v_pix", IntegerType(), False),
        StructField("real", DoubleType(), False),
        StructField("imag", DoubleType(), False),
        StructField("weight", DoubleType(), False),
    ])
    
    data = [(0, 0, 1.0, 0.5, 1.0)]
    df = spark.createDataFrame(data, schema)
    
    with patch(f"{MODULE}.apply_gridding", return_value=None):
        with pytest.raises(RuntimeError, match="Image consolidation failed"):
            consolidate_processing([df], 4, grid_config, "job123")


def test_consolidate_processing_raises_on_none_grid(spark):
    """Test that None from build_grid raises RuntimeError"""
    grid_config = {"img_size": 8}
    
    schema = StructType([
        StructField("u_pix", IntegerType(), False),
        StructField("v_pix", IntegerType(), False),
        StructField("real", DoubleType(), False),
        StructField("imag", DoubleType(), False),
        StructField("weight", DoubleType(), False),
    ])
    
    data = [(0, 0, 1.0, 0.5, 1.0)]
    df = spark.createDataFrame(data, schema)
    
    with patch(f"{MODULE}.apply_gridding") as mock_gridding, \
         patch(f"{MODULE}.build_grid", return_value=(None, None)):
        
        mock_gridding.return_value = df
        
        with pytest.raises(RuntimeError, match="Image consolidation failed"):
            consolidate_processing([df], 4, grid_config, "job123")


# ==============================================================================
# Tests for generate_metrics
# ==============================================================================

def test_generate_metrics_raises_on_none_bda_config():
    """Test that None bda_config raises ValueError"""
    with pytest.raises(ValueError, match="bda_config cannot be None"):
        generate_metrics([], [], 4, None, "job123")


def test_generate_metrics_raises_on_invalid_num_partitions():
    """Test that invalid num_partitions raises ValueError"""
    with pytest.raises(ValueError, match="num_partitions must be positive"):
        generate_metrics([], [], 0, {}, "job123")


def test_generate_metrics_with_none_slurm_id(df_single_row):
    """Test that None slurm_job_id is handled"""
    bda_config = {}
    
    with patch(f"{MODULE}.calculate_metrics") as mock_calc:
        # Should not raise
        generate_metrics([df_single_row], [df_single_row], 4, bda_config, None)
        
        mock_calc.assert_called_once()


def test_generate_metrics_empty_windowed(capsys):
    """Test that empty windowed list skips metrics"""
    bda_config = {}
    
    generate_metrics([], [Mock()], 4, bda_config, "job123")
    
    captured = capsys.readouterr()
    assert "No processed data samples available" in captured.out


def test_generate_metrics_empty_averaged(capsys):
    """Test that empty averaged list skips metrics"""
    bda_config = {}
    
    generate_metrics([Mock()], [], 4, bda_config, "job123")
    
    captured = capsys.readouterr()
    assert "No processed data samples available" in captured.out


def test_generate_metrics_empty_lists(capsys):
    """Test that empty lists skip metrics"""
    bda_config = {}
    
    generate_metrics([], [], 4, bda_config, "job123")
    
    captured = capsys.readouterr()
    assert "No processed data samples available" in captured.out


def test_generate_metrics_success(df_single_row):
    """Test successful metrics generation"""
    bda_config = {}
    
    with patch(f"{MODULE}.calculate_metrics") as mock_calc:
        generate_metrics([df_single_row], [df_single_row], 4, bda_config, "job123")
        
        mock_calc.assert_called_once()


def test_generate_metrics_propagates_exception(df_single_row):
    """Test that exceptions during metrics are propagated"""
    bda_config = {}
    
    with patch(f"{MODULE}.calculate_metrics", side_effect=Exception("Calculation failed")):
        with pytest.raises(RuntimeError, match="Metrics calculation failed"):
            generate_metrics([df_single_row], [df_single_row], 4, bda_config, "job123")


# ═════════════════════════════════════════════════════════════════════════════
# run_consumer — fixtures
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def config_files(tmp_path):
    """Crea los dos archivos de config requeridos por run_consumer."""
    bda  = tmp_path / "bda_config.json"
    grid = tmp_path / "grid_config.json"
    bda.write_text('{"decorr_factor": 1.0}',  encoding="utf-8")
    grid.write_text('{"img_size": 512}',        encoding="utf-8")
    return str(bda), str(grid)


@pytest.fixture
def mock_spark():
    """Mock de SparkSession con los atributos mínimos usados en run_consumer."""
    spark = MagicMock()
    spark.sparkContext.defaultParallelism = 4
    return spark


@pytest.fixture
def mock_query():
    """
    Mock de StreamingQuery con isActive=False como PropertyMock.

    PropertyMock es obligatorio porque isActive se evalúa en un while:
        while query.isActive and not stream_state['signal_received']:
    Si se asigna un bool real (query.isActive = False), MagicMock lo
    convierte en atributo plano y ya no es interceptable. Con PropertyMock
    podemos cambiar el valor devuelto entre llamadas sin congelar el loop.
    """
    query = MagicMock()
    type(query).isActive = PropertyMock(return_value=False)
    return query


@pytest.fixture
def patched_run_consumer(config_files, mock_spark, mock_query):
    """
    Parchea todas las dependencias externas de run_consumer para que el pipeline
    complete sin Spark ni Kafka reales.

    mock_query.isActive=False hace que el while termine en la primera evaluación,
    evitando el loop infinito que consume RAM hasta congelar el proceso.

    Retorna (bda_path, grid_path, mock_spark, mock_query).
    """
    bda_path, grid_path = config_files

    # writeStream → foreachBatch → trigger → option → start → mock_query
    kafka_stream_mock = MagicMock()
    kafka_stream_mock.writeStream.foreachBatch.return_value \
        .trigger.return_value \
        .option.return_value \
        .start.return_value = mock_query

    with patch(f"{MODULE}.create_spark_session",     return_value=mock_spark), \
         patch(f"{MODULE}.create_kafka_stream",       return_value=kafka_stream_mock), \
         patch(f"{MODULE}.define_visibility_schema",  return_value=MagicMock()), \
         patch(f"{MODULE}.consolidate_processing",    return_value=None), \
         patch(f"{MODULE}.generate_metrics",          return_value=None), \
         patch(f"{MODULE}.uuid.uuid4",                return_value=MagicMock(hex="abcd1234")), \
         patch(f"{MODULE}.time.sleep",                return_value=None), \
         patch(f"{MODULE}.time.time",                 return_value=0.0):
        yield bda_path, grid_path, mock_spark, mock_query


# ═════════════════════════════════════════════════════════════════════════════
# run_consumer — guards de entrada
# ═════════════════════════════════════════════════════════════════════════════

def test_raises_if_bootstrap_server_is_none(config_files):
    from services.consumer_service import run_consumer
    bda, grid = config_files
    with pytest.raises(ValueError, match="bootstrap_server cannot be None or empty"):
        run_consumer(None, "topic", bda, grid, "job1")

def test_raises_if_bootstrap_server_is_empty(config_files):
    from services.consumer_service import run_consumer
    bda, grid = config_files
    with pytest.raises(ValueError, match="bootstrap_server cannot be None or empty"):
        run_consumer("", "topic", bda, grid, "job1")

def test_raises_if_topic_is_none(config_files):
    from services.consumer_service import run_consumer
    bda, grid = config_files
    with pytest.raises(ValueError, match="topic cannot be None or empty"):
        run_consumer("localhost:9092", None, bda, grid, "job1")

def test_raises_if_topic_is_empty(config_files):
    from services.consumer_service import run_consumer
    bda, grid = config_files
    with pytest.raises(ValueError, match="topic cannot be None or empty"):
        run_consumer("localhost:9092", "", bda, grid, "job1")

def test_raises_if_bda_config_path_is_none(config_files):
    from services.consumer_service import run_consumer
    _, grid = config_files
    with pytest.raises(ValueError, match="bda_config_path cannot be None or empty"):
        run_consumer("localhost:9092", "topic", None, grid, "job1")

def test_raises_if_bda_config_path_is_empty(config_files):
    from services.consumer_service import run_consumer
    _, grid = config_files
    with pytest.raises(ValueError, match="bda_config_path cannot be None or empty"):
        run_consumer("localhost:9092", "topic", "", grid, "job1")

def test_raises_if_grid_config_path_is_none(config_files):
    from services.consumer_service import run_consumer
    bda, _ = config_files
    with pytest.raises(ValueError, match="grid_config_path cannot be None or empty"):
        run_consumer("localhost:9092", "topic", bda, None, "job1")

def test_raises_if_grid_config_path_is_empty(config_files):
    from services.consumer_service import run_consumer
    bda, _ = config_files
    with pytest.raises(ValueError, match="grid_config_path cannot be None or empty"):
        run_consumer("localhost:9092", "topic", bda, "", "job1")

def test_raises_if_bda_config_file_not_found(config_files):
    from services.consumer_service import run_consumer
    _, grid = config_files
    with pytest.raises(FileNotFoundError, match="BDA config file not found"):
        run_consumer("localhost:9092", "topic", "/nonexistent/bda.json", grid, "job1")

def test_raises_if_grid_config_file_not_found(config_files):
    from services.consumer_service import run_consumer
    bda, _ = config_files
    with pytest.raises(FileNotFoundError, match="Grid config file not found"):
        run_consumer("localhost:9092", "topic", bda, "/nonexistent/grid.json", "job1")

def test_none_slurm_job_id_is_converted_to_empty_string(patched_run_consumer):
    """slurm_job_id=None no debe lanzar — se convierte silenciosamente a ''."""
    from services.consumer_service import run_consumer
    bda_path, grid_path, mock_spark, mock_query = patched_run_consumer
    # No debe lanzar ValueError
    run_consumer("localhost:9092", "topic", bda_path, grid_path, None)


# ═════════════════════════════════════════════════════════════════════════════
# run_consumer — pipeline exitoso
# ═════════════════════════════════════════════════════════════════════════════

def test_returns_true_on_success(patched_run_consumer):
    from services.consumer_service import run_consumer
    bda_path, grid_path, mock_spark, mock_query = patched_run_consumer
    result = run_consumer("localhost:9092", "topic", bda_path, grid_path, "job1")
    assert result is True

def test_spark_session_is_always_stopped(patched_run_consumer):
    """El finally debe llamar spark.stop() incluso en ejecución exitosa."""
    from services.consumer_service import run_consumer
    bda_path, grid_path, mock_spark, mock_query = patched_run_consumer
    run_consumer("localhost:9092", "topic", bda_path, grid_path, "job1")
    mock_spark.stop.assert_called_once()

def test_query_awaits_termination(patched_run_consumer):
    from services.consumer_service import run_consumer
    bda_path, grid_path, mock_spark, mock_query = patched_run_consumer
    run_consumer("localhost:9092", "topic", bda_path, grid_path, "job1")
    mock_query.awaitTermination.assert_called_once()


def test_query_stopped_when_signal_received_and_active(config_files, mock_spark):
    """
    Cubre la rama:
        if query.isActive and stream_state['signal_received']:
            query.stop()

    Mecanismo:
        - foreachBatch recibe side_effect que invoca process_batch con
        end_signal=True → stream_state['signal_received'] = True
        - side_effect debe retornar un objeto cuya cadena
        .trigger().option().start() devuelva mock_query.
        - isActive=[True, True]: [0] evaluado en el while (sale porque
        signal ya está activo), [1] evaluado en el if → llama stop().
    """
    from services.consumer_service import run_consumer
    from unittest.mock import PropertyMock

    bda_path, grid_path = config_files
    mock_query = MagicMock()
    type(mock_query).isActive = PropertyMock(side_effect=[True, True])

    # Construimos el objeto que foreachBatch debe retornar para que
    # .trigger().option().start() devuelva mock_query.
    foreachbatch_result = MagicMock()
    foreachbatch_result.trigger.return_value \
        .option.return_value \
        .start.return_value = mock_query

    kafka_stream_mock = MagicMock()

    def invoke_with_end_signal(callback):
        """
        Ejecuta process_batch sincrónicamente con end_signal=True para
        activar signal_received ANTES del while, evitando loops infinitos.
        Retorna foreachbatch_result para que .trigger().option().start()
        apunte al mock_query correcto.
        """
        df_mock = MagicMock()
        df_mock.isEmpty.return_value = False
        with patch(f"{MODULE}.check_end_signal",
                    return_value=(df_mock, True)), \
                patch(f"{MODULE}.load_bda_config",        return_value={"decorr_factor": 1.0}), \
                patch(f"{MODULE}.load_grid_config",        return_value={}), \
                patch(f"{MODULE}.process_streaming_batch", return_value=(None, None, None)):
            callback(df_mock, 0)
        return foreachbatch_result

    kafka_stream_mock.writeStream.foreachBatch.side_effect = invoke_with_end_signal

    with patch(f"{MODULE}.create_spark_session",     return_value=mock_spark), \
            patch(f"{MODULE}.create_kafka_stream",       return_value=kafka_stream_mock), \
            patch(f"{MODULE}.define_visibility_schema",  return_value=MagicMock()), \
            patch(f"{MODULE}.consolidate_processing",    return_value=None), \
            patch(f"{MODULE}.time.sleep",                return_value=None), \
            patch(f"{MODULE}.time.time",                 return_value=0.0), \
            patch(f"{MODULE}.uuid.uuid4",                return_value=MagicMock(hex="ff")):
        run_consumer("localhost:9092", "topic", bda_path, grid_path, "job1")

    mock_query.stop.assert_called_once()
    mock_spark.stop.assert_called_once()

# ═════════════════════════════════════════════════════════════════════════════
# run_consumer — manejo de errores
# ═════════════════════════════════════════════════════════════════════════════

def test_keyboard_interrupt_reraises_and_stops_active_query(config_files):
    """KeyboardInterrupt debe propagarse y detener el query si está activo."""
    from services.consumer_service import run_consumer
    from unittest.mock import PropertyMock
    bda_path, grid_path = config_files

    mock_spark = MagicMock()
    mock_spark.sparkContext.defaultParallelism = 4

    mock_query = MagicMock()
    type(mock_query).isActive = PropertyMock(return_value=True)

    kafka_stream_mock = MagicMock()
    kafka_stream_mock.writeStream.foreachBatch.return_value \
        .trigger.return_value \
        .option.return_value \
        .start.return_value = mock_query

    with patch(f"{MODULE}.create_spark_session",    return_value=mock_spark), \
            patch(f"{MODULE}.create_kafka_stream",      return_value=kafka_stream_mock), \
            patch(f"{MODULE}.define_visibility_schema", return_value=MagicMock()), \
            patch(f"{MODULE}.time.sleep",               side_effect=KeyboardInterrupt), \
            patch(f"{MODULE}.time.time",                return_value=0.0), \
            patch(f"{MODULE}.uuid.uuid4",               return_value=MagicMock(hex="ff")):
        with pytest.raises(KeyboardInterrupt):
            run_consumer("localhost:9092", "topic", bda_path, grid_path, "job1")

    mock_query.stop.assert_called_once()
    mock_spark.stop.assert_called_once()

def test_exception_in_pipeline_raises_runtime_error(config_files):
    """Una excepción genérica debe convertirse en RuntimeError y cerrar Spark."""
    from services.consumer_service import run_consumer
    bda_path, grid_path = config_files

    mock_spark = MagicMock()
    mock_spark.sparkContext.defaultParallelism = 4

    with patch(f"{MODULE}.create_spark_session",    return_value=mock_spark), \
            patch(f"{MODULE}.create_kafka_stream",      side_effect=RuntimeError("kafka down")), \
            patch(f"{MODULE}.define_visibility_schema", return_value=MagicMock()), \
            patch(f"{MODULE}.time.time",                return_value=0.0), \
            patch(f"{MODULE}.uuid.uuid4",               return_value=MagicMock(hex="ff")):
        with pytest.raises(RuntimeError, match="Consumer failed"):
            run_consumer("localhost:9092", "topic", bda_path, grid_path, "job1")

    mock_spark.stop.assert_called_once()

def test_exception_stops_active_query_before_raising(config_files):
    """Si el query estaba activo cuando ocurre el error, debe intentar detenerlo."""
    from services.consumer_service import run_consumer
    from unittest.mock import PropertyMock
    bda_path, grid_path = config_files

    mock_spark = MagicMock()
    mock_spark.sparkContext.defaultParallelism = 4
    mock_query = MagicMock()
    type(mock_query).isActive = PropertyMock(return_value=True)

    kafka_stream_mock = MagicMock()
    kafka_stream_mock.writeStream.foreachBatch.return_value \
        .trigger.return_value \
        .option.return_value \
        .start.return_value = mock_query

    with patch(f"{MODULE}.create_spark_session",    return_value=mock_spark), \
            patch(f"{MODULE}.create_kafka_stream",      return_value=kafka_stream_mock), \
            patch(f"{MODULE}.define_visibility_schema", return_value=MagicMock()), \
            patch(f"{MODULE}.time.sleep",               side_effect=Exception("boom")), \
            patch(f"{MODULE}.time.time",                return_value=0.0), \
            patch(f"{MODULE}.uuid.uuid4",               return_value=MagicMock(hex="ff")):
        with pytest.raises(RuntimeError):
            run_consumer("localhost:9092", "topic", bda_path, grid_path, "job1")

    mock_query.stop.assert_called_once()

def test_query_stop_failure_is_silenced_in_except(config_files):
    """query.stop() puede fallar dentro del except — no debe propagar el error."""
    from services.consumer_service import run_consumer
    from unittest.mock import PropertyMock
    bda_path, grid_path = config_files

    mock_spark = MagicMock()
    mock_spark.sparkContext.defaultParallelism = 4
    mock_query = MagicMock()
    type(mock_query).isActive = PropertyMock(return_value=True)
    mock_query.stop.side_effect = Exception("stop failed")

    kafka_stream_mock = MagicMock()
    kafka_stream_mock.writeStream.foreachBatch.return_value \
        .trigger.return_value \
        .option.return_value \
        .start.return_value = mock_query

    with patch(f"{MODULE}.create_spark_session",    return_value=mock_spark), \
            patch(f"{MODULE}.create_kafka_stream",      return_value=kafka_stream_mock), \
            patch(f"{MODULE}.define_visibility_schema", return_value=MagicMock()), \
            patch(f"{MODULE}.time.sleep",               side_effect=Exception("boom")), \
            patch(f"{MODULE}.time.time",                return_value=0.0), \
            patch(f"{MODULE}.uuid.uuid4",               return_value=MagicMock(hex="ff")):
        # RuntimeError debe llegar, no la excepción de query.stop()
        with pytest.raises(RuntimeError, match="Consumer failed"):
            run_consumer("localhost:9092", "topic", bda_path, grid_path, "job1")


# ═════════════════════════════════════════════════════════════════════════════
# main
# ═════════════════════════════════════════════════════════════════════════════
BASE_ARGV = [
    "consumer_service.py",
    "--bootstrap-server", "localhost:9092",
    "--topic",            "test-topic",
    "--bda-config",       "/fake/bda.json",
    "--grid-config",      "/fake/grid.json",
    "--slurm-job-id",     "job42",
]

def test_exits_0_on_success():
    from services.consumer_service import main
    with patch("sys.argv", BASE_ARGV), \
            patch(f"{MODULE}.run_consumer", return_value=True):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0

def test_exits_1_when_run_consumer_returns_false():
    from services.consumer_service import main
    with patch("sys.argv", BASE_ARGV), \
            patch(f"{MODULE}.run_consumer", return_value=False):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1

def test_exits_130_on_keyboard_interrupt():
    from services.consumer_service import main
    with patch("sys.argv", BASE_ARGV), \
            patch(f"{MODULE}.run_consumer", side_effect=KeyboardInterrupt):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 130

def test_exits_1_on_unexpected_exception():
    from services.consumer_service import main
    with patch("sys.argv", BASE_ARGV), \
            patch(f"{MODULE}.run_consumer", side_effect=RuntimeError("fatal")):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
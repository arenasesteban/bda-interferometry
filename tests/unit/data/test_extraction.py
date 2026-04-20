import io
import numpy as np
import numpy.testing as npt
import pytest
import msgpack
from unittest.mock import MagicMock, patch

from data.extraction import (
    create_dask_client,
    rechunk_dataset,
    create_kafka_producer,
    close_kafka_producer,
    iter_rows,
    compute_chunk,
    build_payload,
    build_metadata,
    send_visibilities,
    send_end_signal,
    stream_dataset
)

MODULE = "src.data.extraction"
NROWS  = 5
NCHAN  = 4
NCORR  = 2

# Test cases for create_dask_client function

def test_returns_client():
    mock_client = MagicMock()
    with patch(f"{MODULE}.dask.config.set"), \
            patch(f"{MODULE}.Client", return_value=mock_client) as mock_cls:
        result = create_dask_client()
        mock_cls.assert_called_once()
        assert result is mock_client


def test_uses_dask_dir_env_variable(monkeypatch):
    monkeypatch.setenv("DASK_DIR", "/custom/dask/dir")
    mock_client = MagicMock()
    with patch(f"{MODULE}.dask.config.set"), \
            patch(f"{MODULE}.Client", return_value=mock_client) as mock_cls:
        create_dask_client()
        _, kwargs = mock_cls.call_args
        assert kwargs["local_directory"] == "/custom/dask/dir"


def test_uses_default_dask_dir_when_env_not_set(monkeypatch):
    monkeypatch.delenv("DASK_DIR", raising=False)
    mock_client = MagicMock()
    with patch(f"{MODULE}.dask.config.set"), \
            patch(f"{MODULE}.Client", return_value=mock_client) as mock_cls:
        create_dask_client()
        _, kwargs = mock_cls.call_args
        assert kwargs["local_directory"] == "tmp/dask"


# Test cases for rechunk_dataset function

def test_raises_if_dataset_is_none():
    with pytest.raises(ValueError, match="Dataset cannot be None"):
        rechunk_dataset(None)


def test_raises_if_dataset_has_no_data_attribute():
    with pytest.raises(AttributeError, match="data.data"):
        rechunk_dataset(MagicMock(spec=[]))


def test_returns_tuple_of_ten_elements(mock_dataset):
    result = rechunk_dataset(mock_dataset)
    assert len(result) == 10


def test_rechunk_skipped_when_chunks_match(mock_dataset):
    row_chunks = mock_dataset.data.data.chunks[0]
    mock_dataset.antenna1.data.chunks = (row_chunks,)
    result = rechunk_dataset(mock_dataset)
    mock_dataset.antenna1.data.rechunk.assert_not_called()


def test_raises_for_unsupported_dimensionality(mock_dataset):
    bad_array = MagicMock()
    bad_array.chunks = (((99,),),)
    bad_array.shape  = (NROWS, 4, 4, 4)
    mock_dataset.antenna1.data = bad_array
    with pytest.raises(ValueError, match="Unsupported array dimensionality"):
        rechunk_dataset(mock_dataset)


# Test cases for create_kafka_producer function

def test_returns_kafka_producer_instance():
    mock_kp = MagicMock()
    with patch(f"{MODULE}.KafkaProducer", return_value=mock_kp) as mock_cls:
        result = create_kafka_producer()
        mock_cls.assert_called_once()
        assert result is mock_kp


# Test cases for close_kafka_producer function

def test_flushes_and_closes_producer_if_present():
    mock_worker = MagicMock(spec=["_kafka_producer"])
    mock_worker._kafka_producer = MagicMock()
    with patch(f"{MODULE}.get_worker", return_value=mock_worker):
        close_kafka_producer()
        mock_worker._kafka_producer.flush.assert_called_once()
        mock_worker._kafka_producer.close.assert_called_once()


def test_does_nothing_if_worker_has_no_producer():
    mock_worker = MagicMock(spec=[])
    with patch(f"{MODULE}.get_worker", return_value=mock_worker):
        close_kafka_producer()


def test_silences_value_error_from_get_worker():
    with patch(f"{MODULE}.get_worker", side_effect=ValueError("no worker")):
        close_kafka_producer()


def test_prints_warning_on_unexpected_exception(capsys):
    mock_worker = MagicMock(spec=["_kafka_producer"])
    mock_worker._kafka_producer.flush.side_effect = RuntimeError("disk full")
    with patch(f"{MODULE}.get_worker", return_value=mock_worker):
        close_kafka_producer()
    captured = capsys.readouterr()
    assert "Warning" in captured.out


# Test cases for iter_rows function

def test_raises_if_nrows_is_not_int():
    with pytest.raises(ValueError, match="nrows must be a positive integer"):
        list(iter_rows(5.0, 2))


def test_raises_if_nrows_is_zero():
    with pytest.raises(ValueError, match="nrows must be a positive integer"):
        list(iter_rows(0, 2))


def test_raises_if_nrows_is_negative():
    with pytest.raises(ValueError, match="nrows must be a positive integer"):
        list(iter_rows(-1, 2))


def test_raises_if_rows_per_msg_is_not_int():
    with pytest.raises(ValueError, match="rows_per_msg must be a positive integer"):
        list(iter_rows(5, 2.0))


def test_raises_if_rows_per_msg_is_zero():
    with pytest.raises(ValueError, match="rows_per_msg must be a positive integer"):
        list(iter_rows(5, 0))


def test_exact_multiple():
    result = list(iter_rows(6, 2))
    assert result == [(0, 2), (2, 4), (4, 6)]


def test_partial_last_block():
    result = list(iter_rows(5, 2))
    assert result == [(0, 2), (2, 4), (4, 5)]


def test_single_block_when_nrows_less_than_block_size():
    result = list(iter_rows(3, 100))
    assert result == [(0, 3)]


def test_covers_all_rows():
    nrows = 17
    ranges = list(iter_rows(nrows, 4))
    covered = set()
    for start, end in ranges:
        covered.update(range(start, end))
    assert covered == set(range(nrows))


# Test cases for compute_chunk function

def test_raises_if_chunk_delayed_is_none():
    with pytest.raises(ValueError, match="chunk_delayed cannot be None"):
        compute_chunk(None)

def test_raises_if_wrong_number_of_elements():
    with pytest.raises(ValueError, match="10 elements"):
        compute_chunk([MagicMock()] * 5)

def test_returns_12_element_tuple():
    rng  = np.random.default_rng(1)
    uvw  = rng.random((NROWS, 3))
    fake_computed = (
        np.arange(NROWS),     # antenna1
        np.arange(NROWS) + 1, # antenna2
        np.ones(NROWS),       # scan_number
        np.linspace(0, 1, NROWS),  # time
        np.ones(NROWS),       # exposure
        np.ones(NROWS),       # interval
        uvw,                  # uvw
        rng.random((NROWS, NCHAN, NCORR)) + 1j * rng.random((NROWS, NCHAN, NCORR)),
        rng.random((NROWS, NCHAN)),  # weight
        np.zeros((NROWS, NCHAN)),    # flag
    )
    chunk_input = [MagicMock() for _ in range(10)]
    with patch(f"{MODULE}.dask.compute", return_value=fake_computed):
        result = compute_chunk(chunk_input)
    assert len(result) == 12
    npt.assert_array_equal(result[6], uvw[:, 0])  # u
    npt.assert_array_equal(result[7], uvw[:, 1])  # v
    npt.assert_array_equal(result[8], uvw[:, 2])  # w


# Test cases for build_payload function

def test_raises_if_data_is_none():
    with pytest.raises(ValueError, match="tuple of 12 arrays"):
        build_payload(None, 0, 2)

def test_raises_if_data_has_wrong_length():
    with pytest.raises(ValueError, match="tuple of 12 arrays"):
        build_payload(tuple(range(5)), 0, 2)

def test_raises_if_start_is_not_int(valid_data):
    with pytest.raises(ValueError, match="start and end must be integers"):
        build_payload(valid_data, 0.0, 2)

def test_raises_if_end_is_not_int(valid_data):
    with pytest.raises(ValueError, match="start and end must be integers"):
        build_payload(valid_data, 0, 2.0)

def test_raises_if_start_is_negative(valid_data):
    with pytest.raises(ValueError, match="Invalid row range"):
        build_payload(valid_data, -1, 2)

def test_raises_if_end_not_greater_than_start(valid_data):
    with pytest.raises(ValueError, match="Invalid row range"):
        build_payload(valid_data, 2, 2)

def test_returns_bytes(valid_data):
    result = build_payload(valid_data, 0, NROWS)
    assert isinstance(result, bytes)
    assert len(result) > 0

def test_payload_is_valid_npz(valid_data):
    result = build_payload(valid_data, 0, NROWS)
    loaded = np.load(io.BytesIO(result))
    expected_keys = {
        "antenna1", "antenna2", "scan_number", "time",
        "exposure", "interval", "u", "v", "w",
        "visibilities", "weights", "flags",
    }
    assert expected_keys == set(loaded.files)

def test_visibilities_stacked_as_real_imag(valid_data):
    result = build_payload(valid_data, 0, NROWS)
    loaded = np.load(io.BytesIO(result))
    vis = loaded["visibilities"]
    assert vis.shape == (NROWS, NCHAN, NCORR, 2)
    npt.assert_allclose(vis[..., 0], valid_data[7][0:NROWS].real, rtol=1e-6)
    npt.assert_allclose(vis[..., 1], valid_data[7][0:NROWS].imag, rtol=1e-6)

def test_flags_cast_to_int8(valid_data):
    result = build_payload(valid_data, 0, NROWS)
    loaded = np.load(io.BytesIO(result))
    assert loaded["flags"].dtype == np.int8

def test_partial_slice_uses_correct_rows(valid_data):
    result = build_payload(valid_data, 1, 3)
    loaded = np.load(io.BytesIO(result))
    assert loaded["antenna1"].shape[0] == 2


# Test cases for build_metadata function

def test_raises_if_message_id_is_none(valid_metadata):
    with pytest.raises(ValueError, match="message_id cannot be None or empty"):
        build_metadata(None, valid_metadata)

def test_raises_if_message_id_is_empty(valid_metadata):
    with pytest.raises(ValueError, match="message_id cannot be None or empty"):
        build_metadata("", valid_metadata)

def test_raises_if_data_is_none():
    with pytest.raises(ValueError, match="data dictionary cannot be None"):
        build_metadata("msg-1", None)

def test_raises_if_required_field_missing(valid_metadata):
    del valid_metadata["subms_id"]
    with pytest.raises(KeyError, match="subms_id"):
        build_metadata("msg-1", valid_metadata)

def test_returns_list_of_two_tuples(valid_metadata):
    result = build_metadata("msg-1", valid_metadata)
    assert isinstance(result, list)
    assert len(result) == 2

def test_first_header_is_schema(valid_metadata):
    result = build_metadata("msg-1", valid_metadata)
    assert result[0] == ("schema", b"visibilities_blocks")


def test_second_header_contains_msgpack_bytes(valid_metadata):
    result = build_metadata("msg-1", valid_metadata)
    key, value = result[1]
    assert key == "metadata"
    unpacked = msgpack.unpackb(value, raw=False)
    assert unpacked["message_id"]  == "msg-1"
    assert unpacked["n_channels"]  == NCHAN
    assert unpacked["schema"]      == "visibilities_blocks"


# Test cases for send_visibilities function

def _make_chunk():
    rng = np.random.default_rng(2)
    return (
        np.arange(NROWS, dtype=np.int32),
        np.arange(NROWS, dtype=np.int32) + 1,
        np.ones(NROWS, dtype=np.int32),
        np.linspace(0, 1, NROWS),
        np.ones(NROWS),
        np.ones(NROWS),
        rng.random((NROWS, 3)),
        rng.random((NROWS, NCHAN, NCORR)) + 1j * rng.random((NROWS, NCHAN, NCORR)),
        rng.random((NROWS, NCHAN)),
        np.zeros((NROWS, NCHAN), dtype=np.bool_),
        np.zeros(NROWS),
        np.zeros(NROWS),
    )


def test_raises_if_producer_is_none(valid_metadata):
    chunk = _make_chunk()
    with pytest.raises(ValueError, match="Kafka producer cannot be None"):
        send_visibilities(*chunk, chunk_id=0, data=valid_metadata,
                            producer=None, topic="test-topic")


def test_raises_if_topic_is_none(mock_producer, valid_metadata):
    chunk = _make_chunk()
    with pytest.raises(ValueError, match="Kafka topic cannot be None or empty"):
        send_visibilities(*chunk, chunk_id=0, data=valid_metadata,
                            producer=mock_producer, topic=None)


def test_raises_if_topic_is_empty(mock_producer, valid_metadata):
    chunk = _make_chunk()
    with pytest.raises(ValueError, match="Kafka topic cannot be None or empty"):
        send_visibilities(*chunk, chunk_id=0, data=valid_metadata,
                            producer=mock_producer, topic="")


def test_returns_true_on_success(mock_producer, valid_metadata):
    chunk = _make_chunk()
    result = send_visibilities(*chunk, chunk_id=0, data=valid_metadata,
                                producer=mock_producer, topic="test-topic")
    assert result is True


def test_producer_send_called_once_per_block(mock_producer, valid_metadata):
    chunk = _make_chunk()
    send_visibilities(*chunk, chunk_id=0, data=valid_metadata,
                        producer=mock_producer, topic="test-topic")
    assert mock_producer.send.call_count == 1


# Test cases for send_end_signal function

def test_raises_if_producer_is_none():
    with pytest.raises(ValueError, match="Kafka producer cannot be None"):
        send_end_signal(None, "topic", 10)


def test_raises_if_topic_is_none(mock_producer):
    with pytest.raises(ValueError, match="Kafka topic cannot be None or empty"):
        send_end_signal(mock_producer, None, 10)


def test_raises_if_topic_is_empty(mock_producer):
    with pytest.raises(ValueError, match="Kafka topic cannot be None or empty"):
        send_end_signal(mock_producer, "", 10)


def test_raises_if_n_blocks_is_negative(mock_producer):
    with pytest.raises(ValueError, match="non-negative integer"):
        send_end_signal(mock_producer, "topic", -1)


def test_raises_if_n_blocks_is_not_int(mock_producer):
    with pytest.raises(ValueError, match="non-negative integer"):
        send_end_signal(mock_producer, "topic", 1.5)


def test_sends_end_message_with_end_key(mock_producer):
    send_end_signal(mock_producer, "test-topic", 5)
    mock_producer.send.assert_called_once()
    call_kwargs = mock_producer.send.call_args
    assert call_kwargs.kwargs["key"] == b"__END__"
    assert call_kwargs.kwargs["value"] is None


def test_zero_blocks_is_valid(mock_producer):
    send_end_signal(mock_producer, "topic", 0)
    mock_producer.send.assert_called_once()


# Test cases for stream_dataset function

def test_raises_if_dataset_is_none(mock_subms):
    with pytest.raises(ValueError, match="Dataset cannot be None"):
        stream_dataset(None, mock_subms, "topic")


def test_raises_if_subms_is_none(mock_dataset):
    with pytest.raises(ValueError, match="SubMS metadata cannot be None"):
        stream_dataset(mock_dataset, None, "topic")


def test_raises_if_topic_is_none(mock_dataset, mock_subms):
    with pytest.raises(ValueError, match="Kafka topic cannot be None or empty"):
        stream_dataset(mock_dataset, mock_subms, None)


def test_raises_if_topic_is_empty(mock_dataset, mock_subms):
    with pytest.raises(ValueError, match="Kafka topic cannot be None or empty"):
        stream_dataset(mock_dataset, mock_subms, "")


def test_raises_if_dataset_has_no_nrows(mock_subms):
    ds = MagicMock(spec=[])   # sin atributo nrows
    with pytest.raises(ValueError, match="positive nrows"):
        stream_dataset(ds, mock_subms, "topic")


def test_raises_if_nrows_is_zero(mock_dataset, mock_subms):
    mock_dataset.nrows = 0
    with pytest.raises(ValueError, match="positive nrows"):
        stream_dataset(mock_dataset, mock_subms, "topic")


def test_producer_flushed_and_closed_on_success(mock_dataset, mock_subms):
    mock_producer = MagicMock()
    mock_producer.send.return_value.get.return_value = None
    fake_arrays = tuple(np.zeros((NROWS,)) for _ in range(10))

    with patch(f"{MODULE}.create_kafka_producer",  return_value=mock_producer), \
            patch(f"{MODULE}.rechunk_dataset",        return_value=fake_arrays), \
            patch(f"{MODULE}.dask.compute",           return_value=fake_arrays), \
            patch(f"{MODULE}.send_visibilities",      return_value=True), \
            patch(f"{MODULE}.send_end_signal"):
        stream_dataset(mock_dataset, mock_subms, "topic")

    mock_producer.flush.assert_called_once()
    mock_producer.close.assert_called_once()


def test_producer_closed_even_on_exception(mock_dataset, mock_subms):
    mock_producer = MagicMock()
    fake_arrays = tuple(np.zeros((NROWS,)) for _ in range(10))

    with patch(f"{MODULE}.create_kafka_producer",  return_value=mock_producer), \
            patch(f"{MODULE}.rechunk_dataset",        return_value=fake_arrays), \
            patch(f"{MODULE}.dask.compute",           return_value=fake_arrays), \
            patch(f"{MODULE}.send_visibilities",      side_effect=RuntimeError("kafka down")), \
            pytest.raises(RuntimeError):
        stream_dataset(mock_dataset, mock_subms, "topic")

    mock_producer.flush.assert_called_once()
    mock_producer.close.assert_called_once()


def test_send_end_signal_called_with_total_blocks(mock_dataset, mock_subms):
    mock_producer = MagicMock()
    mock_producer.send.return_value.get.return_value = None
    fake_arrays = tuple(np.zeros((NROWS,)) for _ in range(10))

    with patch(f"{MODULE}.create_kafka_producer",  return_value=mock_producer), \
            patch(f"{MODULE}.rechunk_dataset",        return_value=fake_arrays), \
            patch(f"{MODULE}.dask.compute",           return_value=fake_arrays), \
            patch(f"{MODULE}.send_visibilities",      return_value=True), \
            patch(f"{MODULE}.send_end_signal") as mock_end:
        stream_dataset(mock_dataset, mock_subms, "topic")
        mock_end.assert_called_once()
        _, call_args, _ = mock_end.mock_calls[0]
        assert isinstance(call_args[2], int)
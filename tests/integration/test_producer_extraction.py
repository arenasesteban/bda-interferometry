import io
import msgpack
import numpy as np

from services.producer_service import stream_kafka


EXPECTED_PAYLOAD_KEYS = {
    "antenna1",
    "antenna2",
    "scan_number",
    "time",
    "exposure",
    "interval",
    "u",
    "v",
    "w",
    "visibilities",
    "weights",
    "flags",
    "baseline_length",
}


def test_producer_stream_publishes_valid_visibility_block_and_end_signal(
    publication_dataset,
    publication_topic,
    publication_bootstrap_servers,
    publication_run_id,
    mock_kafka_producer,
    monkeypatch,
):
    monkeypatch.setattr(
        "data.extraction.create_kafka_producer",
        lambda: mock_kafka_producer,
    )

    stream_kafka(
        dataset=publication_dataset,
        topic=publication_topic,
        bootstrap_servers=publication_bootstrap_servers,
        run_id=publication_run_id,
    )

    send_calls = mock_kafka_producer.send.call_args_list

    assert len(send_calls) == 2

    data_call = send_calls[0]
    end_call = send_calls[1]

    data_args, data_kwargs = data_call

    assert data_args[0] == publication_topic
    assert data_kwargs["key"] == b"message-0-0"
    assert isinstance(data_kwargs["value"], bytes)

    headers = dict(data_kwargs["headers"])

    assert headers["schema"] == b"visibilities_blocks"

    metadata = msgpack.unpackb(headers["metadata"], raw=False)

    assert metadata["schema"] == "visibilities_blocks"
    assert metadata["message_id"] == "0-0"
    assert metadata["subms_id"] == 0
    assert metadata["field_id"] == 1
    assert metadata["spw_id"] == 2
    assert metadata["polarization_id"] == 0
    assert metadata["n_channels"] == 4
    assert metadata["n_correlations"] == 2

    with io.BytesIO(data_kwargs["value"]) as buffer:
        payload = np.load(buffer)

        assert set(payload.files) == EXPECTED_PAYLOAD_KEYS

        assert payload["antenna1"].shape[0] == 20
        assert payload["antenna2"].shape[0] == 20
        assert payload["scan_number"].shape[0] == 20
        assert payload["time"].shape[0] == 20
        assert payload["exposure"].shape[0] == 20
        assert payload["interval"].shape[0] == 20

        assert payload["u"].shape[0] == 20
        assert payload["v"].shape[0] == 20
        assert payload["w"].shape[0] == 20

        assert payload["visibilities"].shape == (20, 4, 2, 2)
        assert payload["weights"].shape == (20, 4)
        assert payload["flags"].shape == (20, 4)
        assert payload["flags"].dtype == np.int8
        assert payload["baseline_length"].shape == (20,)
        assert np.all(payload["baseline_length"] >= 0.0)

    end_args, end_kwargs = end_call

    assert end_args[0] == publication_topic
    assert end_kwargs["key"] == b"__END__"
    assert end_kwargs["value"] is None

    end_headers = dict(end_kwargs["headers"])

    assert end_headers["schema"] == b"visibilities_blocks"

    end_metadata = msgpack.unpackb(end_headers["metadata"], raw=False)

    assert end_metadata["schema"] == "visibilities_blocks"
    assert end_metadata["total_blocks"] == 1

    mock_kafka_producer.flush.assert_called_once()
    mock_kafka_producer.close.assert_called_once()
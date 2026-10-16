import pytest

from partd.buffer import Buffer
from partd.dict import Dict
from partd.file import File
from partd.pickle import Pickle


@pytest.fixture(params=[Dict, File], ids=["memory", "file"])
def buffer(request):
    with Buffer(Dict(), request.param()) as result:
        yield result


@pytest.mark.parametrize("flush_keys", [None, ["x"]], ids=["all", "selected"])
def test_append_order_across_flush(buffer, flush_keys):
    buffer.append({"x": b"first", "y": b"other"})
    buffer.flush(flush_keys)
    buffer.append({"x": b"second", "y": b"tail"})

    assert buffer.get("x") == b"firstsecond"
    assert buffer.get(["y", "x", "missing"]) == [b"othertail", b"firstsecond", b""]
    assert buffer.get([["x", "missing"], ["y", "x"]]) == [
        [b"firstsecond", b""],
        [b"othertail", b"firstsecond"],
    ]
    with buffer.lock:
        assert buffer.get("x", lock=False) == b"firstsecond"

    buffer.flush()
    assert buffer.get(["x", "y"]) == [b"firstsecond", b"othertail"]


def test_append_order_after_automatic_spill(buffer):
    buffer.available_memory = 4
    buffer.append({"x": b"first"})
    assert buffer.fast.get("x") == b""
    assert buffer.slow.get("x") == b"first"

    buffer.append({"x": b"+2"})
    assert buffer.fast.get("x") == b"+2"
    assert buffer.get("x") == b"first+2"


@pytest.mark.parametrize("flush_every", [1, 2, 5])
def test_append_order_over_repeated_flushes(buffer, flush_every):
    expected = b""
    for i in range(17):
        chunk = bytes([i])
        buffer.append({"x": chunk})
        expected += chunk
        assert buffer.get("x") == expected
        if i % flush_every == 0:
            buffer.flush(["x"])
            assert buffer.get("x") == expected
    buffer.flush()
    assert buffer.get("x") == expected


def test_append_order_with_preexisting_slow_data(buffer):
    buffer.slow.append({"x": b"existing"})
    buffer.append({"x": b"new"})
    assert buffer.get("x") == b"existingnew"


def test_pickle_record_order_across_flush(buffer):
    encoded = Pickle(buffer)
    first = [{"id": 1}, {"id": 2}]
    second = [{"id": 3}, {"id": 4}]
    encoded.append({"x": first})
    buffer.flush()
    encoded.append({"x": second})
    assert encoded.get("x") == first + second
    buffer.flush()
    assert encoded.get("x") == first + second


def test_append_order_with_empty_tiers(buffer):
    assert buffer.get(["missing", "x"]) == [b"", b""]
    buffer.append({"x": b"first"})
    assert buffer.get("x") == b"first"
    buffer.flush()
    assert buffer.get("x") == b"first"
    buffer.flush()
    assert buffer.get("x") == b"first"

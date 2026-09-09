import json

import pytest

from tehillim_cluster.atomic_write import write_json, write_text


def test_write_json_round_trips(tmp_path):
    path = tmp_path / "out" / "payload.json"

    write_json(path, {"rows": [1, 2, 3]})

    assert json.loads(path.read_text(encoding="utf-8")) == {"rows": [1, 2, 3]}


def test_write_json_refuses_nan_which_no_reader_accepts(tmp_path):
    with pytest.raises(ValueError, match="Out of range float"):
        write_json(tmp_path / "p.json", {"x": float("nan")})


def test_a_failed_write_keeps_the_previous_file(tmp_path):
    path = tmp_path / "p.json"
    write_json(path, {"kept": True})

    with pytest.raises(ValueError, match="Out of range float"):
        write_json(path, {"bad": float("nan")})

    assert json.loads(path.read_text(encoding="utf-8")) == {"kept": True}


def test_a_writer_that_fails_midway_leaves_no_temp_behind(tmp_path):
    from tehillim_cluster.atomic_write import replace_atomically

    path = tmp_path / "out.bin"

    def write_then_fail(target):
        target.write_text("half written", encoding="utf-8")
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        replace_atomically(path, write_then_fail)

    assert list(tmp_path.iterdir()) == []


def test_write_text_replaces_atomically(tmp_path):
    path = tmp_path / "page" / "index.html"

    write_text(path, "<html></html>")

    assert path.read_text(encoding="utf-8") == "<html></html>"


def test_the_compact_form_parses_back_to_the_same_payload(tmp_path):
    """Every site payload is written compact, so the compact form is the one that must parse."""
    path = tmp_path / "payload.json"
    payload = {"methods": [{"name": "m", "labels": [0, 1, 1]}], "k": 2}

    write_json(path, payload, compact=True)

    assert json.loads(path.read_text()) == payload


def test_the_compact_form_carries_no_separator_padding(tmp_path):
    """Compact is what keeps the published payloads small, so the padding must actually be gone."""
    path = tmp_path / "payload.json"

    write_json(path, {"a": 1, "b": 2}, compact=True)

    assert path.read_text() == '{"a":1,"b":2}'


def test_a_statistic_that_was_not_computed_is_written_as_null(tmp_path):
    """None is how an uncomputed statistic reaches the reader, and JSON can express it."""
    path = tmp_path / "payload.json"

    write_json(path, {"partition_p_value": None})

    assert json.loads(path.read_text()) == {"partition_p_value": None}


def test_hebrew_is_written_unescaped_so_the_payload_stays_readable(tmp_path):
    path = tmp_path / "payload.json"

    write_json(path, {"incipit": "אשרי"})

    assert "אשרי" in path.read_text()

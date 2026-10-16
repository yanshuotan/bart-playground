"""A transient Windows file lock must not terminate a sampling worker."""

import json
from pathlib import Path

from diagnosis.run_smc import write_json


def test_progress_write_retries_a_reader_lock(tmp_path, monkeypatch):
    path = tmp_path / "progress.json"
    path.write_text('{"stage": 1}')
    original = Path.replace
    calls = []

    def transient_lock(source, destination):
        calls.append(destination)
        if len(calls) < 3:
            raise PermissionError("destination held open by a reader")
        return original(source, destination)

    monkeypatch.setattr(Path, "replace", transient_lock)
    monkeypatch.setattr("diagnosis.run_smc.time.sleep", lambda _: None)
    write_json(path, {"stage": 2})
    assert json.loads(path.read_text()) == {"stage": 2}
    assert len(calls) == 3
    assert not path.with_suffix(".json.tmp").exists()

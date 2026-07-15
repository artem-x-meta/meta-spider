"""Tests for the safe cloud helper: protective invariants are properties of the code, not of discipline.

We mock vast (FakeVastClient) + inject ssh/scp — we never touch real vast.
"""
import json

import pytest

from daimon_loom import cloud


class FakeVastClient:
    """Records create/destroy; show is always running. No network."""

    def __init__(self):
        self.created = []
        self.destroyed = []
        self.next_id = 12345

    def search_offers(self, query):
        return [{"id": 999}]

    def create(self, offer_id, image, disk=50, label="metaloom"):
        self.created.append(offer_id)
        return self.next_id

    def show(self, instance_id):
        return {"id": instance_id, "actual_status": "running",
                "ssh_host": "fake.host", "ssh_port": 22222}

    def destroy(self, instance_id):
        self.destroyed.append(int(instance_id))
        return {"success": True}


def test_provision_arms_killswitch_before_returning(tmp_path):
    """kill-switch (sleep;shutdown) is sent to the instance BEFORE returning; state is recorded."""
    client = FakeVastClient()
    ssh_cmds = []
    sf = tmp_path / "state.json"
    inst = cloud.provision(
        client, gpu_query={}, image="img", deadline_min=1, state_file=sf, poll_sec=0,
        ssh_runner=lambda c: ssh_cmds.append(c) or "", scp_runner=lambda r, l: None,
    )
    assert client.created == [999]
    assert any("shutdown -h now" in c and "sleep 60" in c for c in ssh_cmds), \
        "kill-switch not armed"
    state = json.loads(sf.read_text(encoding="utf-8"))
    assert state[0]["id"] == 12345
    inst.destroy()  # cancel the timer
    assert client.destroyed == [12345]


def test_run_and_fetch_exfiltrates_even_on_failure(tmp_path):
    """run_and_fetch retrieves artifacts in finally — even if the ssh command failed."""
    client = FakeVastClient()
    fetched = []
    inst = cloud.Instance(
        1, "h", 22, client, deadline_min=1, state_file=tmp_path / "s.json",
        ssh_runner=lambda c: (_ for _ in ()).throw(RuntimeError("boom")),
        scp_runner=lambda r, l: fetched.append((r, l)),
    )
    with pytest.raises(RuntimeError):
        inst.run_and_fetch("cmd", artifacts=["/data/x.pt"], local_dir=str(tmp_path))
    assert fetched == [("/data/x.pt", str(tmp_path / "x.pt"))]


def test_context_manager_destroys_on_exception(tmp_path):
    """`with instance(...)` tears down the instance even on an exception in the body."""
    client = FakeVastClient()
    with pytest.raises(ValueError):
        with cloud.instance(
            client, gpu_query={}, image="img", deadline_min=1,
            state_file=tmp_path / "s.json", poll_sec=0,
            ssh_runner=lambda c: "", scp_runner=lambda r, l: None,
        ):
            raise ValueError("work failed")
    assert client.destroyed == [12345]


def test_reap_destroys_orphans_and_clears(tmp_path):
    """reap tears down everything in cloud_state.json (recovery after a crash) and clears the file."""
    client = FakeVastClient()
    sf = tmp_path / "s.json"
    cloud._write_state(sf, [{"id": 111}, {"id": 222}])
    killed = cloud.reap(client, sf)
    assert set(killed) == {111, 222}
    assert set(client.destroyed) == {111, 222}
    assert cloud._read_state(sf) == []


def test_destroy_is_idempotent(tmp_path):
    client = FakeVastClient()
    inst = cloud.Instance(5, "h", 22, client, deadline_min=1, state_file=tmp_path / "s.json")
    inst.destroy()
    inst.destroy()
    assert client.destroyed == [5]


def test_provision_destroys_if_not_running(tmp_path):
    """If the instance didn't reach running within the timeout — tear it down, don't leave it hanging."""
    client = FakeVastClient()
    client.show = lambda iid: {"id": iid, "actual_status": "loading"}  # never running
    with pytest.raises(RuntimeError):
        cloud.provision(client, gpu_query={}, image="img", deadline_min=1,
                        state_file=tmp_path / "s.json", poll_sec=0, timeout_sec=0.05,
                        ssh_runner=lambda c: "", scp_runner=lambda r, l: None)
    assert client.destroyed == [12345]  # tore down the stuck one

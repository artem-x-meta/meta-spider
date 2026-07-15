"""Safe cloud helper (vast.ai): provision/run_and_fetch/destroy with a kill-switch.

Codifies the lesson from the 17.06 incident (idle billing → $0 balance): the protective
invariants are code properties, not discipline:

  1. **The kill-switch is ARMED BEFORE any work.** Right after `running`, an autonomous
     `sleep <deadline>; shutdown` is launched on the instance — it kills GPU billing even if
     the laptop drops off or the agent hangs on a blocking question. Plus a laptop-side timer
     backstop.
  2. **exfiltrate-then-destroy by construction.** `run_and_fetch` pulls artifacts BACK
     (in `finally`, success OR error) before returning; destroy runs in the context manager's
     `finally`.
  3. **Orphan reaping.** `cloud_state.json` is written at creation time → `reap()` (or
     `metaloom cloud reap`) tears down instances orphaned after an agent crash.

Vast has no native time/$ limit per instance (only a stop at $0 balance) — so we hold the
deadline ourselves. The real vast calls are isolated in `VastClient` (mocked in tests);
ssh/scp are injectable.
"""
from __future__ import annotations

import atexit
import json
import re
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Optional

__all__ = ["VastClient", "Instance", "provision", "instance", "reap", "add_args", "run"]

DEFAULT_STATE = Path.home() / ".metaloom" / "cloud_state.json"
_API = "https://console.vast.ai/api/v0"


# ────────────────────────── vast API ──────────────────────────

def _load_api_key(env_path: str = ".env") -> str:
    """VASTAI_API_KEY from .env (not printed)."""
    txt = Path(env_path).read_text(encoding="utf-8")
    m = re.search(r"VASTAI_API_KEY=(\S+)", txt)
    if not m:
        raise RuntimeError("VASTAI_API_KEY not found in .env")
    return m.group(1)


class VastClient:
    """Thin wrapper over vast.ai (urllib + vastai CLI for create). Mocked in tests."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or _load_api_key()

    def _api(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        req = urllib.request.Request(
            f"{_API}{path}", method=method,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            data=(json.dumps(body).encode() if body is not None else None),
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())

    def search_offers(self, query: dict) -> list[dict]:
        q = urllib.parse.quote(json.dumps(query))
        return self._api("GET", f"/bundles/?q={q}").get("offers", [])

    def create(self, offer_id: int, image: str, disk: int = 50,
               label: str = "metaloom") -> int:
        """Create an instance via the vastai CLI (--raw) → instance_id."""
        out = subprocess.run(
            ["vastai", "create", "instance", str(offer_id), "--image", image,
             "--disk", str(disk), "--ssh", "--label", label, "--raw",
             "--api-key", self.api_key],
            capture_output=True, text=True, timeout=120,
        )
        data = json.loads(out.stdout)
        if not data.get("success"):
            raise RuntimeError(f"vast create failed: {out.stdout} {out.stderr}")
        return int(data["new_contract"])

    def show(self, instance_id: int) -> dict:
        for it in self._api("GET", "/instances/").get("instances", []):
            if int(it.get("id")) == int(instance_id):
                return it
        return {}

    def destroy(self, instance_id: int) -> dict:
        return self._api("DELETE", f"/instances/{instance_id}/")


# ────────────────────────── state (orphan reaping) ──────────────────────────

def _read_state(state_file: Path) -> list[dict]:
    if Path(state_file).exists():
        try:
            return json.loads(Path(state_file).read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _write_state(state_file: Path, entries: list[dict]) -> None:
    Path(state_file).parent.mkdir(parents=True, exist_ok=True)
    Path(state_file).write_text(json.dumps(entries, ensure_ascii=False, indent=2),
                                encoding="utf-8")


def _state_add(state_file: Path, entry: dict) -> None:
    e = [x for x in _read_state(state_file) if x.get("id") != entry["id"]]
    e.append(entry)
    _write_state(state_file, e)


def _state_remove(state_file: Path, instance_id: int) -> None:
    _write_state(state_file, [x for x in _read_state(state_file)
                              if int(x.get("id", -1)) != int(instance_id)])


# ────────────────────────── instance ──────────────────────────

def _default_ssh(host: str, port: int) -> Callable[[str], str]:
    def _run(cmd: str) -> str:
        out = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=20",
             "-p", str(port), f"root@{host}", cmd],
            capture_output=True, text=True, timeout=600,
        )
        return out.stdout
    return _run


def _default_scp(host: str, port: int) -> Callable[[str, str], None]:
    def _fetch(remote: str, local: str) -> None:
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["scp", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=20",
             "-P", str(port), f"root@{host}:{remote}", local],
            capture_output=True, text=True, timeout=600,
        )
    return _fetch


class Instance:
    """An active vast instance with an autonomous kill-switch. destroy() is idempotent."""

    def __init__(self, instance_id: int, ssh_host: str, ssh_port: int,
                 client: VastClient, deadline_min: int, state_file: Path,
                 ssh_runner: Optional[Callable] = None,
                 scp_runner: Optional[Callable] = None):
        self.id = int(instance_id)
        self.ssh_host = ssh_host
        self.ssh_port = int(ssh_port)
        self.client = client
        self.deadline_sec = int(deadline_min) * 60
        self.state_file = Path(state_file)
        self._ssh = ssh_runner or _default_ssh(ssh_host, self.ssh_port)
        self._scp = scp_runner or _default_scp(ssh_host, self.ssh_port)
        self._destroyed = False
        self._timer: Optional[threading.Timer] = None

    # — kill-switch (armed BEFORE any work) —
    def arm_killswitch(self) -> None:
        # (1) autonomous timer ON the instance: kills GPU billing even without laptop/agent
        self._ssh(f"setsid sh -c 'sleep {self.deadline_sec}; shutdown -h now' "
                  f"</dev/null >/dev/null 2>&1 &")
        # (2) laptop-side backstop: a clean destroy at the deadline while the process is alive
        self._timer = threading.Timer(self.deadline_sec, self._safe_destroy)
        self._timer.daemon = True
        self._timer.start()
        # (3) atexit: destroy on normal process exit
        atexit.register(self._safe_destroy)

    def ssh(self, cmd: str) -> str:
        return self._ssh(cmd)

    def fetch(self, remote: str, local: str) -> None:
        self._scp(remote, local)

    def run_and_fetch(self, cmd: str, artifacts: Optional[list] = None,
                      local_dir: str = ".") -> str:
        """ssh-run cmd, THEN (in finally) fetch artifacts — exfiltrate before returning."""
        try:
            return self._ssh(cmd)
        finally:
            for remote in (artifacts or []):
                local = str(Path(local_dir) / Path(remote).name)
                try:
                    self._scp(remote, local)
                except Exception:
                    pass

    def _safe_destroy(self) -> None:
        try:
            self.destroy()
        except Exception:
            pass

    def destroy(self) -> None:
        if self._destroyed:
            return
        self._destroyed = True
        if self._timer is not None:
            self._timer.cancel()
        try:
            self.client.destroy(self.id)
        finally:
            _state_remove(self.state_file, self.id)


# ────────────────────────── provision / context / reap ──────────────────────────

def provision(client: VastClient, *, gpu_query: dict, image: str, disk: int = 50,
              deadline_min: int = 120, label: str = "metaloom",
              state_file: Path = DEFAULT_STATE,
              poll_sec: float = 10.0, timeout_sec: float = 900.0,
              ssh_runner: Optional[Callable] = None,
              scp_runner: Optional[Callable] = None) -> Instance:
    """Bring up an instance and IMMEDIATELY arm the kill-switch (before returning). Returns Instance."""
    offers = client.search_offers(gpu_query)
    if not offers:
        raise RuntimeError("no matching vast offers for the query")
    offer_id = offers[0]["id"]
    inst_id = client.create(offer_id, image=image, disk=disk, label=label)

    # wait for running + ssh info
    t0 = time.time()
    info: dict = {}
    while time.time() - t0 < timeout_sec:
        info = client.show(inst_id)
        if info.get("actual_status") == "running" and info.get("ssh_host"):
            break
        time.sleep(poll_sec)
    if info.get("actual_status") != "running":
        # did not come up — tear down so it doesn't linger
        try:
            client.destroy(inst_id)
        except Exception:
            pass
        raise RuntimeError(f"instance {inst_id} did not reach running within {timeout_sec}s")

    inst = Instance(inst_id, info["ssh_host"], info["ssh_port"], client,
                    deadline_min=deadline_min, state_file=state_file,
                    ssh_runner=ssh_runner, scp_runner=scp_runner)
    # write STATE BEFORE arming (so reap finds it even if arm fails), then the kill-switch
    _state_add(state_file, {"id": inst_id, "ssh_host": info["ssh_host"],
                            "ssh_port": info["ssh_port"], "deadline_min": deadline_min,
                            "created": time.time()})
    inst.arm_killswitch()
    return inst


@contextmanager
def instance(client: VastClient, **kwargs):
    """`with cloud.instance(client, ...) as inst:` — destroy guaranteed in finally."""
    inst = provision(client, **kwargs)
    try:
        yield inst
    finally:
        inst.destroy()


def reap(client: VastClient, state_file: Path = DEFAULT_STATE) -> list[int]:
    """Tear down all instances in cloud_state.json (recovery after a crash). Returns the ids."""
    killed = []
    for entry in _read_state(state_file):
        iid = int(entry["id"])
        try:
            client.destroy(iid)
        except Exception:
            pass
        killed.append(iid)
    _write_state(state_file, [])
    return killed


# ────────────────────────── CLI: metaloom cloud ──────────────────────────

def add_args(p) -> None:
    p.add_argument("action", choices=["reap", "destroy", "list"],
                   help="reap = tear down all orphans from cloud_state.json; destroy --id; list")
    p.add_argument("--id", type=int, default=None, help="instance_id for destroy")
    p.add_argument("--state-file", default=str(DEFAULT_STATE))


def run(args) -> None:
    client = VastClient()
    sf = Path(args.state_file)
    if args.action == "reap":
        killed = reap(client, sf)
        print(f"reaped: {killed or 'no orphans'}", flush=True)
    elif args.action == "destroy":
        if args.id is None:
            raise SystemExit("--id is required for destroy")
        client.destroy(args.id)
        _state_remove(sf, args.id)
        print(f"destroyed {args.id}", flush=True)
    elif args.action == "list":
        print(json.dumps(_read_state(sf), ensure_ascii=False, indent=2))

"""RunPod training worker for Synth City (SN50).

A flexible, sandboxed job runner. Agents on the ganglion MCP server submit
jobs here via RunPodBackend — they never talk to this container directly.

The worker can run in two ways:

  1. **Script mode** — the agent submits a Python script (as a string in the
     job spec). The worker validates it for blocked imports, writes it to a
     temp file, and executes it in a subprocess. The script has access to
     baked-in datasets, numpy, properscoring, and a helper library
     (synth_runtime) for loading data and writing results.

  2. **Built-in mode** — for quick runs, the agent can use the built-in
     "gbm" simulator + CRPS validator + backtester without writing any code.

Job spec arrives via:
  - GANGLION_JOB_SPEC env var (JSON) — set by RunPodBackend in dockerArgs
  - /input/spec.json — mounted by RunPod volume
  - Individual env vars as fallback

Results go to /outputs/ (collected by RunPodBackend.collect()).
If Hippius is configured, results are also uploaded to persistent storage
and the agent receives a URL to poll/fetch results even if the pod crashes.
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import math
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("synth-worker")

OUTPUT_DIR = Path(os.environ.get("GANGLION_OUTPUT_DIR", "/outputs"))
CHECKPOINT_DIR = Path(os.environ.get("GANGLION_CHECKPOINT_DIR", "/app/checkpoints"))
DATASET_DIR = Path("/app/datasets")
SCRIPT_TIMEOUT = int(os.environ.get("GANGLION_SCRIPT_TIMEOUT", "1800"))

# ── Baked-in constants (from SN50 spec) ────────────────────

LF_ASSETS = ["BTC", "ETH", "SOL", "XAU", "SPYX", "NVDAX", "TSLAX", "AAPLX", "GOOGLX"]
HF_ASSETS = ["BTC", "ETH", "SOL", "XAU"]

SIGMA_MAP: dict[str, float] = {
    "BTC": 0.00472, "ETH": 0.00695, "SOL": 0.00782, "XAU": 0.00208,
    "SPYX": 0.00156, "NVDAX": 0.00342, "TSLAX": 0.00332,
    "AAPLX": 0.00250, "GOOGLX": 0.00332,
}

ASSET_COEFFICIENTS: dict[str, float] = {
    "BTC": 1.0, "ETH": 0.6715, "SOL": 0.5884, "XAU": 2.262,
    "SPYX": 2.991, "NVDAX": 1.389, "TSLAX": 1.420,
    "AAPLX": 1.865, "GOOGLX": 1.431,
}

LOW_FREQ_SCORING_INTERVALS: dict[str, int] = {
    "5min": 300, "30min": 1800, "3hour": 10800, "24hour_abs": 86400,
}
HIGH_FREQ_SCORING_INTERVALS: dict[str, int] = {
    "1min": 60, "2min": 120, "5min": 300, "15min": 900,
    "30min": 1800, "60min_abs": 3600,
}


# ═══════════════════════════════════════════════════════════
# Hippius — persistent result storage
# ═══════════════════════════════════════════════════════════


class HippiusUploader:
    """Upload results to Hippius IPFS/S3-compatible storage.

    If the pod crashes mid-run, partial results already uploaded are
    still accessible. The agent gets back a results_url it can poll.

    Configure via env vars:
      HIPPIUS_API_URL     — Hippius gateway (e.g. https://api.hippius.network)
      HIPPIUS_API_KEY     — auth key
      HIPPIUS_BUCKET      — storage bucket (default: ganglion-runs)
    """

    def __init__(self) -> None:
        self.api_url = os.environ.get("HIPPIUS_API_URL", "").rstrip("/")
        self.api_key = os.environ.get("HIPPIUS_API_KEY", "")
        self.bucket = os.environ.get("HIPPIUS_BUCKET", "ganglion-runs")
        self.enabled = bool(self.api_url and self.api_key)
        if self.enabled:
            logger.info("Hippius enabled: %s bucket=%s", self.api_url, self.bucket)
        else:
            logger.info("Hippius not configured (set HIPPIUS_API_URL + HIPPIUS_API_KEY to enable)")

    def _job_prefix(self, job_id: str) -> str:
        return f"{self.bucket}/{job_id}"

    async def upload_json(self, job_id: str, filename: str, data: dict) -> str | None:
        """Upload a JSON blob. Returns the access URL or None on failure."""
        if not self.enabled:
            return None
        try:
            import urllib.request

            key = f"{self._job_prefix(job_id)}/{filename}"
            payload = json.dumps(data, indent=2, default=str).encode()
            url = f"{self.api_url}/v1/objects/{key}"

            req = urllib.request.Request(
                url,
                data=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="PUT",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp.read()

            access_url = f"{self.api_url}/v1/objects/{key}"
            logger.info("Uploaded %s → %s", filename, access_url)
            return access_url
        except Exception as e:
            logger.warning("Hippius upload failed for %s: %s", filename, e)
            return None

    async def upload_file(self, job_id: str, filepath: Path) -> str | None:
        """Upload a file (checkpoint, artifact). Returns access URL."""
        if not self.enabled:
            return None
        try:
            import urllib.request

            key = f"{self._job_prefix(job_id)}/{filepath.name}"
            url = f"{self.api_url}/v1/objects/{key}"

            req = urllib.request.Request(
                url,
                data=filepath.read_bytes(),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/octet-stream",
                },
                method="PUT",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                resp.read()

            access_url = f"{self.api_url}/v1/objects/{key}"
            logger.info("Uploaded %s → %s", filepath.name, access_url)
            return access_url
        except Exception as e:
            logger.warning("Hippius upload failed for %s: %s", filepath.name, e)
            return None

    def results_url(self, job_id: str) -> str | None:
        """Return the base URL where all results for this job live."""
        if not self.enabled:
            return None
        return f"{self.api_url}/v1/objects/{self._job_prefix(job_id)}/"


# ═══════════════════════════════════════════════════════════
# Script validation
# ═══════════════════════════════════════════════════════════

BLOCKED_IMPORTS = [
    "subprocess", "os.system", "os.exec", "os.spawn", "os.popen",
    "shutil.rmtree", "socket", "http.server", "http.client",
    "urllib", "requests", "aiohttp", "httpx",
    "ftplib", "smtplib", "telnetlib", "ctypes", "multiprocessing",
]


def validate_script(code: str) -> list[str]:
    """Validate an agent-submitted script. Returns errors (empty = ok)."""
    errors: list[str] = []
    if len(code) > 50_000:
        return [f"Script too large ({len(code)} bytes, max 50000)"]
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"Syntax error: {e}"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for blocked in BLOCKED_IMPORTS:
                    if alias.name == blocked or alias.name.startswith(blocked + "."):
                        errors.append(f"Blocked import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                full = f"{module}.{alias.name}" if module else alias.name
                for blocked in BLOCKED_IMPORTS:
                    if full.startswith(blocked) or module.startswith(blocked) or module == blocked:
                        errors.append(f"Blocked import: {full}")
                        break
    return errors


# ═══════════════════════════════════════════════════════════
# synth_runtime — helper module injected into script env
# ═══════════════════════════════════════════════════════════

RUNTIME_MODULE = '''"""synth_runtime — helper library for agent-submitted scripts.

Available imports:
    from synth_runtime import (
        load_dataset, load_prices, load_checkpoint,
        save_checkpoint, save_result, save_artifact,
        list_datasets, list_checkpoints, competition_params,
        SIGMA_MAP, ASSET_COEFFICIENTS, LF_ASSETS, HF_ASSETS,
        PARAMS,  # dict parsed from GANGLION_PARAMS env var
    )
"""
import json
import os
import numpy as np
from pathlib import Path

DATASET_DIR = Path("{dataset_dir}")
OUTPUT_DIR = Path("{output_dir}")
CHECKPOINT_DIR = Path("{checkpoint_dir}")

LF_ASSETS = {lf_assets}
HF_ASSETS = {hf_assets}
SIGMA_MAP = {sigma_map}
ASSET_COEFFICIENTS = {asset_coefficients}
LOW_FREQ_SCORING_INTERVALS = {lf_intervals}
HIGH_FREQ_SCORING_INTERVALS = {hf_intervals}

# Agent-provided parameters (passed via job spec "params" field)
_raw = os.environ.get("GANGLION_PARAMS", "{{}}")
try:
    PARAMS = json.loads(_raw)
except json.JSONDecodeError:
    PARAMS = {{}}


def list_datasets() -> list[str]:
    """List available .npy dataset files."""
    if not DATASET_DIR.exists():
        return []
    return sorted(p.name for p in DATASET_DIR.glob("*.npy"))


def list_checkpoints() -> list[str]:
    """List saved checkpoint files."""
    if not CHECKPOINT_DIR.exists():
        return []
    return sorted(p.name for p in CHECKPOINT_DIR.glob("*"))


def load_dataset(name: str) -> np.ndarray:
    """Load a .npy dataset by filename."""
    path = DATASET_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {{path}}. Available: {{list_datasets()}}")
    return np.load(str(path))


def load_prices(asset: str, competition: str = "low_freq") -> np.ndarray:
    """Shorthand: load baked-in historical prices for an asset."""
    return load_dataset(f"{{asset.upper()}}_{{competition}}_prices.npy")


def load_checkpoint(name: str) -> np.ndarray:
    """Load a saved checkpoint."""
    path = CHECKPOINT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {{path}}")
    return np.load(str(path), allow_pickle=False)


def save_checkpoint(name: str, data: np.ndarray) -> str:
    """Save a numpy array as checkpoint. Returns path."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / name
    np.save(str(path), data)
    return str(path)


def save_result(result: dict, filename: str = "result.json") -> str:
    """Write JSON result to /outputs. Returns path."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    path.write_text(json.dumps(result, indent=2, default=str))
    return str(path)


def save_artifact(name: str, data) -> str:
    """Save artifact (bytes or ndarray) to /outputs. Returns path."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    if isinstance(data, np.ndarray):
        np.save(str(path), data)
    else:
        path.write_bytes(data)
    return str(path)


def competition_params(competition: str = "low_freq") -> dict:
    """Return time_increment and time_length for a competition."""
    if competition == "high_freq":
        return {{"time_increment": 60, "time_length": 3600}}
    return {{"time_increment": 300, "time_length": 86400}}
'''


def _render_runtime() -> str:
    return RUNTIME_MODULE.format(
        dataset_dir=DATASET_DIR, output_dir=OUTPUT_DIR, checkpoint_dir=CHECKPOINT_DIR,
        lf_assets=repr(LF_ASSETS), hf_assets=repr(HF_ASSETS),
        sigma_map=repr(SIGMA_MAP), asset_coefficients=repr(ASSET_COEFFICIENTS),
        lf_intervals=repr(LOW_FREQ_SCORING_INTERVALS),
        hf_intervals=repr(HIGH_FREQ_SCORING_INTERVALS),
    )


# ═══════════════════════════════════════════════════════════
# Script execution
# ═══════════════════════════════════════════════════════════


async def run_script(
    spec: dict[str, Any], hippius: HippiusUploader, job_id: str,
) -> dict[str, Any]:
    """Execute an agent-submitted Python script in a sandboxed subprocess.

    Spec keys:
        script: str             — Python source code (required)
        params: dict            — arbitrary params available as synth_runtime.PARAMS
        datasets: list[str]     — dataset names to verify exist before running
        timeout: int            — max seconds (default: SCRIPT_TIMEOUT)
    """
    script = spec.get("script", "")
    if not script:
        return {"error": "No 'script' field in job spec"}

    errors = validate_script(script)
    if errors:
        return {"error": "Script validation failed", "validation_errors": errors}

    params = spec.get("params", {})
    requested_datasets = spec.get("datasets", [])
    timeout = spec.get("timeout", SCRIPT_TIMEOUT)

    missing = [ds for ds in requested_datasets if not (DATASET_DIR / ds).exists()]
    if missing:
        available = sorted(p.name for p in DATASET_DIR.glob("*.npy")) if DATASET_DIR.exists() else []
        return {"error": "Requested datasets not found", "missing": missing, "available": available}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    await hippius.upload_json(job_id, "status.json", {
        "status": "running", "started_at": time.time(), "mode": "script",
    })

    with tempfile.TemporaryDirectory(prefix="ganglion-script-") as tmpdir:
        tmppath = Path(tmpdir)
        (tmppath / "synth_runtime.py").write_text(_render_runtime())
        script_path = tmppath / "run.py"
        script_path.write_text(script)

        env = {
            **os.environ,
            "PYTHONPATH": f"{tmpdir}:{os.environ.get('PYTHONPATH', '')}",
            "GANGLION_PARAMS": json.dumps(params, default=str),
            "GANGLION_OUTPUT_DIR": str(OUTPUT_DIR),
            "GANGLION_CHECKPOINT_DIR": str(CHECKPOINT_DIR),
            "GANGLION_DATASET_DIR": str(DATASET_DIR),
        }
        # Strip keys that shouldn't leak into the script subprocess
        for key in ["HIPPIUS_API_KEY", "HIPPIUS_API_URL", "GANGLION_JOB_SPEC"]:
            env.pop(key, None)

        start = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, str(script_path),
                env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                cwd=tmpdir,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            proc.kill()
            result = {
                "error": f"Script timed out after {timeout}s",
                "duration_s": round(time.monotonic() - start, 2),
            }
            await hippius.upload_json(job_id, "result.json", result)
            return result

    elapsed = time.monotonic() - start
    stdout = stdout_bytes.decode(errors="replace")
    stderr = stderr_bytes.decode(errors="replace")
    artifacts = sorted(p.name for p in OUTPUT_DIR.iterdir() if p.is_file())

    result_data = None
    result_path = OUTPUT_DIR / "result.json"
    if result_path.exists():
        try:
            result_data = json.loads(result_path.read_text())
        except json.JSONDecodeError:
            pass

    result = {
        "mode": "script",
        "exit_code": proc.returncode,
        "success": proc.returncode == 0,
        "stdout": stdout[-10_000:],
        "stderr": stderr[-5_000:] if proc.returncode != 0 else "",
        "duration_s": round(elapsed, 2),
        "artifacts": artifacts,
        "result": result_data,
    }

    await hippius.upload_json(job_id, "result.json", result)
    for artifact_name in artifacts:
        await hippius.upload_file(job_id, OUTPUT_DIR / artifact_name)

    result["results_url"] = hippius.results_url(job_id)
    return result


# ═══════════════════════════════════════════════════════════
# Built-in modes (no script needed)
# ═══════════════════════════════════════════════════════════


def _competition_params(comp: str) -> dict[str, int]:
    if comp == "high_freq":
        return {"time_increment": 60, "time_length": 3600}
    return {"time_increment": 300, "time_length": 86400}


def simulate_gbm(
    price: float, sigma: float, time_increment: int,
    time_length: int, n: int,
) -> np.ndarray:
    dt = time_increment / 3600
    steps = time_length // time_increment
    std = sigma * math.sqrt(dt)
    pct = np.random.normal(0, std, size=(n, steps))
    paths = np.empty((n, steps + 1))
    paths[:, 0] = price
    paths[:, 1:] = price * np.cumprod(1 + pct, axis=1)
    return paths


def calculate_crps(
    sim_paths: np.ndarray, real_prices: np.ndarray,
    time_increment: int, scoring_intervals: dict[str, int],
) -> dict[str, Any]:
    try:
        from properscoring import crps_ensemble
    except ImportError:
        return {"error": "properscoring not installed", "total": -1.0}

    breakdown: dict[str, float] = {}
    total = 0.0
    for name, interval_s in scoring_intervals.items():
        is_abs = name.endswith("_abs")
        step = interval_s // time_increment
        if step < 1:
            continue
        sim_at = sim_paths[:, ::step]
        real_at = real_prices[::step]
        if sim_at.shape[1] < 2:
            continue
        if is_abs:
            sim_changes = sim_at[:, 1:]
            real_changes = real_at[1:]
        else:
            denom_sim = sim_at[:, :-1]
            denom_real = real_at[:-1]
            if np.any(denom_sim == 0) or np.any(denom_real == 0):
                breakdown[name] = -1.0
                continue
            sim_changes = (np.diff(sim_at, axis=1) / denom_sim) * 10_000
            real_changes = (np.diff(real_at) / denom_real) * 10_000
        flat_real = real_changes if real_changes.ndim == 1 else real_changes.ravel()
        interval_crps = 0.0
        n_scored = 0
        for t in range(len(flat_real)):
            val = float(flat_real[t])
            if np.isnan(val):
                continue
            c = crps_ensemble(val, sim_changes[:, t].astype(float))
            if is_abs and real_prices[-1] != 0:
                c = c / (real_prices[-1] * 10_000)
            interval_crps += c
            n_scored += 1
        avg = interval_crps / max(n_scored, 1)
        breakdown[name] = round(avg, 8)
        total += avg
    breakdown["total"] = round(total, 8)
    return breakdown


async def run_builtin_train(
    spec: dict[str, Any], hippius: HippiusUploader, job_id: str,
) -> dict[str, Any]:
    """Built-in GBM train.

    Spec keys:
        assets: list[str]       — which assets (default: all for competition)
        competition: str        — "low_freq" | "high_freq"
        num_simulations: int    — paths per asset (default: 1000)
        sigma_overrides: dict   — per-asset sigma
        current_prices: dict    — per-asset spot price
    """
    competition = spec.get("competition", "low_freq")
    params = _competition_params(competition)
    num_sims = spec.get("num_simulations", 1000)
    sigma_overrides = spec.get("sigma_overrides", {})
    current_prices = spec.get("current_prices", {})
    default_assets = HF_ASSETS if competition == "high_freq" else LF_ASSETS
    assets = spec.get("assets", default_assets)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}

    for asset in assets:
        asset = asset.upper()
        start = time.monotonic()
        sigma = sigma_overrides.get(asset, SIGMA_MAP.get(asset, 0.005))
        price = current_prices.get(asset, 100_000.0)
        paths = simulate_gbm(price, sigma, params["time_increment"], params["time_length"], num_sims)
        ckpt_path = CHECKPOINT_DIR / f"{asset}_{competition}_paths.npy"
        np.save(str(ckpt_path), paths)
        await hippius.upload_file(job_id, ckpt_path)
        final = paths[:, -1]
        elapsed = time.monotonic() - start
        results[asset] = {
            "status": "ok", "model": "gbm", "sigma": sigma,
            "start_price": price, "num_simulations": num_sims,
            "num_steps": paths.shape[1],
            "mean_final": round(float(np.mean(final)), 2),
            "std_final": round(float(np.std(final)), 2),
            "p5": round(float(np.percentile(final, 5)), 2),
            "p95": round(float(np.percentile(final, 95)), 2),
            "checkpoint": str(ckpt_path),
            "duration_s": round(elapsed, 2),
        }

    result = {"mode": "train", "competition": competition, "assets": results}
    await hippius.upload_json(job_id, "result.json", result)
    result["results_url"] = hippius.results_url(job_id)
    return result


async def run_builtin_validate(
    spec: dict[str, Any], hippius: HippiusUploader, job_id: str,
) -> dict[str, Any]:
    """Built-in CRPS validate — agent controls the data source.

    Spec keys:
        assets: list[str]
        competition: str
        realized_prices: dict   — per-asset price lists (agent-provided inline)
        dataset_files: dict     — per-asset dataset filenames from /app/datasets
    """
    competition = spec.get("competition", "low_freq")
    params = _competition_params(competition)
    provided_prices = spec.get("realized_prices", {})
    dataset_files = spec.get("dataset_files", {})
    default_assets = HF_ASSETS if competition == "high_freq" else LF_ASSETS
    assets = spec.get("assets", default_assets)
    intervals = HIGH_FREQ_SCORING_INTERVALS if competition == "high_freq" else LOW_FREQ_SCORING_INTERVALS

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}

    for asset in assets:
        asset = asset.upper()
        ckpt_path = CHECKPOINT_DIR / f"{asset}_{competition}_paths.npy"
        if not ckpt_path.exists():
            results[asset] = {"error": f"No checkpoint at {ckpt_path}"}
            continue
        paths = np.load(str(ckpt_path))

        source = "proxy"
        if asset in provided_prices:
            real = np.array(provided_prices[asset], dtype=float)
            source = "provided"
        elif asset in dataset_files:
            ds_path = DATASET_DIR / dataset_files[asset]
            if ds_path.exists():
                real = np.load(str(ds_path))
                source = f"dataset:{dataset_files[asset]}"
            else:
                results[asset] = {"error": f"Dataset file not found: {dataset_files[asset]}"}
                continue
        else:
            baked_path = DATASET_DIR / f"{asset}_{competition}_prices.npy"
            if baked_path.exists():
                real = np.load(str(baked_path))
                source = "baked"
            else:
                real = paths[0, :].copy()

        min_len = min(len(real), paths.shape[1])
        real, paths_trimmed = real[:min_len], paths[:, :min_len]
        start = time.monotonic()
        crps = calculate_crps(paths_trimmed, real, params["time_increment"], intervals)
        elapsed = time.monotonic() - start
        coeff = ASSET_COEFFICIENTS.get(asset, 1.0)
        raw_total = crps.get("total", -1.0)
        weighted = round(raw_total * coeff, 8) if raw_total >= 0 else -1.0
        results[asset] = {
            "status": "ok", "crps_breakdown": crps,
            "crps_total": raw_total, "crps_weighted": weighted,
            "asset_coefficient": coeff, "realized_source": source,
            "duration_s": round(elapsed, 2),
        }

    result = {"mode": "validate", "competition": competition, "assets": results}
    await hippius.upload_json(job_id, "result.json", result)
    result["results_url"] = hippius.results_url(job_id)
    return result


async def run_builtin_backtest(
    spec: dict[str, Any], hippius: HippiusUploader, job_id: str,
) -> dict[str, Any]:
    """Agent-controlled backtest — the agent specifies everything.

    Spec keys:
        asset: str                  — target asset
        competition: str            — "low_freq" | "high_freq"
        simulated_paths: list       — 2D array of paths (num_sims x num_steps)
        realized_prices: list       — 1D array of realised prices
        dataset_file: str           — OR load realized prices from this baked dataset
        checkpoint: str             — OR load simulated paths from this checkpoint
        time_increment: int         — override (default: from competition)
        time_length: int            — override (default: from competition)
        scoring_intervals: dict     — override scoring intervals
    """
    asset = spec.get("asset", "BTC").upper()
    competition = spec.get("competition", "low_freq")
    params = _competition_params(competition)
    ti = spec.get("time_increment", params["time_increment"])
    tl = spec.get("time_length", params["time_length"])

    if "simulated_paths" in spec:
        sim = np.array(spec["simulated_paths"], dtype=float)
    elif "checkpoint" in spec:
        ckpt = CHECKPOINT_DIR / spec["checkpoint"]
        if not ckpt.exists():
            return {"error": f"Checkpoint not found: {spec['checkpoint']}"}
        sim = np.load(str(ckpt))
    else:
        ckpt = CHECKPOINT_DIR / f"{asset}_{competition}_paths.npy"
        if not ckpt.exists():
            return {"error": f"No simulated_paths, checkpoint, or default checkpoint for {asset}"}
        sim = np.load(str(ckpt))

    if "realized_prices" in spec:
        real = np.array(spec["realized_prices"], dtype=float)
    elif "dataset_file" in spec:
        ds = DATASET_DIR / spec["dataset_file"]
        if not ds.exists():
            avail = sorted(p.name for p in DATASET_DIR.glob("*.npy")) if DATASET_DIR.exists() else []
            return {"error": f"Dataset not found: {spec['dataset_file']}", "available": avail}
        real = np.load(str(ds))
    else:
        baked = DATASET_DIR / f"{asset}_{competition}_prices.npy"
        if baked.exists():
            real = np.load(str(baked))
        else:
            return {"error": f"No realized_prices, dataset_file, or baked data for {asset}"}

    intervals = spec.get("scoring_intervals")
    if intervals is None:
        intervals = HIGH_FREQ_SCORING_INTERVALS if competition == "high_freq" else LOW_FREQ_SCORING_INTERVALS

    min_len = min(len(real), sim.shape[1])
    real, sim = real[:min_len], sim[:, :min_len]

    start = time.monotonic()
    crps = calculate_crps(sim, real, ti, intervals)
    elapsed = time.monotonic() - start

    coeff = ASSET_COEFFICIENTS.get(asset, 1.0)
    raw_total = crps.get("total", -1.0)

    result = {
        "mode": "backtest",
        "asset": asset,
        "competition": competition,
        "crps_breakdown": crps,
        "crps_total": raw_total,
        "crps_weighted": round(raw_total * coeff, 8) if raw_total >= 0 else -1.0,
        "asset_coefficient": coeff,
        "num_paths": sim.shape[0],
        "num_steps": min_len,
        "time_increment": ti,
        "time_length": tl,
        "duration_s": round(elapsed, 2),
    }

    await hippius.upload_json(job_id, "result.json", result)
    result["results_url"] = hippius.results_url(job_id)
    return result


# ═══════════════════════════════════════════════════════════
# Job dispatch
# ═══════════════════════════════════════════════════════════


async def run_job(spec: dict[str, Any], hippius: HippiusUploader, job_id: str) -> dict[str, Any]:
    """Dispatch based on mode or presence of 'script' field."""
    if "script" in spec:
        return await run_script(spec, hippius, job_id)

    mode = spec.get("mode", "train")
    if mode == "train":
        return await run_builtin_train(spec, hippius, job_id)
    elif mode == "validate":
        return await run_builtin_validate(spec, hippius, job_id)
    elif mode == "backtest":
        return await run_builtin_backtest(spec, hippius, job_id)
    elif mode == "train_and_validate":
        t = await run_builtin_train(spec, hippius, job_id)
        v = await run_builtin_validate(spec, hippius, job_id)
        result = {"train": t, "validate": v}
        await hippius.upload_json(job_id, "result.json", result)
        result["results_url"] = hippius.results_url(job_id)
        return result
    elif mode == "train_validate_backtest":
        t = await run_builtin_train(spec, hippius, job_id)
        v = await run_builtin_validate(spec, hippius, job_id)
        b = await run_builtin_backtest(spec, hippius, job_id)
        result = {"train": t, "validate": v, "backtest": b}
        await hippius.upload_json(job_id, "result.json", result)
        result["results_url"] = hippius.results_url(job_id)
        return result
    elif mode == "list_datasets":
        avail = sorted(p.name for p in DATASET_DIR.glob("*.npy")) if DATASET_DIR.exists() else []
        ckpts = sorted(p.name for p in CHECKPOINT_DIR.glob("*")) if CHECKPOINT_DIR.exists() else []
        return {"datasets": avail, "checkpoints": ckpts}
    else:
        return {
            "error": f"Unknown mode '{mode}'.",
            "available_modes": [
                "train", "validate", "backtest",
                "train_and_validate", "train_validate_backtest",
                "list_datasets",
            ],
            "tip": "Or pass a 'script' field with Python code for full flexibility.",
        }


def main() -> None:
    import uuid

    spec: dict[str, Any] = {}

    env_spec = os.environ.get("GANGLION_JOB_SPEC", "").strip()
    if env_spec:
        try:
            spec = json.loads(env_spec)
            logger.info("Loaded job spec from GANGLION_JOB_SPEC env var")
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON in GANGLION_JOB_SPEC: %s", e)
            sys.exit(1)

    if not spec:
        spec_file = Path("/input/spec.json")
        if spec_file.exists():
            spec = json.loads(spec_file.read_text())
            logger.info("Loaded job spec from %s", spec_file)

    if not spec:
        spec = {
            "mode": os.environ.get("GANGLION_MODE", "train"),
            "competition": os.environ.get("GANGLION_COMPETITION", "low_freq"),
            "num_simulations": int(os.environ.get("GANGLION_NUM_SIMULATIONS", "1000")),
        }
        assets_env = os.environ.get("GANGLION_ASSETS", "").strip()
        if assets_env:
            spec["assets"] = [a.strip().upper() for a in assets_env.split(",") if a.strip()]
        logger.info("Using env-var fallback spec")

    job_id = spec.get("job_id", f"run-{uuid.uuid4().hex[:8]}")
    hippius = HippiusUploader()

    logger.info("Job %s starting: mode=%s, has_script=%s",
                job_id, spec.get("mode", "?"), "script" in spec)

    # Upload initial status so agent can poll immediately
    try:
        asyncio.run(hippius.upload_json(job_id, "status.json", {
            "status": "starting", "job_id": job_id, "started_at": time.time(),
        }))
    except Exception:
        pass

    try:
        result = asyncio.run(run_job(spec, hippius, job_id))
    except Exception as e:
        logger.error("Job %s failed: %s", job_id, e, exc_info=True)
        result = {
            "error": str(e),
            "traceback": traceback.format_exc()[-3000:],
            "job_id": job_id,
        }
        # Upload failure to Hippius so agent can see what happened
        try:
            asyncio.run(hippius.upload_json(job_id, "result.json", result))
            asyncio.run(hippius.upload_json(job_id, "status.json", {
                "status": "failed", "job_id": job_id, "error": str(e),
            }))
        except Exception:
            pass

    result["job_id"] = job_id
    if hippius.enabled:
        result["results_url"] = hippius.results_url(job_id)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = OUTPUT_DIR / "result.json"
    result_path.write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, default=str))
    logger.info("Job %s complete. Results at %s", job_id, result_path)


if __name__ == "__main__":
    main()

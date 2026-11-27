#!/usr/bin/env python3
"""Capture a golden input of OHMRAKEL42 on a live installation (snapshot e9f39ad5).

The script runs the parameter preparation and the planning of the OHMRAKEL42 snapshot once,
on a copy of the live EOS directory, and records every call the preparation makes to the
measurements, the predictions and the presence provider together with its answer. It writes a
golden input file for ``tests/test_ohmrakel42golden.py``: the anonymized configuration, the
recorded calls and the outputs of this run. The golden test replays the recorded calls and
checks that it gives the same outputs; the port of OHMRAKEL42 to main has to give them, too.

It never calls the energy management run: no adapter is read or written, no device is
controlled. Prediction providers are updated in the copy as a live run would do (network).

Usage, on the live host in the environment of the installed snapshot::

    python capture_ohmrakel42_golden.py --eos-dir /path/to/EOS_DIR --name live_house

Review the written file before you commit it: it holds the configuration and the measured
histories of the installation (secrets, location and server/adapter settings are removed).
"""

import argparse
import asyncio
import json
import math
import os
import re
import shutil
import sys
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

# Keys of the configuration whose values are replaced, matched case-insensitively.
SECRET_KEY_PATTERN = re.compile(r"token|password|secret|api_?key|credential|auth", re.IGNORECASE)
# Sections that do not reach the planning and may identify the installation.
DROPPED_SECTIONS = ("server", "adapter", "logging", "cache")
DROPPED_GENERAL_KEYS = ("latitude", "longitude", "data_folder_path", "data_output_subpath")
ANONYMOUS = "<removed>"


def jsonable(value: Any) -> Any:
    """Convert an argument or an answer into plain JSON values, NaN to None."""
    # Imported here: the module must be importable before EOS_DIR is set.
    import numpy as np
    import pandas as pd
    from pydantic import BaseModel

    if isinstance(value, BaseModel):
        return jsonable(value.model_dump(mode="json"))
    if isinstance(value, pd.Series):
        return {
            "index": [pd.Timestamp(stamp).isoformat() for stamp in value.index],
            "values": jsonable(value.tolist()),
        }
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def call_key(name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Return the replay key of a call: its name and its arguments as canonical JSON."""
    return json.dumps(
        {"call": name, "args": jsonable(list(args)), "kwargs": jsonable(kwargs)},
        sort_keys=True,
    )


class Recorder:
    """Forward calls to a real object and record each call with its answer or exception."""

    def __init__(self, target: Any, prefix: str, calls: dict[str, Any]) -> None:
        self._target = target
        self._prefix = prefix
        self._calls = calls

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._target, name)
        if not callable(attribute):
            raise AttributeError(
                f"{self._prefix}.{name} is not a method; the capture cannot record it."
            )
        qualified = f"{self._prefix}.{name}"

        if name == "provider_by_id":

            def provider(provider_id: str) -> "Recorder":
                return Recorder(attribute(provider_id), f"{qualified}:{provider_id}", self._calls)

            return provider

        if asyncio.iscoroutinefunction(attribute):

            async def record_async(*args: Any, **kwargs: Any) -> Any:
                key = call_key(qualified, args, kwargs)
                try:
                    answer = await attribute(*args, **kwargs)
                except Exception as exc:
                    self._calls[key] = {
                        "raises": type(exc).__name__,
                        "message": str(exc),
                        "async": True,
                    }
                    raise
                self._calls[key] = {
                    "returns": jsonable(answer),
                    "type": type(answer).__name__,
                    "async": True,
                }
                return answer

            return record_async

        def record(*args: Any, **kwargs: Any) -> Any:
            key = call_key(qualified, args, kwargs)
            try:
                answer = attribute(*args, **kwargs)
            except Exception as exc:
                self._calls[key] = {
                    "raises": type(exc).__name__,
                    "message": str(exc),
                    "async": False,
                }
                raise
            self._calls[key] = {
                "returns": jsonable(answer),
                "type": type(answer).__name__,
                "async": False,
            }
            return answer

        return record


def anonymize(settings: Any, path: str = "") -> Any:
    """Remove secrets, the location and the sections that do not reach the planning."""
    if isinstance(settings, dict):
        result: dict[str, Any] = {}
        for key, value in settings.items():
            if not path and key in DROPPED_SECTIONS:
                continue
            if path == "general" and key in DROPPED_GENERAL_KEYS:
                continue
            if SECRET_KEY_PATTERN.search(key) and value not in (None, "", [], {}):
                result[key] = ANONYMOUS
                continue
            result[key] = anonymize(value, f"{path}.{key}" if path else key)
        return result
    if isinstance(settings, list):
        return [anonymize(item, path) for item in settings]
    return settings


async def capture(name: str, originals: dict[Path, Path]) -> dict[str, Any]:
    """Run preparation and planning once with recording proxies; return the golden input.

    Args:
        name: Name of the case.
        originals: Copied directory to live directory; the configuration is read from the
            live file, because EOS may migrate the copy when it starts.
    """
    from akkudoktoreos.core.coreabc import (
        get_config,
        get_ems,
        get_measurement,
        get_prediction,
        singletons_init,
    )
    from akkudoktoreos.optimization.planner import run_planning
    from akkudoktoreos.optimization.request import PlanningRequest
    from akkudoktoreos.optimization.request.devices import DeviceResolver
    from akkudoktoreos.optimization.request.thermal import ThermalResolver
    from akkudoktoreos.utils.datetimeutil import to_datetime

    singletons_init()  # as the server does; constructs the adapters but never runs them
    # Restore the persisted measurements and predictions as the server does at startup
    # (load_eos_state in server/eos.py); without it the measurement store is empty: no SoC,
    # no histories. The file cache is not loaded: its index holds absolute paths into the
    # live EOS_DIR, and it only caches provider downloads.
    if not await get_measurement().load():
        raise SystemExit("The measurements of the installation could not be loaded.")
    await get_prediction().load()
    config = get_config()
    ems = get_ems()
    calls: dict[str, Any] = {}
    measurement = Recorder(get_measurement(), "measurement", calls)
    prediction = Recorder(get_prediction(), "prediction", calls)
    DeviceResolver.measurement = measurement  # type: ignore[assignment]
    ThermalResolver.measurement = measurement  # type: ignore[assignment]
    ThermalResolver.prediction = prediction  # type: ignore[assignment]
    PlanningRequest.prediction = prediction  # type: ignore[assignment]

    config_file = config.general.config_file_path
    live_file = None
    for copy, original in originals.items():
        if config_file is not None and Path(config_file).is_relative_to(copy):
            live_file = original / Path(config_file).relative_to(copy)
    if live_file is None or not live_file.exists():
        raise SystemExit(f"No configuration file of the live installation found ({config_file}).")
    with live_file.open("r", encoding="utf-8") as handle:
        settings = anonymize(json.load(handle))
    # The default time zone may depend on the host; the replay must use the live one.
    settings.setdefault("general", {})["timezone"] = str(config.general.timezone)

    now = to_datetime(in_timezone=config.general.timezone)
    ems.set_start_datetime(now)
    request = PlanningRequest()
    outputs: list[dict[str, Any]]
    try:
        task = await request.resolve()
        outcome = run_planning(task, None, [], config.thermal.plan_ttl_seconds)
    except Exception as exc:
        outputs = [{"error": f"{type(exc).__name__}: {exc}"}]
    else:
        outputs = [
            {
                "error": None,
                "solution": outcome.solution.model_dump(mode="json"),
                "optimization_solution": outcome.optimization_solution.model_dump(mode="json"),
                "plan": outcome.energy_management_plan.model_dump(mode="json"),
                "thermal_plans": [plan.model_dump(mode="json") for plan in outcome.thermal_plans],
            }
        ]
    return {
        "description": f"Captured on a live installation ({name}), snapshot e9f39ad5.",
        "settings": settings,
        "runs": [{"start": now.isoformat(), "request": {}, "changed_devices": []}],
        "measurements": {},
        "outdoor_temperature_c": None,
        "presence": None,
        "recorded_calls": calls,
        "captured_outputs": jsonable(outputs),
    }


def main(argv: Optional[list[str]] = None) -> int:
    """Copy the EOS directory, capture one run in the copy and write the golden input."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eos-dir", required=True, type=Path, help="Live EOS_DIR (only read).")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Live EOS_CONFIG_DIR, if it is not inside the EOS directory (only read).",
    )
    parser.add_argument(
        "--name", required=True, help="Case name, lowercase letters, digits and '_'."
    )
    parser.add_argument("--out", type=Path, default=Path.cwd(), help="Output directory.")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[a-z0-9_]+", args.name):
        parser.error("--name may contain lowercase letters, digits and '_' only.")

    # On Windows, EOS may still hold a cache file of the copy open; a cleanup error must
    # not lose the capture.
    with tempfile.TemporaryDirectory(prefix="eos-capture-", ignore_cleanup_errors=True) as scratch:
        eos_dir = Path(scratch) / "eos"
        shutil.copytree(args.eos_dir, eos_dir)
        originals = {eos_dir: args.eos_dir.resolve()}
        os.environ["EOS_DIR"] = str(eos_dir)
        os.environ.pop("EOS_DATA_DIR", None)
        if args.config_dir is not None:
            config_dir = Path(scratch) / "config"
            shutil.copytree(args.config_dir, config_dir)
            originals[config_dir] = args.config_dir.resolve()
            os.environ["EOS_CONFIG_DIR"] = str(config_dir)
        else:
            os.environ.pop("EOS_CONFIG_DIR", None)
        case = asyncio.run(capture(args.name, originals))

    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / f"golden_{args.name}_input.json"
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(case, handle, indent=1)
        handle.write("\n")
    error = case["captured_outputs"][0]["error"]
    print(f"Wrote {target} ({len(case['recorded_calls'])} recorded calls).")
    if error:
        print(f"The run raised: {error}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

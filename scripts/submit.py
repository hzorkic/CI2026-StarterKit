#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Built for the CI 2026 hackathon starter kit

r'''
Run forecasts for a test suite and submit them to the portal.

Inherits all forecast options from forecast.yaml, runs the forward
pass for all four regions, then POSTs the resulting netCDF files to
the hackathon submission portal.

Run from the starter_kit root directory::

    python scripts/submit.py email=you@example.com

Override the experiment or any forecast option::

    python scripts/submit.py \
        exp_name=baseline_mlp \
        device=cuda \
        email=you@example.com

Skip forecasting if the netCDF files already exist::

    python scripts/submit.py \
        skip_forecast=true \
        email=you@example.com
'''

# System modules
import logging
import sys
from pathlib import Path
from typing import Dict

# External modules
import hydra
import requests
from omegaconf import DictConfig, OmegaConf

# Internal modules
from forecast import run_forecast


main_logger = logging.getLogger(__name__)

_REGIONS = [
    "era5_region1",
    "era5_region2",
    "aimip_region1",
    "aimip_region2",
]

_PORTAL_FIELDS = {
    "era5_region1": "file_era5_region1",
    "era5_region2": "file_era5_region2",
    "aimip_region1": "file_aimip_region1",
    "aimip_region2": "file_aimip_region2",
}


def _run_all_forecasts(cfg: DictConfig) -> None:
    r'''
    Run forecasts for all four regions using the given test suite.

    For each region the base config is merged with the region-specific
    input_path and output_path, then passed directly to run_forecast.

    Parameters
    ----------
    cfg : DictConfig
        Full submit configuration tree.
    '''
    for region in _REGIONS:
        region_paths = cfg.regions[region]
        region_cfg = OmegaConf.merge(cfg, region_paths)
        main_logger.info("Forecasting %s …", region)
        run_forecast(region_cfg)


def _collect_forecast_files(cfg: DictConfig) -> Dict[str, Path]:
    r'''
    Locate the four forecast netCDF files defined in the config.

    Parameters
    ----------
    cfg : DictConfig
        Full submit configuration tree.

    Returns
    -------
    Dict[str, Path]
        Mapping from region name to absolute netCDF path.

    Raises
    ------
    FileNotFoundError
        If any of the four expected output files is missing.
    '''
    files: Dict[str, Path] = {}
    missing = []
    for region in _REGIONS:
        output_path = cfg.regions[region].output_path
        path = Path(output_path)
        if not path.exists():
            missing.append(str(path))
        else:
            files[region] = path
    if missing:
        raise FileNotFoundError(
            "Missing forecast files:\n"
            + "\n".join(f"  {p}" for p in missing)
        )
    return files


def _submit_to_portal(
    email: str,
    portal_url: str,
    forecast_files: Dict[str, Path],
) -> None:
    r'''
    POST the four forecast files to the submission portal.

    Parameters
    ----------
    email : str
        Registered submitter email address.
    portal_url : str
        Base URL of the hackathon submission portal.
    forecast_files : Dict[str, Path]
        Mapping from region name to netCDF file path.

    Raises
    ------
    SystemExit
        If the server returns a non-2xx response.
    '''
    url = portal_url.rstrip("/") + "/api/v1/submissions"
    handles = {}
    try:
        for region, path in forecast_files.items():
            handles[region] = open(path, "rb")
        files = {
            _PORTAL_FIELDS[region]: (
                path.name,
                handles[region],
                "application/octet-stream",
            )
            for region, path in forecast_files.items()
        }
        main_logger.info("Submitting to %s …", url)
        response = requests.post(
            url,
            data={"email": email},
            files=files,
            timeout=120,
        )
    finally:
        for handle in handles.values():
            handle.close()

    if response.ok:
        payload = response.json()
        unique_idx = payload.get("unique_idx")
        main_logger.info("Submission accepted.")
        main_logger.info(
            "  unique_idx : %s", unique_idx
        )
        main_logger.info(
            "  status     : %s", payload.get("status")
        )
        main_logger.info(
            "  queue pos  : %s", payload.get("queue_position")
        )
        main_logger.info(
            "  est. wait  : %s",
            payload.get("estimated_wait_formatted"),
        )
        main_logger.info(
            "Check status with:\n  curl -s %s/api/v1/submissions/%s"
            " | python -m json.tool",
            portal_url.rstrip("/"),
            unique_idx,
        )
    else:
        main_logger.error(
            "Submission failed: %d %s",
            response.status_code,
            response.text,
        )
        sys.exit(1)


@hydra.main(
    config_path="../configs",
    config_name="submit",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    r'''
    Entry point: forecast all regions and submit to the portal.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra configuration tree.
    '''
    if not cfg.skip_forecast:
        _run_all_forecasts(cfg)

    forecast_files = _collect_forecast_files(cfg)
    for region, path in forecast_files.items():
        main_logger.info("  %s: %s", region, path)

    _submit_to_portal(
        cfg.email,
        cfg.url_portal,
        forecast_files,
    )


if __name__ == "__main__":
    main()

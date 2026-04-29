#!/bin/env python
# -*- coding: utf-8 -*-
#
# Built for the CI 2026 hackathon starter kit

# System modules
import logging

# External modules
import torch

# Internal modules


main_logger = logging.getLogger(__name__)


_RD = 287.0597
_RV = 461.51
_EPSILON = _RD / _RV


_MAGNUS_SETTINGS = dict(
    a1_w=611.21,
    a3_w=17.502,
    a4_w=32.19,
    a1_i=611.21,
    a3_i=22.587,
    a4_i=-0.7,
    t0=273.16,
    tice=250.16,
)


def _estimate_sat(temp, a1, a3, a4):
    return a1 * torch.exp(a3 * ((temp - _MAGNUS_SETTINGS["t0"]) / (temp - a4)))


def _interp_sat(temp):
    sat_w = _estimate_sat(
        temp,
        _MAGNUS_SETTINGS["a1_w"],
        _MAGNUS_SETTINGS["a3_w"],
        _MAGNUS_SETTINGS["a4_w"],
    )
    sat_i = _estimate_sat(
        temp,
        _MAGNUS_SETTINGS["a1_i"],
        _MAGNUS_SETTINGS["a3_i"],
        _MAGNUS_SETTINGS["a4_i"],
    )
    alpha = (
        (temp - _MAGNUS_SETTINGS["tice"])
        / (_MAGNUS_SETTINGS["t0"] - _MAGNUS_SETTINGS["tice"])
    ) ** 2
    alpha = torch.clamp(alpha, 0, 1)
    return alpha * sat_w + (1 - alpha) * sat_i


def estimate_relative_humidity(
    temperature: torch.Tensor, specific_humidity: torch.Tensor, pressure: torch.Tensor
) -> torch.Tensor:
    r"""
    Estimate relative humidity from temperature, specific humidity, and
    pressure using the formula:

    .. math:: RH = \\frac{q}{q_{sat}(T, P)}

    where :math:`q` is the specific humidity and :math:`q_{sat}` is the
    saturation specific humidity computed from temperature and pressure using
    the Magnus formula.


    Parameters
    ----------
    temperature : torch.Tensor
        Temperature in Kelvin.
    specific_humidity : torch.Tensor
        Specific humidity in kg/kg.
    pressure : torch.Tensor
        Pressure in Pa.

    Returns
    -------
    torch.Tensor
        Relative humidity as a fraction between 0 and 1.
    """
    vapor_pressure = (specific_humidity * pressure) / (
        _EPSILON + specific_humidity * (1.0 - _EPSILON)
    )

    saturation_pressure = _interp_sat(temperature)

    rh = vapor_pressure / (saturation_pressure + 1e-12)
    return rh.clamp(0.0, 1.0)


def approximate_surface_pressure(
    geopotential_at_surface: torch.Tensor,
    reference_pressure: float = 1013_25.0,
    reference_temperature: float = 288.15,
) -> torch.Tensor:
    r"""
    Approximate surface pressure from the reference pressure, reference
    temperature, and surface geopotential using the barometric formula:

    .. math:: P = P_0 \\exp\\left(-\\frac{g \\cdot z}{R_d \\cdot T_0}\\right).

    Parameters
    ----------
    geopotential_at_surface : torch.Tensor
        Surface geopotential in m^2/s^2.
    reference_pressure : float, optional
        Reference pressure at sea level in Pa (default = 101325 Pa).
    reference_temperature : float, optional
        Reference temperature at sea level in K (default = 288.15 K).

    Returns
    -------
    torch.Tensor
        Estimated surface pressure in Pa. Same shape as input.
    """
    return reference_pressure * torch.exp(
        -geopotential_at_surface / (_RD * reference_temperature)
    )

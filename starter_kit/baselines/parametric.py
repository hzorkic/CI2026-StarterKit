#!/bin/env python
# -*- coding: utf-8 -*-
#
# Built for the CI 2026 hackathon starter kit

# System modules
import logging
from typing import Any, Dict

# External modules
import torch

# Internal modules
from starter_kit.model import BaseModel
from .utils import estimate_relative_humidity, approximate_surface_pressure


main_logger = logging.getLogger(__name__)


class ParametricNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_parameter(
            "level_logscale", torch.nn.Parameter(torch.full((7, 1, 1), 10.0).log())
        )
        self.register_parameter(
            "level_logshift", torch.nn.Parameter(torch.full((7, 1, 1), 0.7).log())
        )
        self.register_buffer(
            "pressure_levels",
            torch.tensor(
                [1000_00, 850_00, 700_00, 500_00, 250_00, 100_00, 50_00]
            ).reshape(-1, 1, 1),
        )

    def forward(
        self, input_level: torch.Tensor, input_auxiliary: torch.Tensor
    ) -> torch.Tensor:
        # Cloud cover at levels
        level_rh = estimate_relative_humidity(
            temperature=input_level[:, 0:1],
            specific_humidity=input_level[:, 1:2],
            pressure=self.pressure_levels,
        )
        level_cloud_cover = torch.sigmoid(
            (level_rh - self.level_logshift.exp()) * self.level_logscale.exp()
        )

        # Mask where pressure is lower than surface pressure
        surface_pressure = approximate_surface_pressure(input_auxiliary[:, 1:2])
        valid_mask = self.pressure_levels < surface_pressure.unsqueeze(2)

        # Randomly distributed cloud cover
        level_clear_sky_fraction = 1 - level_cloud_cover * valid_mask
        total_cloud_cover = 1 - torch.prod(level_clear_sky_fraction, dim=2)
        return total_cloud_cover


class ParametricModel(BaseModel):
    def estimate_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        prediction = self.network(
            input_level=batch["input_level"], input_auxiliary=batch["input_auxiliary"]
        )
        loss = (prediction - batch["target"]).pow(2)
        loss = (loss * self.lat_weights).mean()
        return {"loss": loss, "prediction": prediction}

    def estimate_auxiliary_loss(
        self, batch: Dict[str, torch.Tensor], outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        mae = (outputs["prediction"] - batch["target"]).abs()
        mae = (mae * self.lat_weights).mean()
        prediction_bool = (outputs["prediction"] > 0.5).float()
        target_bool = (batch["target"] > 0.5).float()
        accuracy = (prediction_bool == target_bool).float()
        accuracy = (accuracy * self.lat_weights).mean()
        return {"mae": mae, "accuracy": accuracy}

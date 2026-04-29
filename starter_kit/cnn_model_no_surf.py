#!/bin/env python
# -*- coding: utf-8 -*-
#
# Built for the CI 2026 hackathon starter kit

# System modules
import logging
from typing import Any, Dict

# External modules
import torch
import torch.nn as nn

# Internal modules
from starter_kit.baselines.utils import estimate_relative_humidity
from starter_kit.layers import InputNormalisation
from starter_kit.model import BaseModel

main_logger = logging.getLogger(__name__)

r"""
Pre-computed per-channel normalisation statistics, plus the convolutional
cloud-cover model that consumes them.

Channel layout going into the CNN (37 channels total), in the same order as
the entries in `_normalisation_mean` / `_normalisation_std`:

    0-6    temperature (K) at 7 pressure levels (1000, 850, 700, 500, 250,
           100, 50 hPa)
    7-13   specific humidity (kg/kg) at the same 7 levels
    14-20  zonal wind u (m/s) at the same 7 levels
    21-27  meridional wind v (m/s) at the same 7 levels
    28     land-sea mask (0-1, dimensionless)
    29     surface geopotential (m^2/s^2)
    30-36  derived relative humidity (fraction) at the 7 levels

Mean and std were computed from the training data (1979-2018, ERA5 region 1)
across all spatial locations and time steps, weighted by latitude to correct
for grid-cell area distortion in the lat-lon mesh.
"""

_normalisation_mean = [
    # temperature (K) at 7 pressure levels (1000 -> 50 hPa)
    294.531359,
    287.010605,
    278.507482,
    262.805241,
    227.580722,
    201.364517,
    209.719502,
    # specific humidity (kg/kg)
    0.010667,
    0.006922,
    0.003784,
    0.001229,
    0.000088,
    0.000003,
    0.000003,
    # zonal wind u (m/s)
    -1.412110,
    -0.914917,
    0.431349,
    3.504875,
    11.699176,
    6.758849,
    -1.214763,
    # meridional wind v (m/s)
    0.167424,
    -0.105374,
    -0.172138,
    -0.022648,
    0.030789,
    0.281048,
    -0.094608,
    # land-sea mask, then surface geopotential
    # 0.410844,
    # 2129.684371,
    # relative humidity
    0.001953,
    0.001983,
    0.001815,
    0.001981,
    0.002994,
    0.004023,
    0.000892,
]
_normalisation_std = [
    # temperature (K)
    62.864550,
    61.180621,
    58.938862,
    56.016099,
    47.532073,
    32.281805,
    38.084321,
    # specific humidity (kg/kg)
    0.006102,
    0.004648,
    0.003013,
    0.001266,
    0.000080,
    0.000001,
    0.000000,
    # zonal wind u (m/s)
    4.661358,
    6.159993,
    7.763541,
    9.877940,
    16.068963,
    11.681901,
    10.705570,
    # meridional wind v (m/s)
    4.119853,
    4.318767,
    4.810067,
    6.209760,
    10.585627,
    5.680168,
    2.978756,
    # land-sea mask, then surface geopotential
    0.498762,
    3602.712270,
    # relative humidity
    0.000637,
    0.000847,
    0.000996,
    0.001389,
    0.001727,
    0.003555,
    0.000679,
]

# Pressure of each model level in Pa (1000, 850, 700, 500, 250, 100, 50 hPa).
# These are dataset constants -- the levels are fixed by the data spec, never
# learned. They feed the RH formula and must stay in the same order as the
# level dimension of `input_level`.
_PRESSURE_LEVELS_PA = [100000.0, 85000.0, 70000.0, 50000.0, 25000.0, 10000.0, 5000.0]


class MLPNetwork(nn.Module):
    r"""
    Fully-convolutional cloud-cover network.

    Pipeline (see `forward`):

      1. Derive relative humidity per level from T, q, and the fixed pressure
         of each level.
      2. Flatten the (vars, levels) dimensions of the pressure-level input
         into channels and concatenate the two used auxiliary static fields.
      3. Append the 7 RH channels to give a 37-channel image.
      4. Whiten each channel with `InputNormalisation` using precomputed
         training statistics.
      5. Run through `n_layers` Conv-Norm-SiLU blocks and a final 1x1
         convolutional head.

    Same-padded convolutions and the absence of pooling preserve the spatial
    extent at every step, so a per-pixel target shape ``(B, 1, H, W)`` can be
    produced directly without any upsampling.

    Parameters
    ----------
    input_dim : int, optional, default = 30
        Number of base channels (4 vars x 7 levels + 2 auxiliary). Relative
        humidity adds 7 more channels internally, for 37 total.
    hidden_dim : int, optional, default = 64
        Width of the intermediate feature maps in each conv block.
    n_layers : int, optional, default = 4
        Number of Conv-Norm-SiLU blocks before the 1x1 prediction head. The
        first block also handles the channel projection from ``total_in``
        to ``hidden_dim``.
    kernel_size : int, optional, default = 5
        Spatial kernel size for all conv blocks. Same-padding uses
        ``kernel_size // 2`` so spatial extent is preserved end-to-end.
    """

    def __init__(
        self,
        input_dim: int = 30,
        hidden_dim: int = 64,
        n_layers: int = 4,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        # Per-channel whitening. Reshape stats to (C, 1, 1) so they broadcast
        # against the channels-first (B, C, H, W) tensor used by Conv2d -- no
        # axis movement required around the normalisation step.
        self.normalisation = InputNormalisation(
            mean=torch.tensor(_normalisation_mean).view(-1, 1, 1),
            std=torch.tensor(_normalisation_std).view(-1, 1, 1),
        )
        # Pressure levels are dataset constants. Registering them as a buffer
        # puts them in `state_dict()`, moves them with `.to(device)`, and
        # keeps them out of the optimiser (they are not learnable).
        self.register_buffer(
            "pressure_levels",
            torch.tensor(_PRESSURE_LEVELS_PA).view(1, -1, 1, 1),
        )

        # Total input channels = base features + one RH channel per level.
        n_levels = self.pressure_levels.numel()
        total_in = input_dim + n_levels
        # "Same" padding: keeps the input H,W constant through every conv,
        # which is required for the per-pixel output to align with the input.
        padding = kernel_size // 2

        # First block: project the 37 input channels into `hidden_dim`.
        layers = [
            nn.Conv2d(total_in, hidden_dim, kernel_size, padding=padding),
            nn.SiLU(),
        ]
        # Inner blocks: Norm -> Conv -> activation. GroupNorm with one group is
        # equivalent to LayerNorm over (C, H, W) per sample -- batch-size
        # independent and stable at small batches (important since validation
        # is run region-by-region).
        for _ in range(n_layers - 1):
            layers.append(nn.GroupNorm(num_groups=1, num_channels=hidden_dim))
            layers.append(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding)
            )
            layers.append(nn.SiLU())

        # 1x1 prediction head: mixes the final feature channels into a single
        # cloud-cover scalar per pixel. Tiny weight init + bias=0.5 makes the
        # network start near the marginal mean (~0.5) so it does not saturate
        # the [0, 1] clamp during the first few optimisation steps.
        head = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        nn.init.normal_(head.weight, std=1e-6)
        nn.init.constant_(head.bias, 0.5)
        layers.append(head)

        self.cnn = nn.Sequential(*layers)

    def forward(
        self,
        input_level: torch.Tensor,
        input_auxiliary: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Map atmospheric state to per-pixel cloud cover.

        Steps
        -----
        1. Derive RH per level from T, q, and the fixed pressure of each level
           (a strong physical predictor of cloud cover).
        2. Collapse the (vars, levels) dimensions of `input_level` into a
           single channel dimension.
        3. Slice off the two used auxiliary fields (LSM, geopotential).
        4. Concatenate base features (30 ch) and RH (7 ch) into a 37-channel
           image.
        5. Whiten with the precomputed per-channel mean/std.
        6. Run the convolutional stack to produce the per-pixel prediction.

        Parameters
        ----------
        input_level : torch.Tensor
            Pressure-level fields, shape ``(B, C_l, L, H, W)``. Channel 0 is
            temperature (K), channel 1 is specific humidity (kg/kg).
        input_auxiliary : torch.Tensor
            Auxiliary static fields, shape ``(B, C_a, H, W)``. The first two
            channels are land-sea mask and surface geopotential.

        Returns
        -------
        torch.Tensor
            Cloud-cover prediction of shape ``(B, 1, H, W)``. Not yet clamped
            to ``[0, 1]`` -- the wrapping `ZSModel.estimate_loss` enforces
            that range.
        """
        B, _, _, H, W = input_level.shape

        # 1) Per-level relative humidity from T, q and the fixed pressure of
        #    each level. RH is the most direct physical predictor of cloud
        #    cover, so passing it explicitly gives the network a strong prior
        #    rather than asking it to learn the Magnus formula from scratch.
        level_rh = estimate_relative_humidity(
            temperature=input_level[:, 0],
            specific_humidity=input_level[:, 1],
            pressure=self.pressure_levels,
        )  # (B, L, H, W)

        # 2) Collapse (vars, levels) -> channels: (B, 4, 7, H, W) -> (B, 28, H, W).
        flattened_input_level = input_level.reshape(B, -1, H, W)
        # 3) Keep only the auxiliary fields used by this model: land-sea mask
        #    and surface geopotential. The remaining auxiliary channels (lat,
        #    lon, land-cover) are intentionally dropped.
        # sliced_auxiliary = input_auxiliary[:, :2]
        # 4) Stack base atmospheric channels (28 + 2 = 30), then append RH (7)
        #    for the 37-channel input that matches the precomputed
        #    normalisation statistics.
        cnn_input = torch.cat([flattened_input_level, level_rh], dim=1)
        
        # 5) Per-channel whitening so each variable enters the network with
        #    roughly zero mean and unit variance, regardless of its physical
        #    scale (T ~ 250 K vs q ~ 1e-3 kg/kg vs geopotential ~ 2000).
        cnn_input = self.normalisation(cnn_input)

        # 6) Convolutional stack -> 1x1 head -> per-pixel prediction.
        return self.cnn(cnn_input)


class ZSModel(BaseModel):
    r"""
    Training and evaluation wrapper for `MLPNetwork`.

    The training loss is a latitude-weighted mean absolute error (MAE),
    matching the official ERA5 scoring metric. Predictions are clamped to
    ``[0, 1]`` (cloud cover is a fraction). Auxiliary metrics include MSE
    and a thresholded "cloudy vs clear" accuracy at 0.5; these are computed
    only on the validation set, not used to update parameters.
    """

    def estimate_loss(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        r"""
        Compute the latitude-weighted MAE loss for one batch.

        The clamp-then-MAE pattern matches the ERA5 evaluation protocol: the
        scoring system never sees out-of-range predictions, so neither
        should the loss. Predictions outside ``[0, 1]`` therefore receive
        zero gradient on the violating side -- a deliberate trade-off for
        boundary stability over edge-case learning signal.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Batch dictionary containing ``input_level``, ``input_auxiliary``,
            and ``target`` tensors.

        Returns
        -------
        Dict[str, Any]
            ``{"loss": ..., "prediction": ...}``. ``loss`` is the
            latitude-weighted MAE; ``prediction`` is the clamped network
            output, reused by `estimate_auxiliary_loss` to avoid a second
            forward pass.
        """
        # Forward through the network -> per-pixel prediction in raw range,
        # shape (B, 1, H, W).
        prediction = self.network(
            input_level=batch["input_level"],
            input_auxiliary=batch["input_auxiliary"],
        )
        # Cloud cover is a physical fraction: enforce the [0, 1] range here,
        # before the loss is computed and before the value is exposed to any
        # downstream consumer.
        prediction = prediction.clamp(0.0, 1.0)
        # Pixel-wise absolute error against the target.
        loss = (prediction - batch["target"]).abs()
        # Latitude weighting corrects for the unequal physical area of
        # lat-lon grid cells (cells near the poles cover less area than at
        # the equator); the final mean is therefore an area-weighted global
        # average rather than an unweighted pixel mean.
        loss = (loss * self.lat_weights).mean()
        return {"loss": loss, "prediction": prediction}

    def estimate_auxiliary_loss(
        self,
        batch: Dict[str, torch.Tensor],
        outputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        r"""
        Compute auxiliary validation-only metrics from a cached prediction.

        These metrics are not used to update parameters; they live here for
        monitoring along complementary axes:

        * ``mse`` penalises large pointwise errors more aggressively than
          MAE does, useful as a stability sanity-check during training.
        * ``accuracy`` thresholds prediction and target at 0.5 to give a
          binary "cloudy vs clear" agreement rate -- a coarse but
          interpretable companion metric.

        The clamped prediction is taken from the `estimate_loss` output to
        avoid re-running the network forward pass.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Batch dictionary containing the ground-truth ``target`` tensor.
        outputs : Dict[str, Any]
            The dict returned by `estimate_loss`; ``prediction`` is reused
            as-is.

        Returns
        -------
        Dict[str, Any]
            ``{"mse": ..., "accuracy": ...}``, each a latitude-weighted
            scalar over the batch.
        """
        # Pixel-wise squared error, then latitude-weighted mean.
        mse = (outputs["prediction"] - batch["target"]).pow(2)
        mse = (mse * self.lat_weights).mean()
        # Binarise prediction and target at 0.5 to compute pixel-wise
        # agreement on "cloudy vs clear".
        prediction_bool = (outputs["prediction"] > 0.5).float()
        target_bool = (batch["target"] > 0.5).float()
        accuracy = (prediction_bool == target_bool).float()
        # Latitude-weighted mean -> global area-weighted accuracy.
        accuracy = (accuracy * self.lat_weights).mean()
        return {"mse": mse, "accuracy": accuracy}

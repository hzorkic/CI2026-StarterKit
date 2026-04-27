#!/bin/env python
# -*- coding: utf-8 -*-
#
# Built for the CI 2026 hackathon starter kit

# System modules
import logging
from typing import Dict, Union, Optional, Callable

# External modules
from torch.utils.data import Dataset

import numpy as np
import xarray as xr
import tensorstore as ts

# Internal modules


main_logger = logging.getLogger(__name__)


def _ensure_3d(arr: np.ndarray) -> np.ndarray:
    r'''
    Ensure that the input array has three spatial dimensions (C, H, W).

    If the input array has only two spatial dimensions (H, W), a channel
    dimension of size 1 is added at the front.

    Parameters
    ----------
    arr : np.ndarray
        Input array of shape (H, W) or (C, H, W).

    Returns
    -------
    np.ndarray
        Output array of shape (C, H, W).
    '''
    if arr.ndim < 3:
        n_missing = 3 - arr.ndim
        new_shape = (1,) * n_missing + arr.shape
        return arr.reshape(new_shape)
    else:
        return arr


class TestDataset(Dataset):
    r'''
    Dataset for loading zarr-backed input variables and auxiliary data.

    This dataset reads time-indexed fields from a zarr archive and stores
    auxiliary variables in memory. It supports optional data augmentation on
    each item.
    '''

    _VARS_TO_LOAD = ["input_level"]
    _input_auxiliary: np.ndarray
    _n_time: int

    def __init__(
            self,
            data_path: str,
            threads_limit: Union[str, int] = "shared",
            augmentation: Optional[Callable] = None,
    ) -> None:
        r'''
        Initialize the dataset.

        Parameters
        ----------
        data_path : str
            Path to the zarr dataset root directory.
        threads_limit : Union[str, int], optional
            Concurrency limit for tensorstore file and data copy
            operations, by default "shared".
        augmentation : Callable, optional
            Optional callable applied to each returned example.
        '''
        self._datasets = {}
        self.data_path = data_path
        self.threads_limit = threads_limit
        self.augmentation = augmentation

        self._load_metadata()

    @property
    def datasets(self) -> Dict[str, ts.TensorStore]:
        r'''
        Return the opened tensorstore datasets.

        Returns
        -------
        Dict[str, ts.TensorStore]
            Mapping from variable name to the opened tensorstore array.
        '''
        if not self._datasets:
            self._datasets = self._setup_datasets()
        return self._datasets

    def _setup_datasets(self) -> Dict[str, ts.TensorStore]:
        r'''
        Load the tensorstore datasets per variable.

        Opens one tensorstore array per variable under ``data_path``, sharing a
        single :class:`tensorstore.Context` so concurrency limits are
        pooled across variables. Opens are dispatched together and resolved at
        the end to amortise metadata reads.

        Called on each worker process after forking, for more
        efficient memory usage and to avoid issues with
        multiprocessing and zarr datasets.

        Returns
        -------
        Dict[str, Any]
            Mapping from variable name to the opened tensorstore
            array.
        '''
        context = ts.Context({
            'file_io_concurrency': {'limit': self.threads_limit},
            'data_copy_concurrency': {'limit': self.threads_limit},
        })
        futures = {
            name: ts.open(
                {
                    'driver': 'zarr',
                    'metadata_key': '.zarray',
                    'kvstore': {
                        'driver': 'file',
                        'path': f'{self.data_path}/{name}',
                    },
                },
                context=context,
            )
            for name in self._VARS_TO_LOAD
        }
        return {
            name: future.result()
            for name, future in futures.items()
        }

    def _load_metadata(self) -> None:
        r'''
        Load auxiliary metadata from the zarr dataset.

        Reads the in-memory auxiliary field and the length of the time
        dimension.
        '''
        with xr.open_zarr(self.data_path) as ds:
            self._input_auxiliary = ds['input_auxiliary'].values
            try:
                self._n_time = ds.sizes['time']
            except KeyError:
                self._n_time = ds.sizes["sample"]

    def __len__(self) -> int:
        r'''
        Return the number of time samples in the dataset.

        Returns
        -------
        int
            Number of time steps available in the dataset.
        '''
        return self._n_time

    def _get_data(self, idx: int) -> Dict[str, np.ndarray]:
        r'''
        Read data for a single time index from opened datasets.

        Parameters
        ----------
        idx : int
            Time index to load.

        Returns
        -------
        Dict[str, np.ndarray]
            Mapping from variable name to the loaded array for the given index.
        '''
        interm_dict = {
            var: _ensure_3d(self.datasets[var][idx].read().result())
            for var in self._VARS_TO_LOAD
        }
        interm_dict['time'] = self.datasets.coords['time'][idx].values
        return interm_dict

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        r'''
        Retrieve a dataset item by index.

        Parameters
        ----------
        idx : int
            Time index to retrieve.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing input fields and auxiliary data.
        '''
        loaded_data = self._get_data(idx)
        loaded_data['input_auxiliary'] = _ensure_3d(self._input_auxiliary)
        if self.augmentation is not None:
            loaded_data = self.augmentation(loaded_data)
        return loaded_data


class TrainDataset(TestDataset):
    r'''
    Training dataset that includes the target variable.
    '''

    _VARS_TO_LOAD = ["input_level", "target"]

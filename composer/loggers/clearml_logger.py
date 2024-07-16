# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Log to `ClearML`."""

from __future__ import annotations

import copy
import os
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch

from composer.loggers.logger import Logger
from composer.loggers.logger_destination import LoggerDestination
from composer.utils import MissingConditionalImportError, dist

if TYPE_CHECKING:
    from composer.core import State

__all__ = ['ClearMLLogger']


class ClearMLLogger(LoggerDestination):
    """Log to `ClearML`.

    Args:
        project (str, optional): WandB project name.
        group (str, optional): WandB group name.
        name (str, optional): WandB run name.
            If not specified, the :attr:`.State.run_name` will be used.
        entity (str, optional): WandB entity name.
        tags (List[str], optional): WandB tags.
        log_artifacts (bool, optional): Whether to log
            `artifacts <https://docs.wandb.ai/ref/python/artifact>`_ (Default: ``False``).
        rank_zero_only (bool, optional): Whether to log only on the rank-zero process.
            When logging `artifacts <https://docs.wandb.ai/ref/python/artifact>`_, it is
            highly recommended to log on all ranks.  Artifacts from ranks ≥1 will not be
            stored, which may discard pertinent information. For example, when using
            Deepspeed ZeRO, it would be impossible to restore from checkpoints without
            artifacts from all ranks (default: ``True``).
        init_kwargs (Dict[str, Any], optional): Any additional init kwargs
            ``wandb.init`` (see
            `WandB documentation <https://docs.wandb.ai/ref/python/init>`_).
    """

    def __init__(
        self,
        project: Optional[str] = None,
        group: Optional[str] = None,
        name: Optional[str] = None,
        entity: Optional[str] = None,
        tags: Optional[List[str]] = None,
        log_artifacts: bool = False,
        rank_zero_only: bool = True,
        init_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            import clearml
        except ImportError as e:
            raise MissingConditionalImportError(
                extra_deps_group='clearml',
                conda_package='clearml',
                conda_channel='conda-forge',
            ) from e

        if log_artifacts and rank_zero_only and dist.get_world_size() > 1:
            warnings.warn((
                'When logging artifacts, `rank_zero_only` should be set to False. '
                'Artifacts from other ranks will not be collected, leading to a loss of information required to '
                'restore from checkpoints.'
            ))
        self._enabled = (not rank_zero_only) or dist.get_global_rank() == 0

        self._rank_zero_only = rank_zero_only
        self._is_in_atexit = False

        if self._enabled:
            if "CLEARML_PROJECT_NAME" in os.environ:
                self.project_name = os.environ["CLEARML_PROJECT_NAME"]
            else:
                raise ValueError("CLEARML_PROJECT_NAME environment variable not set")

            if "CLEARML_TASK_NAME" in os.environ:
                self.task_name = os.environ["CLEARML_TASK_NAME"]
            else:
                raise ValueError("CLEARML_TASK_NAME environment variable not set")

            clearml.Task.init(project_name=self.project_name, task_name=self.task_name, reuse_last_task_id=False)

    def _set_is_in_atexit(self):
        self._is_in_atexit = True

    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        if self._enabled:
            import clearml
            clearml.Task.current_task().connect(hyperparameters)

    def log_table(
        self,
        columns: List[str],
        rows: List[List[Any]],
        name: str = 'Table',
        step: Optional[int] = None,
    ) -> None:
          raise NotImplementedError('I have not implemented ClearML logging for tables')

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if self._enabled:
            import clearml

            # wandb.log alters the metrics dictionary object, so we deepcopy to avoid
            # side effects.
            metrics_copy = copy.deepcopy(metrics)
            for k, v in metrics_copy.items():
                clearml.Task.current_task().get_logger().report_scalar(
                    title=k,
                    series=clearml.Task.current_task().name,
                    value=v,
                    iteration=step,
                )

    def log_images(
        self,
        images: Union[np.ndarray, torch.Tensor, Sequence[Union[np.ndarray, torch.Tensor]]],
        name: str = 'Images',
        channels_last: bool = False,
        step: Optional[int] = None,
        masks: Optional[Dict[str, Union[np.ndarray, torch.Tensor, Sequence[Union[np.ndarray, torch.Tensor]]]]] = None,
        mask_class_labels: Optional[Dict[int, str]] = None,
        use_table: bool = False,
    ):
        raise NotImplementedError('I have not implemented ClearML logging for images')

    def state_dict(self) -> Dict[str, Any]:
        return {}
        # raise NotImplementedError('I have not implemented ClearML state_dict')

    def init(self, state: State, logger: Logger) -> None:
        del logger  # unused

        # if self.name is None:
        #     self.name = state.run_name

        # # Adjust name and group based on `rank_zero_only`.
        # if not self._rank_zero_only:
        #     self.name += f'-rank{dist.get_global_rank()}'

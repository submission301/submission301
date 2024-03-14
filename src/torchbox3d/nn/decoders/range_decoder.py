"""Network decoder."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union, cast

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.functional import Tensor

from torchbox3d.math.conversions import BCHW_to_BKC
from torchbox3d.math.linalg.lie.SO3 import yaw_to_quat
from torchbox3d.math.ops.coding import decode_range_view
from torchbox3d.math.ops.index import ravel_multi_index
from torchbox3d.math.ops.nms import batched_multiclass_nms


@dataclass
class RangeDecoder:
    enable_azimuth_invariant_targets: bool
    enable_sample_by_range: bool = False

    def decode(
        self,
        multiscale_outputs: Dict[Union[int, str], Dict[str, Tensor]],
        post_processing_config: DictConfig,
        task_config: DictConfig,
        use_nms: bool = True,
        **kwargs: Dict[int, Any],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Decode the length, width, height cuboid parameterization."""
        predictions_list = []
        for k1, multiscale_outputs in multiscale_outputs.items():
            stride = cast(int, k1)
            cart = multiscale_outputs["cart"]
            mask = multiscale_outputs["mask"]

            task_offset = 0
            for k2, task_group in task_config.items():
                task_id = cast(int, k2)
                outputs = multiscale_outputs[task_id]

                classification_scores = outputs["logits"].sigmoid()
                scores = classification_scores * mask

                # BE CAREFUL. ONLY USE FOR ANALYSIS!
                if kwargs.get("use_oracle", False):
                    classification_labels = kwargs["data"][stride][task_id][
                        "classification_labels"
                    ]
                    classification_labels = F.one_hot(
                        classification_labels, len(task_group) + 1
                    ).permute(0, 3, 1, 2)[:, :-1]
                    scores = classification_labels
                    # outputs["regressands"] = kwargs["data"][stride][task_id][
                    #     "regression_targets"
                    # ]

                scores, categories = cast(
                    Tuple[Tensor, Tensor], scores.max(dim=1, keepdim=True)
                )

                cuboids = decode_range_view(
                    regressands=outputs["regressands"],
                    cart=cart,
                    enable_azimuth_invariant_targets=self.enable_azimuth_invariant_targets,
                )

                if self.enable_sample_by_range:
                    scores, categories, cuboids = _slower_sampler(
                        scores, categories, cuboids, cart
                    )
                else:

                    # scores, categories, cuboids = _fast_sampler(
                    #     scores, categories, cuboids, cart
                    # )

                    scores = BCHW_to_BKC(scores).squeeze(-1)
                    cuboids = BCHW_to_BKC(cuboids)
                    categories = (BCHW_to_BKC(categories)).squeeze(-1)

                categories += task_offset
                task_offset += len(task_group)

                predictions_list.append(
                    {
                        "scores": scores,
                        "cuboids": cuboids,
                        "categories": categories,
                    }
                )

        collated_predictions = defaultdict(list)
        for stride_predictions in predictions_list:
            for k, v in stride_predictions.items():
                collated_predictions[k].append(v)

        predictions = {k: torch.cat(v, dim=1) for k, v in collated_predictions.items()}

        params = predictions["cuboids"]
        scores = predictions["scores"]
        categories = predictions["categories"]
        if use_nms:
            params, scores, categories, batch_index = batched_multiclass_nms(
                params,
                scores,
                categories,
                num_pre_nms=post_processing_config["num_pre_nms"],
                num_post_nms=post_processing_config["num_post_nms"],
                iou_threshold=post_processing_config["nms_threshold"],
                min_confidence=post_processing_config["min_confidence"],
                nms_mode=post_processing_config["nms_mode"],
            )
        else:
            B, N, _ = params.shape
            batch_index = torch.arange(0, B, device=params.device).repeat_interleave(N)
            params = params.flatten(0, 1)
            scores = scores.flatten(0, 1)
            categories = categories.flatten(0, 1)

            t = scores >= post_processing_config["min_confidence"]
            params = params[t]
            scores = scores[t]
            categories = categories[t]
            batch_index = batch_index[t]

        quats_wxyz = yaw_to_quat(params[:, -1:])
        params = torch.cat([params[:, :-1], quats_wxyz], dim=-1)
        return params, scores, categories, batch_index


def _slower_sampler(
    scores: Tensor, categories: Tensor, cuboids: Tensor, cart: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    scores_list = []
    categories_list = []
    cuboids_list = []

    dists = cart.norm(dim=1, keepdim=True)
    lower_bounds = torch.as_tensor((0, 15, 30), device=scores.device).view(1, -1, 1, 1)
    upper_bounds = torch.as_tensor((15, 30, torch.inf), device=scores.device).view(
        1, -1, 1, 1
    )
    partitions = torch.logical_and(dists > lower_bounds, dists <= upper_bounds)

    rates = [4, 2, 1]
    for i, partition in enumerate(partitions.transpose(1, 0)):
        rate = rates[i]
        scores_list.append(
            ((scores * partition.unsqueeze(1))[:, :, ::, ::rate]).flatten(2)
        )
        categories_list.append(categories[:, :, ::, ::rate].flatten(2))
        cuboids_list.append(cuboids[:, :, ::, ::rate].flatten(2))

    scores = torch.cat(scores_list, dim=-1)
    categories = torch.cat(categories_list, dim=-1)
    cuboids = torch.cat(cuboids_list, dim=-1)

    return scores.squeeze(1), categories.squeeze(1), cuboids.transpose(2, 1)


def _fast_sampler(
    scores: Tensor, categories: Tensor, cuboids: Tensor, cart: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    B, C, H, W = cuboids.shape
    dists = cart.norm(dim=1, keepdim=True)

    lower_bounds = torch.as_tensor((0, 10, 15, 20, 30, 45), device=scores.device).view(
        1, -1, 1, 1
    )
    upper_bounds = torch.as_tensor(
        (15, 20, 30, 40, 60, torch.inf), device=scores.device
    ).view(1, -1, 1, 1)
    partitions = torch.logical_and(dists > lower_bounds, dists <= upper_bounds)

    # counts = partitions.sum(dim=[2, 3])

    scores = scores[:, :, None] * partitions[:, None]
    categories = categories[:, :, None] * partitions[:, None]
    cuboids = cuboids[:, :, None] * partitions[:, None]

    num_partitions = lower_bounds.shape[1]

    # H * W (number of total sampling indices)
    all_indices = (
        torch.arange(0, H * W, device=scores.device)
        .view(1, -1)
        .repeat_interleave(num_partitions, 0)
    )

    # Map H * W -> H * W * factor
    # 2^5, 2^4, ... 1
    factors = 2 ** torch.arange(0, num_partitions, device=scores.device).flip(0).view(
        -1, 1
    )

    # up = counts.flatten() // factors.flatten()

    # 2^5 * (0, 1, 2, ..., H * W)
    strided_indices = (factors * all_indices).flatten()

    # Filter out indices that were mapped our of domain.
    partition_indices = torch.arange(
        0, num_partitions, device=scores.device
    ).repeat_interleave(H * W)

    mask = strided_indices < H * W
    partition_indices = partition_indices[mask]
    spatial_indices = strided_indices[mask]

    K = spatial_indices.shape[0]

    batch_indices = torch.arange(0, B, device=scores.device)
    cuboid_indices = torch.arange(0, C, device=scores.device)

    indices = torch.stack(
        [batch_indices.repeat_interleave(K), partition_indices, spatial_indices], dim=-1
    )
    cuboid_indices = torch.stack(
        [
            batch_indices.repeat(C * K),
            cuboid_indices.repeat_interleave(B * K),
            partition_indices.repeat(C),
            spatial_indices.repeat(C),
        ],
        dim=-1,
    )

    raveled_indices = ravel_multi_index(indices, shape=[B, num_partitions, H * W])
    raveled_cuboids_indices = ravel_multi_index(
        cuboid_indices, shape=[B, C, num_partitions, H * W]
    )

    scores = scores.flatten().gather(0, raveled_indices).view(B, -1)

    # scores_upper = (scores.flatten() > 0).sum()
    categories = categories.flatten().gather(0, raveled_indices).view(B, -1)
    cuboids = (
        cuboids.flatten()
        .gather(0, raveled_cuboids_indices)
        .view(B, C, -1)
        .permute(0, 2, 1)
    )
    return scores, categories, cuboids

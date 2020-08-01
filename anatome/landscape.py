from copy import deepcopy
from typing import List, Tuple, Callable

import torch
from torch import nn, Tensor
from tqdm import tqdm

from .utils import _evaluate

EPS = 1e-8


def _filter_normed_random_direction(model: nn.Module
                                    ) -> List[Tensor]:
    # applies filter normalization proposed in Li+2018
    def _filter_norm(dirs: Tensor,
                     params: Tensor
                     ) -> Tensor:
        d_norm = dirs.view(dirs.size(0), -1).norm(dim=-1)
        p_norm = params.view(params.size(0), -1).norm(dim=-1)
        ones = [1 for _ in range(dirs.dim() - 1)]
        return dirs.mul_(p_norm.view(-1, *ones) / (d_norm.view(-1, *ones) + EPS))

    directions = [(params, torch.randn_like(params)) for params in model.parameters()]
    directions = [_filter_norm(dirs, params) for params, dirs in directions]
    return directions


def _get_perturbed_model(model: nn.Module,
                         direction: List[Tensor] or Tuple[List[Tensor], List[Tensor]],
                         step_size: float or Tuple[float, float]
                         ) -> nn.Module:
    # perturb the weight of model along direction with step size
    new_model = deepcopy(model)
    if len(direction) == 2:
        # 2d
        perturbation = [d_0 * step_size[0] + d_1 * step_size[1] for d_0, d_1 in zip(*direction)]
    else:
        # 1d
        perturbation = [d_0 * step_size for d_0 in direction]

    for param, pert in zip(new_model.parameters(), perturbation):
        if param.data.dim() <= 1:
            # ignore biasbn in the original code
            continue
        param.data.add_(pert)

    return new_model


@torch.no_grad()
def landscape1d(model: nn.Module,
                data: Tuple[Tensor, Tensor],
                criterion: Callable[[Tensor, Tensor], Tensor],
                x_range: Tuple[float, float],
                step_size: float,
                ) -> Tuple[Tensor, Tensor]:
    """ Compute loss landscape along a random direction X. The landscape is

    [{criterion(input, target) at Θ+iX} for i in range(x_min, x_max, α)]

    Args:
        model: Trained model, parameterized by Θ
        data: Pairs of [input, target] to compute criterion
        criterion: Criterion of (input, target) -> scalar value
        x_range: (x_min, x_max)
        step_size: α

    Returns: x-coordinates, landscape

    """

    x_coord = torch.arange(x_range[0], x_range[1] + step_size, step_size, dtype=torch.float)
    x_direction = _filter_normed_random_direction(model)
    loss_values = torch.zeros_like(x_coord, device=torch.device('cpu'))
    for i, x in enumerate(tqdm(x_coord.tolist(), ncols=80)):
        new_model = _get_perturbed_model(model, x_direction, x)
        loss_values[i] = _evaluate(new_model, data, criterion)
    return x_coord, loss_values


@torch.no_grad()
def landscape2d(model: nn.Module,
                data: Tuple[Tensor, Tensor],
                criterion: Callable[[Tensor, Tensor], Tensor],
                x_range: Tuple[float, float],
                y_range: Tuple[float, float],
                step_size: float or Tuple[float, float],
                ) -> Tuple[Tensor, Tensor, Tensor]:
    """  Compute loss landscape along two random directions X and Y. The landscape is

    [
     [{criterion(input, target) at Θ+iX+jY}
     for i in range(x_min, x_max, α)]
     for j in range(y_min, y_max, β)]
    ]

    Args:
        model: Trained model, parameterized by Θ
        data: Pairs of [input, target] to compute criterion
        criterion: Criterion of (input, target) -> scalar value
        x_range: (x_min, x_max)
        y_range: (y_min, y_max)
        step_size: α, β

    Returns: x-coordinates, y-coordinates, landscape

    """
    if isinstance(step_size, float):
        step_size = (step_size, step_size)
    x_coord = torch.arange(x_range[0], x_range[1] + step_size[0], step_size[0], dtype=torch.float)
    y_coord = torch.arange(y_range[0], y_range[1] + step_size[1], step_size[1], dtype=torch.float)
    x_coord, y_coord = torch.meshgrid(x_coord, y_coord)
    shape = x_coord.shape
    x_coord, y_coord = x_coord.flatten(), y_coord.flatten()
    x_direction = _filter_normed_random_direction(model)
    y_direction = _filter_normed_random_direction(model)
    loss_values = torch.zeros_like(x_coord, device=torch.device('cpu'))
    # To enable tqdm
    for i, (x, y) in enumerate(zip(
            tqdm(x_coord.tolist(), ncols=80),
            y_coord.tolist())
    ):
        new_model = _get_perturbed_model(model, (x_direction, y_direction), (x, y))
        loss_values[i] = _evaluate(new_model, data, criterion)
    return x_coord.view(shape), y_coord.view(shape), loss_values.view(shape)

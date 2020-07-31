from typing import Tuple, Callable

from torch import nn, Tensor


def _evaluate(model: nn.Module,
              data: Tuple[Tensor, Tensor],
              criterion: Callable[[Tensor, Tensor], Tensor]
              ) -> float:
    # evaluate model with given data points using the criterion
    input, target = data
    return criterion(model(input), target).item()

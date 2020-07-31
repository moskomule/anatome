from typing import Tuple, Callable

import torch
from torch import nn, Tensor

AUTO_CAST = False

def use_auto_cast():
    global AUTO_CAST
    AUTO_CAST = True

def _evaluate(model: nn.Module,
              data: Tuple[Tensor, Tensor],
              criterion: Callable[[Tensor, Tensor], Tensor]
              ) -> float:
    # evaluate model with given data points using the criterion
    with torch.cuda.amp.autocast(AUTO_CAST):
        input, target = data
        return criterion(model(input), target).item()

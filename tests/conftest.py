import random

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def fixed_seed():
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)

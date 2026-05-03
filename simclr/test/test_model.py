import pytest
import torch

from simclr.model.simclr_model import SimCLR
from simclr.utils.misc import load_config

@pytest.fixture
def cfg():
    return load_config("simclr/config/simclr_config.yml")

@pytest.mark.parametrize("bs, n_ch, h, w", [(1, 3, 224, 224)])
def test_model(cfg, bs, n_ch, h, w):
    x = torch.randn([bs, n_ch, h, w])
    model = SimCLR(cfg)
    out = model(x)

    assert out.shape == (bs, cfg.MODEL.HEAD.dim2)
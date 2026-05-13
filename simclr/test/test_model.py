import pytest
import torch

from simclr.model.simclr_model import SimCLRModel
from simclr.utils.misc import load_config


@pytest.mark.parametrize("bs, n_ch, h, w", [(1, 3, 32, 32)])
def test_model_encoder(bs, n_ch, h, w):
    x = torch.randn([bs, n_ch, h, w])
    model = SimCLRModel().eval()
    out = model(x, encoder_out=True)

    assert out.shape == (bs, 256)

@pytest.mark.parametrize("bs, n_ch, h, w", [(1, 3, 32, 32)])
def test_model_complete(bs, n_ch, h, w):
    x = torch.randn([bs, n_ch, h, w])
    model = SimCLRModel().eval()
    out = model(x)

    assert out.shape == (bs, 128)

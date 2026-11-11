import pytest


@pytest.mark.parametrize("stride,kernel", [(2, 4), (3, 6), (7, 14), (1, 1)])
def test_grouped_transpose_matches_published_codec(stride, kernel):
    torch = pytest.importorskip("torch")
    pytest.importorskip("nanocodec_torch")
    from nanocodec_torch.models.decoder import CausalConvTranspose1d

    from kani_tts_mcp.models import grouped_transpose

    torch.manual_seed(123)
    layer = CausalConvTranspose1d(8, 4, kernel, stride)
    with torch.no_grad():
        layer.weight.normal_()
        layer.bias.normal_()
        x = torch.randn(2, 8, 11)
        expected = layer(x)
        actual = grouped_transpose(layer, x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert actual.shape == (2, 4, 11 * stride)

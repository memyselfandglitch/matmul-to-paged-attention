import unittest

import torch
from torch import nn

from imbps_bench.hf_opt import (
    HFOPTIMBPSMLP,
    HFOPTReferenceMLP,
    OPTModelIMBPSPatcher,
)


class HFOPTMLPTests(unittest.TestCase):
    def _make_linears(self):
        torch.manual_seed(17)
        fc1 = nn.Linear(11, 29, bias=True)
        fc2 = nn.Linear(29, 11, bias=True)
        return fc1.eval(), fc2.eval()

    def test_real_linear_layout_bias_and_uneven_splits(self):
        fc1, fc2 = self._make_linears()
        activation = nn.ReLU()
        reference = HFOPTReferenceMLP(fc1, activation, fc2).eval()
        x = torch.randn(2, 7, 11)
        with torch.inference_mode():
            expected = reference(x).clone()
            for layout in ("views", "prepacked"):
                for split_k in (1, 2, 5, 7):
                    with self.subTest(layout=layout, split_k=split_k):
                        split = HFOPTIMBPSMLP(
                            fc1, activation, fc2, split_k, layout, "relu"
                        ).eval()
                        actual = split(x).clone()
                        self.assertTrue(
                            torch.allclose(actual, expected, rtol=2e-5, atol=2e-6),
                            msg="layout=%s K=%d max_error=%g"
                            % (layout, split_k, (actual - expected).abs().max().item()),
                        )

    def test_down_bias_is_added_once(self):
        fc1, fc2 = self._make_linears()
        fc1.weight.data.zero_()
        fc1.bias.data.zero_()
        fc2.weight.data.zero_()
        fc2.bias.data.fill_(3.0)
        x = torch.randn(4, 11)
        split = HFOPTIMBPSMLP(fc1, nn.ReLU(), fc2, 5, "prepacked", "relu")
        with torch.inference_mode():
            actual = split(x)
        self.assertTrue(torch.equal(actual, torch.full_like(actual, 3.0)))

    def test_configured_gelu_is_preserved(self):
        fc1, fc2 = self._make_linears()
        activation = nn.GELU()
        x = torch.randn(6, 11)
        with torch.inference_mode():
            expected = HFOPTReferenceMLP(fc1, activation, fc2)(x).clone()
            actual = HFOPTIMBPSMLP(
                fc1, activation, fc2, 5, "prepacked", "gelu"
            )(x).clone()
        self.assertTrue(torch.allclose(actual, expected, rtol=2e-5, atol=2e-6))

    def test_workspace_shrinks_with_k(self):
        fc1, fc2 = self._make_linears()
        x = torch.randn(9, 11)
        one = HFOPTIMBPSMLP(fc1, nn.ReLU(), fc2, 1, "views", "relu")
        five = HFOPTIMBPSMLP(fc1, nn.ReLU(), fc2, 5, "views", "relu")
        with torch.inference_mode():
            one(x)
            five(x)
        self.assertLess(five.workspace_bytes, one.workspace_bytes)

    def test_fp32_partial_accumulation_returns_input_dtype(self):
        fc1, fc2 = self._make_linears()
        fc1 = fc1.to(dtype=torch.bfloat16)
        fc2 = fc2.to(dtype=torch.bfloat16)
        x = torch.randn(9, 11, dtype=torch.bfloat16)
        split = HFOPTIMBPSMLP(
            fc1,
            nn.ReLU(),
            fc2,
            5,
            "prepacked",
            "relu",
            accumulation_dtype="fp32",
        )
        with torch.inference_mode():
            output = split(x)
        self.assertEqual(output.dtype, x.dtype)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(any(block.down_weight_t_accum is not None for block in split.blocks))

    def test_fp32_sum_preserves_bf16_gemm_weights(self):
        fc1, fc2 = self._make_linears()
        fc1 = fc1.to(dtype=torch.bfloat16)
        fc2 = fc2.to(dtype=torch.bfloat16)
        x = torch.randn(9, 11, dtype=torch.bfloat16)
        split = HFOPTIMBPSMLP(
            fc1,
            nn.ReLU(),
            fc2,
            5,
            "prepacked",
            "relu",
            accumulation_dtype="fp32_sum",
        )
        with torch.inference_mode():
            output = split(x)
        self.assertEqual(output.dtype, x.dtype)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(all(block.down_weight_t_accum is None for block in split.blocks))


class _FakeOPTLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 19, bias=True)
        self.activation_fn = nn.ReLU()
        self.fc2 = nn.Linear(19, 8, bias=True)

    def forward(self, x):
        return self.fc2(self.activation_fn(self.fc1(x)))


class _Container(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_FakeOPTLayer(), _FakeOPTLayer()])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ModelPatcherTests(unittest.TestCase):
    def test_toggles_complete_layer_chain_without_rewriting_layer_forward(self):
        torch.manual_seed(3)
        model = _Container().eval()
        x = torch.randn(5, 8)
        patcher = OPTModelIMBPSPatcher(model, 4, "prepacked", "relu")
        with torch.inference_mode():
            patcher.enable_reference()
            expected = model(x).clone()
            patcher.enable_imbps()
            actual = model(x).clone()
            patcher.enable_reference()
            restored = model(x).clone()
        self.assertTrue(torch.allclose(actual, expected, rtol=2e-5, atol=2e-6))
        self.assertTrue(torch.equal(restored, expected))


if __name__ == "__main__":
    unittest.main()

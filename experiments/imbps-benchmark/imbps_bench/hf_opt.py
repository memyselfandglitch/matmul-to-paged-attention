"""Hugging Face OPT MLP adapters for reference and IMBPS execution.

This module intentionally imports only PyTorch.  Loading Hugging Face models is
handled by :mod:`imbps_bench.hf_runner`, so the original kernel benchmark keeps
working when ``transformers`` is not installed.

Hugging Face ``nn.Linear`` weights use ``[out_features, in_features]`` layout:

* ``fc1.weight``: ``[intermediate, hidden]``
* ``fc2.weight``: ``[hidden, intermediate]``

IMBPS partitions rows of ``fc1`` and columns of ``fc2``.  The down-projection
bias is added once after all partial products have been accumulated.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from imbps_bench.kernels import split_ranges, tensor_nbytes


Activation = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class HFOPTBlock:
    """One intermediate-dimension block in matrix-ready layouts."""

    start: int
    end: int
    # Native Hugging Face Linear layout: [block_width, hidden].
    up_weight: torch.Tensor
    up_bias: Optional[torch.Tensor]
    # Mathematical down layout used by addmm: [block_width, hidden].
    down_weight_t: torch.Tensor
    # Optional FP32 copy used to avoid rounding each partial sum to BF16.
    down_weight_t_accum: Optional[torch.Tensor] = None

    @property
    def width(self) -> int:
        return self.end - self.start


def _clone_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone(memory_format=torch.contiguous_format)


def _parameter_bytes(linear: nn.Linear) -> int:
    total = tensor_nbytes(linear.weight)
    if linear.bias is not None:
        total += tensor_nbytes(linear.bias)
    return total


class HFOPTReferenceMLP(nn.Module):
    """The exact Hugging Face OPT MLP composition used as the baseline."""

    def __init__(self, fc1: nn.Linear, activation_fn: Activation, fc2: nn.Linear):
        super().__init__()
        self.fc1 = fc1
        self.activation_fn = activation_fn
        self.fc2 = fc2

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation_fn(self.fc1(hidden_states)))


class HFOPTIMBPSMLP(nn.Module):
    """Sequential IMBPS replacement for a Hugging Face OPT MLP.

    The module is designed for CPU inference. Persistent activation and output
    buffers are cached by row count, so prefill and decode shapes are both reused.
    The returned output aliases the active workspace and is overwritten by the
    next call with the same shape. This is safe in an ordinary OPT decoder because
    the caller immediately forms the residual result before the layer is invoked
    again.
    """

    def __init__(
        self,
        fc1: nn.Linear,
        activation_fn: Activation,
        fc2: nn.Linear,
        split_k: int,
        weight_layout: str = "prepacked",
        activation_name: str = "unknown",
        accumulation_dtype: str = "input",
    ) -> None:
        super().__init__()
        if not isinstance(fc1, nn.Linear) or not isinstance(fc2, nn.Linear):
            raise TypeError("OPT fc1 and fc2 must be torch.nn.Linear modules")
        if fc1.in_features != fc2.out_features:
            raise ValueError("fc1 input and fc2 output dimensions must match")
        if fc1.out_features != fc2.in_features:
            raise ValueError("fc1 output and fc2 input dimensions must match")
        if fc1.weight.device.type != "cpu" or fc2.weight.device.type != "cpu":
            raise ValueError("the Hugging Face OPT benchmark currently supports CPU tensors only")
        if fc1.weight.dtype != fc2.weight.dtype:
            raise ValueError("fc1 and fc2 weights must use the same dtype")
        if weight_layout not in ("prepacked", "views"):
            raise ValueError("weight_layout must be 'prepacked' or 'views'")
        if accumulation_dtype not in ("input", "fp32", "fp32_sum"):
            raise ValueError("accumulation_dtype must be 'input', 'fp32', or 'fp32_sum'")

        self.hidden_size = int(fc1.in_features)
        self.intermediate_size = int(fc1.out_features)
        self.split_k = int(split_k)
        self.weight_layout = weight_layout
        self.activation_name = activation_name
        self.accumulation_dtype = accumulation_dtype
        self.activation_fn = activation_fn
        self.canonical_parameter_bytes = _parameter_bytes(fc1) + _parameter_bytes(fc2)

        started = perf_counter_ns()
        blocks: List[HFOPTBlock] = []
        packed_parameter_bytes = 0
        for start, end in split_ranges(self.intermediate_size, self.split_k):
            up_weight = fc1.weight[start:end, :].detach()
            up_bias = fc1.bias[start:end].detach() if fc1.bias is not None else None
            # fc2 is [hidden, intermediate].  Transpose the selected columns so
            # addmm sees the mathematical [block_width, hidden] down matrix.
            down_weight_t = fc2.weight[:, start:end].t().detach()
            if weight_layout == "prepacked":
                up_weight = _clone_contiguous(up_weight)
                down_weight_t = _clone_contiguous(down_weight_t)
                if up_bias is not None:
                    up_bias = _clone_contiguous(up_bias)
                packed_parameter_bytes += tensor_nbytes(up_weight) + tensor_nbytes(down_weight_t)
                if up_bias is not None:
                    packed_parameter_bytes += tensor_nbytes(up_bias)
            down_weight_t_accum = None
            if accumulation_dtype == "fp32" and down_weight_t.dtype != torch.float32:
                down_weight_t_accum = down_weight_t.float().contiguous()
                packed_parameter_bytes += tensor_nbytes(down_weight_t_accum)
            blocks.append(
                HFOPTBlock(
                    start=start,
                    end=end,
                    up_weight=up_weight,
                    up_bias=up_bias,
                    down_weight_t=down_weight_t,
                    down_weight_t_accum=down_weight_t_accum,
                )
            )

        self.blocks: Sequence[HFOPTBlock] = tuple(blocks)
        self.max_width = max(block.width for block in self.blocks)
        self.packing_ms = (perf_counter_ns() - started) / 1_000_000.0
        self.packed_parameter_bytes = packed_parameter_bytes
        self.down_bias = fc2.bias.detach() if fc2.bias is not None else None
        if weight_layout == "prepacked" and self.down_bias is not None:
            self.down_bias = _clone_contiguous(self.down_bias)
            self.packed_parameter_bytes += tensor_nbytes(self.down_bias)
        self.down_bias_accum = None
        if (
            accumulation_dtype in ("fp32", "fp32_sum")
            and self.down_bias is not None
            and self.down_bias.dtype != torch.float32
        ):
            self.down_bias_accum = self.down_bias.float().contiguous()
            self.packed_parameter_bytes += tensor_nbytes(self.down_bias_accum)

        self.register_buffer("_up_workspace", None, persistent=False)
        self.register_buffer("_output_workspace", None, persistent=False)
        self.register_buffer("_accum_up_workspace", None, persistent=False)
        self.register_buffer("_partial_output_workspace", None, persistent=False)
        self.register_buffer("_cast_workspace", None, persistent=False)
        self._workspace_cache: Dict[
            Tuple[int, torch.dtype, torch.device],
            Tuple[
                torch.Tensor,
                torch.Tensor,
                Optional[torch.Tensor],
                Optional[torch.Tensor],
                Optional[torch.Tensor],
            ],
        ] = {}

    @property
    def workspace_bytes(self) -> int:
        return sum(
            sum(tensor_nbytes(tensor) for tensor in workspace if tensor is not None)
            for workspace in self._workspace_cache.values()
        )

    def _ensure_workspace(self, rows: int, source: torch.Tensor) -> None:
        key = (rows, source.dtype, source.device)
        workspace = self._workspace_cache.get(key)
        if workspace is None:
            up = torch.empty(
                (rows, self.max_width), dtype=source.dtype, device=source.device
            )
            output_dtype = (
                torch.float32
                if self.accumulation_dtype in ("fp32", "fp32_sum")
                else source.dtype
            )
            output = torch.empty(
                (rows, self.hidden_size), dtype=output_dtype, device=source.device
            )
            accum_up = (
                torch.empty(
                    (rows, self.max_width), dtype=torch.float32, device=source.device
                )
                if self.accumulation_dtype == "fp32" and source.dtype != torch.float32
                else None
            )
            partial_output = (
                torch.empty(
                    (rows, self.hidden_size), dtype=source.dtype, device=source.device
                )
                if self.accumulation_dtype == "fp32_sum" and source.dtype != torch.float32
                else None
            )
            cast = (
                torch.empty(
                    (rows, self.hidden_size), dtype=source.dtype, device=source.device
                )
                if output_dtype != source.dtype
                else None
            )
            workspace = (up, output, accum_up, partial_output, cast)
            self._workspace_cache[key] = workspace
        (
            self._up_workspace,
            self._output_workspace,
            self._accum_up_workspace,
            self._partial_output_workspace,
            self._cast_workspace,
        ) = workspace

    def _activate_in_place(self, tensor: torch.Tensor) -> None:
        name = self.activation_name.lower()
        if name == "relu":
            tensor.relu_()
            return
        # Preserve arbitrary Hugging Face activation semantics.  Most functions
        # allocate a block-sized result; it is copied back so the reusable buffer
        # remains the operand consumed by the down projection.
        activated = self.activation_fn(tensor)
        if activated.data_ptr() != tensor.data_ptr():
            tensor.copy_(activated)

    def _functional_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Autograd-safe fallback used outside inference/no-grad contexts."""

        shape = hidden_states.shape
        x = hidden_states.reshape(-1, self.hidden_size)
        partials = []
        for block in self.blocks:
            up = torch.nn.functional.linear(x, block.up_weight, block.up_bias)
            up = self.activation_fn(up)
            if self.accumulation_dtype == "fp32":
                down = block.down_weight_t_accum
                if down is None:
                    down = block.down_weight_t
                partials.append(torch.mm(up.float(), down))
            elif self.accumulation_dtype == "fp32_sum":
                partials.append(torch.mm(up, block.down_weight_t).float())
            else:
                partials.append(torch.mm(up, block.down_weight_t))
        output = torch.stack(partials, dim=0).sum(dim=0)
        if self.down_bias is not None:
            bias = self.down_bias_accum if self.down_bias_accum is not None else self.down_bias
            output = output + bias
        return output.to(dtype=hidden_states.dtype).view(*shape[:-1], self.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                "expected hidden dimension %d, got %d"
                % (self.hidden_size, hidden_states.shape[-1])
            )
        if torch.is_grad_enabled():
            return self._functional_forward(hidden_states)

        shape = hidden_states.shape
        x = hidden_states.reshape(-1, self.hidden_size)
        self._ensure_workspace(x.shape[0], x)
        assert self._up_workspace is not None
        assert self._output_workspace is not None
        output = self._output_workspace
        output.zero_()
        for block in self.blocks:
            up = self._up_workspace[:, : block.width]
            torch.mm(x, block.up_weight.t(), out=up)
            if block.up_bias is not None:
                up.add_(block.up_bias)
            self._activate_in_place(up)
            if self.accumulation_dtype == "fp32":
                accum_up = self._accum_up_workspace
                if accum_up is None:
                    accum_up = up
                else:
                    accum_up = accum_up[:, : block.width]
                    accum_up.copy_(up)
                down = block.down_weight_t_accum
                if down is None:
                    down = block.down_weight_t
                output.addmm_(accum_up, down)
            elif self.accumulation_dtype == "fp32_sum":
                partial_output = self._partial_output_workspace
                if partial_output is None:
                    output.addmm_(up, block.down_weight_t)
                else:
                    torch.mm(up, block.down_weight_t, out=partial_output)
                    output.add_(partial_output)
            else:
                output.addmm_(up, block.down_weight_t)
        if self.down_bias is not None:
            bias = self.down_bias_accum if self.down_bias_accum is not None else self.down_bias
            output.add_(bias)
        if self._cast_workspace is not None:
            self._cast_workspace.copy_(output)
            output = self._cast_workspace
        return output.view(*shape[:-1], self.hidden_size)


@dataclass
class _LayerPatch:
    layer: nn.Module
    fc1: nn.Linear
    activation_fn: Activation
    fc2: nn.Linear
    split_mlp: HFOPTIMBPSMLP


def find_opt_decoder_layers(model: nn.Module) -> Sequence[nn.Module]:
    """Return Hugging Face OPT decoder layers without importing Transformers."""

    candidates = [
        getattr(getattr(getattr(model, "model", None), "decoder", None), "layers", None),
        getattr(getattr(model, "decoder", None), "layers", None),
        getattr(model, "layers", None),
    ]
    for layers in candidates:
        if layers is None:
            continue
        result = list(layers)
        if result and all(
            hasattr(layer, "fc1") and hasattr(layer, "fc2") and hasattr(layer, "activation_fn")
            for layer in result
        ):
            return result
    raise ValueError("could not locate Hugging Face OPT decoder layers")


class OPTModelIMBPSPatcher:
    """Toggle all selected OPT layers between native and IMBPS MLP execution."""

    def __init__(
        self,
        model: nn.Module,
        split_k: int,
        weight_layout: str,
        activation_name: str,
        accumulation_dtype: str = "input",
        layer_indices: Optional[Sequence[int]] = None,
    ) -> None:
        layers = find_opt_decoder_layers(model)
        selected = set(range(len(layers))) if layer_indices is None else set(layer_indices)
        if not selected or min(selected) < 0 or max(selected) >= len(layers):
            raise ValueError("layer_indices must identify at least one existing decoder layer")

        patches: List[_LayerPatch] = []
        for index, layer in enumerate(layers):
            if index not in selected:
                continue
            fc1 = layer.fc1
            fc2 = layer.fc2
            activation_fn = layer.activation_fn
            split_mlp = HFOPTIMBPSMLP(
                fc1=fc1,
                activation_fn=activation_fn,
                fc2=fc2,
                split_k=split_k,
                weight_layout=weight_layout,
                activation_name=activation_name,
                accumulation_dtype=accumulation_dtype,
            )
            patches.append(
                _LayerPatch(
                    layer=layer,
                    fc1=fc1,
                    activation_fn=activation_fn,
                    fc2=fc2,
                    split_mlp=split_mlp,
                )
            )
        self.patches = tuple(patches)
        self.enabled = False

    @property
    def packing_ms(self) -> float:
        return sum(patch.split_mlp.packing_ms for patch in self.patches)

    @property
    def packed_parameter_bytes(self) -> int:
        return sum(patch.split_mlp.packed_parameter_bytes for patch in self.patches)

    @property
    def workspace_bytes(self) -> int:
        return sum(patch.split_mlp.workspace_bytes for patch in self.patches)

    def enable_imbps(self) -> None:
        for patch in self.patches:
            patch.layer.fc1 = nn.Identity()
            patch.layer.activation_fn = nn.Identity()
            patch.layer.fc2 = patch.split_mlp
        self.enabled = True

    def enable_reference(self) -> None:
        for patch in self.patches:
            patch.layer.fc1 = patch.fc1
            patch.layer.activation_fn = patch.activation_fn
            patch.layer.fc2 = patch.fc2
        self.enabled = False

    def close(self) -> None:
        self.enable_reference()

    def __enter__(self) -> "OPTModelIMBPSPatcher":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

"""Reference and split MLP kernels used by the benchmark.

Weights use their mathematical layouts:

* up and gate: [hidden, intermediate]
* down: [intermediate, hidden]

The prepacked IMBPS representation stores each split as a contiguous tensor.
Packing time is measured separately and never included in steady-state latency.
"""

from dataclasses import dataclass
from time import perf_counter_ns
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


MLP_KINDS = ("opt", "swiglu")


@dataclass(frozen=True)
class MLPWeights:
    kind: str
    up: torch.Tensor
    down: torch.Tensor
    gate: Optional[torch.Tensor] = None

    @property
    def hidden_size(self) -> int:
        return int(self.up.shape[0])

    @property
    def intermediate_size(self) -> int:
        return int(self.up.shape[1])

    @property
    def dtype(self) -> torch.dtype:
        return self.up.dtype

    def validate(self) -> None:
        if self.kind not in MLP_KINDS:
            raise ValueError("unsupported MLP kind: %s" % self.kind)
        h, intermediate = self.up.shape
        if tuple(self.down.shape) != (intermediate, h):
            raise ValueError("down must have shape [intermediate, hidden]")
        if self.kind == "swiglu":
            if self.gate is None or tuple(self.gate.shape) != (h, intermediate):
                raise ValueError("SwiGLU requires gate shape [hidden, intermediate]")
        elif self.gate is not None:
            raise ValueError("OPT MLP does not use a gate weight")
        tensors = [self.up, self.down]
        if self.gate is not None:
            tensors.append(self.gate)
        if any(t.device.type != "cpu" for t in tensors):
            raise ValueError("the CPU benchmark only accepts CPU tensors")
        if any(t.dtype != self.up.dtype for t in tensors):
            raise ValueError("all weights must have the same dtype")


@dataclass(frozen=True)
class MLPBlock:
    start: int
    end: int
    up: torch.Tensor
    down: torch.Tensor
    gate: Optional[torch.Tensor] = None

    @property
    def width(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class PackedMLP:
    kind: str
    blocks: Sequence[MLPBlock]
    layout: str
    packing_ms: float
    packed_bytes: int

    @property
    def split_k(self) -> int:
        return len(self.blocks)

    @property
    def max_width(self) -> int:
        return max(block.width for block in self.blocks)


@dataclass
class ReferenceWorkspace:
    up: torch.Tensor
    output: torch.Tensor
    gate: Optional[torch.Tensor] = None

    @property
    def nbytes(self) -> int:
        tensors = [self.up, self.output]
        if self.gate is not None:
            tensors.append(self.gate)
        return sum(tensor_nbytes(t) for t in tensors)


@dataclass
class SplitWorkspace:
    up: torch.Tensor
    output: torch.Tensor
    gate: Optional[torch.Tensor] = None

    @property
    def nbytes(self) -> int:
        tensors = [self.up, self.output]
        if self.gate is not None:
            tensors.append(self.gate)
        return sum(tensor_nbytes(t) for t in tensors)


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def weights_nbytes(weights: MLPWeights) -> int:
    total = tensor_nbytes(weights.up) + tensor_nbytes(weights.down)
    if weights.gate is not None:
        total += tensor_nbytes(weights.gate)
    return total


def split_ranges(total: int, split_k: int) -> List[Tuple[int, int]]:
    if total <= 0:
        raise ValueError("total must be positive")
    if split_k <= 0:
        raise ValueError("split_k must be positive")
    if split_k > total:
        raise ValueError("split_k cannot exceed the intermediate dimension")
    quotient, remainder = divmod(total, split_k)
    ranges: List[Tuple[int, int]] = []
    start = 0
    for index in range(split_k):
        width = quotient + (1 if index < remainder else 0)
        end = start + width
        ranges.append((start, end))
        start = end
    return ranges


def make_weights(
    kind: str,
    hidden_size: int,
    intermediate_size: int,
    dtype: torch.dtype,
    seed: int,
    std: float = 0.02,
) -> MLPWeights:
    if kind not in MLP_KINDS:
        raise ValueError("unsupported MLP kind: %s" % kind)
    if hidden_size <= 0 or intermediate_size <= 0:
        raise ValueError("hidden and intermediate sizes must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    def normal(shape: Tuple[int, int]) -> torch.Tensor:
        result = torch.empty(shape, dtype=dtype, device="cpu")
        return result.normal_(mean=0.0, std=std, generator=generator)

    up = normal((hidden_size, intermediate_size))
    down = normal((intermediate_size, hidden_size))
    gate = normal((hidden_size, intermediate_size)) if kind == "swiglu" else None
    weights = MLPWeights(kind=kind, up=up, down=down, gate=gate)
    weights.validate()
    return weights


def make_input(
    tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    seed: int,
    std: float = 0.02,
) -> torch.Tensor:
    if tokens <= 0 or hidden_size <= 0:
        raise ValueError("tokens and hidden_size must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    result = torch.empty((tokens, hidden_size), dtype=dtype, device="cpu")
    return result.normal_(mean=0.0, std=std, generator=generator)


def pack_weights(weights: MLPWeights, split_k: int, layout: str) -> PackedMLP:
    """Create split views or persistent contiguous block tensors."""
    weights.validate()
    if layout not in ("prepacked", "views"):
        raise ValueError("layout must be 'prepacked' or 'views'")
    started = perf_counter_ns()
    blocks: List[MLPBlock] = []
    packed_bytes = 0
    for start, end in split_ranges(weights.intermediate_size, split_k):
        up = weights.up[:, start:end]
        down = weights.down[start:end, :]
        gate = weights.gate[:, start:end] if weights.gate is not None else None
        if layout == "prepacked":
            up = up.contiguous()
            down = down.contiguous()
            if gate is not None:
                gate = gate.contiguous()
            packed_bytes += tensor_nbytes(up) + tensor_nbytes(down)
            if gate is not None:
                packed_bytes += tensor_nbytes(gate)
        blocks.append(MLPBlock(start=start, end=end, up=up, down=down, gate=gate))
    elapsed_ms = (perf_counter_ns() - started) / 1_000_000.0
    return PackedMLP(
        kind=weights.kind,
        blocks=tuple(blocks),
        layout=layout,
        packing_ms=elapsed_ms,
        packed_bytes=packed_bytes,
    )


def make_reference_workspace(
    kind: str,
    tokens: int,
    hidden_size: int,
    intermediate_size: int,
    dtype: torch.dtype,
) -> ReferenceWorkspace:
    up = torch.empty((tokens, intermediate_size), dtype=dtype, device="cpu")
    output = torch.empty((tokens, hidden_size), dtype=dtype, device="cpu")
    gate = (
        torch.empty((tokens, intermediate_size), dtype=dtype, device="cpu")
        if kind == "swiglu"
        else None
    )
    return ReferenceWorkspace(up=up, output=output, gate=gate)


def make_split_workspace(
    kind: str,
    tokens: int,
    hidden_size: int,
    max_width: int,
    dtype: torch.dtype,
) -> SplitWorkspace:
    up = torch.empty((tokens, max_width), dtype=dtype, device="cpu")
    output = torch.empty((tokens, hidden_size), dtype=dtype, device="cpu")
    gate = (
        torch.empty((tokens, max_width), dtype=dtype, device="cpu")
        if kind == "swiglu"
        else None
    )
    return SplitWorkspace(up=up, output=output, gate=gate)


def _gelu_in_place(tensor: torch.Tensor, approximate: str) -> None:
    # aten.gelu_ is available in supported PyTorch releases. The fallback keeps
    # portability for builds which omit the public in-place binding.
    try:
        torch.ops.aten.gelu_.default(tensor, approximate=approximate)
    except (AttributeError, RuntimeError, TypeError):
        tensor.copy_(F.gelu(tensor, approximate=approximate))


def reference_forward_into(
    x: torch.Tensor,
    weights: MLPWeights,
    workspace: ReferenceWorkspace,
    gelu_approximate: str = "none",
) -> torch.Tensor:
    """Run the unsplit MLP with persistent activation/output buffers."""
    weights.validate()
    torch.mm(x, weights.up, out=workspace.up)
    if weights.kind == "opt":
        _gelu_in_place(workspace.up, gelu_approximate)
        torch.mm(workspace.up, weights.down, out=workspace.output)
        return workspace.output

    if workspace.gate is None or weights.gate is None:
        raise ValueError("SwiGLU workspace and gate weight are required")
    torch.mm(x, weights.gate, out=workspace.gate)
    F.silu(workspace.gate, inplace=True)
    workspace.gate.mul_(workspace.up)
    torch.mm(workspace.gate, weights.down, out=workspace.output)
    return workspace.output


def imbps_forward_into(
    x: torch.Tensor,
    packed: PackedMLP,
    workspace: SplitWorkspace,
    gelu_approximate: str = "none",
) -> torch.Tensor:
    """Run IMBPS sequentially, accumulating each split into one output buffer."""
    workspace.output.zero_()
    for block in packed.blocks:
        up = workspace.up[:, : block.width]
        torch.mm(x, block.up, out=up)
        if packed.kind == "opt":
            _gelu_in_place(up, gelu_approximate)
            workspace.output.addmm_(up, block.down)
            continue

        if workspace.gate is None or block.gate is None:
            raise ValueError("SwiGLU workspace and gate block are required")
        gate = workspace.gate[:, : block.width]
        torch.mm(x, block.gate, out=gate)
        F.silu(gate, inplace=True)
        gate.mul_(up)
        workspace.output.addmm_(gate, block.down)
    return workspace.output


def theoretical_flops(kind: str, tokens: int, hidden_size: int, intermediate_size: int) -> int:
    gemm_count = 2 if kind == "opt" else 3
    return int(gemm_count * 2 * tokens * hidden_size * intermediate_size)


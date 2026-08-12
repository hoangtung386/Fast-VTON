"""Static inspection of the released SwiftEdit checkpoints.

Runs entirely on CPU and never executes a model: safetensors headers are read directly
and the pickled adapter is memory-mapped. Answers three questions the paper leaves open:

1. Was the generator frozen during training? (Compare ``ip_adapter.bin`` to ``sbv2_0.5``.)
2. How far did the inversion network drift from its initialisation?
3. What is the real parameter budget for each fine-tuning strategy?

Also serves as the verification tool for the Stage 1 checklist: after training, rerun
``--compare-generator`` to confirm no frozen tensor was accidentally updated.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from collections.abc import Callable
from pathlib import Path

import torch
from safetensors import safe_open

from src.vton.config import CheckpointConfig


def _rule(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def _tensor_names(path: Path) -> list[str]:
    """List tensor names in a safetensors file without reading the payload."""
    with safe_open(path, framework="pt") as handle:
        return list(handle.keys())


def compare_unets(reference: Path, candidate: Path) -> None:
    """Report how far two safetensors UNets diverge, tensor by tensor."""
    ref_names, cand_names = _tensor_names(reference), _tensor_names(candidate)
    print(f"  tensors: {len(ref_names)} vs {len(cand_names)}")
    print(f"  identical key set: {set(ref_names) == set(cand_names)}")

    identical = 0
    worst_name, worst_drift = None, 0.0
    with safe_open(reference, framework="pt") as ref, safe_open(candidate, framework="pt") as cand:
        for name in ref_names:
            left, right = ref.get_tensor(name), cand.get_tensor(name)
            if torch.equal(left, right):
                identical += 1
                continue
            drift = (left - right).abs().mean().item() / (left.abs().mean().item() + 1e-12)
            if drift > worst_drift:
                worst_name, worst_drift = name, drift

    print(f"  bit-identical    : {identical}/{len(ref_names)}")
    print(f"  differing        : {len(ref_names) - identical}/{len(ref_names)}")
    if worst_name is not None:
        print(f"  largest relative drift: {worst_drift:.3f} at {worst_name}")


def inspect_adapter(adapter_path: Path, generator_weights: Path) -> None:
    """Break ``ip_adapter.bin`` down by group and check the UNet half against disk."""
    state = torch.load(str(adapter_path), map_location="cpu", mmap=True, weights_only=True)

    counts: collections.Counter[str] = collections.Counter()
    params: collections.Counter[str] = collections.Counter()
    for key, value in state.items():
        group = key.split(".")[0]
        counts[group] += 1
        params[group] += value.numel()

    print(f"  total keys: {len(state)}")
    for group in sorted(counts):
        print(f"    {group:20s} {counts[group]:5d} tensors  {params[group] / 1e6:9.2f} M params")

    print("\n  image projection shapes:")
    for key, value in state.items():
        if key.startswith("image_proj_model"):
            print(f"    {key:42s} {tuple(value.shape)}")

    _rule("Was the generator frozen during training?")
    reference = set(_tensor_names(generator_weights))
    unet_keys = [k for k in state if k.startswith("unet.")]
    matched = identical = 0
    differing: list[str] = []

    with safe_open(generator_weights, framework="pt") as handle:
        for key in unet_keys:
            bare = key[len("unet.") :]
            if bare not in reference:
                continue
            matched += 1
            saved = handle.get_tensor(bare)
            if saved.shape == state[key].shape and torch.equal(saved, state[key].to(saved.dtype)):
                identical += 1
            else:
                differing.append(bare)

    print(f"  unet.* tensors in adapter : {len(unet_keys)}")
    print(f"  matched by name           : {matched}")
    print(f"  bit-identical             : {identical}")
    print(f"  differing                 : {len(differing)}")
    for name in differing[:10]:
        print(f"      {name}")

    unique = sum(
        value.numel()
        for key, value in state.items()
        if not key.startswith("adapter_modules.")  # alias of tensors already under unet.*
    )
    expected_bytes = unique * 4
    actual_bytes = adapter_path.stat().st_size
    print(f"\n  unique parameters : {unique / 1e6:.2f} M")
    print(f"  implied file size : {expected_bytes / 1e9:.3f} GB")
    print(f"  actual file size  : {actual_bytes / 1e9:.3f} GB")
    print(f"  consistent        : {math.isclose(expected_bytes, actual_bytes, rel_tol=1e-3)}")


def report_budget(inversion: Path, generator: Path, adapter: Path) -> None:
    """Print the parameter budget for each fine-tuning strategy."""

    def total(path: Path, predicate: Callable[[str], bool] = lambda _: True) -> int:
        with safe_open(path, framework="pt") as handle:
            return sum(
                math.prod(handle.get_slice(name).get_shape())
                for name in handle.keys()  # noqa: SIM118 - safe_open is not a mapping
                if predicate(name)
            )

    generator_params = total(generator)
    inversion_params = total(inversion)
    prompt_kv = total(generator, lambda n: n.endswith(("attn2.to_k.weight", "attn2.to_v.weight")))

    state = torch.load(str(adapter), map_location="cpu", mmap=True, weights_only=True)
    projection = sum(v.numel() for k, v in state.items() if k.startswith("image_proj_model"))
    image_kv = sum(v.numel() for k, v in state.items() if k.startswith("adapter_modules."))

    rows = [
        ("generator UNet (G)", generator_params),
        ("inversion UNet (F)", inversion_params),
        ("image_proj_model", projection),
        ("to_k_ip / to_v_ip", image_kv),
        ("attn2 to_k/to_v (W_y)", prompt_kv),
        ("full fine-tune (G + F)", generator_params + inversion_params),
        ("stage 1 adapters", projection + prompt_kv),
    ]
    for label, value in rows:
        print(f"  {label:26s} {value / 1e6:9.2f} M")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-root", type=Path, default=CheckpointConfig().root)
    parser.add_argument(
        "--compare-generator",
        action="store_true",
        help="only verify the generator is untouched, then exit",
    )
    parser.add_argument("--json", action="store_true", help="print machine-readable configs")
    return parser.parse_args()


def main() -> None:
    """Run the requested inspections."""
    args = parse_args()
    checkpoints = CheckpointConfig(root=args.weights_root)
    checkpoints.validate()

    generator_weights = checkpoints.generator_dir / "diffusion_pytorch_model.safetensors"
    inversion_weights = (
        checkpoints.inversion_dir / "unet_ema" / "diffusion_pytorch_model.safetensors"
    )

    if args.compare_generator:
        _rule("Generator integrity check")
        inspect_adapter(checkpoints.ip_adapter_path, generator_weights)
        return

    _rule("UNet configurations")
    generator_config = json.loads((checkpoints.generator_dir / "config.json").read_text())
    inversion_config = json.loads(
        (checkpoints.inversion_dir / "unet_ema" / "config.json").read_text()
    )
    if args.json:
        print(json.dumps({"generator": generator_config, "inversion": inversion_config}, indent=2))
    else:
        for key in sorted(set(generator_config) | set(inversion_config)):
            left = generator_config.get(key, "<absent>")
            right = inversion_config.get(key, "<absent>")
            marker = "" if left == right else "   <<< differs"
            print(f"  {key:26s} G={str(left)[:30]:32s} F={str(right)[:30]}{marker}")

    _rule("Inversion network vs generator initialisation")
    compare_unets(generator_weights, inversion_weights)

    _rule("Contents of ip_adapter.bin")
    inspect_adapter(checkpoints.ip_adapter_path, generator_weights)

    _rule("Parameter budget")
    report_budget(inversion_weights, generator_weights, checkpoints.ip_adapter_path)


if __name__ == "__main__":
    main()

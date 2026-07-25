"""CUDA sanity check.

Answers one narrow question: will this torch install actually run work on this GPU?

`torch.cuda.is_available()` is not that question. It returns True whenever a driver
and a device are visible, including on a wheel built without kernels for the device's
compute capability -- which then fails at the first real op with:

    CUDA error: no kernel image is available for execution on the device

That is the failure this project cares about, because it is silent until late: `uv
sync` succeeds, `import torch` works, `ruff` and `pytest` pass, and the problem
surfaces at the diffusion backend. So every stage below runs real work instead of
asking torch how it feels.

Usage:

    uv run python scripts/cuda_check.py

Exit code 0 means usable, 1 means not. Diagnostic only -- nothing in the package
imports this, and it is not part of the test suite (it needs a GPU).
"""

import sys

FAILURES: list[str] = []


def check(label: str, fn) -> None:
    """Run one probe, reporting the exception instead of aborting the report."""
    try:
        print(f"  {label}: {fn()}")
    except Exception as exc:
        print(f"  {label}: FAILED -- {type(exc).__name__}: {exc}")
        FAILURES.append(label)


def main() -> int:
    import torch

    print("build")
    print(f"  torch: {torch.__version__}")
    print(f"  built against CUDA: {torch.version.cuda}")
    print(f"  cuDNN: {torch.backends.cudnn.version()}")

    if not torch.cuda.is_available():
        print("\nCUDA NOT AVAILABLE -- torch sees no usable device.")
        print("On a CPU-only wheel, 'built against CUDA' above prints None.")
        return 1

    print("\ndevice")
    print(f"  name: {torch.cuda.get_device_name(0)}")
    major, minor = torch.cuda.get_device_capability(0)
    device_arch = f"sm_{major}{minor}"
    print(f"  compute capability: {major}.{minor}  ({device_arch})")
    print(f"  memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")

    # The actual "is my GPU too new for this wheel" test.
    arch_list = torch.cuda.get_arch_list()
    print("\nkernel coverage")
    print(f"  wheel compiled for: {' '.join(arch_list)}")
    if device_arch in arch_list:
        print(f"  {device_arch}: native kernels present  <- what you want")
    else:
        ptx = [a for a in arch_list if a.startswith("compute_")]
        if ptx:
            print(f"  {device_arch}: no native kernels; relies on PTX JIT from {ptx}")
            print("    Works, but pays first-call JIT latency and is not guaranteed for newer ops.")
        else:
            print(f"  {device_arch}: no native kernels and no PTX to JIT from.")
            print("    This is the 'no kernel image is available' failure -- wrong wheel.")
            FAILURES.append("kernel coverage")

    def matmul_matches_cpu() -> str:
        a, b = torch.randn(512, 512), torch.randn(512, 512)
        expected = a @ b
        got = (a.cuda() @ b.cuda()).cpu()
        max_diff = (expected - got).abs().max().item()
        assert torch.allclose(expected, got, atol=1e-3), f"max diff {max_diff}"
        return f"max diff {max_diff:.2e}"

    def half_precision() -> str:
        x = torch.ones(256, 256, device="cuda", dtype=torch.float16)
        out = x @ x
        # Accumulate in fp32: 256**3 = 16777216 overflows fp16's ~65504 ceiling, so
        # summing in half precision reports inf and reads like a GPU fault.
        total = out.float().sum().item()
        assert out[0, 0].item() == 256, f"element wrong: {out[0, 0].item()}"
        return f"element 256 ok, total {total:.0f} (expected 16777216)"

    def cudnn_conv() -> tuple[int, ...]:
        conv = torch.nn.Conv2d(3, 8, 3).cuda()
        return tuple(conv(torch.randn(1, 3, 64, 64, device="cuda")).shape)

    print("\nreal work")
    check("allocate + copy", lambda: tuple(torch.ones(1024, 1024, device="cuda").shape))
    check("fp32 matmul vs CPU", matmul_matches_cpu)
    check("fp16 matmul (tensor cores)", half_precision)
    check("cudnn conv2d", cudnn_conv)
    torch.cuda.synchronize()

    if FAILURES:
        print(f"\nFAILED: {', '.join(FAILURES)}")
        return 1
    print("\nOK -- CUDA is usable for real work on this device.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

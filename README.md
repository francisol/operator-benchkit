# operator-benchkit

An extensible HIP/CUDA operator microbenchmark tool for PyTorch, LightOp, AITER, or custom kernels.

## Installation

Python 3.9+ is required. On Hygon/ROCm/DTK systems, **install the environment-compatible PyTorch wheel first** using your platform's installation instructions; this project deliberately does not pin a `torch` version. `lightop` is needed only for `act-quant`.

```bash
python -m pip install -e .
opbench --help
# Alternative without a console script:
python -m benchkit --help
```

For development:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

## Usage

```bash
opbench topk --tokens 1 --tokens 16 --tokens 64 \
  --dims 1024 --dims 4096 --ks 1 --ks 8 \
  --dtype bf16 --mode graph --check --csv topk.csv

opbench act-quant --tokens 1 --tokens 16 --dims 1024 \
  --dtype bf16 --mode graph --check --csv act_quant.csv

# Adapter run(x, k, values, indices) must write to the supplied output tensors.
opbench topk --impl my_adapter:run --tokens 64 --dims 4096 --ks 8
```

Click `multiple=True` flags must be repeated for each value. See `opbench <operator> --help`.

### Measurement methodology

`eager` times repeated Python-dispatched calls using HIP/CUDA events. `graph` captures `graph_iters` calls and measures graph replays with events, dividing by `graph_iters * repeat` for per-call latency in microseconds. Warmup and optional correctness checks are excluded from timing. Outputs include median/min/max across rounds and *logical* GB/s (derived from declared tensor sizes; **not** measured HBM bandwidth). Graph mode requires capture-safe kernels and stable tensor addresses. Custom adapters may have different output conventions, so verify those contracts before benchmarking.

## Adding an operator

Add `src/benchkit/ops/rmsnorm.py` with a Click command named `cli`. The main CLI automatically discovers it. Nothing needs to be registered centrally.

```python
import click
from benchkit.core import Case, common_options, get_dtypes, run_cases


def make_case(tokens, dim, dtype):
    import torch
    x = torch.randn((tokens, dim), device='cuda', dtype=dtype)
    out = torch.empty_like(x)

    def fn():
        # Call your RMSNorm implementation with x and out here.
        raise NotImplementedError

    def check():
        # Compare to a correctness reference outside the timed region.
        raise NotImplementedError

    return Case(fn=fn, check=check,
                logical_bytes=x.numel() * x.element_size() * 2,
                info={'tokens': tokens, 'dim': dim})


@click.command('rmsnorm')
@click.option('--tokens', multiple=True, type=click.IntRange(min=1), default=(1, 16))
@click.option('--dims', multiple=True, type=click.IntRange(min=1), default=(1024,))
@common_options
def cli(tokens, dims, **common):
    def cases():
        for dtype in get_dtypes(common['dtype']):
            for t in tokens:
                for d in dims:
                    yield str(dtype), make_case(t, d, dtype)
    run_cases('rmsnorm', cases(), **common)
```

The example requires implementing `fn` and `check` before execution. Keep GPU-specific imports inside the operator functions so `--help` and CPU CI remain usable without GPU packages.

## Project structure

```text
src/benchkit/cli.py         auto-discovered CLI commands
src/benchkit/core.py        shared event timing, graph timing, verification, CSV
src/benchkit/ops/          one Click command per operator
tests/                     CPU-only discovery and timing tests
.github/workflows/ci.yml   GitHub Actions CI
pyproject.toml             package metadata and opbench console script
```

## GPU validation

CI runs packaging and CPU-compatible tests. Runtime GPU latency and graph-capture compatibility must be checked on the actual target device (e.g. Hygon BW1000 + target DTK/PyTorch/LightOp versions).

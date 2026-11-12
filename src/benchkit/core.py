"""Common benchmarking, validation hooks, CSV reporting, and CLI options."""
import csv
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import click


@dataclass
class Case:
    fn: Callable[[], None]
    check: Callable[[], None]
    logical_bytes: int
    info: dict = field(default_factory=dict)
    # Keep tensors alive for the full duration of measurement, especially Graph capture.


def common_options(fn):
    """Click options inherited by every operator command."""
    options = [
        click.option('--dtype', type=click.Choice(['bf16', 'fp16', 'fp32', 'both']), default='bf16', show_default=True),
        click.option('--mode', type=click.Choice(['graph', 'eager']), default='graph', show_default=True),
        click.option('--warmup', type=click.IntRange(min=0), default=100, show_default=True),
        click.option('--graph-iters', type=click.IntRange(min=1), default=100, show_default=True),
        click.option('--repeat', type=click.IntRange(min=1), default=20, show_default=True),
        click.option('--rounds', type=click.IntRange(min=1), default=10, show_default=True),
        click.option('--seed', type=int, default=1234, show_default=True),
        click.option('--check/--no-check', default=False, show_default=True),
        click.option('--csv', 'csv_path', type=click.Path(path_type=Path, dir_okay=False)),
    ]
    for option in reversed(options):
        fn = option(fn)
    return fn


def get_dtypes(dtype):
    import torch
    result = {'bf16': [torch.bfloat16], 'fp16': [torch.float16],
              'fp32': [torch.float32], 'both': [torch.float16, torch.bfloat16]}
    return result[dtype]


def measure(fn, *, mode, warmup, graph_iters, repeat, rounds):
    """Return (median_us, samples_us); event timing runs on PyTorch HIP/CUDA stream."""
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    if mode == 'graph':
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(graph_iters):
                fn()
        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize()
        run = graph.replay
        iterations = graph_iters * repeat
    else:
        run = fn
        iterations = repeat
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(rounds):
        start.record()
        for _ in range(repeat):
            run()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / iterations)
    return statistics.median(samples), samples


def run_cases(op, cases, *, dtype, mode, warmup, graph_iters, repeat,
              rounds, seed, check, csv_path):
    """cases: iterable of (dtype_name, Case). Consume lazily to limit VRAM use."""
    import torch
    if not torch.cuda.is_available():
        raise click.ClickException('No HIP/CUDA device detected by PyTorch')
    torch.manual_seed(seed)
    click.echo(f'GPU={torch.cuda.get_device_name(0)} PyTorch={torch.__version__} HIP={torch.version.hip}')
    click.echo(f'op={op} mode={mode} warmup={warmup} graph_iters={graph_iters if mode == "graph" else "-"} '
               f'repeat={repeat} rounds={rounds} check={check}')
    rows = []
    for dtype_name, case in cases:
        if check:
            case.check()
            torch.cuda.synchronize()
        median, samples = measure(case.fn, mode=mode, warmup=warmup,
                                  graph_iters=graph_iters, repeat=repeat, rounds=rounds)
        gbps = case.logical_bytes / (median * 1e-6) / 1e9 if median > 0 else float('nan')
        row = {'op': op, 'dtype': dtype_name, **case.info, 'mode': mode,
               'median_us': round(median, 6), 'min_us': round(min(samples), 6),
               'max_us': round(max(samples), 6), 'logical_GBs': round(gbps, 6)}
        rows.append(row)
        details = ' '.join(f'{key}={value}' for key, value in case.info.items())
        click.echo(f'{dtype_name:>12} {details} median={median:.3f}us '
                   f'min={min(samples):.3f}us max={max(samples):.3f}us logical={gbps:.2f} GB/s', color=False)
        # Release references and cached graph between cases.
        del case
    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        if rows:
            columns = list(dict.fromkeys(key for row in rows for key in row))
            with csv_path.open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
            click.echo(f'CSV saved: {csv_path}')
        else:
            click.echo('No cases to write', err=True)
    click.echo('Logical GB/s is an estimate based on tensor sizes, not measured HBM traffic.')
    return rows

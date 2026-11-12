"""Torch or user-provided Top-K implementation."""
import importlib
import importlib.util
from pathlib import Path
import click
from benchkit.core import Case, common_options, get_dtypes, run_cases


def load_adapter(spec):
    if ':' not in spec:
        raise click.BadParameter('expected module:function or /path/file.py:function', param_hint='--impl')
    module_name, function_name = spec.rsplit(':', 1)
    if module_name.endswith('.py'):
        path = Path(module_name)
        module_spec = importlib.util.spec_from_file_location('_bench_topk_adapter', path)
        if module_spec is None or module_spec.loader is None:
            raise click.ClickException(f'Cannot load adapter: {path}')
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)
    fn = getattr(module, function_name)
    if not callable(fn):
        raise click.ClickException(f'Adapter {spec} is not callable')
    return fn


def make_case(tokens, dim, k, dtype, adapter, impl):
    import torch
    x = torch.randn((tokens, dim), device='cuda', dtype=dtype).contiguous()
    values = torch.empty((tokens, k), device='cuda', dtype=dtype)
    indices = torch.empty((tokens, k), device='cuda', dtype=torch.long)
    if adapter is None:
        def fn():
            torch.topk(x, k, dim=-1, largest=True, sorted=True, out=(values, indices))
    else:
        def fn():
            adapter(x, k, values, indices)

    def check():
        fn()
        torch.cuda.synchronize()
        assert bool(((indices >= 0) & (indices < dim)).all().item()), 'invalid indices'
        torch.testing.assert_close(values, torch.gather(x, -1, indices), rtol=0, atol=0)
        ref = torch.topk(x, k, dim=-1, largest=True, sorted=True).values
        torch.testing.assert_close(torch.sort(values, dim=-1, descending=True).values, ref, rtol=0, atol=0)

    logical_bytes = (x.numel() * x.element_size() + values.numel() * values.element_size()
                     + indices.numel() * indices.element_size())
    return Case(fn, check, logical_bytes, {'tokens': tokens, 'dim': dim, 'k': k, 'implementation': impl})


def resolve_dims(dims, m_start, m_stop, m_step):
    """Resolve either explicitly listed M values or an inclusive M sweep."""
    if m_start is None and m_stop is None:
        return dims or (1024, 4096)
    if m_start is None or m_stop is None:
        raise click.UsageError('--m-start and --m-stop must be supplied together')
    if dims:
        raise click.UsageError('Use --dims or --m-start/--m-stop, not both')
    if m_start > m_stop:
        raise click.BadParameter('must be <= --m-stop', param_hint='--m-start')
    values = list(range(m_start, m_stop + 1, m_step))
    if values[-1] != m_stop:
        values.append(m_stop)
    return tuple(values)


@click.command('topk')
@click.option('--tokens', multiple=True, type=click.IntRange(min=1), default=(1, 16, 64), show_default=True)
@click.option('--dims', multiple=True, type=click.IntRange(min=1), default=(), help='Explicit M sizes; repeat for multiple values')
@click.option('--m-start', type=click.IntRange(min=1), default=None, help='First M for range sweep')
@click.option('--m-stop', type=click.IntRange(min=1), default=None, help='Inclusive last M for range sweep')
@click.option('--m-step', type=click.IntRange(min=1), default=1024, show_default=True, help='M increment in sweep')
@click.option('--ks', multiple=True, type=click.IntRange(min=1), default=(1, 8, 32), show_default=True)
@click.option('--impl', default='torch', show_default=True, help='torch or module:function or /path/file.py:function')
@common_options
def cli(tokens, dims, m_start, m_stop, m_step, ks, impl, **common):
    """Benchmark Top-K values and indices along the final dimension."""
    dims = resolve_dims(dims, m_start, m_stop, m_step)
    for dim in dims:
        for k in ks:
            if k > dim:
                raise click.BadParameter(f'k={k} must be <= dim={dim}', param_hint='--ks')
    adapter = None if impl == 'torch' else load_adapter(impl)
    dtypes = get_dtypes(common['dtype'])
    def cases():
        for dtype in dtypes:
            for tokens_value in tokens:
                for dim in dims:
                    for k in ks:
                        yield str(dtype), make_case(tokens_value, dim, k, dtype, adapter, impl)
    run_cases('topk', cases(), **common)

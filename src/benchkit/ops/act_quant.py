"""LightOp fused SiLU + Mul + INT8 quantization."""
import click
from benchkit.core import Case, common_options, get_dtypes, run_cases


def make_case(tokens, d, dtype, expert_ids_enabled):
    import torch
    # Lazy import: other benchmark commands do not require LightOp.
    from lightop.activation import fuse_silu_mul_quant
    x = torch.randn((tokens, 2 * d), device='cuda', dtype=dtype).contiguous()
    out = torch.empty((tokens, d), device='cuda', dtype=torch.int8)
    scales = torch.empty((tokens, 1), device='cuda', dtype=torch.float32)
    expert_ids = torch.zeros((tokens,), device='cuda', dtype=torch.int32) if expert_ids_enabled else None

    def fn():
        fuse_silu_mul_quant(x, output=out, scales=scales, expert_ids=expert_ids)

    def check():
        fn()
        torch.cuda.synchronize()
        assert out.shape == (tokens, d) and out.dtype == torch.int8
        assert scales.shape == (tokens, 1) and scales.dtype == torch.float32
        assert torch.isfinite(scales).all().item(), 'non-finite scales'
        assert (scales >= 0).all().item(), 'negative scales'

    logical_bytes = x.numel() * x.element_size() + out.numel() * out.element_size() + scales.numel() * scales.element_size()
    if expert_ids is not None:
        logical_bytes += expert_ids.numel() * expert_ids.element_size()
    return Case(fn, check, logical_bytes, {'tokens': tokens, 'dim': d, 'implementation': 'lightop', 'expert_ids': expert_ids_enabled})


@click.command('act-quant')
@click.option('--tokens', multiple=True, type=click.IntRange(min=1), default=(1, 16, 64, 256, 1024), show_default=True)
@click.option('--dims', multiple=True, type=click.IntRange(min=1), default=(1024,), show_default=True)
@click.option('--expert-ids/--no-expert-ids', default=True, show_default=True)
@common_options
def cli(tokens, dims, expert_ids, **common):
    """Benchmark LightOp fused SiLU+Mul+INT8 quantization."""
    if common['dtype'] == 'fp32':
        raise click.BadParameter('act-quant supports bf16/fp16 only', param_hint='--dtype')
    dtypes = get_dtypes(common['dtype'])
    def cases():
        for dtype in dtypes:
            for tokens_value in tokens:
                for dim in dims:
                    yield str(dtype), make_case(tokens_value, dim, dtype, expert_ids)
    run_cases('act_quant', cases(), **common)

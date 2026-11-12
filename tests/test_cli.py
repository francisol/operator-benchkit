from click.testing import CliRunner

from benchkit.cli import AutoOpsGroup, cli


def test_discovery():
    group = AutoOpsGroup()
    assert 'topk' in group.list_commands(None)
    assert 'act-quant' in group.list_commands(None)


def test_cli_help():
    result = CliRunner().invoke(cli, ['--help'])
    assert result.exit_code == 0, result.output
    assert 'topk' in result.output
    assert 'act-quant' in result.output


def test_subcommand_help_without_torch():
    for command in ('topk', 'act-quant'):
        result = CliRunner().invoke(cli, [command, '--help'])
        assert result.exit_code == 0, result.output
        assert '--mode' in result.output
        assert '--csv' in result.output


def test_unregistered_command():
    result = CliRunner().invoke(cli, ['does-not-exist'])
    assert result.exit_code != 0


def test_topk_invalid_k():
    result = CliRunner().invoke(cli, ['topk', '--dims', '8', '--ks', '16'])
    assert result.exit_code != 0
    assert 'must be <=' in result.output

def test_topk_range_resolution():
    from benchkit.ops.topk import resolve_dims
    assert resolve_dims((), 1024, 8196, 1024) == (
        1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 8196
    )
    assert resolve_dims((1024, 8196), None, None, 1024) == (1024, 8196)


def test_topk_range_validation():
    from click.testing import CliRunner
    result = CliRunner().invoke(cli, ['topk', '--m-start', '8196', '--m-stop', '1024'])
    assert result.exit_code != 0
    result = CliRunner().invoke(cli, ['topk', '--dims', '4096', '--m-start', '1024', '--m-stop', '8196'])
    assert result.exit_code != 0

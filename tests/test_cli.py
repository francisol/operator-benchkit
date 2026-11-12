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

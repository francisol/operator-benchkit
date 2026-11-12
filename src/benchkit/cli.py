"""CLI that automatically discovers operator benchmark modules."""
import importlib
import pkgutil

import click

from benchkit import ops


class AutoOpsGroup(click.Group):
    """Load each src/benchkit/ops/<name>.py exporting a `cli` command."""

    def _modules(self):
        return {
            module.name.replace('_', '-'): module.name
            for module in pkgutil.iter_modules(ops.__path__)
            if not module.name.startswith('_') and not module.ispkg
        }

    def list_commands(self, ctx):
        return sorted(self._modules())

    def get_command(self, ctx, cmd_name):
        module_name = self._modules().get(cmd_name)
        if module_name is None:
            return None
        try:
            module = importlib.import_module(f'benchkit.ops.{module_name}')
        except ImportError as exc:
            raise click.ClickException(f'Cannot load operator {cmd_name}: {exc}') from exc
        command = getattr(module, 'cli', None)
        if not isinstance(command, click.Command):
            raise click.ClickException(f'{module.__name__} must export a Click command named cli')
        return command


@click.group(cls=AutoOpsGroup, context_settings={'help_option_names': ['-h', '--help']})
@click.version_option(package_name='operator-benchkit')
def cli():
    """HIP/CUDA operator benchmarks with automatically discovered commands."""


if __name__ == '__main__':
    cli()

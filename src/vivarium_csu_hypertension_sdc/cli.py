from pathlib import Path

import click
from loguru import logger
from vivarium.framework.utilities import handle_exceptions

from vivarium_csu_hypertension_sdc.components import builder

@click.command()
@click.argument('location')
def build_htn_artifact(location):
    output_root = Path('/share/costeffectiveness/artifacts/vivarium_csu_hypertension_sdc/')
    main = handle_exceptions(builder.build_artifact, logger, with_debugger=True)
    main(output_root, location)

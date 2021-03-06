from collections import namedtuple
from pathlib import Path
import re
from typing import List, Optional, Iterable

import click
from jinja2 import Template
from loguru import logger
from vivarium.framework.utilities import handle_exceptions


PROJECT_NAME = 'vivarium_csu_hypertension_sdc'
BASE_DIR = Path(__file__).parent.resolve()
MODEL_SPEC_DIR = BASE_DIR / 'model_specifications'
ARTIFACT_DIR = BASE_DIR / 'artifacts'
Location = namedtuple('Location', ['proper', 'sanitized'])


@click.command()
@click.argument('location')
def build_htn_artifact(location):
    from vivarium_csu_hypertension_sdc.components import builder
    output_root = Path(f'/share/costeffectiveness/artifacts/{PROJECT_NAME}/')
    main = handle_exceptions(builder.build_artifact, logger, with_debugger=True)
    main(output_root, location)


@click.command()
@click.option('-l', '--locations-file',
              type=click.Path(dir_okay=False),
              help=('The file with the location parameters for the template. If no locations file is provided '
                    'and no single location is provided with the ``-s`` flag, this will default to the locations '
                    f'file located at {str(MODEL_SPEC_DIR / "locations.txt")}.'))
@click.option('-t', '--template',
              default=str(MODEL_SPEC_DIR / 'model_spec.in'),
              show_default=True,
              type=click.Path(exists=True, dir_okay=False),
              help='The model specification template file.')
@click.option('-s', '--single-location',
              default='',
              help='Specify a single location name.')
@click.option('-o', '--output-dir',
              default=str(MODEL_SPEC_DIR),
              show_default=True,
              type=click.Path(exists=True),
              help='Specify an output directory. Directory must exist.')
def make_specs(template: str, locations_file: str, single_location: str, output_dir: str) -> None:
    """Generate model specifications based on a template.

    The default template lives here:

    ``vivarium_csu_hypertension_sdc/src/vivarium_csu_hypertension_sdc/model_specification/model_spec.in``

    Supply the locations for which you want a model spec generated by filling
    in the empty 'locations.txt' file. A template for this file can be found at

    ``vivarium_csu_hypertension_sdc/src/vivarium_csu_hypertension_sdc/model_specification/locations.txt``

    with instructions for it's use.

    """
    template = Path(template)
    output_dir = Path(output_dir)
    locations = parse_locations(locations_file, single_location)

    with template.open() as infile:
        jinja_temp = Template(infile.read())

    logger.info(f'Writing model spec(s) to "{output_dir}"')

    for location in sanitize(*locations):
        filespec = output_dir / f'{location.sanitized}.yaml'
        with filespec.open('w+') as outfile:
            logger.info(f'   Writing {filespec.name}')
            outfile.write(jinja_temp.render(
                location_proper=location.proper,
                location_sanitized=location.sanitized,
                artifact_directory=str(ARTIFACT_DIR)))


def sanitize(*locations: str) -> Iterable[Location]:
    """Processes locations into tuples of proper and sanitized names.

    Sanitized location strings are all lower case, have spaces replaced
    by underscores, and have apostrophes replaced by dashes.

    Parameters
    ----------
    locations
        The locations to process formatted as proper location names.
        Proper location names should come from GBD location set 1, the
        location reporting hierarchy wherever possible. If using sub-national
        locations, they may be found in GBD location set 35, the model
        results location hierarchy.

    Yields
    -------
        Named tuples with both the proper and sanitized location names

    Examples
    --------
    >>> sanitize("Nigeria")
    Location(proper="Nigeria", sanitized="nigeria")

    >>> sanitize("Burkina Faso")
    Location(proper="Burkina Faso", sanitized="burkina_faso")

    >>> sanitize("Cote d'Ivoire")
    Location(proper="Cote d'Ivoire", sanitized="cote_d-ivoire")

    """
    # TODO: Check that they're in a call to get_location_id from
    #    vivarium_gbd_access.gbd.  Not doing this now because it's not
    #    specified as a proper dependency in the setup.py and would be a
    #    bit of a pain to do now.
    for location in locations:
        proper = location.strip()
        sanitized = re.sub("[- ,.&']", '_', proper).lower()
        yield Location(proper, sanitized)


def parse_locations(locations_file: Optional[str], single_location: Optional[str]) -> List[str]:
    """Parses location inputs into a list of location strings.

    If no arguments are provided, this will default to the ``locations.txt``
    file located in the repository model_specifications directory.

    Parameters
    ----------
    locations_file
        Path to a file containing a list of locations to generate model
        specifications for.
    single_location
        Optional single location to generate a model specification for.

    Returns
    -------
        A list of location strings.

    Raises
    ------
    ValueError
        If both ``locations_file`` and ``single_loc`` are provided or if
        a ``locations_file`` with no locations is provided.

    """
    if locations_file and single_location:
        raise ValueError('You provided both a locations file and a single location to make_specs.')

    if single_location:
        return [single_location]

    locations_file = Path(locations_file) if locations_file else MODEL_SPEC_DIR / 'locations.txt'
    with locations_file.open() as f:
        # Interpret each line that doesn't start with a '#' as a single location.
        locations = [l for l in f.readlines() if not l.startswith('#')]
    if not locations:
        raise ValueError(f'No locations provided in location file {str(locations_file)}.')

    return locations

import sys

import click
from loguru import logger

from vivarium_cluster_tools.psimulate.utilities import get_drmaa
from . import proportion_hypertensive


@click.command()
@click.argument('location')
def pcalculate_proportion_hypertensive(location):
    """Calculate 1000 draws of the proportion of the population that has a SBP
    above the hypertensive threshold (SBP of 140) in parallel and aggregate
    to a single hdf file saved in the central vivarium artifact store as
    ``proportion_hypertensive/location.hdf``
    LOCATION should be specified as all lower-case, with underscores replacing
    spaces (i.e., the same way the model artifacts are named),
    e.g., russian_federation
    """
    drmaa = get_drmaa()

    num_draws = 1000

    data_file = (proportion_hypertensive.HYPERTENSION_DATA_FOLDER / f'{location}.hdf')
    if data_file.exists():
        # I don't want to write over b/c of issue where writing to same hdf key makes the files huge
        logger.info(f'Existing data found for {location}. Removing and re-calculating.')
        data_file.unlink()

    output_path = proportion_hypertensive.HYPERTENSION_DATA_FOLDER / location
    output_path.mkdir(parents=True)

    proportion_hypertensive.prep_input_data(output_path, location)

    logger.info('Submitting jobs.')
    with drmaa.Session() as s:
        jt = s.createJobTemplate()
        jt.remoteCommand = sys.executable
        jt.nativeSpecification = '-l m_mem_free=1G,fthread=1,h_rt=00:05:00 -q all.q -P proj_cost_effect'
        jt.args = [proportion_hypertensive.__file__, location, 'draw']
        jt.jobName = f'{location}_prop_hypertensive_draw'

        draw_jids = s.runBulkJobs(jt, 1, num_draws, 1)
        draw_jid_base = draw_jids[0].split('.')[0]

        jt.nativeSpecification = f'-l m_mem_free=3G,fthread=1,h_rt=00:15:00 ' \
            f'-q all.q -P proj_cost_effect -hold_jid {draw_jid_base}'
        jt.args = [proportion_hypertensive.__file__, location, 'aggregate']
        jt.jobName = f'{location}_prop_hypertensive_aggregate'

        agg_jid = s.runJob(jt)

        logger.info(f'Draws for {location} have been submitted with jid {draw_jid_base}. '
                    f'They will be aggregated by jid {agg_jid}.')
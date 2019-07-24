from bdb import BdbQuit
import sys
from pathlib import Path

import click
import drmaa
from loguru import logger

from vivarium_gbd_access.gbd import ARTIFACT_FOLDER
from vivarium_inputs.data_artifact import utilities
from .utilities import patch_external_data, build_and_patch
from . import proportion_hypertensive


@click.command()
@click.argument('model_specification')
@click.option('--append', '-a', is_flag=True,
              help="Preserve existing artifact and append to it")
@click.option('--verbose', '-v', is_flag=True,
              help="Turn on debug mode for logging")
@click.option('--pdb', 'debugger', is_flag=True, help='Drop the debugger if an error occurs')
def build_hypertension_artifact(model_specification, append, verbose, debugger):
    """
    build_hypertension_artifact is a program for building data artifacts locally
    for the hypertension_mgt model.

    MODEL_SPECIFICATION should be the name of the model specification you wish
    to build an artifact for, e.g., bangladesh.yaml and should be available in
    the model_specifications folder of this repository.

    It requires access to the J drive and /ihme. If you are running this job
    from a qlogin on the cluster, you must specifically request J drive access
    when you qlogin by adding "-l archive=TRUE" to your command.

    Please have at least 20GB of memory on your qlogin."""
    model_specification_path = Path(__file__).parent.parent / 'model_specifications' / model_specification
    output_root = ARTIFACT_FOLDER / 'vivarium_dcpn_hypertension_mgt'

    utilities.setup_logging(output_root, verbose, None, model_specification_path, append)

    try:
        build_and_patch(model_specification_path, output_root, append)
    except (BdbQuit, KeyboardInterrupt):
        raise
    except Exception as e:
        logger.exception("Uncaught exception: %s", e)
        if debugger:
            import pdb
            import traceback
            traceback.print_exc()
            pdb.post_mortem()
        else:
            raise


@click.command()
def update_external_data_artifacts():
    """
    update_external_data_artifacts will update all data artifacts stored in the
    central costeffectiveness data artifact folder with external data from the
    external_data folder in this repository. Useful if external data has changed.
    """
    artifact_folder = ARTIFACT_FOLDER / 'vivarium_dcpn_hypertension_mgt'
    artifact_files = [f for f in artifact_folder.iterdir() if f.suffix == '.hdf']

    for f in artifact_files:
        patch_external_data(f)


@click.command()
@click.argument('location')
def pcalculate_proportion_hypertensive(location):
    """Calculate 1000 draws of the proportion of the population that has a SBP
    above the hypertensive threshold (SBP of 140) in parallel and aggregate
    to a single hdf file saved in the central vivarium artifact store as
    ``proportion_hypertensive/location.hdf``
    """
    num_draws = 1000

    output_path = Path(ARTIFACT_FOLDER / f'vivarium_dcpn_hypertension_mgt/proportion_hypertensive/{location}')
    output_path.mkdir(parents=True)

    with drmaa.Session() as s:
        jt = s.createJobTemplate()
        jt.remoteCommand = sys.executable
        jt.nativeSpecification = '-l m_mem_free=1G,fthread=1,h_rt=00:30:00 -q all.q -P proj_cost_effect_dcpn'
        jt.args = [proportion_hypertensive.__file__, location, 'draw']
        jt.jobName = f'{location}_prop_hypertensive_draw'

        draw_jids = s.runBulkJobs(jt, 1, num_draws, 1)
        draw_jid_base = draw_jids[0].split('.')

        jt.nativeSpecification = f'-l m_mem_free=10G,fthread=1,h_rt=01:30:00 ' \
            f'-q all.q -P proj_cost_effect_dcpn -hold_jid {draw_jid_base}'
        jt.args = [proportion_hypertensive.__file__, location, 'aggregate']
        jt.jobName = f'{location}_prop_hypertensive_aggregate'

        agg_jid = s.runJob(jt)

        logger.info(f'Draws for {location} have been submitted with jid {draw_jid_base}. '
                    f'They will be aggregated by jid {agg_jid}.')

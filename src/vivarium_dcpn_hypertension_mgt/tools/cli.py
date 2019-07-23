from bdb import BdbQuit
import logging
from pathlib import Path

import click

from vivarium_gbd_access.gbd import ARTIFACT_FOLDER
from vivarium_inputs.data_artifact import utilities
from .utilities import patch_external_data, build_and_patch


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
        logging.exception("Uncaught exception: %s", e)
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




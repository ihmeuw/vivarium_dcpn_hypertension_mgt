from bdb import BdbQuit
import logging
from pathlib import Path

import click

from vivarium_gbd_access.gbd import ARTIFACT_FOLDER
from vivarium_inputs.data_artifact import utilities
from vivarium_inputs.data_artifact.cli import main
from vivarium_public_health.dataset_manager import Artifact
from .utilities import patch_artifact, prep_external_data, get_external_data_files


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
        main(str(model_specification_path), output_root, None, append)
        artifact_path = output_root / model_specification.replace('yaml', 'hdf')
        patch_artifact(artifact_path)
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

    external_data_files = get_external_data_files()
    external_data_keys = [f'health_technology.hypertension_drugs.{f.stem}' for f in external_data_files]
    external = zip(external_data_keys, external_data_files)

    for f in artifact_files:
        art = Artifact(str(f))
        location = art.load('metadata.locations')[0]

        for k, data_file in external:
            data = prep_external_data(data_file, location)
            if k in art:
                art.replace(k, data)
            else:
                art.write(k, data)



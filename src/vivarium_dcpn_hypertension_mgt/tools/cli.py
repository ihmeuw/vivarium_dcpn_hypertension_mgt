from bdb import BdbQuit
import logging
from pathlib import Path
import yaml

import click
import pandas as pd

from vivarium_gbd_access.gbd import ARTIFACT_FOLDER
from vivarium_inputs.data_artifact import utilities
from vivarium_inputs.data_artifact.cli import main
from vivarium_public_health.dataset_manager import Artifact


@click.command()
@click.argument('model_specification',
                help='Name of the model specification you wish to build an artifact for, e.g., bangladesh.yaml '
                     'Should be available in the model_specifications folder of this repository.')
@click.option('--append', '-a', is_flag=True,
              help="Preserve existing artifact and append to it")
@click.option('--verbose', '-v', is_flag=True,
              help="Turn on debug mode for logging")
@click.option('--pdb', 'debugger', is_flag=True, help='Drop the debugger if an error occurs')
def build_hypertension_artifact(model_specification, append, verbose, debugger):
    """
    build_washout_artifact is a program for building data artifacts locally
    for the obesity_washout model.

    It will build an artifact for the ``vivarium_ihme_obesity_washout.yaml``
    model specification file stored in the repository.

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
        _patch_artifact(artifact_path, model_specification_path)
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

    external_data_files = _get_external_data_files()
    external_data_keys = [f'health_technology.hypertension_drugs.{f.stem}' for f in external_data_files]
    external = zip(external_data_keys, external_data_files)

    for f in artifact_files:
        art = Artifact(f)
        location =
        for k, f in external:
            if k in art:
                art.remove(k)
                df =



def _patch_artifact(artifact_path: Path, model_specification: Path):
    art = Artifact(str(artifact_path))
    location = yaml.safe_load(model_specification.read_text())['configuration']['input_data']['location']

    data_files = _get_external_data_files()
    for file in data_files:
        df = _prep_external_data(file, location)
        name = file.stem
        art.write(f'health_technology.hypertension_drugs.{name}', df)


def _prep_external_data(data_file, location):
    data = pd.read_csv(data_file)
    data.location = data.location.apply(lambda s: s.strip())  # some locs have trailing spaces so won't match
    data = data[data.location == location]
    if 'sex' in data and len(data.sex.unique() == 3):
        # we have both sex and age specific values - we are only using sex specific for now
        data = data[data.sex != 'Both']

    return data


def _get_external_data_files():
    external_data_path = Path(__file__).parent.parent / 'external_data'
    return [f for f in external_data_path.iterdir() if f.suffix == '.csv']

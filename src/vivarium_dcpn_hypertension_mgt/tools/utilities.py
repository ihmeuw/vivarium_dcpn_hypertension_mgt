from pathlib import Path
import pandas as pd

from vivarium_public_health.dataset_manager import Artifact


def patch_artifact(artifact_path: Path):
    art = Artifact(str(artifact_path))
    location = art.load('metadata.locations')[0]

    data_files = get_external_data_files()
    for file in data_files:
        data = prep_external_data(file, location)
        name = file.stem
        art.write(f'health_technology.hypertension_drugs.{name}', data)


def prep_external_data(data_file, location):
    data = pd.read_csv(data_file)
    if 'location' in data:
        data.location = data.location.apply(lambda s: s.strip())  # some locs have trailing spaces so won't match
        data = data[data.location == location]
    if 'sex' in data and len(data.sex.unique()) == 3:
        # we have both sex and age specific values - we are defaulting to using age specific for now
        data.sex = data.sex.apply(lambda s: s.strip())
        data = data[data.sex == 'Both']

    return data


def get_external_data_files():
    external_data_path = Path(__file__).parent.parent / 'external_data'
    return [f for f in external_data_path.iterdir() if f.suffix == '.csv']

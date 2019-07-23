from pathlib import Path
import numpy as np
import pandas as pd

from loguru import logger

from risk_distributions import Normal
from vivarium_inputs.data_artifact.cli import main as build_artifact
from vivarium_public_health.dataset_manager import Artifact

DRAW_COLUMNS = [f'draw_{i}' for i in range(1000)]

CI_WIDTH_MAP = {99: 2.58, 95: 1.96, 90: 1.65, 68: 1}

RANDOM_SEED = 123456

TRANSFORMATION_SPECIFICATION = {
    'baseline_treatment_coverage': {
        'measures': ['treated_among_hypertensive', 'control_among_treated'],
        'columns': ['location', 'sex', 'age_group_start', 'age_group_end',
                    'measure', 'hypertension_threshold'] + DRAW_COLUMNS,
    },
    'baseline_therapy_categories': {
        'measures': ['percentage_among_treated'],
        'columns': ['location', 'therapy_category', 'measure'] + DRAW_COLUMNS
    },
    'baseline_treatment_profiles': {
        'measures': ['percentage_among_therapy_category'],
        'columns': ['location', 'hypertension_drug_category', 'DU', 'BB',
                    'ACEI', 'ARB', 'CCB', 'other', 'measure'] + DRAW_COLUMNS
    },
    'drug_efficacy': {
        'measures': ['half_dose_efficacy_mean', 'standard_dose_efficacy_mean', 'double_dose_efficacy_mean'],
        'columns': ['medication', 'sd_mean', 'dosage'] + DRAW_COLUMNS
    }
}


def patch_artifact(artifact_path: Path):
    art = Artifact(str(artifact_path))
    location = art.load('metadata.locations')[0]
    logger.info(f'Beginning external data for {location}.')

    external_data_files = get_external_data_files()
    external_data_keys = [f'health_technology.hypertension_drugs.{f.stem}' for f in external_data_files]

    for k, data_file in zip(external_data_keys, external_data_files):
        data = prep_external_data(data_file, location)
        if data.empty:
            logger.warning(f'No data found for {k} in {location}. This may be '
                           f'because external data has not yet been prepped for {location}.')
        else:
            if k in art:
                art.replace(k, data)
            else:
                art.write(k, data)


def prep_external_data(data_file, location):
    data_file = Path(data_file)
    data = pd.read_csv(data_file)
    if 'location' in data:
        data.location = data.location.apply(lambda s: s.strip())  # some locs have trailing spaces so won't match
        data = data[data.location == location]
    if 'sex' in data and len(data.sex.unique()) == 3:
        # we have both sex and age specific values - we are defaulting to using age specific for now
        data.sex = data.sex.apply(lambda s: s.strip())
        data = data[data.sex == 'Both']

    data = transform_data(data_file.stem, data)

    if 'sex' in data and set(data.sex) == {'Both'}:  # duplicate for male, female
        male = data
        male.sex = 'Male'
        female = data.copy()
        female.sex = 'Female'
        data = pd.concat([male, female])

    return data


def transform_data(data_type, data):
    spec = TRANSFORMATION_SPECIFICATION[data_type]
    measure_data = []

    for m in spec['measures']:
        measure_data.append(create_draw_level_data(data, m, spec['columns']))

    return pd.concat(measure_data)


def create_draw_level_data(data, measure, columns_to_keep):
    data['measure'] = measure
    measure_columns = {c: c.replace(f'{measure}_', '') for c in data if measure in c}

    if 'efficacy' in measure:
        measure_columns[f'{"_".join(measure.split("_")[:3])}_sd_mean'] = 'sd_mean'
        measure_columns['name'] = 'medication'
        measure_columns['measure'] = 'dosage'
        data.measure = data.measure.apply(lambda s: s.split('_')[0])

    data = data.rename(columns=measure_columns)

    # FIXME: shouldn't default like this but this is just a temp measure until I get the actual levels from MW
    if 'uncertainty_level' not in data:
        data['uncertainty_level'] = 95

    no_ci_to_convert = data.uncertainty_level.isnull()

    to_draw = data[~no_ci_to_convert]

    ci_widths = to_draw.uncertainty_level.map(lambda l: CI_WIDTH_MAP.get(l, 0) * 2)

    data.loc[~no_ci_to_convert, 'sd'] = (to_draw['ub'] - to_draw['lb']) / ci_widths

    if measure == 'percentage_among_therapy_category':
        data = collapse_other_drug_profiles(data)

    draws = pd.DataFrame(data=np.transpose(np.tile(data['mean'].values, (1000, 1))),
                         columns=DRAW_COLUMNS, index=data.index)

    data = pd.concat([data, draws], axis=1)

    np.random.seed(RANDOM_SEED)
    d = np.random.random(1000)
    for row in data.loc[~no_ci_to_convert].iterrows():
        dist = Normal(mean=row[1]['mean'], sd=row[1]['sd'])
        draws = dist.ppf(d)
        data.loc[row[0], DRAW_COLUMNS] = draws

    return data.filter(columns_to_keep + DRAW_COLUMNS)


def collapse_other_drug_profiles(data):
    """Baseline drug profile data may include various other, non-guideline
    drug categories within a single therapy category, e.g., there may be 2
    different non-guideline mono therapy drug profiles. For our purposes, we
    want to collapse these to a single category, propagating any uncertainty.
    """
    guideline_profiles = data[data.other == 0]
    non_guideline_profiles = data[data.other == 1]

    prepped_profiles = [guideline_profiles]

    for category in non_guideline_profiles.hypertension_drug_category.unique():
        category_profiles = non_guideline_profiles[non_guideline_profiles.hypertension_drug_category == category]

        collapsed_profile = category_profiles.iloc[[0]]
        collapsed_profile['drug_class'] = 'other'
        collapsed_profile['mean'] = category_profiles['mean'].sum()
        collapsed_profile['sd'] = ((category_profiles.sd ** 2).sum()) ** 0.5

        prepped_profiles.append(collapsed_profile)

    return pd.concat(prepped_profiles)


def get_external_data_files():
    external_data_path = Path(__file__).parent.parent / 'external_data'
    return [f for f in external_data_path.iterdir() if f.suffix == '.csv']


def build_and_patch(model_spec, output_root, append):
    build_artifact(str(model_spec), output_root, None, append)
    artifact_path = output_root / model_spec.stem.replace('yaml', 'hdf')
    patch_artifact(artifact_path)

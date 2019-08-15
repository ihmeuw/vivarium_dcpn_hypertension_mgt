from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

from loguru import logger

from vivarium_inputs.data_artifact.cli import main as build_artifact
from vivarium_inputs.utilities import reshape
from vivarium_public_health.dataset_manager import Artifact
from .proportion_hypertensive import HYPERTENSION_DATA_FOLDER, HYPERTENSION_HDF_KEY


DRAW_COLUMNS = [f'draw_{i}' for i in range(1000)]
CI_WIDTH_MAP = {99: 2.58, 95: 1.96, 90: 1.65, 68: 1}
RANDOM_SEED = 123456

# used to map the measures that need draws and the columns to keep after creating draws for each external data source
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
        'columns': ['location', 'therapy_category', 'thiazide_type_diuretics',
                    'beta_blockers', 'ace_inhibitors', 'angiotensin_ii_blockers',
                    'calcium_channel_blockers', 'other', 'measure'] + DRAW_COLUMNS
    },
    'drug_efficacy': {
        'measures': ['half_dose_efficacy_mean', 'standard_dose_efficacy_mean', 'double_dose_efficacy_mean'],
        'columns': ['medication', 'sd_mean', 'dosage'] + DRAW_COLUMNS
    }
}

RANDOM_SEEDS_BY_MEASURE = {
    'treated_among_hypertensive': 12345,
    'control_among_treated': 23456,
    'percentage_among_treated': 34567,
    'percentage_among_therapy_category': 45678,
    'half_dose_efficacy_mean': 56789,  # use the same seed to correlate the draws for different dosages
    'standard_dose_efficacy_mean': 56789,
    'double_dose_efficacy_mean': 56789
}


def patch_external_data(art):
    location = art.load('metadata.locations')[0]
    logger.info(f'Beginning external data for {location}.')

    external_data_files = get_external_data_files()
    external_data_keys = [f'health_technology.hypertension_drugs.{f.stem}' for f in external_data_files]

    for k, data_file in zip(external_data_keys, external_data_files):
        data = prep_external_data(data_file, location)
        if data.empty and data.index.empty:
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

    def prep(df):
        # strip all string columns to prevent the pesky leading/trailing spaces that may have crept in
        df_obj = df.select_dtypes(['object'])
        df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())

        if 'location' in df:
            df = df[df.location == location]
        if 'sex' in df and len(df.sex.unique()) == 3:
            # we have both sex and age specific values - we are defaulting to using age specific for now
            df = df[df.sex == 'Both']

        df = transform_data(data_file.stem, df)

        if 'sex' in df and set(df.sex) == {'Both'}:  # duplicate for male, female
            male = df
            male.sex = 'Male'
            female = df.copy()
            female.sex = 'Female'
            df = pd.concat([male, female])
        return df

    if data_file.stem == 'ramp_transition_filters':  # we need to prep (e.g., normalize over sex) for each ramp
        data = data.groupby(by=['guideline', 'from_ramp']).apply(prep).reset_index(drop=True)
    else:
        data = prep(data)

    return reshape(data)


def transform_data(data_type, data):
    spec = TRANSFORMATION_SPECIFICATION.get(data_type, {'measures': [], 'columns': []})
    measure_data = [] if spec['measures'] else [data]

    for m in spec['measures']:
        np.random.seed(RANDOM_SEEDS_BY_MEASURE[m])
        d = np.random.random(1000)
        df = clean_data(data, m)
        measure_data.append(create_draw_level_data(df, m, spec['columns'], random_draws=d))

    return pd.concat(measure_data)


def clean_data(data, measure):
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

    return data


def create_draw_level_data(data, measure, columns_to_keep, random_draws):
    no_ci_to_convert = data.uncertainty_level.isnull()

    to_draw = data[~no_ci_to_convert]
    ci_widths = to_draw.uncertainty_level.map(lambda l: CI_WIDTH_MAP.get(l, 0) * 2)
    data.loc[~no_ci_to_convert, 'sd'] = (to_draw['ub'] - to_draw['lb']) / ci_widths

    if measure == 'percentage_among_therapy_category':
        data = collapse_other_drug_profiles(data)

    draws = pd.DataFrame(data=np.transpose(np.tile(data['mean'].values, (1000, 1))),
                         columns=DRAW_COLUMNS, index=data.index)

    data = pd.concat([data, draws], axis=1)

    for row in data.loc[~no_ci_to_convert].iterrows():
        dist = norm(loc=row[1]['mean'], scale=row[1]['sd'])
        draws = dist.ppf(random_draws)
        data.loc[row[0], DRAW_COLUMNS] = draws

    return data.filter(columns_to_keep)


def collapse_other_drug_profiles(data):
    """Baseline drug profile data may include various other, non-guideline
    drug categories within a single therapy category, e.g., there may be 2
    different non-guideline mono therapy drug profiles. For our purposes, we
    want to collapse these to a single category, propagating any uncertainty.
    """
    guideline_profiles = data[data.other == 0]
    non_guideline_profiles = data[data.other == 1]

    prepped_profiles = [guideline_profiles]

    for category in non_guideline_profiles.therapy_category.unique():
        category_profiles = non_guideline_profiles[non_guideline_profiles.therapy_category == category]

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
    logger.info('GBD artifact built successfully.')
    artifact_path = output_root / f'{model_spec.stem}.hdf'
    art = Artifact(str(artifact_path))
    patch_external_data(art)
    logger.info('External data patched.')
    patch_proportion_hypertensive(art)
    logger.info('Proportion hypertensive patched.')


def patch_proportion_hypertensive(art):
    location = art.load('metadata.locations')[0].replace(' ', '_').replace("'", "-").lower()
    data = pd.read_hdf(HYPERTENSION_DATA_FOLDER / f'{location}.hdf', HYPERTENSION_HDF_KEY)
    key = f'risk_factor.high_systolic_blood_pressure.{HYPERTENSION_HDF_KEY}'
    if key in art:
        art.replace(key, data)
    else:
        art.write(key, data)

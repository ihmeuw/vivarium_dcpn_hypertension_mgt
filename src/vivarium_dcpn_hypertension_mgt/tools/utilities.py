from pathlib import Path
import pandas as pd

from vivarium_public_health.dataset_manager import Artifact

CONFIDENCE_INTERVALS_TO_CONVERT = {
    'baseline_treatment_coverage': ['treated_among_hypertensive', 'control_among_hypertensive', 'control_among_treated'],
    'baseline_therapy_categories': ['percentage_among_treated'],
    'baseline_treatment_profiles': ['percentage_among_therapy_category']
}


def patch_artifact(artifact_path: Path):
    art = Artifact(str(artifact_path))
    location = art.load('metadata.locations')[0]

    data_files = get_external_data_files()
    for file in data_files:
        data = prep_external_data(file, location)
        name = file.stem
        art.write(f'health_technology.hypertension_drugs.{name}', data)


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

    if data_file.stem in CONFIDENCE_INTERVALS_TO_CONVERT:
        for k in CONFIDENCE_INTERVALS_TO_CONVERT[data_file.stem]:
            data = convert_confidence_interval(data, k)

    if data_file.stem == 'baseline_treatment_profiles':
        data = collapse_other_drug_profiles(data)
    return data


def convert_confidence_interval(data, key):
    data[f'{key}_sd'] = 0.0
    no_ci_to_convert = (data[f'{key}_uncertainty_level'].isnull())
    no_ci = data[no_ci_to_convert]

    ci = data[~no_ci_to_convert]
    ci_width_map = {99: 2.58, 95: 1.96, 90: 1.65, 68: 1}

    ci_widths = ci[f'{key}_uncertainty_level'].map(lambda l: ci_width_map[l] * 2)
    ci[f'{key}_sd'] = (ci[f'{key}_ub'] - ci[f'{key}_lb']) / ci_widths

    data = pd.concat([ci, no_ci])
    return data.drop(columns=[f'{key}_lb', f'{key}_ub', f'{key}_uncertainty_level'])


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
        collapsed_profile['percentage_among_therapy_category_mean'] = \
            category_profiles.percentage_among_therapy_category_mean.sum()
        collapsed_profile['percentage_among_therapy_category_sd'] = \
            (category_profiles.percentage_among_therapy_category_sd ** 2).sum()

        prepped_profiles.append(collapsed_profile)

    return pd.concat(prepped_profiles)


def get_external_data_files():
    external_data_path = Path(__file__).parent.parent / 'external_data'
    return [f for f in external_data_path.iterdir() if f.suffix == '.csv']

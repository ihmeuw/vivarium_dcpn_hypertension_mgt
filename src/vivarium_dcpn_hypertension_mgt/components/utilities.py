import itertools
from loguru import logger

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

import numpy as np
import pandas as pd

from .globals import HYPERTENSION_DRUGS


def convert_filter_transition_to_query_string(row):
    return f'{row.age_group_start} <= age and age < {row.age_group_end} and sex == {row.sex} and ' \
        f'{row.systolic_blood_pressure_start} <= systolic_blood_pressure ' \
        f'and systolic_blood_pressure < {row.systolic_blood_pressure_end} and ' \
        f'cvd_risk_cat == {row.cvd_risk_cat}'


def load_domain_filters(builder) -> pd.DataFrame:
    guideline = builder.configuration['hypertension_drugs']['guideline']

    if guideline == 'baseline':
        ramp_filter_transitions = build_baseline_ramp_filter_transitions()
    else:
        ramp_filter_transitions = builder.data.load('health_technology.hypertension_drugs.ramp_transition_filters')
        ramp_filter_transitions = ramp_filter_transitions[ramp_filter_transitions.guideline == guideline]

    ramp_filter_transitions['domain_filter'] = ramp_filter_transitions.apply(convert_filter_transition_to_query_string,
                                                                             axis=1)
    return ramp_filter_transitions.set_index(['from_ramp', 'to_ramp'])


def build_baseline_ramp_filter_transitions() -> pd.DataFrame:
    no_tx = build_full_domain_to_null_filter_transitions('no_treatment')
    initial = build_full_domain_to_null_filter_transitions('initial')
    return pd.concat([no_tx, initial])


def build_full_domain_to_null_filter_transitions(from_ramp) -> pd.DataFrame:
    # build full domain queries
    full_domain = {'sex': ['Male', 'Female'] * 2, 'age_group_start': 0, 'age_group_end': 125,
                   'systolic_blood_pressure_start': 60, 'systolic_blood_pressure_end': 300,
                   'cvd_risk_cat': [0] * 2 + [1] * 2}

    filter_transitions = pd.DataFrame(full_domain)
    filter_transitions['from_ramp'] = from_ramp
    filter_transitions['to_ramp'] = 'null_state'
    return filter_transitions


def load_efficacy_data(builder) -> pd.DataFrame:
    efficacy_data = builder.data.load('health_technology.hypertension_drugs.drug_efficacy')
    efficacy_data.dosage = efficacy_data.dosage.map({'half': 0.5, 'standard': 1.0, 'double': 2.0})

    zero_dosage = efficacy_data.loc[efficacy_data.dosage == 0.5].copy()
    zero_dosage.dosage = 0.0
    zero_dosage.append({'dosage': 'none', 'medication': 'other'}, ignore_index=True)
    zero_dosage.sd_mean = 0.0
    zero_dosage.value = 0.0

    other_efficacies = pd.Series(builder.configuration['hypertension_drugs']['other_drugs_efficacy'].to_dict())
    other_efficacies.name = 'value'
    other_efficacies.index.name = 'dosage'
    other_efficacies = other_efficacies.reset_index()
    other_efficacies['medication'] = 'other'
    other_efficacies['sd_mean'] = 0

    efficacy_data = pd.concat([zero_dosage, efficacy_data, other_efficacies])

    return efficacy_data.set_index(['dosage', 'medication'])


def load_treatment_profiles(builder) -> pd.DataFrame:
    columns = HYPERTENSION_DRUGS + ['ramp_position', 'ramp_name']

    initial_profiles = load_initial_profiles(builder)[columns]
    guideline_profiles = load_guideline_profiles(builder)[columns]
    no_treatment_profile = make_no_treatment_profile()

    return pd.concat([guideline_profiles, initial_profiles, no_treatment_profile])


def load_initial_profiles(builder) -> pd.DataFrame:
    profile_data = builder.data.load(f'health_technology.hypertension_drugs.baseline_treatment_profiles')

    # make a choice based on config for profiles marked for a choice between ace_inhibitors and angiotensin_ii_blockers
    choice = builder.configuration['hypertension_drugs']['ace_inhibitors_or_angiotensin_ii_blockers']
    other = 'ace_inhibitors' if choice == 'angiotensin_ii_blockers' else 'angiotensin_ii_blockers'
    profile_data.loc[profile_data[choice] == 'parameter', choice] = 1
    profile_data.loc[profile_data[other] == 'parameter', other] = 0

    profile_data.loc[profile_data.other == 1, 'other'] = profile_data.loc[profile_data.other == 1,
                                                                            'therapy_category']
    profile_data.loc[profile_data.other == 0, 'other'] = 'none'

    profile_data = profile_data.astype({choice: 'int', other: 'int', 'other': str})

    profile_data['ramp_name'] = 'initial'
    profile_data['ramp_position'] = pd.Series(range(len(profile_data)), index=profile_data.index) + 1  # ramp positions start from 1
    return profile_data


def load_guideline_profiles(builder) -> pd.DataFrame:
    profile_data = builder.data.load('health_technology.hypertension_drugs.guideline_ramp_profiles')
    guideline = builder.configuration['hypertension_drugs']['guideline']
    profile_data['other'] = '0'
    return profile_data[profile_data.guideline == guideline]


def make_no_treatment_profile() -> pd.DataFrame:
    profile_data = {d: 0 for d in HYPERTENSION_DRUGS}
    profile_data['ramp_name'] = 'no_treatment'
    profile_data['ramp_position'] = 1
    return pd.DataFrame(profile_data, index=[0])


def calculate_pop_efficacy(drug_dosages: dict, efficacy_data: pd.DataFrame) -> float:
    drug_dosages = pd.Series(drug_dosages)

    drug_dosages.name = 'dosage'
    drug_dosages.index.name = 'medication'
    drugs_idx = drug_dosages.reset_index().set_index(['dosage', 'medication']).index

    return efficacy_data.loc[drugs_idx].fillna(0).value.sum()


def get_closest_in_efficacy_in_ramp(current_efficacy: float, profiles: pd.Series, ramp: str):
    """Get up to two profiles in given ramp closest in efficacy to the current
    efficacy such that their efficacy is equal to or greater than current
    efficacy."""
    ramp_profiles = profiles.filter(like=ramp).sort_values()

    closest_idx = np.digitize([current_efficacy], ramp_profiles, right=True)[0]

    if closest_idx >= len(ramp_profiles):
        closest_profiles = pd.Series()
    else:
        closest_profiles = ramp_profiles[closest_idx: closest_idx+2]  # may be 1 or 2 profiles

    return closest_profiles


def get_position_profile_in_ramp(treatment_profiles: dict, ramp: str, highest_or_lowest: str = 'lowest'):
    ramp_keys = sorted([k for k in treatment_profiles if ramp in k], key=lambda s: int(s.split('_')[-1]))
    key = ramp_keys[0] if highest_or_lowest == 'lowest' else ramp_keys[-1]
    return treatment_profiles[key]


def get_state_domain_filters(domain_filters: pd.DataFrame, ramp: str, position: int,
                             ramp_profiles: pd.DataFrame, ramp_transitions: dict) -> pd.Series:
    """I enumerated the standard set of filter transitions but that breaks down
    for the last position in a ramp if the only ramp to transition to is the
    current ramp. In that case, just add a filter to the null state that covers
    the full domain."""
    if position == ramp_profiles.ramp_position.max():
        if ramp_transitions[ramp] == [ramp]:
            # can only transition w/i ramp and we've hit the end
            filter_transitions = build_full_domain_to_null_filter_transitions(ramp)
            filter_transitions['domain_filter'] = filter_transitions.apply(convert_filter_transition_to_query_string,
                                                                           axis=1)
            profile_domain_filters = filter_transitions.set_index(['from_ramp', 'to_ramp']).domain_filter
        else:  # doing this here because I don't want to go change format + re-validate input filter transitions rn
            profile_domain_filters = domain_filters.query("from_ramp == @ramp").reset_index()
            profile_domain_filters.loc[profile_domain_filters.to_ramp == ramp, 'to_ramp'] = 'null_state'
            profile_domain_filters = profile_domain_filters.set_index(['from_ramp', 'to_ramp']).domain_filter
    else:
        profile_domain_filters = domain_filters.query("from_ramp == @ramp").domain_filter

    return profile_domain_filters


def get_domain_filters_between_ramps(domain_filters: pd.Series, from_ramp: str, to_ramp: str):
    domain_filter_idx = pd.MultiIndex.from_tuples([(from_ramp, to_ramp)])
    domain_filters_between_ramps = domain_filters.loc[domain_filter_idx]
    return domain_filters_between_ramps


def draw_rectangles(ax, state_filters, facecolor='r', edgecolor='b', alpha=0.4):
    filters = []
    for r in state_filters.iterrows():
        xy = (r[1].age_start, r[1].systolic_blood_pressure_start)
        rect = Rectangle(xy, r[1].width, r[1].height)
        filters.append(rect)

    pc = PatchCollection(filters, facecolor=facecolor, alpha=alpha, edgecolor=edgecolor)

    ax.add_collection(pc)


def plot_profile_domain_filters(data, figure_name):
    i = 1

    fig = plt.figure()

    for sex, cvd_risk_cat in itertools.product(('Male', 'Female'), ('0', '1')):
        ax = fig.add_subplot(2, 2, i)
        if data is not None and not data.empty:
            draw_rectangles(ax, data[(data.sex == sex) & (data.cvd_risk_cat == cvd_risk_cat)])
            ax.set_ylim(0, 360)
            ax.set_xlim(-10, 135)

        container = Rectangle((0, 60), 125, 240, fill=False, edgecolor='black', lw=4)
        ax.add_patch(container)
        ax.title.set_text(f'({sex}, {cvd_risk_cat})')
        i += 1

    fig.suptitle(figure_name)
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    return fig
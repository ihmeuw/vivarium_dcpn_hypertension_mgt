import pandas as pd

from .globals import HYPERTENSION_DRUGS


def load_domain_filters(builder) -> pd.Series:
    guideline = builder.configuration['hypertension_drugs']['guideline']

    if guideline == 'baseline':
        ramp_filter_transitions = build_baseline_ramp_filter_transitions()
    else:
        ramp_filter_transitions = builder.data.load('health_technology.hypertension_drugs.ramp_transition_filters')
        ramp_filter_transitions = ramp_filter_transitions[ramp_filter_transitions.guideline == guideline]

    # convert to query strings
    def convert_to_query_string(row):
        return f'{row.age_group_start} <= age and age < {row.age_group_end} and sex == {row.sex} and ' \
            f'{row.systolic_blood_pressure_start} <= systolic_blood_pressure ' \
            f'and systolic_blood_pressure < {row.systolic_blood_pressure_end} and ' \
            f'cvd_risk_cat == {row.cvd_risk_cat}'

    ramp_filter_transitions['domain_filter'] = ramp_filter_transitions.apply(convert_to_query_string, axis=1)
    domain_filters = (ramp_filter_transitions.set_index(['from_ramp', 'to_ramp'])).domain_filter
    return domain_filters


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

    other_efficacies = pd.Series(builder.configuration['hypertension_drugs']['other_drugs_efficacy'])
    other_efficacies.name = 'value'
    other_efficacies.index.name = 'dosage'
    other_efficacies = other_efficacies.reset_index()
    other_efficacies['medication'] = 'other'
    other_efficacies['sd_mean'] = 0

    efficacy_data = pd.concat([efficacy_data, other_efficacies])

    return efficacy_data.set_index(['dosage', 'medication'])


def load_treatment_profiles(builder) -> pd.DataFrame:
    columns = HYPERTENSION_DRUGS + ['ramp_position']

    initial_profiles = load_initial_profiles(builder)[columns]
    guideline_profiles = load_guideline_profiles(builder)[columns]
    no_treatment_profile = make_no_treatment_profile()

    return pd.concat([guideline_profiles, initial_profiles, no_treatment_profile])


def load_initial_profiles(builder) -> pd.DataFrame:
    profile_data = builder.data.load(f'health_technology.hypertension_drugs.baseline_treatment_profiles')

    # make a choice based on config for profiles marked for a choice between ace_inhibitors and angiotensin_ii_blockers
    choice = builder.configuration['hypertension_drugs']['ace_inhibitors_or_angiotensin_ii_blockers']
    other = 'ace_inhibitors' if choice == 'angiotensin_ii_blockers' else 'ace_inhibitors'
    profile_data.loc[profile_data[choice] == 'parameter', choice] = 1
    profile_data.loc[profile_data[other] == 'parameter', other] = 0
    profile_data = profile_data.astype({choice: 'int', other: 'int'})
    profile_data['ramp_position'] = pd.Series(range(len(profile_data)), index=profile_data.index) + 1  # ramp positions start from 1
    return profile_data


def load_guideline_profiles(builder) -> pd.DataFrame:
    profile_data = builder.data.load('health_technology.hypertension_drugs.guideline_ramp_profiles')
    guideline = builder.configuration['hypertension_drugs']['guideline']

    return profile_data[profile_data.guideline == guideline]


def make_no_treatment_profile() -> pd.DataFrame:
    profile_data = {d: 0 for d in HYPERTENSION_DRUGS}
    profile_data['ramp_position'] = 1
    return pd.DataFrame(profile_data, index=[0])


def calculate_pop_efficacy(drug_dosages: dict, efficacy_data: pd.DataFrame) -> float:
    drug_dosages = pd.Series(drug_dosages)

    drug_dosages = drug_dosages[drug_dosages.index != 'other']
    drug_dosages.name = 'dosage'
    drug_dosages.index.name = 'medication'
    drugs_idx = drug_dosages.reset_index().set_index(['dosage', 'medication'])

    return efficacy_data.loc[drugs_idx].fillna(0).value.sum()


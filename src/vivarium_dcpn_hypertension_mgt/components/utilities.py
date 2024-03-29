from collections import namedtuple
import itertools

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

import numpy as np
import pandas as pd

from .globals import HYPERTENSION_DRUGS


def convert_filter_transition_to_query_string(row):
    return f'{row.age_group_start} <= age and age < {row.age_group_end} and sex == "{row.sex}" and ' \
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
    zero_dosage = zero_dosage.append({'dosage': 'none', 'medication': 'other'}, ignore_index=True)
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
    columns = HYPERTENSION_DRUGS + ['ramp_position', 'ramp_name', 'probability_given_treated']
    initial_profiles = load_initial_profiles(builder)[columns]
    guideline_profiles = load_guideline_profiles(builder)[columns]
    no_treatment_profile = make_no_treatment_profile()[columns]

    profiles = pd.concat([guideline_profiles, initial_profiles, no_treatment_profile])
    profiles['profile_name'] = profiles.apply(lambda r: r.ramp_name + '_' + str(r.ramp_position), axis=1)
    return profiles.reset_index(drop=True)


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

    therapy_cat_data = (builder.data.load('health_technology.hypertension_drugs.baseline_therapy_categories')
                        .filter(['therapy_category', 'value']))

    profile_data = profile_data.merge(therapy_cat_data, on='therapy_category')
    profile_data['probability_given_treated'] = profile_data.value_x / 100 * profile_data.value_y / 100
    profile_data['probability_given_treated'] /= profile_data['probability_given_treated'].sum()

    return profile_data


def load_guideline_profiles(builder) -> pd.DataFrame:
    profile_data = builder.data.load('health_technology.hypertension_drugs.guideline_ramp_profiles')
    guideline = builder.configuration['hypertension_drugs']['guideline']
    profile_data['other'] = 'none'
    profile_data['probability_given_treated'] = 0
    return profile_data[profile_data.guideline == guideline]


def make_no_treatment_profile() -> pd.DataFrame:
    profile_data = {d: (0.0 if d != 'other' else 'none') for d in HYPERTENSION_DRUGS}
    profile_data['ramp_name'] = 'no_treatment'
    profile_data['ramp_position'] = 1
    profile_data['probability_given_treated'] = 0
    return pd.DataFrame(profile_data, index=[0])


def load_coverage_data(builder) -> pd.DataFrame:
    coverage_data = (builder.data.load('health_technology.hypertension_drugs.baseline_treatment_coverage')
                     .pivot_table(index=['sex', 'age_group_start', 'age_group_end'],
                                  columns='measure', values='value').reset_index())
    coverage_data.treated_among_hypertensive /= 100  # convert from percent
    coverage_data.control_among_treated /= 100  # convert from percent
    return coverage_data


def probability_profile_given_sbp_level(sbp_level, proportion_high_sbp, coverage, profiles):
    hypertensive = proportion_high_sbp / (1 - coverage.control_among_treated * coverage.treated_among_hypertensive)

    if sbp_level == 'below_140':
        prob_treated = (coverage.control_among_treated * coverage.treated_among_hypertensive
                        * hypertensive / (1 - proportion_high_sbp))

    elif sbp_level == 'above_140':
        prob_treated = ((1 - coverage.control_among_treated) * coverage.treated_among_hypertensive
                        * hypertensive / proportion_high_sbp)

    else:
        raise ValueError(f'The only acceptable sbp levels are "below_140" or "above_140". '
                         f'You provided {sbp_level}.')

    profile_names = list(profiles['profile_name'])
    no_treatment_idx = profile_names.index('no_treatment_1')

    def get_profile_probabilities(p_treated):
        p_profile = p_treated * profiles.probability_given_treated.values
        p_profile[no_treatment_idx] = 1.0 - np.sum(p_profile)
        return p_profile

    prob_profiles = prob_treated.apply(get_profile_probabilities)

    return prob_profiles, profile_names


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
    if position == ramp_profiles.ramp_position.max() or not(ramp_transitions[ramp]):
        if ramp_transitions[ramp] == [ramp] or not ramp_transitions[ramp]:
            # can only transition w/i ramp and we've hit the end or we're in baseline and no transitions allowed
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


# duration type is one of 'constant', 'options', or 'range' and duration values
# is respectively a single value, a list of values or a tuple with two values:
# start of range and end of range both inclusive. All values in days.
FollowupDuration = namedtuple('FollowupDuration', 'duration_type duration_values')
ConditionalFollowup = namedtuple('ConditionalFollowup', 'age measured_sbp followup_duration')


def get_dict_for_guideline(guideline, dictionary_choice):
    """If a dictionary has a 'guideline' key, indicating multiple options
    based on guideline, return the dictionary with only the options for the
    given guideline."""
    thresholds = {
        'icu': 180,
        'guideline': {
            'baseline': {
                'immediate_tx': None,
                'controlled': 140,
            },
            'aha': {
                'immediate_tx': 160,
                'controlled': 130,
            },
            'china': {
                'immediate_tx': None,
                'controlled': [(pd.Interval(0, 80, closed='left'), 140),  # (age interval applies to, threshold)
                               (pd.Interval(80, 125, closed='left'), 150)],
            },
            'who': {
                'immediate_tx': 160,
                'controlled': 140,
            }
        }
    }

    followup_schedules = {
        # top-level keys are visit type during which followup (of type in sub-keys) is being scheduled
        'background': {
            'maintenance': FollowupDuration('constant', 28),
            'confirmatory': FollowupDuration('options', [2 * 7, 3 * 7, 4 * 7])  # 2, 3, or 4 weeks
        },
        'maintenance': {
            'maintenance': {
                'transitioned': FollowupDuration('constant', 28),
                'untransitioned': {
                    'guideline': {
                        'who': FollowupDuration('range', (3 * 28, 6 * 28)),  # 3-6 months
                        'aha': FollowupDuration('range', (3 * 28, 6 * 28)),  # 3-6 months
                        'china': FollowupDuration('constant', 3 * 28),  # 3 months
                        'baseline': FollowupDuration('range', (3 * 28, 6 * 28)),  # 3-6 months
                    }
                }
            }
        },
        'confirmatory': {
            'maintenance': FollowupDuration('constant', 28),
            'reassessment': {
                'guideline': {
                    'baseline': None,
                    'who': None,
                    'aha': [(ConditionalFollowup(age=pd.Interval(0, 125, closed='left'),
                                                 measured_sbp=pd.Interval(130, 180, closed='left'),
                                                 followup_duration=FollowupDuration('range', (3 * 28, 6 * 28))), # 3-6 mos
                             )],
                    'china': [ConditionalFollowup(age=pd.Interval(0, 80, closed='left'),
                                                  measured_sbp=pd.Interval(140, 180, closed='left'),
                                                  followup_duration=FollowupDuration('range', (1 * 28, 3 * 28))), # 1-3 mos
                              ConditionalFollowup(age=pd.Interval(80, 125, closed='left'),
                                                  measured_sbp=pd.Interval(60, 180, closed='left'),
                                                  followup_duration=FollowupDuration('constant', 3 * 28))],  # 3 mos
                }
            }
        },
        'reassessment': {
            'maintenance': FollowupDuration('constant', 28),
            'reassessment': {
                'guideline': {
                    'baseline': None,
                    'who': None,
                    'aha': [ConditionalFollowup(age=pd.Interval(0, 125, closed='left'),
                                                measured_sbp=pd.Interval(120, 180, closed='left'),
                                                followup_duration=FollowupDuration('range', (3 * 28, 6 * 28))), # 3-6 mos
                            ConditionalFollowup(age=pd.Interval(0, 125, closed='left'),
                                                measured_sbp=pd.Interval(60, 120, closed='left'),
                                                followup_duration=FollowupDuration('constant', 365.25))],  # 1 yr
                    'china': [ConditionalFollowup(age=pd.Interval(0, 125, closed='left'),
                                                  measured_sbp=pd.Interval(130, 180, closed='left'),
                                                  followup_duration=FollowupDuration('range', (1 * 28, 3 * 28))), # 1-3 mos
                              ConditionalFollowup(age=pd.Interval(0, 125, closed='left'),
                                                  measured_sbp=pd.Interval(60, 130, closed='left'),
                                                  followup_duration=FollowupDuration('constant', 365.25))],  # 1 yr
                }
            }
        },
        'intensive_care_unit': {
            'maintenance': FollowupDuration('constant', 28),
        }
    }

    def collapse_dict(to_collapse):
        collapsed = dict()
        for k, v in to_collapse.items():
            if k == 'guideline':
                # won't have guideline nested w/i guideline so can just
                # collapse once when we hit guideline and not recurse
                g = v[guideline]
                if isinstance(g, dict):  # bring up above guideline
                    for sub_k, sub_v in g.items():
                        collapsed[sub_k] = sub_v
                else:
                    collapsed = g
            elif isinstance(v, dict):
                collapsed[k] = collapse_dict(v)
            else:
                collapsed[k] = v
        return collapsed

    if dictionary_choice == 'thresholds':
        return collapse_dict(thresholds)
    elif dictionary_choice == 'followup_schedules':
        return collapse_dict(followup_schedules)
    else:
        raise ValueError(f'The only acceptable dictionary choices are "thresholds" or "followup_schedules". You'
                         f'provided {dictionary_choice}.')


def get_durations_in_range(randomness, low: int, high: int, index: pd.Index, randomness_key=None):
    """Get pd.Timedelta durations between low and high days, both inclusive for
    given index using giving randomness."""
    durations = pd.Series([])
    if not index.empty:
        to_time_delta = np.vectorize(lambda d: pd.Timedelta(days=d))
        np.random.seed(randomness.get_seed(randomness_key))
        durations = pd.Series(to_time_delta(np.random.random_integers(low=low, high=high, size=len(index))), index=index)
    return durations



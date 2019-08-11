import pandas as pd

from typing import Dict, List

from vivarium.framework.state_machine import State, Transition, Machine
from . import utilities
from .globals import HYPERTENSION_DRUGS


class TreatmentProfile(State):

    def __init__(self, ramp: str, position: int, drug_dosages: Dict[str, float], domain_filters: List[str]):
        self.ramp = ramp
        self.position = position
        self.drug_dosages = drug_dosages
        self._domain_filters = {f: [] for f in domain_filters}

        super().__init__(f'{self.ramp}_{self.position}')  # state_id = ramp_position e.g., elderly_3

    @property
    def name(self):
        return f'treatment_profile(ramp={self.ramp}, position={self.position})'

    def add_transition(self, output, probability_func=lambda index: pd.Series(1, index=index), domain_filter: str = ""):
        probability = probability_func([0])[0]  # assuming probability func will always return same probability
        if domain_filter not in self._domain_filters:
            raise ValueError(f'The given domain filter {domain_filter} is invalid for this state.')
        elif sum(self._domain_filters[domain_filter]) + probability > 1:
            raise ValueError(f'The given domain filter cannot be added with probability {probability} because it'
                             f'would push the summed probabilities for this domain filter over 1.')

        self._domain_filters[domain_filter].append(probability)

        t = FilterTransition(self, output, probability_func=probability_func, domain_filter=domain_filter)
        self.transition_set.append(t)

        return t

    def is_valid(self):
        for domain_filter, probabilities in self._domain_filters.items():
            if sum(probabilities) != 1:
                return False
            return True


class FilterTransition(Transition):

    def __init__(self, input_profile, output_profile, probability_func=lambda index: pd.Series(1, index=index),
                 domain_filter: str = ""):

        self.domain_filter = domain_filter

        super().__init__(input_profile, output_profile, probability_func)

    @property
    def name(self):
        # because a domain_filter describes a continuous region there may be
        # multiple transitions between two states, each with a different domain filter
        return super().name + "_" + self.domain_filter

    def setup(self, builder):
        self.population_view = builder.population.get_view(['age', 'sex'])
        self.blood_pressure = builder.value.get_value('high_systolic_blood_pressure.exposure')
        self.cvd_risk_cat = builder.value.get_value('cvd_risk_category')

    def probability(self, index: pd.Index):
        """Apply filter to index to determine who's eligible for this
        transition. Everyone else gets probability 0."""
        p = pd.Series(0, index=index)

        characteristics = self.population_view.get(index)
        characteristics['systolic_blood_pressure'] = self.blood_pressure(index)
        characteristics['cvd_risk_cat'] = self.cvd_risk_cat(index)

        in_domain_index = characteristics.query(self.domain_filter).index

        p.loc[in_domain_index] = super().probability(in_domain_index)
        return p


class NullStateError(Exception):
    """Exception raised when simulants are transitioned into the null state."""
    pass


class NullTreatmentProfile(TreatmentProfile):
    """Marker state to indicate something has gone wrong. Simulants should never
    enter this state."""

    def __init__(self):
        super().__init__(ramp="null_state", position=0, drug_dosages={}, domain_filters=[])

    @property
    def name(self):
        return 'null_treatment_profile'

    def add_transition(self, output, probability_func=lambda index: pd.Series(1, index=index), domain_filter: str = ""):
        raise NotImplementedError('Transitions cannot be added to the null state.')

    def transition_effect(self, index, event_time, population_view):
        raise NullStateError(f'{len(index)} simulants are attempting to transition '
                             f'into the null treatment profile state.')

    def is_valid(self):
        """The null state is valid so long as there are no transitions from it."""
        return not len(self.transition_set)


class CVDRiskAttribute:

    @property
    def name(self):
        return 'cvd_risk_attribute'

    def setup(self, builder):

        self.bmi = builder.value.get_value('high_body_mass_index_in_adults.exposure')
        self.fpg = builder.value.get_value('high_fasting_plasma_glucose_continuous.exposure')

        self.population_view = builder.population.get_view(['ischemic_heart_disease_event_time'])

        self.clock = builder.time.clock()

        builder.value.register_value_producer('cvd_risk_category', source=self.get_cvd_risk_category)

    def get_cvd_risk_category(self, index):
        """CVD risk category is 0 if bmi and fpg are not within risk ranges and
        there has not been an ihd event in the last year; 1 otherwise.

        Risk Ranges:
            bmi >= 28 kg/m2
            fpg: 6.1 - 6.9 mmol/L
            time since last ihd event: <= 1 yr
        """

        cvd_risk = pd.Series(0, index=index)

        bmi = self.bmi(index)
        fpg = self.fpg(index)
        time_since_ihd = self.clock() - self.population_view.get(index).ischemic_heart_disease_event_time

        cvd_risk.loc[(28 <= bmi) | ((6.1 <= fpg) & (fpg <= 6.9)) | (time_since_ihd <= pd.Timedelta(days=365.25))] = 1

        return cvd_risk


class TreatmentProfileModel(Machine):

    configuration_defaults = {
        'hypertension_drugs': {
            'ace_inhibitors_or_angiotensin_ii_blockers': 'ace_inhibitors',
            'other_drugs_efficacy': {
                'mono': 6,
                'dual': 6,
                'triple': 6,
                'quad': 6,
            },
            'guideline': 'baseline'  # one of: ["baseline", "china", "aha", "who"]
        }
    }

    ramps = ['elderly', 'mono_starter', 'combo_starter', 'initial', 'no_treatment']
    ramp_transitions = {'elderly': ['elderly'],
                        'mono_starter': ['mono_starter', 'elderly'],
                        'combo_starter': ['combo_starter', 'elderly'],
                        'initial': ['mono_starter', 'combo_starter', 'elderly'],
                        'no_treatment': ['mono_starter', 'combo_starter', 'elderly']}

    def __init__(self):
        super().__init__('treatment_profile', states=[])

    def setup(self, builder):
        """Build TreatmentProfiles in specific order so that transitions will
        be only to already built TreatmentProfiles:
        0. null state profile (used to ensure only valid simulants are transitioned)
        1. elderly ramp profiles (if exists)
        2. mono_starter ramp profiles
        3. combo_starter ramp profiles (if exists)
        4. initial profiles
        5. no treatment profile

        Within each ramp, build profiles in reverse order (e.g., build position
        3 then 2 then 1 etc.)
        """
        self.treatment_profiles = {}

        profiles = utilities.load_treatment_profiles(builder)
        domain_filters = utilities.load_domain_filters(builder)
        efficacy_data = utilities.load_efficacy_data(builder)

        self.ramps = [r for r in TreatmentProfileModel.ramps if r in set(profiles.ramp_name)]
        self.ramp_transitions = {k: [r for r in v if r in self.ramps]
                                 for k, v in TreatmentProfileModel.ramp_transitions.items()}

        # 0. add null state profile
        self.treatment_profiles['null_state'] = NullTreatmentProfile()

        # used to find next profile when changing btw ramps
        profile_efficacies = pd.Series()

        # 1-5. add ramp profiles in order
        for ramp in self.ramps:
            ramp_profiles = profiles[profiles.ramp_name == ramp].sort_values(by='ramp_position', ascending=False)

            for profile in ramp_profiles.iterrows():
                profile_domain_filters = utilities.get_state_domain_filters(domain_filters, ramp,
                                                                            profile[1].ramp_position, ramp_profiles,
                                                                            self.ramp_transitions)
                tx_profile = TreatmentProfile(ramp, profile[1].ramp_position, profile[1][HYPERTENSION_DRUGS],
                                              list(profile_domain_filters))

                # record the current efficacy to be used in finding next states for other profiles
                efficacy = utilities.calculate_pop_efficacy(tx_profile.drug_dosages, efficacy_data)
                profile_efficacies = profile_efficacies.append(pd.Series(efficacy, index=[tx_profile.state_id]))

                # add transitions to next treatment profile states
                for next_profile, probability in get_next_states(tx_profile, self.ramp_transitions[ramp].copy(),
                                                                 self.treatment_profiles, profile_efficacies):
                    domain_filter_idx = pd.MultiIndex.from_tuples([(ramp, next_profile.ramp)])
                    domain_filter = profile_domain_filters.loc[domain_filter_idx][0]
                    tx_profile.add_transition(next_profile,
                                              probability_func=lambda index: pd.Series(probability, index=index),
                                              domain_filter=domain_filter)

                # add transitions to null state
                for domain_filter in profile_domain_filters.filter(like='null_state'):
                    tx_profile.add_transition(self.treatment_profiles['null_state'], domain_filter=domain_filter)

                self.treatment_profiles[tx_profile.state_id] = tx_profile

        self.add_states(self.treatment_profiles.values())
        super().setup(builder)

    def validate(self):
        for state in self.states:
            assert state.is_valid(), f'State {state.state_id} is invalid.'


def get_next_states(current_profile: TreatmentProfile, next_ramps: List[str],
                    tx_profiles: Dict[str, TreatmentProfile], profile_efficacies: pd.Series):
    next_profiles = []  # list of tuples of form (next_profile, probability)
    current_efficacy = profile_efficacies.loc[current_profile.state_id]

    if 'mono_starter' in next_ramps and 'combo_starter' in next_ramps and current_profile.ramp == 'initial':
        mono_next = utilities.get_closest_in_efficacy_in_ramp(current_efficacy, profile_efficacies, 'mono_starter')
        combo_next = utilities.get_closest_in_efficacy_in_ramp(current_efficacy, profile_efficacies, 'combo_starter')
        both_next = mono_next.append(combo_next)

        # we want the two closest in efficacy but if there are ties, add both and split the probability
        for unique_efficacy in [e for e in both_next.groupy(both_next)][:2]:
            next_profiles.append([(tx_profiles[n], 1/len(unique_efficacy)) for n in unique_efficacy.index])

        next_ramps.remove('mono_starter')
        next_ramps.remove('combo_starter')

    if current_profile.ramp in next_ramps:
        if f'{current_profile.ramp}_{current_profile.position + 1}' in tx_profiles:  # we aren't at the last position
            next_in_ramp = tx_profiles[f'{current_profile.ramp}_{current_profile.position + 1}']
            next_profiles.append((next_in_ramp, 1.0))
        next_ramps.remove(current_profile.ramp)

    for ramp in next_ramps:
        if current_profile.ramp == 'no_treatment':
            next_in_ramp = tx_profiles[ramp]
            next_profiles.append((next_in_ramp, 1.0))
        else:
            next_in_ramp = utilities.get_closest_in_efficacy_in_ramp(current_efficacy, profile_efficacies, ramp)
            next_profiles.extend([(tx_profiles[n], 1 / len(next_in_ramp)) for n in next_in_ramp])

    return next_profiles



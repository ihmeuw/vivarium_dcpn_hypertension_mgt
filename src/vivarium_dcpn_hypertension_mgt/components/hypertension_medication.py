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

    def __eq__(self, other):
        """Used to determine identical drug profiles (possibly across ramps)
        in determining possible next profiles for initial non-guideline
        profiles."""
        return self.drug_dosages == other.drug_dosages

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

    def setup(self, builder):
        self.population_view = builder.population.get_view(requires_columns=['age', 'sex'])
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
        super().__init__(ramp="", position=0, drug_dosages={}, domain_filters=[])

    @property
    def name(self):
        return 'null_treatment_profile'

    def add_transition(self, output, probability_func=lambda index: pd.Series(1, index=index), domain_filter: str = ""):
        raise NotImplementedError('Transitions cannot be added to the null state.')

    def transition_effect(self, index, event_time, population_view):
        raise NullStateError(f'{len(index)} simulants are attempting to transition '
                             f'into the null treatment profile state.')


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


RAMPS = ['elderly', 'mono_starter', 'combo_starter', 'initial', 'no_treatment']
RAMP_TRANSITIONS = {'elderly': ['elderly'],
                    'mono_starter': ['mono_starter', 'elderly'],
                    'combo_starter': ['combo_starter', 'elderly'],
                    'initial': ['mono_starter', 'combo_starter', 'elderly'],
                    'no_treatment': ['mono_starter', 'combo_starter', 'elderly']}


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

        ramps_to_build = [r for r in RAMPS if r in set(profiles.ramp_name)]

        # 0. add null state profile
        self.treatment_profiles['null_state'] = NullTreatmentProfile()

        # used to find next profile when changing btw ramps
        profile_efficacies = pd.Series()

        # 1-5. add ramp profiles in order
        for ramp in ramps_to_build:
            ramp_profiles = profiles[profiles.ramp_name == ramp].sort_values(by='ramp_position', ascending=False)

            profile_domain_filters = domain_filters.query("from_ramp == @ramp")

            for profile in ramp_profiles.iterrows():
                tx_profile = TreatmentProfile(ramp, profile[1].ramp_position, profile[1][HYPERTENSION_DRUGS],
                                              list(profile_domain_filters.domain_filter))

                efficacy = utilities.calculate_pop_efficacy(tx_profile.drug_dosages, efficacy_data)
                profile_efficacies = profile_efficacies.append(pd.Series(efficacy, index=[tx_profile.state_id]))

                self.treatment_profiles[tx_profile.state_id] = tx_profile

        super().setup(builder)



def get_next_states(tx_profile: TreatmentProfile, domain_filters: pd.Series):

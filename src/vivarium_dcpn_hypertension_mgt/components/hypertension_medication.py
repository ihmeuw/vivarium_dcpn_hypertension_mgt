import pandas as pd

from typing import Dict, List

from vivarium.framework.state_machine import State, Transition, Machine


class TreatmentProfile(State):

    def __init__(self, ramp: str, position: int, drug_dosages: Dict[str, float], domain_filters: List[str]):
        self.ramp = ramp
        self.position = position
        self.drug_dosages = drug_dosages
        self._domain_filters = {f: [] for f in domain_filters}

        super().__init__(f'{self.ramp}_{self.position}')

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
            bmi >= 28 km/m2
            fpg: 6.1 - 6.9 mmol/L
            time since last ihd event: <= 1 yr
        """

        cvd_risk = pd.Series(0, index=index)

        bmi = self.bmi(index)
        fpg = self.fpg(index)
        time_since_ihd = self.clock() - self.population_view.get(index).ischemic_heart_disease_event_time

        cvd_risk.loc[(bmi >= 28) | ((fpg >= 6.1) & (fpg <= 6.9)) | (time_since_ihd <= pd.Timedelta(days=365.25))] = 1

        return cvd_risk


HYPERTENSION_DRUGS = ['thiazide_type_diuretics', 'beta_blockers', 'ace_inhibitors',
                      'angiotensin_ii_blockers', 'calcium_channel_blockers', 'other']

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

def load_domain_filters(builder):
    guideline = builder.configuration['hypertension_drugs']['guideline']

    if guideline == 'baseline':
        # build full domain queries
        full_domain = {'sex': ['Male', 'Female']*2, 'age_group_start': 0, 'age_group_end': 125,
                       'systolic_blood_pressure_start': 60, 'systolic_blood_pressure_end': 300,
                       'cvd_risk_cat': [0]*2 + [1]*2}

        no_tx = pd.DataFrame(full_domain)
        no_tx['from_ramp'] = 'no_treatment'
        no_txt['to_ramp'] = 'null'
    else:
        # load from data

def load_efficacy_data(builder):
    efficacy_data = builder.data.load('health_technology.hypertension_drugs.drug_efficacy')
    efficacy_data.dosage = efficacy_data.dosage.map({'half': 0.5, 'standard': 1.0, 'double': 2.0})
    return efficacy_data.set_index(['dosage', 'medication'])


def load_treatment_profiles(builder):
    columns = HYPERTENSION_DRUGS + ['ramp_position']

    initial_profiles = load_initial_profiles(builder)[columns]
    guideline_profiles = load_guideline_profiles(builder)[columns]
    no_treatment_profile = make_no_treatment_profile()

    return pd.concat([guideline_profiles, initial_profiles, no_treatment_profile])


def load_initial_profiles(builder):
    profile_data = builder.data.load(f'health_technology.hypertension_drugs.baseline_treatment_profiles')
    profile_data.value /= 100  # convert from percent

    # make a choice based on config for profiles marked for a choice between ace_inhibitors and angiotensin_ii_blockers
    choice = builder.configuration['hypertension_drugs']['ace_inhibitors_or_angiotensin_ii_blockers']
    other = 'ace_inhibitors' if choice == 'angiotensin_ii_blockers' else 'ace_inhibitors'
    profile_data.loc[profile_data[choice] == 'parameter', choice] = 1
    profile_data.loc[profile_data[other] == 'parameter', other] = 0
    profile_data = profile_data.astype({choice: 'int', other: 'int'})
    profile_data['ramp_position'] = pd.Series(range(len(profile_data)), index=profile_data.index) + 1
    return profile_data


def load_guideline_profiles(builder):
    profile_data = builder.data.load('health_technology.hypertension_drugs.guideline_ramp_profiles')
    guideline = builder.configuration['hypertension_drugs']['guideline']

    return profile_data[profile_data.guideline == guideline]


def make_no_treatment_profile():
    profile_data = {d: 0 for d in HYPERTENSION_DRUGS}
    profile_data['ramp_position'] = 1
    return pd.DataFrame(profile_data, index=[0])
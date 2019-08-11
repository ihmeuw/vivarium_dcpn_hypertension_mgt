import pandas as pd

from typing import Dict, List

from vivarium.framework.state_machine import State, Transition


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







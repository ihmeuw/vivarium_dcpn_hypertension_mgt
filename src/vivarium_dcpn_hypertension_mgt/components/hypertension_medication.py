import pandas as pd

from typing import Dict

from vivarium.framework.state_machine import State, Transition


class TreatmentProfile(State):

    def __init__(self, ramp: str, position: int, drug_dosages: Dict[str, float]):
        self.ramp = ramp
        self.position = position
        self.drug_dosages = drug_dosages

        super().__init__(f'{self.ramp}_{self.position}')

    def add_transition(self, output, probability_func=lambda index: pd.Series(1, index=index), filters: dict = None):
        t = FilterTransition(self, output, probability_func=probability_func, filters=filters)
        self.transition_set.append(t)
        return t

    def __eq__(self, other):
        """Used to determine identical drug profiles (possibly across ramps)
        in determining possible next profiles for initial non-guideline
        profiles."""
        return self.drug_dosages == other.drug_dosages


class FilterTransition(Transition):

    def __init__(self, input_profile, output_profile, probability_func=lambda index: pd.Series(1, index=index),
                 filters: dict = None):

        self.filters = filters if filters else dict()

        acceptable_filters = {'age', 'systolic_blood_pressure', 'sex'}
        extra_filters = set(filters.keys()).difference(acceptable_filters)
        if extra_filters:
            raise ValueError(f'The only acceptable filter terms are based on "age", "sex", '
                             f'and "systolic_blood_pressure". You included {extra_filters}.')

        # TODO: check that filters is in form: <filter_term>: {'start': <#>, 'end': <#>} for age/bp
        #  and "sex": {"Male" or "Female"} for sex

        super().__init__(input_profile, output_profile, probability_func)

    def setup(self, builder):
        self.population_view = builder.population.get_view(requires_columns=['age', 'sex'])
        self.blood_pressure = builder.value.get_value('high_systolic_blood_pressure.exposure')

    def probability(self, index: pd.Index):
        """Apply all filters (and-ed together) to index to determine who's
        eligible for this transition."""
        filtered_index = index

        p = pd.Series(0, index=index)

        if 'age' in self.filters:
            age = self.population_view(index).age
            in_age_index = ((age >= self.filters['age']['start'])
                            & (age < self.filters['age']['end'])).index
            filtered_index = filtered_index.intersection(in_age_index)
        if 'sex' in self.filters:
            sex = self.population_view(index).sex
            sex_index = (sex == self.filters['sex']).index
            filtered_index = filtered_index.intersection(sex_index)
        if 'systolic_blood_pressure' in self.filters:
            sbp = self.blood_pressure(index)
            bp_index = ((sbp >= self.filters['systolic_blood_pressure']['start'])
                        & sbp < self.filters['systolic_blood_pressure']['end']).index
            filtered_index = filtered_index.intersection(bp_index)

        p.loc[filtered_index] = super().probability(filtered_index)
        return p


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







import pandas as pd

from typing import Dict, List

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
                 domain: FilterDomain = None):

        self.domain = FilterDomain
        super().__init__(input_profile, output_profile, probability_func)

    def probability(self, index: pd.Index):
        """Use domain to determine who's eligible for this transition."""
        p = pd.Series(0, index=index)
        filtered_index = self.domain.filter_index(index)
        p.loc[filtered_index] = super().probability(filtered_index)
        return p


class FilterDomain:
    range_dimensions = {'age', 'systolic_blood_pressure', 'cvd_risk_number'}
    set_dimensions = {'sex'}

    def __init__(self, dimensions: dict = None):
        self.dimensions = FilterDomain.get_dimensions(dimensions)

    def setup(self, builder):
        self.population_view = builder.population.get_view(requires_columns=['age', 'sex'])
        self.blood_pressure = builder.value.get_value('high_systolic_blood_pressure.exposure')
        # TODO: add cvd risk pipeline - need to clarify w/ MW whether the risk thresholds vary by ramp or by guideline

    def filter_index(self, index: pd.Index):
        filtered_index = index

        if 'age' in self.dimensions:
            age = self.population_view(index).age
            in_age_index = ((age >= self.dimensions['age']['start'])
                            & (age < self.dimensions['age']['end'])).index
            filtered_index = filtered_index.intersection(in_age_index)
        if 'sex' in self.dimensions:
            sex = self.population_view(index).sex
            sex_index = (sex.isin(self.dimensions['sex'])).index
            filtered_index = filtered_index.intersection(sex_index)
        if 'systolic_blood_pressure' in self.dimensions:
            sbp = self.blood_pressure(index)
            bp_index = ((sbp >= self.dimensions['systolic_blood_pressure']['start'])
                        & sbp < self.dimensions['systolic_blood_pressure']['end']).index
            filtered_index = filtered_index.intersection(bp_index)
        # TODO: add cvd risk num filter

        return filtered_index

    def __eq__(self, other):
        return self.dimensions == other.dimensions

    def __contains__(self, other):
        for d in FilterDomain.range_dimensions:
            if (self.dimensions[d]['start'] > other.dimensions[d]['start']
                    or self.dimensions[d]['end'] <= other.dimensions[d]['end']):
                return False
        for d in FilterDomain.set_dimensions:
            if not other.dimensions[d].issubset(self.dimensions[d]):
                return False
        return True

    def get_subdomain(self, sub_dimensions: dict):
        subdomain = FilterDomain(sub_dimensions)
        if subdomain not in self:
            raise ValueError('Subdomains must be contained within the parent domain.')

        subdomain.population_view = self.population_view
        subdomain.blood_pressure = self.blood_pressure
        # TODO: share cvd risk num pipeline

        return subdomain

    def is_covered(self, sub_domains: List):


    @staticmethod
    def get_dimensions(dimensions):
        # for ranges: start is always inclusive, end is always exclusive
        default_dimensions = {'age': {'start': 0, 'end': 125},
                              'systolic_blood_pressure': {'start': 60, 'end': 300},
                              'cvd_risk_number': {0, 4},
                              'sex': {'Male', 'Female'}}

        if dimensions:
            acceptable_dimensions = {'age', 'systolic_blood_pressure', 'cvd_risk_number', 'sex'}
            extra_dimensions = set(dimensions.keys()).difference(acceptable_dimensions)
            if extra_dimensions:
                raise ValueError(f'The only acceptable dimensions are: {acceptable_dimensions}. '
                                 f'You included {extra_dimensions}.')

            for k, v in dimensions.items():
                if k in FilterDomain.range_dimensions:
                    default_dimensions[k].update(v)
                else:
                    default_dimensions[k] = v

        return default_dimensions


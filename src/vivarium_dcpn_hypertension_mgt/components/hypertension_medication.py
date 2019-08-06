import copy
import itertools

import pandas as pd

from typing import Dict, List, Tuple

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


class FilterDomain:

    def __init__(self, dimensions: dict = None, key: str = None):
        self._dimensions = FilterDomain.get_dimensions(dimensions)
        self.key = key

    @property
    def name(self):
        return f'filter_domain_{self.key}'

    @property
    def dimensions(self):
        return copy.deepcopy(self._dimensions)

    def setup(self, builder):
        self.population_view = builder.population.get_view(requires_columns=['age', 'sex'])
        self.blood_pressure = builder.value.get_value('high_systolic_blood_pressure.exposure')
        # TODO: add cvd risk pipeline

    def filter_index(self, index: pd.Index):
        in_domain = pd.Series(True, index=index)

        if 'age' in self.dimensions:
            age = self.population_view(index).age
            in_age = age.apply(lambda a: a in self.dimensions['age'])
            in_domain &= in_age
        if 'sex' in self.dimensions:
            sex = self.population_view(index).sex
            in_sex = (sex.isin(self.dimensions['sex']))
            in_domain &= in_sex
        if 'systolic_blood_pressure' in self.dimensions:
            sbp = self.blood_pressure(index)
            in_bp = sbp.apply(lambda bp: bp in self.dimensions['systolic_blood_pressure'])
            in_domain &= in_bp
        # TODO: add cvd risk num filter

        return index[in_domain]

    def __eq__(self, other):
        return self.dimensions == other.dimensions

    def __contains__(self, other):
        """Checks that other is fully contained within self."""
        if self.dimensions.keys() != other.dimensions.keys():
            return False
        for d in self.dimensions:
            val, other_val = self.dimensions[d], other.dimensions[d]
            if isinstance(val, pd.Interval) and not check_interval_contains(val, other_val):
                return False
            elif isinstance(val, set) and not other_val.issubset(val):
                return False
        return True

    def __hash__(self):
        sorted_dimensions = {k: self.dimensions[k] for k in sorted(self.dimensions)}
        return hash(str(sorted_dimensions))

    def get_subdomain(self, sub_dimensions: dict):
        subdomain = FilterDomain(sub_dimensions)
        if subdomain not in self:
            raise ValueError('Sub-domains must be contained within the parent domain.')

        subdomain.population_view = self.population_view
        subdomain.blood_pressure = self.blood_pressure
        # TODO: pass on cvd risk num pipeline

        return subdomain

    def is_covered(self, sub_domains: List):
        subs = []
        for s in sub_domains:
            subs.append(s.flatten_dimensions())
        return check_subdomains_complete(pd.concat(subs), self.dimensions)

    @staticmethod
    def get_dimensions(dimensions):
        # for ranges: start is always inclusive, end is always exclusive
        default_dimensions = {'age': pd.Interval(0, 125, closed='left'),
                              'systolic_blood_pressure': pd.Interval(60, 300, closed='left'),
                              'cvd_risk_number': pd.Interval(0, 4, closed='left'),
                              'sex': {'Male', 'Female'}}

        if dimensions:
            acceptable_dimensions = {'age', 'systolic_blood_pressure', 'cvd_risk_number', 'sex'}
            extra_dimensions = set(dimensions.keys()).difference(acceptable_dimensions)
            if extra_dimensions:
                raise ValueError(f'The only acceptable dimensions are: {acceptable_dimensions}. '
                                 f'You included {extra_dimensions}.')

            default_dimensions.update(dimensions)

        return default_dimensions

    def flatten_dimensions(self):
        set_dimensions = {k: v for k, v in self.dimensions.items() if isinstance(v, set)}
        set_combinations = itertools.product(*set_dimensions.values())
        set_df = pd.DataFrame(set_combinations, columns=set_dimensions.keys())

        range_dimensions = {k: v for k, v in self.dimensions.items() if isinstance(v, pd.Interval)}
        range_df = pd.DataFrame(range_dimensions, index=set_df.index)

        return pd.concat([set_df, range_df], axis=1)


def check_subdomains_complete(sub_domains_data: pd.DataFrame, full_domain: dict):
    """Verifies that sub-domains span full_domain in all dimensions with no
    overlaps or gaps and do not exceed full_domain in any dimensions."""
    for d in full_domain.keys():
        other_dimensions = set(full_domain.keys()).difference({d})
        sub_tables = sub_domains_data.groupby(list(other_dimensions))

        for _, table in sub_tables:

            if isinstance(full_domain[d], pd.Interval):
                sub_dimension = collapse_intervals(table[d])
            else:  # set dimension
                sub_dimension = set(table[d])

            if sub_dimension != full_domain[d]:
                import pdb; pdb.set_trace()
                return False

    return True


INTERVAL_CLOSED = {'left': [1, 0], 'right': [0, 1], 'both': [1, 1], 'neither': [0, 0]}


def collapse_intervals(intervals: pd.Series) -> pd.Interval:
    """Collapses a series of intervals into a single interval, checking that
    there are no overlaps or gaps within the intervals while doing so."""
    intervals = list(intervals.sort_values(ascending=False))

    collapsed = intervals.pop()

    while intervals:
        next_interval = intervals.pop()

        if collapsed.overlaps(next_interval):
            raise ValueError('Overlapping intervals found.')
        if collapsed.right != next_interval.left or (collapsed.right == next_interval.left
                                                     and collapsed.closed in {'left', 'neither'}
                                                     and next_interval.closed in {'right', 'neither'}):
            raise ValueError('Gaps in intervals found.')

        right = next_interval.right

        closed_combo = INTERVAL_CLOSED[collapsed.closed] + INTERVAL_CLOSED[next_interval.closed]

        if closed_combo[0] == 0:
            closed = 'right' if closed_combo[-1] else 'neither'
        else:
            closed = 'left' if not closed_combo[-1] else 'both'

        collapsed = pd.Interval(collapsed.left, right, closed=closed)

    return collapsed


def check_interval_contains(a, b):
    """Returns true if b is fully contained in a; false otherwise"""
    if a.left > b.left or a.right < b.right:
        return False
    if a.left == b.left and not INTERVAL_CLOSED[a.closed][0] and INTERVAL_CLOSED[b.closed][0]:
        return False
    if a.right == b.right and not INTERVAL_CLOSED[a.closed][1] and INTERVAL_CLOSED[b.closed][1]:
        return False
    return True

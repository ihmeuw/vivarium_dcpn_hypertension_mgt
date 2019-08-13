from loguru import logger
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

    def add_transition(self, output, probability_value: float = 1, domain_filter: str = "", **kwargs):
        if 'probability_func' in kwargs:
            logger.warning('Probability functions cannot be used for transitions '
                           'added to a treatment profile state. Only single '
                           'value probabilities are allowed.')

        if domain_filter not in self._domain_filters:
            raise ValueError(f'The given domain filter {domain_filter} is invalid for this state.')
        elif sum(self._domain_filters[domain_filter]) + probability_value > 1:
            raise ValueError(f'The given domain filter cannot be added with probability {probability_value} because it '
                             f'would push the summed probabilities for this domain filter over 1.')

        self._domain_filters[domain_filter].append(probability_value)

        t = FilterTransition(self, output, probability_func=lambda index: pd.Series(probability_value, index=index),
                             domain_filter=domain_filter, **kwargs)
        self.transition_set.append(t)

        return t

    def is_valid(self):
        for domain_filter, probabilities in self._domain_filters.items():
            if sum(probabilities) != 1:
                return False
            return True

    def graph_domain_filters(self):
        def parse_query_string(f):
            s = f.split(' <= ')
            if len(s) == 2:
                return f'{s[1]}_start', int(s[0])

            s = f.split(' < ')
            if len(s) == 2:
                return f'{s[0]}_end', int(s[1])

            s = f.split(' == ')
            return s[0], s[1]

        dfs = []
        for d, v in self._domain_filters.items():
            d_split = d.split(" and ")
            parsed_filter = [parse_query_string(part) for part in d_split]
            d_filter = {k: v for k, v in parsed_filter}
            d_filter['probability'] = sum(v)
            dfs.append(pd.DataFrame(d_filter, index=[0]))

        df = pd.concat(dfs).reset_index(drop=True)
        df['width'] = df.age_end - df.age_start
        df['height'] = df.systolic_blood_pressure_end - df.systolic_blood_pressure_start

        return utilities.plot_profile_domain_filters(df, self.state_id)


class FilterTransition(Transition):

    def __init__(self, input_profile, output_profile, probability_func=lambda index: pd.Series(1, index=index),
                 domain_filter: str = "", **kwargs):

        self.domain_filter = domain_filter

        super().__init__(input_profile, output_profile, probability_func, **kwargs)

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

    def add_transition(self, output, probability_value: int = 1, domain_filter: str = "", **kwargs):
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
                    tx_profile.add_transition(next_profile, probability_value=probability, domain_filter=domain_filter)

                # add transitions to null state
                for domain_filter in profile_domain_filters.filter(like='null_state'):
                    tx_profile.add_transition(self.treatment_profiles['null_state'], domain_filter=domain_filter)

                self.treatment_profiles[tx_profile.state_id] = tx_profile

        self.add_states(self.treatment_profiles.values())
        super().setup(builder)

    def validate(self):
        for state in self.states:
            assert state.is_valid(), f'State {state.state_id} is invalid.'

    def to_dot(self, domain_filter_labels=False):
        """Produces a ball and stick graph of this state machine.

        Returns
        -------
        `graphviz.Digraph`
            A ball and stick visualization of this state machine.
        """
        from graphviz import Digraph
        dot = Digraph(format='png')

        for state in self.states:
            if isinstance(state, NullTreatmentProfile):
                dot.node(state.state_id, style='dashed')
            else:
                dot.node(state.state_id)
            for transition in state.transition_set:
                label = transition.domain_filter if domain_filter_labels else None
                color = 'red' if isinstance(transition.output_state, NullTreatmentProfile) else 'green'

                dot.edge(state.state_id, transition.output_state.state_id, label, color=color)
        return dot


def get_next_states(current_profile: TreatmentProfile, next_ramps: List[str],
                    tx_profiles: Dict[str, TreatmentProfile], profile_efficacies: pd.Series):
    next_profiles = []  # list of tuples of form (next_profile, probability)
    current_efficacy = profile_efficacies.loc[current_profile.state_id]

    if 'mono_starter' in next_ramps and 'combo_starter' in next_ramps and current_profile.ramp == 'initial':
        mono_next = utilities.get_closest_in_efficacy_in_ramp(current_efficacy, profile_efficacies, 'mono_starter')
        combo_next = utilities.get_closest_in_efficacy_in_ramp(current_efficacy, profile_efficacies, 'combo_starter')
        both_next = mono_next.append(combo_next)

        next_mono_combo = []
        # we want the two closest in efficacy but if there are ties, add both and split the probability
        unique_efficacies = [e for e in both_next.groupby(both_next)][:2]
        for unique_efficacy in unique_efficacies:
            next_mono_combo.extend([(tx_profiles[n], (1/len(unique_efficacies))/len(unique_efficacy[1]))
                                    for n in unique_efficacy[1].index])

        if len(next_mono_combo) == 0:
            # FIXME: for now putting at position on ramp with highest efficacy if nothing has greater efficacy
            logger.warning(f'There are no mono_starter or combo_starter states with efficacy >= current '
                           f'efficacy for initial profile {current_profile.state_id}.')

            next_mono_combo = [(utilities.get_highest_position_profile_in_ramp(tx_profiles, r), 0.5)
                               for r in ['mono_starter', 'combo_starter']]

        next_profiles.extend(next_mono_combo)

        next_ramps.remove('mono_starter')
        next_ramps.remove('combo_starter')

    if current_profile.ramp in next_ramps:
        if f'{current_profile.ramp}_{current_profile.position + 1}' in tx_profiles:  # we aren't at the last position
            next_in_ramp = tx_profiles[f'{current_profile.ramp}_{current_profile.position + 1}']
            next_profiles.append((next_in_ramp, 1.0))
        next_ramps.remove(current_profile.ramp)

    for ramp in next_ramps:
        if current_profile.ramp == 'no_treatment':
            next_in_ramp = tx_profiles[f'{ramp}_1']
            next_profiles.append((next_in_ramp, 1.0))
        else:
            next_in_ramp = utilities.get_closest_in_efficacy_in_ramp(current_efficacy, profile_efficacies, ramp)

            if len(next_in_ramp) == 0:
                # FIXME: for now putting at position on ramp with highest efficacy if nothing has greater efficacy
                logger.warning(f'There are no {ramp} states with efficacy >= current '
                               f'efficacy for profile {current_profile.state_id}.')
                next_profiles.extend([(utilities.get_highest_position_profile_in_ramp(tx_profiles, ramp), 1.0)])
            else:
                next_profiles.extend([(tx_profiles[n], 1 / len(next_in_ramp)) for n in next_in_ramp.index])

    return next_profiles



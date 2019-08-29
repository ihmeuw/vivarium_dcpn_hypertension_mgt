from scipy import stats

from loguru import logger
import numpy as np
import pandas as pd

from typing import Dict, List

from vivarium.framework.state_machine import State, Transition, Machine
from vivarium_dcpn_hypertension_mgt.components import utilities
from vivarium_dcpn_hypertension_mgt.components.globals import HYPERTENSION_DRUGS


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

    def get_valid_filter(self):
        valid_transitions = []
        for transition in self.transition_set:
            if not isinstance(transition.output_state, NullTreatmentProfile):
                valid_transitions.append(transition.domain_filter)
        return " or ".join(set(valid_transitions))

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
        self.population_view = builder.population.get_view(['age', 'sex', 'high_systolic_blood_pressure_measurement'])
        self.cvd_risk_cat = builder.value.get_value('cvd_risk_category')

    def probability(self, index: pd.Index):
        """Apply filter to index to determine who's eligible for this
        transition. Everyone else gets probability 0."""
        p = pd.Series(0, index=index)

        characteristics = self.population_view.get(index)
        characteristics = characteristics.rename(columns={'high_systolic_blood_pressure_measurement':
                                                              'systolic_blood_pressure'})
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
        if not index.empty:
            raise NullStateError(f'{len(index)} simulants are attempting to transition '
                                 f'into the null treatment profile state.')

    def is_valid(self):
        """The null state is valid so long as there are no transitions from it."""
        return not len(self.transition_set)

    def graph_domain_filters(self):
        return utilities.plot_profile_domain_filters(None, self.state_id)


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
            fpg >= 6.1 mmol/L
            time since last ihd event: <= 1 yr
        """

        cvd_risk = pd.Series(0, index=index)

        bmi = self.bmi(index)
        fpg = self.fpg(index)
        time_since_ihd = self.clock() - self.population_view.get(index).ischemic_heart_disease_event_time

        cvd_risk.loc[(28 <= bmi) | (6.1 <= fpg) | (time_since_ihd <= pd.Timedelta(days=365.25))] = 1

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
        self.profiles = utilities.load_treatment_profiles(builder)
        domain_filters = utilities.load_domain_filters(builder)
        efficacy_data = utilities.load_efficacy_data(builder)

        self.ramps = [r for r in TreatmentProfileModel.ramps if r in set(self.profiles.ramp_name)]
        self.ramp_transitions = {k: [r for r in v if r in self.ramps]
                                 for k, v in TreatmentProfileModel.ramp_transitions.items()}

        self.treatment_profiles = build_states(self.ramps, self.ramp_transitions,
                                               self.profiles, domain_filters, efficacy_data)

        self.add_states(self.treatment_profiles.values())
        super().setup(builder)

        self.coverage = builder.lookup.build_table(utilities.load_coverage_data(builder),
                                                   parameter_columns=[('age', 'age_group_start', 'age_group_end')])
        self.proportion_above_hypertensive_threshold = builder.lookup.build_table(
            builder.data.load('risk_factor.high_systolic_blood_pressure.proportion_above_hypertensive_threshold'))

        sbp = builder.value.get_value('high_systolic_blood_pressure.exposure')
        self.raw_sbp = lambda index: pd.Series(sbp.source(index), index=index)

        self.randomness = builder.randomness.get_stream('initial_treatment_profile')

        self.population_view = builder.population.get_view(['age', 'sex', 'high_systolic_blood_pressure_measurement',
                                                            self.state_column])
        builder.population.initializes_simulants(self.on_initialize_simulants, creates_columns=[self.state_column])
        builder.value.register_value_producer('prescribed_medications', source=self.get_prescribed_medications)
        
        self.cvd_risk_cat = builder.value.get_value('cvd_risk_category')
        self.valid_transition_filters = self.get_valid_transition_filters()

    def on_initialize_simulants(self, pop_data):
        initial_tx_profiles = self.get_initial_profiles(pop_data.index)
        initial_tx_profiles.name = self.state_column
        self.population_view.update(initial_tx_profiles)

    def get_initial_profiles(self, index):
        raw_sbp = self.raw_sbp(index)
        below_140 = raw_sbp[raw_sbp < 140].index
        above_140 = raw_sbp[raw_sbp >= 140].index

        profile_prob_below_140, profile_names = utilities.probability_profile_given_sbp_level('below_140',
                                    self.proportion_above_hypertensive_threshold(below_140),
                                    self.coverage(below_140), self.profiles)
        profile_prob_above_140, profile_names = utilities.probability_profile_given_sbp_level('above_140',
                                    self.proportion_above_hypertensive_threshold(above_140),
                                    self.coverage(above_140), self.profiles)

        profile_probabilities = np.stack(pd.concat([profile_prob_below_140, profile_prob_above_140])
                                         .sort_index().values, axis=0)
        profile_choices = self.randomness.choice(index, choices=profile_names, p=profile_probabilities)

        return profile_choices

    def get_valid_transition_filters(self):
        valid_filters = dict()
        for state in self.states:
            valid_filters[state.state_id] = state.get_valid_filter()
        return valid_filters

    def filter_for_next_valid_state(self, index):
        if not self.valid_transition_filters:
            return pd.Index([])

        characteristics = self.population_view.subview(['age', 'sex',
                                                        'high_systolic_blood_pressure_measurement']).get(index)
        characteristics = characteristics.rename(columns={'high_systolic_blood_pressure_measurement':
                                                              'systolic_blood_pressure'})
        characteristics['cvd_risk_cat'] = self.cvd_risk_cat(index)

        valid_index = pd.Index([])
        for state, pop_in_state in self._get_state_pops(index):
            if not pop_in_state.empty and self.valid_transition_filters[state.state_id]:
                valid_in_state = characteristics.loc[pop_in_state.index].query(self.valid_transition_filters[state.state_id]).index
                valid_index = valid_index.union(valid_in_state)
        return valid_index

    def filter_for_prescribed_treatment(self, index):
        population = self.population_view.subview([self.state_column]).get(index)
        return population.loc[population[self.state_column] != 'no_treatment_1'].index

    def get_prescribed_medications(self, index):
        df = pd.DataFrame({d: (0.0 if d != 'other' else 'none') for d in HYPERTENSION_DRUGS}, index=index)

        for state, pop_in_state in self._get_state_pops(index):
            if not pop_in_state.empty:
                df.loc[pop_in_state.index, HYPERTENSION_DRUGS] = pd.DataFrame(state.drug_dosages, index=pop_in_state.index)

        return df

    def _get_state_pops(self, index):
        population = self.population_view.subview([self.state_column]).get(index)
        return [[state, population[population[self.state_column] == state.state_id]] for state in self.states]

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


def build_states(ramps, ramp_transitions, profiles, domain_filters, efficacy_data):
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
    treatment_profiles = dict()

    # 0. add null state profile
    treatment_profiles['null_state'] = NullTreatmentProfile()

    # used to find next profile when changing btw ramps
    profile_efficacies = pd.Series()

    # 1-5. add ramp profiles in order
    for ramp in ramps:
        ramp_profiles = profiles[profiles.ramp_name == ramp].sort_values(by='ramp_position', ascending=False)

        for profile in ramp_profiles.iterrows():
            profile_domain_filters = utilities.get_state_domain_filters(domain_filters, ramp,
                                                                        profile[1].ramp_position, ramp_profiles,
                                                                        ramp_transitions)
            tx_profile = TreatmentProfile(ramp, profile[1].ramp_position, (profile[1][HYPERTENSION_DRUGS]).to_dict(),
                                          list(profile_domain_filters))

            # record the current efficacy to be used in finding next states for other profiles
            efficacy = utilities.calculate_pop_efficacy(tx_profile.drug_dosages, efficacy_data)
            profile_efficacies = profile_efficacies.append(pd.Series(efficacy, index=[tx_profile.state_id]))

            # add transitions to next treatment profile states
            for next_profile, probability, transition_domain_filters in get_next_states(tx_profile,
                                                                                        ramp_transitions[
                                                                                            ramp].copy(),
                                                                                        treatment_profiles,
                                                                                        profile_efficacies,
                                                                                        profile_domain_filters):

                for domain_filter in transition_domain_filters:
                    tx_profile.add_transition(next_profile,
                                              probability_value=probability, domain_filter=domain_filter)

            # add transitions to null state
            for domain_filter in profile_domain_filters.filter(like='null_state'):
                tx_profile.add_transition(treatment_profiles['null_state'], domain_filter=domain_filter)

            treatment_profiles[tx_profile.state_id] = tx_profile

    return treatment_profiles


def get_next_states(current_profile: TreatmentProfile, next_ramps: List[str],
                    tx_profiles: Dict[str, TreatmentProfile], profile_efficacies: pd.Series,
                    ramp_domain_filters: pd.Series):
    next_profiles = []  # list of tuples of form (next_profile, probability, domain_filters)
    current_efficacy = profile_efficacies.loc[current_profile.state_id]

    if 'mono_starter' in next_ramps and 'combo_starter' in next_ramps and current_profile.ramp == 'initial':
        mono_next = utilities.get_closest_in_efficacy_in_ramp(current_efficacy, profile_efficacies, 'mono_starter')
        combo_next = utilities.get_closest_in_efficacy_in_ramp(current_efficacy, profile_efficacies, 'combo_starter')
        both_next = mono_next.append(combo_next)

        next_mono_combo = []
        # we want the two closest in efficacy but if there are ties, add both and split the probability
        unique_efficacies = [e for e in both_next.groupby(both_next)][:2]
        for unique_efficacy in unique_efficacies:
            profiles = [tx_profiles[n] for n in unique_efficacy[1].index]
            probability = (1 / len(unique_efficacies)) / len(unique_efficacy[1])  # profiles w/ same eff have same prob
            domain_filters = [utilities.get_domain_filters_between_ramps(ramp_domain_filters, current_profile.ramp,
                                                                         p.ramp) for p in profiles]
            next_mono_combo.extend([(p, probability, f) for p, f in zip(profiles, domain_filters)])

        if len(next_mono_combo) == 0:  # no profiles with efficacy >= current
            logger.warning(f'There are no mono_starter or combo_starter states with efficacy >= current '
                           f'efficacy for initial profile {current_profile.state_id}.')
            domain_filters = utilities.get_domain_filters_between_ramps(ramp_domain_filters, current_profile.ramp,
                                                                        'mono_starter')
            next_mono_combo = [(tx_profiles['null_state'], 1.0, domain_filters)]

        next_profiles.extend(next_mono_combo)

        next_ramps.remove('mono_starter')
        next_ramps.remove('combo_starter')

    if current_profile.ramp in next_ramps:
        if f'{current_profile.ramp}_{current_profile.position + 1}' in tx_profiles:  # we aren't at the last position
            next_in_ramp = tx_profiles[f'{current_profile.ramp}_{current_profile.position + 1}']
            domain_filters = utilities.get_domain_filters_between_ramps(ramp_domain_filters, current_profile.ramp,
                                                                        current_profile.ramp)

            next_profiles.append((next_in_ramp, 1.0, domain_filters))
        next_ramps.remove(current_profile.ramp)

    for ramp in next_ramps:
        domain_filters = utilities.get_domain_filters_between_ramps(ramp_domain_filters, current_profile.ramp, ramp)

        if current_profile.ramp == 'no_treatment':
            next_in_ramp = utilities.get_position_profile_in_ramp(tx_profiles, ramp, 'lowest')
            next_profiles.append((next_in_ramp, 1.0, domain_filters))
        else:
            next_in_ramp = utilities.get_closest_in_efficacy_in_ramp(current_efficacy, profile_efficacies, ramp)

            if len(next_in_ramp) == 0:
                logger.warning(f'There are no {ramp} states with efficacy >= current '
                               f'efficacy for profile {current_profile.state_id}.')

                next_profiles.extend([(tx_profiles['null_state'], 1.0, domain_filters)])
            else:
                next_profiles.extend([(tx_profiles[n], 1 / len(next_in_ramp), domain_filters)
                                      for n in next_in_ramp.index])

    return next_profiles


class TreatmentEffect:

    @property
    def name(self):
        return 'hypertension_drugs_treatment_effect'

    def setup(self, builder):
        self.medications = HYPERTENSION_DRUGS
        self.efficacy_data = utilities.load_efficacy_data(builder).reset_index()

        self.medication_efficacy = pd.Series(index=pd.MultiIndex(levels=[[], [], []],
                                                                 labels=[[], [], []],
                                                                 names=['simulant', 'medication', 'dosage']))

        self.shift_column = 'hypertension_drugs_baseline_shift'

        self.population_view = builder.population.get_view([self.shift_column])
        builder.population.initializes_simulants(self.on_initialize_simulants, creates_columns=[self.shift_column])
        self.prescribed_medications = builder.value.get_value('prescribed_medications')

        self.randomness = builder.randomness.get_stream('dose_efficacy')
        self.medication_effects = {m: builder.value.register_value_producer(f'{m}.effect_size',
                                                                            self.get_medication_effect)
                                   for m in self.medications}

        self.prescription_filled = builder.value.get_value('rx_fill.currently_filled')

        self.treatment_effect = builder.value.register_value_producer('hypertension_drugs.effect_size',
                                                                      self.get_treatment_effect)

        builder.value.register_value_modifier('high_systolic_blood_pressure.exposure', self.treat_sbp)

    def on_initialize_simulants(self, pop_data):
        self.medication_efficacy = self.medication_efficacy.append(self.determine_medication_efficacy(pop_data.index))
        prescribed_meds = self.prescribed_medications(pop_data.index)
        effects = [self.get_medication_effect(prescribed_meds[m], m) for m in self.medications]
        self.population_view.update(sum(effects))

    def determine_medication_efficacy(self, index):
        efficacy = []

        for med in self.medications:
            efficacy_draw = self.randomness.get_draw(index, additional_key=med)
            med_efficacy = self.efficacy_data.query('medication == @med')
            for dose in med_efficacy.dosage.unique():
                dose_efficacy_parameters = med_efficacy.loc[med_efficacy.dosage == dose, ['value', 'sd_mean']].values[0]
                dose_index = pd.MultiIndex.from_product((index, [med], [dose]),
                                                        names=('simulant', 'medication', 'dosage'))
                if dose_efficacy_parameters[1] == 0.0:  # if sd is 0, no need to draw
                    dose_efficacy = pd.Series(dose_efficacy_parameters[0], index=dose_index)
                else:
                    dose_efficacy = pd.Series(stats.norm.ppf(efficacy_draw, loc=dose_efficacy_parameters[0],
                                                                   scale=dose_efficacy_parameters[1]),
                                              index=dose_index)
                efficacy.append(dose_efficacy)

        return pd.concat(efficacy)

    def get_medication_effect(self, dosages, medication):
        lookup_index = pd.MultiIndex.from_arrays((dosages.index.values,
                                                  np.tile(medication, len(dosages)),
                                                  dosages.values),
                                                 names=('simulant', 'medication', 'dosage'))

        efficacy = self.medication_efficacy.loc[lookup_index]
        efficacy.index = efficacy.index.droplevel(['medication', 'dosage'])
        return efficacy * self.prescription_filled(dosages.index)

    def get_treatment_effect(self, index):
        prescribed_meds = self.prescribed_medications(index)
        return sum([self.get_medication_effect(prescribed_meds[m], m) for m in self.medications])

    def treat_sbp(self, index, exposure):
        baseline_shift = self.population_view.get(index)[self.shift_column]
        return exposure + baseline_shift - self.treatment_effect(index)

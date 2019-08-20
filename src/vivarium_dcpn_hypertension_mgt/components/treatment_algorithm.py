import scipy
import numpy as np
import pandas as pd

from vivarium.framework.engine import Builder
from vivarium.framework.engine import Event
from vivarium_dcpn_hypertension_mgt.components import Adherence, MeasuredSBP, TreatmentProfileModel


class Adherence:
    @property
    def name(self):
        # TreatmentAlgorithmAdherence
        return "hypertension_drugs_ta_adherence"

    def __init__(self):
        self._thresholds = None
        self.rand_rx_fill = None
        self.rand_appt_followup = None
        self.rand_threshold_creation = None
        self.df_ad_data = None

    def setup(self, builder: Builder):
        self._thresholds = builder.data.load('health_technology.hypertension_drugs.adherence_thresholds')

        self.rand_rx_fill = builder.randomness.get_stream("hypertension_rx_coverage")
        self.rand_appt_followup = builder.randomness.get_stream("hypertension_apt_fup_coverage")
        self.rand_threshold_creation = builder.randomness.get_stream("hypertension_threshold")

        self.df_ad_data = pd.DataFrame()

        builder.population.initializes_simulants(self.on_initialize_simulants)

        builder.value.register_value_producer('rx_fill.adherence', source=self.get_rx_fill)
        builder.value.register_value_producer('appt_followup.adherence', source=self.get_appt_followup)

    def on_initialize_simulants(self, pop_data):
        # bin everyone into 1 of the 4 adherence catagories
        mask = self.rand_threshold_creation.choice(pop_data.index, self._thresholds.index, self._thresholds.proportion)
        self.df_ad_data['rx_ad'] = mask.apply(lambda x: self._thresholds.loc[x].prescription_fill)
        self.df_ad_data['appt_ad'] = mask.apply(lambda x: self._thresholds.loc[x].follow_up)

    def get_rx_fill(self, index) -> pd.Series:
        return self.rand_rx_fill.get_draw(index) < self.df_ad_data.rx_ad.loc[index]

    def get_appt_followup(self, index) -> pd.Series:
        return self.rand_appt_followup.get_draw(index) < self.df_ad_data.appt_ad.loc[index]


class MeasuredSBP:
    @property
    def name(self):
        return "high_systolic_blood_pressure_measurement"

    configuration_defaults = {
        'measurement': {
            'error_sd': 6,
        }
    }

    def __init__(self):
        self.configuration_defaults = {
            f'{self.name}_measurement': MeasuredSBP.configuration_defaults['measurement']
        }

    def setup(self, builder: Builder):
        self.measurement_error = builder.configuration[f'{self.name}_measurement'].error_sd
        self.randomness = builder.randomness.get_stream(f'{self.name}.measurement')
        self.true_exposure = builder.value.get_value(f'{self.name}.exposure')

        self.measurement_column = f'{self.name}_measurement'
        columns_created = [self.measurement_column]

        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 creates_columns=columns_created)

        self.population_view = builder.population.get_view(columns_created)

        builder.event.register_listener('time_step__prepare', self.on_time_step_prepare)

    def on_initialize_simulants(self, pop_data):
        self.population_view.update(
            pd.Series(np.nan, name=self.measurement_column, index=pop_data.index)
        )

    def on_time_step_prepare(self, event: Event):
        self.population_view.update(pd.Series(np.nan, index=event.index, name=self.measurement_column))

    def __call__(self, idx_measure: pd.Index, idx_record_these: pd.Index, measure_type_average: bool = False):
        draw = self.randomness.get_draw(idx_measure)
        if self.measurement_error:
            noise = scipy.stats.norm.ppf(draw, scale=self.measurement_error)
        else:
            noise = 0

        true_exp = self.true_exposure(idx_measure)
        detect_zero = true_exp[true_exp==0]
        hypertension_measurement = true_exp + noise
        hypertension_measurement.loc[detect_zero.index] = 0.0

        if measure_type_average:
            hypertension_measurement = (hypertension_measurement +
                                        self.population_view.get(idx_measure)[self.measurement_column]) / 2

        measurement = self.population_view.get(idx_measure)[self.measurement_column]
        measurement.loc[idx_record_these] = hypertension_measurement
        self.population_view.update(measurement)

        return hypertension_measurement


class HealthcareUtilization:

    @property
    def name(self):
        return 'healthcare_utilization'

    def setup(self, builder):
        self.utilization_data = builder.lookup.build_table(
            builder.data.load('health_technology.hypertension_drugs.healthcare_access'),
            parameter_columns=[('age', 'age_group_start', 'age_group_end')])

        self.randomness = builder.randomness.get_stream('healthcare_utilization_propensity')
        self._propensity = pd.Series()
        builder.population.initializes_simulants(self.on_initialize_simulants)

        builder.value.register_value_producer('healthcare_utilization_rate',
                                              source=self.get_utilization_rate)

    def on_initialize_simulants(self, pop_data):
        self._propensity = self._propensity.append(self.randomness.get_draw(pop_data.index))

    def get_utilization_rate(self, index):
        params = self.utilization_data(index)
        distributions = scipy.stats.norm(loc=params.value, scale=params.sd_individual_heterogeneity)
        utilization = pd.Series(distributions.ppf(self._propensity[index]), index=index)
        utilization.loc[utilization < 0] = 0.0
        return utilization


class TreatmentAlgorithm:
    configuration_defaults = {
        'high_systolic_blood_pressure_measurement': {
            'probability': 1.0,
        }
    }

    @property
    def name(self):
        return 'hypertension_treatment_algorithm'

    def setup(self, builder):
        self.measure_sbp = MeasuredSBP()
        self.treatment_profile_model = TreatmentProfileModel()
        builder.components.add_components([Adherence(), HealthcareUtilization(), self.measure_sbp])

        self.clock = builder.time.clock()

        self.sim_start = pd.Timestamp(**builder.configuration.time.start)
        columns_created = ['followup_date', 'followup_duration', 'followup_type', 'ICU']
        columns_required = ['treatment_profile']
        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 requires_columns=columns_required,
                                                 creates_columns=columns_created)
        self.population_view = builder.population.get_view(columns_required + columns_created)

        self.randomness = {'initial_followup_scheduling': builder.randomness.get_stream('initial_followup_scheduling'),
                           'background_visit_attendance': builder.randomness.get_stream('background_visit_attendance'),
                           'sbp_measured': builder.randomness.get_stream('sbp_measured')}

        self.sbp_measurement_probability = builder.configuration.high_systolic_blood_pressure_measurement.probability
        self.guideline = builder.configuration.hypertension_drugs.guideline

        self.healthcare_utilization = builder.value.get_value('healthcare_utilization_rate')
        self.followup_adherence = builder.value.get_value('appt_followup.adherence')

        self._prescription_filled = pd.Series()
        self.prescription_filled = builder.value.register_value_producer('rx_fill.currently_filled',
                                                                         source=self.get_prescriptions_filled)

        builder.event.register_listener('time_step', self.on_time_step)

    def on_initialize_simulants(self, pop_data):
        sims_on_tx = (self.population_view.subview(['treatment_profile']).get(pop_data.index)
                      .query("treatment_profile != 'no_treatment_1'")).index

        initialize = pd.DataFrame({'followup_date': pd.NaT, 'followup_duration': pd.NaT, 'followup_type': None,
                                   'ICU': 0},
                                  index=pop_data.index)

        initialize.loc[sims_on_tx, ['followup_duration', 'followup_type']] = pd.Timedelta(days=6*28), 'maintenance'

        to_sim_date = np.vectorize(lambda d: self.sim_start + pd.Timedelta(days=d))
        np.random.seed(self.randomness['initial_followup_scheduling'].get_seed())
        initialize.loc[sims_on_tx, 'followup_date'] = to_sim_date(np.random.random_integers(low=0, high=6*28,
                                                                                            size=len(sims_on_tx)))
        self._prescription_filled = self._prescription_filled.append(pd.Series(0, pop_data.index))
        self._prescription_filled.loc[sims_on_tx] = 1

        self.population_view.update(initialize)

    def on_time_step(self, event):
        pop = self.population_view.get(event.index)

        followup_scheduled = (self.clock() < pop.followup_date) & (pop.followup_date <= event.time)

        followup_pop = event.index[followup_scheduled]
        followup_attendance = self.followup_adherence(followup_pop)
        self.attend_followup(followup_pop[followup_attendance])
        self.reschedule_followup(followup_pop[~followup_attendance])

        background_eligible = event.index[~followup_scheduled]
        background_attending = (self.randomness['background_visit_attendance']
                                .filter_for_rate(background_eligible,
                                                 self.healthcare_utilization(background_eligible).values))

        self.attend_background(background_attending, event.time)

    def reschedule_followup(self, index):
        followups = self.population_view.subview(['followup_date', 'followup_duration']).get(index)
        followups.followup_date = followups.followup_date + followups.followup_duration
        self.population_view.update(followups)
        self._prescription_filled.loc[index] = 0

    def schedule_followup(self, index: pd.Index, followup_type: str, duration: pd.Timedelta,
                          current_date: pd.Timestamp):
        followups = self.population_view.subview(['followup_date', 'followup_duration', 'followup_type']).get(index)
        followups['followup_type'] = followup_type
        followups['followup_duration'] = duration
        followups['followup_date'] = current_date + duration
        self.population_view.update(followups)

    def attend_followup(self, index):
        pass

    def attend_background(self, index, visit_date):
        # check who we measure SBP for - send everyone else home w/o doing anything else
        sbp_measured_idx = self.randomness['sbp_measured'].filter_for_probability(index,
                                                                                  self.sbp_measurement_probability)
        # we don't want to record any SBP measurement for people already on hypertension visit plan
        no_followup_scheduled = sbp_measured_idx[self.population_view.subview(['followup_date'])
            .get(sbp_measured_idx)['followup_date'].isnull()]

        sbp = self.measure_sbp(sbp_measured_idx, idx_record_these=no_followup_scheduled, measure_type_average=False)

        # send everyone w/ sbp >= 180 to ICU and don't treat them further
        icu_threshold = 180
        send_to_icu = sbp.loc[sbp >= icu_threshold].index
        icu = self.population_view.subview(['ICU']).get(send_to_icu)['ICU'] + 1
        self.population_view.update(icu)
        # anyone who has a hypertension followup scheduled or was sent to the ICU
        # should not continue treatment on a background visit
        sbp = sbp.loc[no_followup_scheduled & (sbp < icu_threshold)]

        # some guidelines have an immediate treatment threshold
        if self.guideline in ['aha', 'who']:
            immediate_treatment_threshold = 160
            immediately_treat = sbp.loc[sbp >= immediate_treatment_threshold].index
            treated = self.transition_treatment(immediately_treat)
            self.schedule_followup(treated, 'maintenance', pd.Timedelta(days=28), visit_date)
            sbp = sbp.loc[sbp < immediate_treatment_threshold]







    def transition_treatment(self, index):
        """Transition treatment for everyone who has a next available tx."""
        to_transition = self.treatment_profile_model.filter_for_next_valid_state(index)
        self.treatment_profile_model.transition(to_transition)
        return to_transition

    def get_prescriptions_filled(self, index):
        return self._prescription_filled[index]

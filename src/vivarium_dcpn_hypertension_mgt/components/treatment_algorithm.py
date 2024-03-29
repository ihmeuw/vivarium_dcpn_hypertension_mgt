import scipy
import numpy as np
import pandas as pd

from typing import Union

from vivarium.framework.engine import Builder
from vivarium_dcpn_hypertension_mgt.components.utilities import (get_dict_for_guideline, get_durations_in_range,
                                                                 FollowupDuration)


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
    configuration_defaults = {
        'high_systolic_blood_pressure_measurement': {
            'error_sd': 6,
        }
    }

    def __init__(self):
        self.risk = 'high_systolic_blood_pressure'

    @property
    def name(self):
        return f"{self.risk}_measurement"

    def setup(self, builder: Builder):
        self.measurement_error = builder.configuration[self.name].error_sd
        self.randomness = builder.randomness.get_stream(f'{self.risk}.measurement')
        self.true_exposure = builder.value.get_value(f'{self.risk}.exposure')

        self.measurement_column = self.name
        self.measurement_time_column = f'{self.risk}_last_measurement_date'
        columns_created = [self.measurement_column, self.measurement_time_column]

        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 creates_columns=columns_created)

        self.population_view = builder.population.get_view(columns_created)

    def on_initialize_simulants(self, pop_data):
        self.population_view.update(
            pd.DataFrame({self.measurement_column: np.nan, self.measurement_time_column: pd.NaT}, index=pop_data.index)
        )

    def __call__(self, idx_measure: pd.Index, idx_record_these: pd.Index, idx_average_these: pd.Index,
                 event_time: pd.Timestamp):
        draw = self.randomness.get_draw(idx_measure)
        if self.measurement_error:
            noise = scipy.stats.norm.ppf(draw, scale=self.measurement_error)
        else:
            noise = 0

        true_exp = self.true_exposure(idx_measure)
        assert np.all(true_exp > 0), '0 values detected for high systolic blood pressure exposure. ' \
                                     'Verify your age ranges are valid for this risk factor.'
        hypertension_measurement = true_exp + noise

        hypertension_measurement.loc[idx_average_these] = (hypertension_measurement.loc[idx_average_these] +
               self.population_view.get(idx_measure).loc[idx_average_these, self.measurement_column]) / 2

        measurement = self.population_view.get(idx_measure)
        measurement.loc[idx_record_these, self.measurement_column] = hypertension_measurement
        measurement.loc[:, self.measurement_time_column] = event_time
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

        builder.value.register_rate_producer('healthcare_utilization_rate',
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
        guideline = builder.configuration.hypertension_drugs.guideline
        self.guideline_thresholds = get_dict_for_guideline(guideline, 'thresholds')
        self.followup_schedules = get_dict_for_guideline(guideline, 'followup_schedules')

        self.measure_sbp = MeasuredSBP()
        builder.components.add_components([Adherence(), HealthcareUtilization(),
                                           self.measure_sbp])
        self.treatment_profile_model = builder.components.get_component('machine.treatment_profile')

        self.clock = builder.time.clock()

        self.sim_start = pd.Timestamp(**builder.configuration.time.start)
        columns_created = ['followup_date', 'followup_duration', 'followup_type', 'last_visit_date',
                           'intensive_care_unit_visits_count', 'last_visit_type', 'last_missed_appt_date']
        columns_required = ['treatment_profile']
        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 requires_columns=columns_required,
                                                 creates_columns=columns_created)
        self.population_view = builder.population.get_view(columns_required + columns_created + ['alive'],
                                                           query='alive == "alive"')

        self.randomness = {'followup_scheduling': builder.randomness.get_stream('followup_scheduling'),
                           'background_visit_attendance': builder.randomness.get_stream('background_visit_attendance'),
                           'sbp_measured': builder.randomness.get_stream('sbp_measured')
                           }

        self.sbp_measurement_probability = builder.configuration.high_systolic_blood_pressure_measurement.probability

        self.healthcare_utilization = builder.value.get_value('healthcare_utilization_rate')
        self.followup_adherence = builder.value.get_value('appt_followup.adherence')

        self.prescription_adherence = builder.value.get_value('rx_fill.adherence')
        self._prescription_filled = pd.Series()
        self.prescription_filled = builder.value.register_value_producer('rx_fill.currently_filled',
                                                                         source=self.get_prescriptions_filled)

        builder.event.register_listener('time_step', self.on_time_step)

    def on_initialize_simulants(self, pop_data):
        sims_on_tx = (self.population_view.subview(['treatment_profile']).get(pop_data.index)
                      .query("treatment_profile != 'no_treatment_1'")).index

        initialize = pd.DataFrame({'followup_date': pd.NaT, 'followup_duration': pd.NaT, 'followup_type': None,
                                   'last_visit_date': pd.NaT, 'last_visit_type': None, 'last_missed_appt_date': pd.NaT,
                                   'intensive_care_unit_visits_count': 0},
                                  index=pop_data.index)

        initialize.loc[sims_on_tx, ['followup_duration', 'followup_type']] = pd.Timedelta(days=180), 'maintenance'

        durations = get_durations_in_range(self.randomness['followup_scheduling'],
                                           low=0, high=180,
                                           index=sims_on_tx)
        initialize.loc[sims_on_tx, 'followup_date'] = durations + self.sim_start
        initialize.loc[sims_on_tx, 'last_visit_date'] = self.sim_start - durations
        initialize.loc[sims_on_tx, 'last_visit_type'] = 'maintenance'

        self._prescription_filled = self._prescription_filled.append(pd.Series(False, pop_data.index))
        self._prescription_filled.loc[sims_on_tx] = True

        self.population_view.update(initialize)

    def on_time_step(self, event):
        pop = self.population_view.get(event.index)

        followup_scheduled = (self.clock() < pop.followup_date) & (pop.followup_date <= event.time)

        followup_pop = pop.index[followup_scheduled]  # event index includes untracked sims so use pop index instead
        followup_attendance = self.followup_adherence(followup_pop)

        pop.loc[followup_pop[followup_attendance], 'last_visit_type'] = \
            pop.loc[followup_pop[followup_attendance], 'followup_type']
        self.attend_followup(followup_pop[followup_attendance], event.time)
        self.reschedule_followup(followup_pop[~followup_attendance])
        pop.loc[followup_pop[~followup_attendance], 'last_missed_appt_date'] = event.time

        background_eligible = pop.index[~followup_scheduled]
        background_attending = (self.randomness['background_visit_attendance']
                                .filter_for_rate(background_eligible,
                                                 self.healthcare_utilization(background_eligible).values))

        self.attend_background(background_attending, event.time)

        pop.loc[background_attending.union(followup_pop[followup_attendance]), 'last_visit_date'] = event.time
        pop.loc[background_attending, 'last_visit_type'] = 'background'
        self.population_view.update(pop.loc[:, ['last_visit_type', 'last_visit_date', 'last_missed_appt_date']])

    def reschedule_followup(self, index):
        followups = self.population_view.subview(['followup_date', 'followup_duration']).get(index)
        followups.followup_date = followups.followup_date + followups.followup_duration
        self.population_view.update(followups)
        self._prescription_filled.loc[index] = False

    def schedule_followup(self, index: pd.Index, followup_type: str,
                          followup_duration: FollowupDuration, current_date: pd.Timestamp, from_visit: str):
        if not index.empty and followup_duration:
            if followup_duration.duration_type == 'constant':
                durations = pd.Timedelta(days=followup_duration.duration_values)
            elif followup_duration.duration_type == 'range':
                durations = get_durations_in_range(self.randomness['followup_scheduling'],
                                                   low=followup_duration.duration_values[0],
                                                   high=followup_duration.duration_values[1],
                                                   index=index, randomness_key=f'{from_visit}_to_{followup_type}')
            else:  # options
                options = [pd.Timedelta(days=x) for x in followup_duration.duration_values]
                durations = self.randomness['followup_scheduling'].choice(index, options,
                                                                          additional_key=f'{from_visit}_to_{followup_type}')

            followups = self.population_view.subview(['followup_date', 'followup_duration', 'followup_type']).get(index)
            followups['followup_type'] = followup_type
            followups['followup_duration'] = durations
            followups['followup_date'] = current_date + durations
            self.population_view.update(followups)

    def attend_followup(self, index, visit_date):
        pop = self.population_view.subview(['followup_type']).get(index)
        followup_groups = pop.groupby(['followup_type']).apply(lambda g: g.index)

        # measure sbp and use the average of this measurement + last for those here for confirmatory visit
        to_average = followup_groups['confirmatory'] if 'confirmatory' in followup_groups.index else pd.Index([])
        sbp = self.measure_sbp(index, idx_record_these=index, idx_average_these=to_average, event_time=visit_date)
        # send everyone w/ sbp >= icu threshold to ICU and don't treat them further here
        sent_to_icu = self.send_to_icu(sbp, visit_date)
        sbp = sbp.loc[~sbp.index.isin(sent_to_icu)]

        # transition everyone who has a treatment available
        transitioned = self.transition_treatment(sbp.index, visit_date)

        # fill prescriptions for everyone who is on a prescribed treatment -
        # this may include more sims than just transitioned because some sims
        # may not have a next treatment available and so will stay on current tx
        self.fill_prescriptions(self.treatment_profile_model.filter_for_prescribed_treatment(sbp.index))

        self.schedule_followups_from_followup(transitioned, sbp.index.difference(transitioned), followup_groups,
                                              visit_date, sbp)

    def schedule_followups_from_followup(self, transitioned: pd.Index, untransitioned: pd.Index,
                                         followup_groups: pd.Series, visit_date: pd.Timestamp, measured_sbp: pd.Series):
        scheduled = pd.Index([])
        # set up followups
        for visit_type, idx in followup_groups.iteritems():
            followups = self.followup_schedules[visit_type]

            # schedule maintenance for everyone who was treated
            if visit_type == 'maintenance':  # diff schedules for sims who transitioned vs those who didn't
                transition_schedule = followups['maintenance']['transitioned']

                to_schedule = untransitioned.intersection(idx)
                self.schedule_followup(to_schedule, 'maintenance', followups['maintenance']['untransitioned'],
                                       visit_date, from_visit=visit_type)
                scheduled = scheduled.union(to_schedule)
            else:  # only schedule maintenance for transitioned sims
                transition_schedule = followups['maintenance']

            to_schedule = transitioned.intersection(idx)
            self.schedule_followup(to_schedule, 'maintenance', transition_schedule,
                                   visit_date, from_visit=visit_type)
            scheduled = scheduled.union(to_schedule)

            if 'reassessment' in followups:
                # schedule reassessment for everyone who was not treated
                reassessment_schedules = followups['reassessment']
                to_schedule = untransitioned.intersection(idx)
                if isinstance(reassessment_schedules, FollowupDuration):
                    self.schedule_followup(to_schedule, 'reassessment',
                                           followups['reassessment'], visit_date, from_visit=visit_type)
                    scheduled = scheduled.union(to_schedule)
                elif isinstance(reassessment_schedules, list):  # list of ConditionalFollowups
                    pop = self.population_view.subview(['age']).get(untransitioned)
                    for cf in reassessment_schedules:
                        conditional_grp = (pop[pop.age.apply(lambda a: a in cf.age)].index
                                           .union(measured_sbp[measured_sbp.apply(lambda s: s in cf.measured_sbp)].index))
                        self.schedule_followup(conditional_grp.intersection(to_schedule), 'reassessment',
                                               cf.followup_duration, visit_date, from_visit=visit_type)
                        scheduled = scheduled.union(conditional_grp)
                else:  # guideline doesn't mandate any reassessment visits scheduled from this visit_type
                    pass

        # clear the scheduled followup date for simulants not scheduled for a
        # follow up so they will get properly processed on future background visits
        self.population_view.update(pd.Series(pd.NaT, index=measured_sbp.index.difference(scheduled),
                                              name='followup_date'))

    def send_to_icu(self, sbp, visit_date):
        icu_threshold = self.guideline_thresholds['icu']
        send_to_icu = sbp.loc[sbp >= icu_threshold].index

        _ = self.transition_treatment(send_to_icu, visit_date)
        # people sent to icu will always get their meds
        self.fill_prescriptions(send_to_icu, fully_adherent=True)
        self.schedule_followup(send_to_icu, 'maintenance', self.followup_schedules['intensive_care_unit']['maintenance'],
                               visit_date, 'intensive_care_unit')

        icu = (self.population_view.subview(['intensive_care_unit_visits_count'])
               .get(send_to_icu).loc[:, 'intensive_care_unit_visits_count'] + 1)
        self.population_view.update(icu)
        return send_to_icu

    def attend_background(self, index, visit_date):
        followups = self.followup_schedules['background']

        # check who we measure SBP for - send everyone else home w/o doing anything else
        sbp_measured_idx = self.randomness['sbp_measured'].filter_for_probability(index,
                                                                                  self.sbp_measurement_probability)
        # we don't want to record any SBP measurement for people already on hypertension visit plan
        no_followup_scheduled = (self.population_view.subview(['followup_date'])
                                 .get(sbp_measured_idx)['followup_date'].isnull())

        sbp = self.measure_sbp(sbp_measured_idx, idx_record_these=sbp_measured_idx[no_followup_scheduled],
                               idx_average_these=pd.Index([]), event_time=visit_date)

        # send everyone w/ sbp >= icu threshold to ICU and don't treat them further here
        sent_to_icu = self.send_to_icu(sbp, visit_date)
        # anyone who has a hypertension followup scheduled or was sent to the ICU
        # should not continue treatment on a background visit
        sbp = sbp.loc[no_followup_scheduled & (~sbp.index.isin(sent_to_icu))]

        # some guidelines have an immediate treatment threshold
        immediate_treatment_threshold = self.guideline_thresholds['immediate_tx']
        if immediate_treatment_threshold:
            immediately_treat = sbp.loc[sbp >= immediate_treatment_threshold].index
            treated = self.transition_treatment(immediately_treat, visit_date)
            self.schedule_followup(treated, 'maintenance', followups['maintenance'], visit_date,
                                   from_visit='background')
            self.fill_prescriptions(treated)
            sbp = sbp.loc[sbp < immediate_treatment_threshold]

        # schedule a confirmatory followup for everyone above controlled threshold for guideline
        uncontrolled = self.get_uncontrolled_population(sbp.index)
        if not uncontrolled.empty:
            self.schedule_followup(uncontrolled, 'confirmatory', followups['confirmatory'], visit_date,
                                   from_visit='background')

    def get_uncontrolled_population(self, index):
        pop = self.population_view.subview(['age', 'high_systolic_blood_pressure_measurement']).get(index)
        threshold = self.guideline_thresholds['controlled']
        if isinstance(threshold, (int, float)):
            uncontrolled = pop[pop.high_systolic_blood_pressure_measurement >= threshold].index
        else:  # list of (age interval threshold applies to, threshold)
            above_threshold = pd.Series(False, index=index)
            for ct in threshold:
                in_age = pop.age.apply(lambda a: a in ct[0])
                over_sbp = pop.high_systolic_blood_pressure_measurement >= ct[1]
                above_threshold |= (in_age & over_sbp)
            uncontrolled = pop[above_threshold].index
        return uncontrolled

    def transition_treatment(self, index, event_time):
        """Transition treatment for everyone who has a next available tx."""
        to_transition = self.treatment_profile_model.filter_for_next_valid_state(index)
        self.treatment_profile_model.transition(to_transition, event_time)
        return to_transition

    def fill_prescriptions(self, index, fully_adherent: bool = None):
        # prescription_adherence is a boolean pipeline with T for sims who filled prescription, F for sims who didn't
        rx_adherence = self.prescription_adherence(index) if fully_adherent is None else fully_adherent
        self._prescription_filled.loc[index] = rx_adherence

    def get_prescriptions_filled(self, index):
        return self._prescription_filled[index]

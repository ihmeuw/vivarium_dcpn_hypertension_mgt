import scipy
import numpy as np
import pandas as pd

from vivarium_dcpn_hypertension_mgt.components import Adherence


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

    @property
    def name(self):
        return 'hypertension_treatment_algorithm'

    def setup(self, builder):
        builder.components.add_components([Adherence(), HealthcareUtilization()])

        self.clock = builder.time.clock()

        self.sim_start = pd.Timestamp(**builder.configuration.time.start)
        columns_created = ['followup_date', 'followup_duration', 'followup_type']
        columns_required = ['treatment_profile']
        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 requires_columns=columns_required,
                                                 creates_columns=columns_created)
        self.population_view = builder.population.get_view(columns_required + columns_created)

        self.randomness = {'initial_followup_scheduling': builder.randomness.get_stream('initial_followup_scheduling'),
                           'background_visit_attendance': builder.randomness.get_stream('background_visit_attendance')}

        self.healthcare_utilization = builder.value.get_value('healthcare_utilization_rate')
        self.followup_adherence = builder.value.get_value('appt_followup.adherence')

        self._prescription_filled = pd.Series()
        self.prescription_filled = builder.value.register_value_producer('rx_fill.currently_filled',
                                                                         source=self.get_prescriptions_filled)

        builder.event.register_listener('time_step', self.on_time_step)

    def on_initialize_simulants(self, pop_data):
        sims_on_tx = (self.population_view.subview(['treatment_profile']).get(pop_data.index)
                      .query("treatment_profile != 'no_treatment_1'")).index

        followups = pd.DataFrame({'followup_date': pd.NaT, 'followup_duration': pd.NaT, 'followup_type': None},
                                 index=pop_data.index)

        followups.loc[sims_on_tx, ['followup_duration', 'followup_type']] = pd.Timedelta(days=6*28), 'maintenance'

        to_sim_date = np.vectorize(lambda d: self.sim_start + pd.Timedelta(days=d))
        np.random.seed(self.randomness['initial_followup_scheduling'].get_seed())
        followups.loc[sims_on_tx, 'followup_date'] = to_sim_date(np.random.random_integers(low=0, high=6*28,
                                                                                           size=len(sims_on_tx)))
        self._prescription_filled = self._prescription_filled.append(pd.Series(0, pop_data.index))
        self._prescription_filled.loc[sims_on_tx] = 1

        self.population_view.update(followups)

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

        self.attend_background(background_attending)

    def reschedule_followup(self, index):
        followups = self.population_view.subview(['followup_date', 'followup_duration']).get(index).copy()
        followups.followup_date = followups.followup_date + followups.followup_duration
        self.population_view.update(followups)
        self._prescription_filled.loc[index] = 0

    def attend_followup(self, index):
        pass

    def attend_background(self, index):
        pass

    def get_prescriptions_filled(self, index):
        return self._prescription_filled[index]

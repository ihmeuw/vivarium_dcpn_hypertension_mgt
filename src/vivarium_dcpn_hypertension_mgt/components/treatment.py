import scipy
import pandas as pd


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
        return pd.Series(distributions.ppf(self._propensity[index]), index=index)

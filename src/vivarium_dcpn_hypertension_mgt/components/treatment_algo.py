# from loguru import logger
import pandas as pd
import numpy as np
import scipy
from vivarium.framework.engine import Builder
from vivarium.framework.engine import Event

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
        return "high_systolic_blood_pressure"

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
        hypertension_measurement = self.true_exposure(idx_measure) + noise
        hypertension_measurement.loc[detect_zero.index] = 0.0

        if measure_type_average:
            hypertension_measurement = (hypertension_measurement +
                                        self.population_view.get(idx_measure)[self.measurement_column]) / 2

        measurement = self.population_view.get(idx_measure)[self.measurement_column]
        measurement.loc[idx_record_these] = hypertension_measurement
        self.population_view.update(measurement)

        return hypertension_measurement


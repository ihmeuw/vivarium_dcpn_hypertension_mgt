# from loguru import logger
import pandas as pd
from vivarium.framework.engine import Builder

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

        builder.value.register_value_producer('rx_fill', source=self.get_rx_fill)
        builder.value.register_value_producer('appt_followup', source=self.get_appt_followup)

    def on_initialize_simulants(self, pop_data):
        # bin everyone into 1 of the 4 adherence catagories
        self.df_ad_data = self.rand_threshold_creation.choice(pop_data.index, self._thresholds.index,
                                                              self._thresholds.proportion)

    def get_rx_fill(self, index) -> pd.Series:
        return self.df_ad_data.loc[self.rand_rx_fill.get_draw(index) < self.df_ad_data.rx].index

    def get_appt_followup(self, index) -> pd.Series:
        pass


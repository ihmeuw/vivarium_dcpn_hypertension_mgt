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
        mask = self.rand_threshold_creation.choice(pop_data.index, self._thresholds.index, self._thresholds.proportion)
        self.df_ad_data['rx_ad'] = mask.apply(lambda x: self._thresholds.loc[x].prescription_fill)
        self.df_ad_data['appt_ad'] = mask.apply(lambda x: self._thresholds.loc[x].follow_up)

    def get_rx_fill(self, index) -> pd.Series:
        return self.rand_rx_fill.get_draw(index) < self.df_ad_data.rx_ad

    def get_appt_followup(self, index) -> pd.Series:
        return self.rand_appt_followup.get_draw(index) < self.df_ad_data.appt_ad


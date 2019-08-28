import pandas as pd

from vivarium_public_health.utilities import EntityString


class DummyRisk:
    def __init__(self, risk, default_value):
        self.risk = EntityString(risk)
        self.default = float(default_value)

    @property
    def name(self):
        return f'dummy_risk.{self.risk}'

    def setup(self, builder):
        builder.value.register_value_producer(f'{self.risk.name}.exposure',
                                              source=lambda index: pd.Series(self.default, index=index))

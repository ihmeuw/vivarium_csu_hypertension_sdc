import numpy as np
import pandas as pd

from vivarium_public_health.utilities import EntityString


class DummyContinuousRisk:
    def __init__(self, risk, low, high):
        self.risk = EntityString(risk)
        self.low = int(low)
        self.high = int(high)
        self.exp = pd.Series()

    @property
    def name(self):
        return f'dummy_risk.{self.risk}'

    def setup(self, builder):
        builder.population.initializes_simulants(self.on_initialize_simulants)
        builder.value.register_value_producer(f'{self.risk.name}.exposure',
                                              source=lambda index: self.exp.loc[index])

    def on_initialize_simulants(self, pop_data):
        s = pd.Series(np.random.randint(self.low, self.high, size=len(pop_data.index)), index=pop_data.index)
        assert np.all(s > 0)
        self.exp = self.exp.append(s)


class DummyAdherence:
    def __init__(self, adherence_value):
        self.adherence = float(adherence_value)

    @property
    def name(self):
        return f'dummy_adherence({self.adherence})'

    def setup(self, builder):
        self.adherence_pipeline = builder.value.register_value_producer('hypertension_meds.pdc',
                                              source=lambda index: pd.Series(self.adherence, index=index))

        builder.value.register_value_modifier('hypertension_meds.effect_size', self.modify_meds_effect)

    def modify_meds_effect(self, index, effect_size):
        return effect_size * self.adherence_pipeline(index)
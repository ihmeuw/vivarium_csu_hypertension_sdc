from vivarium_public_health.risks import RiskEffect
from vivarium_public_health.risks.data_transformations import (get_relative_risk_data,
                                                               get_population_attributable_fraction_data)
from vivarium_public_health.utilities import TargetString


class BetterRiskEffect(RiskEffect):

    def load_relative_risk_data(self, builder):
        if self.target.measure == 'incidence_rate':
            return super().load_relative_risk_data(builder)
        elif self.target.measure == 'relapse_rate':
            hacked_target = TargetString(f'{self.target.type}.{self.target.name}.incidence_rate')
            return get_relative_risk_data(builder, self.risk, hacked_target, self.randomness)

    def load_population_attributable_fraction_data(self, builder):
        if self.target.measure == 'incidence_rate':
            return super().load_population_attributable_fraction_data(builder)
        elif self.target.measure == 'relapse_rate':
            hacked_target = TargetString(f'{self.target.type}.{self.target.name}.incidence_rate')
            return get_population_attributable_fraction_data(builder, self.risk, hacked_target, self.randomness)

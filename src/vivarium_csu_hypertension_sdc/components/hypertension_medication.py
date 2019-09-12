import numpy as np
import pandas as pd

from vivarium_csu_hypertension_sdc.components import utilities
from vivarium_csu_hypertension_sdc.components.globals import (HYPERTENSION_DRUGS, HYPERTENSIVE_CONTROLLED_THRESHOLD)


class BaselineCoverage:

    def __init__(self):
        self._dosage_columns = [f'{d}_dosage' for d in HYPERTENSION_DRUGS]
        self._single_pill_columns = [f'{d}_in_single_pill' for d in HYPERTENSION_DRUGS]

    @property
    def name(self):
        return 'baseline_coverage'

    def setup(self, builder):
        self.coverage = (builder.data.load('health_technology.hypertension_medication.treatment_coverage')
                         .set_index('measure').value)
        self.therapy_categories = (builder.data.load('health_technology.hypertension_medication.therapy_category')
                                   .set_index('therapy_category').value)
        self.med_probabilities = builder.data.load('health_technology.hypertension_medication.medication_probabilities')

        self.proportion_above_hypertensive_threshold = builder.lookup.build_table(
            builder.data.load('risk_factor.high_systolic_blood_pressure.proportion_above_hypertensive_threshold'))

        sbp = builder.value.get_value('high_systolic_blood_pressure.exposure')
        self.raw_sbp = lambda index: pd.Series(sbp.source(index), index=index)

        self.randomness = builder.randomness.get_stream('initial_treatment')

        self.population_view = builder.population.get_view(self._dosage_columns + self._single_pill_columns)
        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 creates_columns=(self._dosage_columns + self._single_pill_columns),
                                                 requires_columns=[],)
                                                 #requires_values=['high_systolic_blood_pressure.exposure'])

    def on_initialize_simulants(self, pop_data):
        medications = pd.DataFrame(0, columns=self._dosage_columns + self._single_pill_columns, index=pop_data.index)

        initial_tx_cats = self.get_initial_treatment_category(pop_data.index)

        initially_treated = initial_tx_cats[initial_tx_cats != 'none']
        cat_groups = initially_treated.groupby(initially_treated).apply(lambda g: g.index)

        # choose drug/pill combination first
        drugs = pd.DataFrame(columns=HYPERTENSION_DRUGS + self._single_pill_columns, index=initially_treated.index)
        for cat, idx in cat_groups.iteritems():
            drugs.loc[idx] = utilities.get_initial_drugs_given_category(self.med_probabilities, cat,
                                                                        idx, self.randomness)

        # then select dosages
        drugs.loc[:, HYPERTENSION_DRUGS] = utilities.get_initial_dosages(drugs, self.randomness)

        medications.loc[initially_treated.index] = drugs.rename(columns={d: f'{d}_dosage' for d in HYPERTENSION_DRUGS})

        self.population_view.update(medications)

    def get_initial_treatment_category(self, index):
        raw_sbp = self.raw_sbp(index)
        below_threshold = raw_sbp[raw_sbp < HYPERTENSIVE_CONTROLLED_THRESHOLD].index
        above_threshold = raw_sbp[raw_sbp >= HYPERTENSIVE_CONTROLLED_THRESHOLD].index

        category_prob_below_threshold, cat_names = utilities.probability_treatment_category_given_sbp_level('below_threshold',
                                                            self.proportion_above_hypertensive_threshold(below_threshold),
                                                            self.coverage, self.therapy_categories)
        category_prob_above_threshold, cat_names = utilities.probability_treatment_category_given_sbp_level('above_threshold',
                                                            self.proportion_above_hypertensive_threshold(above_threshold),
                                                            self.coverage, self.therapy_categories)

        cat_probabilities = np.stack(pd.concat([category_prob_below_threshold, category_prob_above_threshold])
                                     .sort_index().values, axis=0)

        category_choices = self.randomness.choice(index, choices=cat_names, p=cat_probabilities,
                                                  additional_key='treatment_category')
        return category_choices


from scipy import stats

import numpy as np
import pandas as pd

from vivarium_csu_hypertension_sdc.components import utilities, data_transformations
from vivarium_csu_hypertension_sdc.components.globals import (HYPERTENSION_DRUGS, HYPERTENSIVE_CONTROLLED_THRESHOLD,
                                                              SINGLE_PILL_COLUMNS, DOSAGE_COLUMNS, MIN_PDC_FOR_ADHERENT)


class BaselineCoverage:

    def __init__(self):
        self.pdc_dist_for_adherent = stats.uniform(loc=MIN_PDC_FOR_ADHERENT, scale=(1 - MIN_PDC_FOR_ADHERENT))
        self.pdc_dist_for_non_adherent = stats.uniform(loc=0, scale=MIN_PDC_FOR_ADHERENT)

    @property
    def name(self):
        return 'baseline_coverage'

    def setup(self, builder):
        self.coverage = (builder.data.load('health_technology.hypertension_medication.treatment_coverage')
                         .set_index('measure').value)
        self.therapy_categories = (builder.data.load('health_technology.hypertension_medication.therapy_category')
                                   .set_index('therapy_category').value)
        self.med_probabilities = builder.data.load('health_technology.hypertension_medication.medication_probabilities')
        self.adherent_thresholds = data_transformations.load_adherent_thresholds(builder)

        proportion_data = builder.data.load('risk_factor.high_systolic_blood_pressure.proportion_above_hypertensive_threshold')

        proportion_data.value += 0.0005  # Account for discrepencies between ppf and cdf functions in sbp dist.
        self.proportion_above_hypertensive_threshold = builder.lookup.build_table(proportion_data,
                                                                                  key_columns=['sex'],
                                                                                  parameter_columns=['age', 'year'])

        self.raw_sbp = builder.value.get_value('high_systolic_blood_pressure.exposure').source

        self.randomness = builder.randomness.get_stream('initial_treatment')

        adherence_columns = ['adherent_propensity', 'pdc_propensity']
        self.population_view = builder.population.get_view(DOSAGE_COLUMNS + SINGLE_PILL_COLUMNS + adherence_columns)
        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 creates_columns=(DOSAGE_COLUMNS + SINGLE_PILL_COLUMNS
                                                                  + adherence_columns),
                                                 requires_columns=[],
                                                 requires_values=['high_systolic_blood_pressure.propensity'],
                                                 requires_streams=['initial_treatment'])

        self.pdc = builder.value.register_value_producer('hypertension_meds.pdc', self.get_pdc,
                                                         requires_columns=(DOSAGE_COLUMNS + SINGLE_PILL_COLUMNS
                                                                             + adherence_columns),
                                                         requires_values=[])
        builder.value.register_value_modifier('hypertension_meds.effect_size', self.modify_meds_effect_with_pdc)

    def on_initialize_simulants(self, pop_data):
        medications = pd.DataFrame(0, columns=DOSAGE_COLUMNS + SINGLE_PILL_COLUMNS, index=pop_data.index)

        initial_tx_cats = self.get_initial_treatment_category(pop_data.index)

        initially_treated = initial_tx_cats[initial_tx_cats != 'none']
        cat_groups = initially_treated.groupby(initially_treated).apply(lambda g: g.index)

        # choose drug/pill combination first
        drugs = pd.DataFrame(columns=HYPERTENSION_DRUGS + SINGLE_PILL_COLUMNS, index=initially_treated.index, dtype=float)
        for cat, idx in cat_groups.iteritems():
            drugs.loc[idx] = utilities.get_initial_drugs_given_category(self.med_probabilities, cat,
                                                                        idx, self.randomness)
        # then select dosages
        drugs.loc[:, HYPERTENSION_DRUGS] = utilities.get_initial_dosages(drugs, self.randomness)

        medications.loc[initially_treated.index] = drugs.rename(columns={d: f'{d}_dosage' for d in HYPERTENSION_DRUGS})

        medications['adherent_propensity'] = self.randomness.get_draw(pop_data.index, 'adherent_propensity')
        medications['pdc_propensity'] = self.randomness.get_draw(pop_data.index, 'pdc_propensity')

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

    def get_pdc(self, index):
        pop = self.population_view.get(index)

        thresholds = {k: v(index) for k, v in self.adherent_thresholds.items()}
        num_pills = utilities.get_num_pills(pop.loc[:, DOSAGE_COLUMNS + SINGLE_PILL_COLUMNS])
        # pdc is 0 for everyone not on treatment and everyone on treatment but not adherent
        pdc = pd.Series(0, index=index)

        on_tx = num_pills > 0
        pill_cats = num_pills.loc[on_tx].apply(lambda n: 'multiple' if n > 1 else 'single')
        pop = pop.loc[on_tx]

        for cat, threshold in thresholds.items():
            pop_in_cat = pop.loc[pill_cats == cat]
            adherent = pop_in_cat.adherent_propensity <= threshold.loc[pop_in_cat.index, 'value']
            pdc.loc[pop_in_cat[adherent].index] = self.pdc_dist_for_adherent.ppf(pop_in_cat.loc[adherent, 'pdc_propensity'])

        return pdc

    def modify_meds_effect_with_pdc(self, index, effect_size):
        return effect_size * self.pdc(index)


class TreatmentEffect:

    @property
    def name(self):
        return 'hypertension_meds_treatment_effect'

    def setup(self, builder):
        self.drugs = HYPERTENSION_DRUGS
        self.efficacy_data = data_transformations.load_efficacy_data(builder).reset_index()

        self.drug_efficacy = pd.Series(index=pd.MultiIndex(levels=[[], [], []],
                                                                 codes=[[], [], []],
                                                                 names=['simulant', 'drug', 'dosage']))

        self.shift_column = 'hypertension_meds_baseline_shift'

        self.population_view = builder.population.get_view([self.shift_column] + DOSAGE_COLUMNS)
        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 creates_columns=[self.shift_column],
                                                 requires_columns=DOSAGE_COLUMNS,
                                                 requires_streams=['dose_efficacy'])

        self.randomness = builder.randomness.get_stream('dose_efficacy')
        self.drug_effects = {m: builder.value.register_value_producer(f'{m}.effect_size', self.get_drug_effect)
                             for m in self.drugs}

        self.treatment_effect = builder.value.register_value_producer('hypertension_meds.effect_size',
                                                                      self.get_treatment_effect,
                                                                      requires_columns=DOSAGE_COLUMNS)

        builder.value.register_value_modifier('high_systolic_blood_pressure.exposure', self.treat_sbp,
                                              requires_columns=[self.shift_column],
                                              requires_values=['hypertension_meds.effect_size'])

    def on_initialize_simulants(self, pop_data):
        self.drug_efficacy = self.drug_efficacy.append(self.determine_drug_efficacy(pop_data.index))
        effects = self.treatment_effect(pop_data.index)
        effects.name = self.shift_column
        self.population_view.update(effects)

    def determine_drug_efficacy(self, index):
        efficacy = []

        for drug in self.drugs:
            efficacy_draw = self.randomness.get_draw(index, additional_key=drug)
            med_efficacy = self.efficacy_data.query('drug == @drug')
            for dose in med_efficacy.dosage.unique():
                dose_efficacy_parameters = med_efficacy.loc[med_efficacy.dosage == dose, ['value', 'individual_sd']].values[0]
                dose_index = pd.MultiIndex.from_product((index, [drug], [dose]),
                                                        names=('simulant', 'drug', 'dosage'))
                if dose_efficacy_parameters[1] == 0.0:  # if sd is 0, no need to draw
                    dose_efficacy = pd.Series(dose_efficacy_parameters[0], index=dose_index)
                else:
                    dose_efficacy = pd.Series(stats.norm.ppf(efficacy_draw, loc=dose_efficacy_parameters[0],
                                                             scale=dose_efficacy_parameters[1]),
                                              index=dose_index)
                efficacy.append(dose_efficacy)

        return pd.concat(efficacy)

    def get_drug_effect(self, dosages, drug):
        lookup_index = pd.MultiIndex.from_arrays((dosages.index.values,
                                                  np.tile(drug, len(dosages)),
                                                  dosages.values),
                                                 names=('simulant', 'drug', 'dosage'))

        efficacy = self.drug_efficacy.loc[lookup_index]
        efficacy.index = efficacy.index.droplevel(['drug', 'dosage'])
        return efficacy

    def get_treatment_effect(self, index):
        prescribed_meds = self.population_view.subview(DOSAGE_COLUMNS).get(index)
        return sum([self.get_drug_effect(prescribed_meds[f'{d}_dosage'], d) for d in self.drugs])

    def treat_sbp(self, index, exposure):
        baseline_shift = self.population_view.subview([self.shift_column]).get(index)[self.shift_column]
        return exposure + baseline_shift - self.treatment_effect(index)

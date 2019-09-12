from itertools import combinations

import numpy as np
import pandas as pd

from vivarium.framework.randomness import RandomnessStream
from vivarium_csu_hypertension_sdc.components.globals import (HYPERTENSION_DRUGS, HYPERTENSION_DOSAGES,
                                                              ILLEGAL_DRUG_COMBINATION)


def probability_treatment_category_given_sbp_level(sbp_level: str, proportion_high_sbp: pd.DataFrame,
                                                   coverage: pd.Series, categories: pd.Series) -> (pd.Series, list):
    controlled_among_treated = coverage.loc['controlled_among_treated']
    treated_among_hypertensive = coverage.loc['treated_among_hypertensive']
    hypertensive = proportion_high_sbp / (1 - controlled_among_treated * treated_among_hypertensive)

    if sbp_level == 'below_threshold':
        prob_treated = (controlled_among_treated * treated_among_hypertensive
                        * hypertensive / (1 - proportion_high_sbp))

    elif sbp_level == 'above_threshold':
        prob_treated = ((1 - controlled_among_treated) * treated_among_hypertensive
                        * hypertensive / proportion_high_sbp)

    else:
        raise ValueError(f'The only acceptable sbp levels are "below_threshold" or "above_threshold". '
                         f'You provided {sbp_level}.')

    # treatment_categories are: {'mono', 'dual', '3+', 'none'}
    category_names = categories.index.to_list() + ['none']

    def get_category_probabilities(p_treated):
        p_profile = p_treated * categories.values
        p_profile[-1] = 1.0 - np.sum(p_profile)
        return p_profile

    prob_categories = prob_treated.apply(get_category_probabilities)

    return prob_categories, category_names


def get_mono_dosages(mono_index: pd.Index, med_probabilities: pd.DataFrame,
                     randomness: RandomnessStream) -> pd.DataFrame:
    mono_options = med_probabilities.loc[med_probabilities.measure == 'individual_drug_probability']
    # normalize probabilities to sum to 1
    mono_options['value'] /= mono_options['value'].sum()
    assert len(mono_options) == len(HYPERTENSION_DRUGS), "Medication probabilities don't line up with set of drugs."

    drug_choices = randomness.choice(mono_index, choices=mono_options.index, p=mono_options.value,
                                     additional_key='drug_choice')
    chosen_drugs = mono_options.loc[drug_choices, HYPERTENSION_DRUGS].set_index(mono_index)

    dosage_choices = randomness.choice(mono_index, choices=HYPERTENSION_DOSAGES,
                                       additional_key='dosage_choice')

    choices = chosen_drugs.multiply(dosage_choices, axis=0)
    return choices.rename(columns={c: f'{c}_dosage' for c in HYPERTENSION_DRUGS})


def get_single_pill_combinations(med_probabilities: pd.DataFrame, num_drugs_in_profile: int) -> pd.DataFrame:
    """Profiles consisting of num_drugs_in_profile drugs, all packaged into 1 pill."""
    drug_combinations = med_probabilities.loc[(med_probabilities.measure == 'single_pill_combination_probability') &
                                              (med_probabilities[HYPERTENSION_DRUGS].sum(axis=1) == num_drugs_in_profile)]
    drug_combinations = pd.concat(
        [drug_combinations, pd.DataFrame(0, columns=[f'{d}_in_single_pill' for d in HYPERTENSION_DRUGS],
                                         index=drug_combinations.index)], axis=1)

    for row in drug_combinations.iterrows():
        drugs = row[1][HYPERTENSION_DRUGS]
        in_pill = [f'{d}_in_single_pill' for d in drugs[drugs > 0].index]
        drug_combinations.loc[row[0], in_pill] = 1

    return drug_combinations


def get_individual_pill_combinations(med_probabilities: pd.DataFrame, num_drugs_in_profile: int) -> pd.DataFrame:
    """Profiles consisting of num_drugs_in_profile pills."""
    drug_combinations = pd.DataFrame(columns=HYPERTENSION_DRUGS + [f'{d}_in_single_pill' for d in HYPERTENSION_DRUGS])
    individual_drug_probs = med_probabilities.loc[med_probabilities.measure == 'individual_drug_probability']

    for c in combinations(HYPERTENSION_DRUGS, num_drugs_in_profile):
        if not ILLEGAL_DRUG_COMBINATION.issubset(c):
            total_prob = sum([individual_drug_probs.loc[individual_drug_probs[d] == 1, 'value'].values[0] for d in c])
            combo_drugs = {d: 1 for d in c}
            combo_drugs['value'] = total_prob
            drug_combinations = drug_combinations.append(combo_drugs, ignore_index=True)

    return drug_combinations


def get_single_pill_individual_pill_combinations(med_probabilities: pd.DataFrame,
                                                 num_drugs_in_profile: int) -> pd.DataFrame:
    """Profiles consisting of 1 pill with (num_drugs_in_profile - 1) drugs
    packaged into it + 1 additional pill with another drug."""
    drug_combinations = pd.DataFrame(columns=HYPERTENSION_DRUGS + [f'{d}_in_single_pill' for d in HYPERTENSION_DRUGS])
    individual_drug_probs = med_probabilities.loc[med_probabilities.measure == 'individual_drug_probability']

    single_pills_in_combo = med_probabilities.loc[(med_probabilities.measure == 'single_pill_combination_probability') &
                                                  ((num_drugs_in_profile - med_probabilities[HYPERTENSION_DRUGS].sum(
                                                      axis=1)) > 0)]

    # assuming we only ever have 1 single pill + 1 other drug in a combo

    for row in single_pills_in_combo.iterrows():
        drugs = row[1][HYPERTENSION_DRUGS]
        drugs_in_single_pill = list(drugs.loc[drugs > 0].index)

        for drug in set(HYPERTENSION_DRUGS).difference(drugs_in_single_pill):
            combo = set(drugs_in_single_pill + [drug])
            if not ILLEGAL_DRUG_COMBINATION.issubset(combo):
                total_prob = row[1]['value'] * \
                             individual_drug_probs.loc[individual_drug_probs[drug] == 1, 'value'].values[0]
                combo_drugs = {d: 1 for d in combo}
                combo_drugs['value'] = total_prob
                drug_combinations = drug_combinations.append(combo_drugs, ignore_index=True)

    return drug_combinations


def generate_category_drug_combos(med_probabilities: pd.DataFrame, category: str) -> pd.DataFrame:
    category_num_drugs = {'mono': 1, 'dual': 2, '3+': 3}
    num_drugs_in_profile = category_num_drugs[category]

    single_pill_combos = get_single_pill_combinations(med_probabilities, num_drugs_in_profile)
    individual_pill_combos = get_individual_pill_combinations(med_probabilities, num_drugs_in_profile)
    single_individual_pill_combos = get_single_pill_individual_pill_combinations(med_probabilities,
                                                                                 num_drugs_in_profile)

    drug_combinations = pd.concat([single_pill_combos, individual_pill_combos, single_individual_pill_combos], axis=0)

    drug_combinations = drug_combinations.reset_index(drop=True).fillna(0)

    # normalize probabilities so they sum to 1
    drug_combinations['value'] /= drug_combinations['value'].sum()

    return drug_combinations


def get_dosages_for_num_pills(drugs: pd.DataFrame, num_pills: int, randomness: RandomnessStream) -> pd.DataFrame:
    if num_pills == 1:
        dosages = randomness.choice(drugs.index, HYPERTENSION_DOSAGES, additional_key='dosage_choice')
        drugs.loc[HYPERTENSION_DRUGS] *= dosages
    elif num_pills == 2:
        dose_combinations = [c for c in combinations(HYPERTENSION_DOSAGES * num_pills, num_pills)]
        dosages = randomness.choice(drugs.index, dose_combinations, additional_key='dosage_choice')
        drugs.loc[]

import pandas as pd


def load_efficacy_data(builder) -> pd.DataFrame:
    efficacy_data = builder.data.load('health_technology.hypertension_medication.drug_efficacy')

    zero_dosage = efficacy_data.loc[efficacy_data.dosage == 0.5].copy()
    zero_dosage.dosage = 0.0
    zero_dosage.individual_sd = 0.0
    zero_dosage.value = 0.0

    efficacy_data = pd.concat([zero_dosage, efficacy_data])
    return efficacy_data.set_index(['dosage', 'drug'])


def load_adherent_thresholds(builder) -> dict:
    data = builder.data.load('health_technology.hypertension_medication.adherence')

    pill_categories = ['single', 'multiple']
    adherence_data = {c: builder.lookup.build_table(data.loc[data.pill_category == c],
                                                    parameter_columns=['age'])
                      for c in pill_categories}
    return adherence_data

import pandas as pd


def load_efficacy_data(builder) -> pd.DataFrame:
    efficacy_data = builder.data.load('health_technology.hypertension_medication.drug_efficacy')

    zero_dosage = efficacy_data.loc[efficacy_data.dosage == 0.5].copy()
    zero_dosage.dosage = 0.0
    zero_dosage.individual_sd = 0.0
    zero_dosage.value = 0.0

    efficacy_data = pd.concat([zero_dosage, efficacy_data])
    return efficacy_data.set_index(['dosage', 'drug'])
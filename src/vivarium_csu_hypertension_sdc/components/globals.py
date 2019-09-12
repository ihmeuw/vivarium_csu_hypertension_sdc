HYPERTENSION_DRUGS = ['thiazide_type_diuretics', 'beta_blockers', 'ace_inhibitors',
                      'angiotensin_ii_blockers', 'calcium_channel_blockers']

HYPERTENSION_DOSAGES = [0.5, 1.0, 2.0]

HYPERTENSIVE_CONTROLLED_THRESHOLD = 140

ILLEGAL_DRUG_COMBINATION = {'ace_inhibitors', 'angiotensin_ii_blockers'}

SINGLE_PILL_COLUMNS = [f'{d}_in_single_pill' for d in HYPERTENSION_DRUGS]

DOSAGE_COLUMNS = [f'{d}_dosage' for d in HYPERTENSION_DRUGS]

ADHERENT_THRESHOLD = 0.8

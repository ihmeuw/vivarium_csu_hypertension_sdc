from scipy import stats

import numpy as np
import pandas as pd

from vivarium_csu_hypertension_sdc.components import utilities
from vivarium_csu_hypertension_sdc.components.globals import (DOSAGE_COLUMNS, HYPERTENSIVE_CONTROLLED_THRESHOLD,
                                                              SINGLE_PILL_COLUMNS, HYPERTENSION_DOSAGES,
                                                              HYPERTENSION_DRUGS, ILLEGAL_DRUG_COMBINATION)


class TreatmentAlgorithm:
    configuration_defaults = {
        'hypertension_treatment': {
            'high_systolic_blood_pressure_measurement': {
                'error_sd': 6,
            },
            'therapeutic_inertia': {
                'mean': 0.136,
                'sd': 0.0136,
            },
            'adverse_events': {
                'mean': 0.1373333333,
                'sd': 0.03069563849,
            },
            'followup_visit_interval': 90,  # days
            'treatment_ramp': 'low_and_slow'  # one of ["low_and_slow", "free_choice", "fixed_dose_combination"]
        }
    }

    @property
    def name(self):
        return 'hypertension_treatment_algorithm'

    def setup(self, builder):
        self.clock = builder.time.clock()

        self.sim_start = pd.Timestamp(**builder.configuration.time.start)

        self.config = builder.configuration.hypertension_treatment

        self.followup_visit_interval_days = self.config.followup_visit_interval

        self.med_probabilities = builder.data.load('health_technology.hypertension_medication.medication_probabilities')

        columns_created = ['followup_date', 'last_visit_date', 'last_visit_type',
                           'high_systolic_blood_pressure_measurement',
                           'high_systolic_blood_pressure_last_measurement_date']
        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 requires_columns=DOSAGE_COLUMNS,
                                                 creates_columns=columns_created,
                                                 requires_streams=['followup_scheduling'])
        self.population_view = builder.population.get_view(DOSAGE_COLUMNS + columns_created + ['alive'],
                                                           query='alive == "alive"')

        self.randomness = {'followup_scheduling': builder.randomness.get_stream('followup_scheduling'),
                           'background_visit_attendance': builder.randomness.get_stream('background_visit_attendance'),
                           'sbp_measurement': builder.randomness.get_stream('sbp_measurement'),
                           'therapeutic_inertia': builder.randomness.get_stream('therapeutic_intertia'),
                           'treatment_transition': builder.randomness.get_stream('treatment_transition')
                           }

        self.ti_probability = utilities.get_therapeutic_inertia_probability(self.config.therapeutic_inertia.mean,
                                                                            self.config.therapeutic_inertia.sd,
                                                                            self.randomness['therapeutic_inertia'])

        self.utilization_data = builder.lookup.build_table(
            builder.data.load('healthcare_entity.outpatient_visits.utilization_rate'))
        self.healthcare_utilization = builder.value.register_rate_producer('healthcare_utilization_rate',
                                                                           source=lambda index: self.utilization_data(index))

        self.sbp = builder.value.get_value('high_systolic_blood_pressure.exposure')

        builder.event.register_listener('time_step', self.on_time_step)

    def on_initialize_simulants(self, pop_data):
        drug_dosages = self.population_view.subview(DOSAGE_COLUMNS).get(pop_data.index)
        sims_on_tx = drug_dosages.loc[drug_dosages.sum(axis=1) > 0].index

        initialize = pd.DataFrame({'followup_date': pd.NaT, 'last_visit_date': pd.NaT, 'last_visit_type': None,
                                   'high_systolic_blood_pressure_measurement': np.nan,
                                   'high_systolic_blood_pressure_last_measurement_date': pd.NaT},
                                  index=pop_data.index)

        durations = utilities.get_days_in_range(self.randomness['followup_scheduling'],
                                                low=0, high=self.config.followup_visit_interval,
                                                index=sims_on_tx)
        initialize.loc[sims_on_tx, 'followup_date'] = durations + self.sim_start
        initialize.loc[sims_on_tx, 'last_visit_date'] = (self.sim_start
                                                         - pd.Timedelta(self.config.followup_visit_interval))
        self.population_view.update(initialize)

    def on_time_step(self, event):
        pop = self.population_view.get(event.index)

        followup_scheduled = (self.clock() < pop.followup_date) & (pop.followup_date <= event.time)

        self.attend_followup(pop.index[followup_scheduled], event.time)
        pop.loc[followup_scheduled, 'last_visit_type'] = 'follow_up'

        background_eligible = pop.index[~followup_scheduled]
        background_attending = (self.randomness['background_visit_attendance']
                                .filter_for_rate(background_eligible,
                                                 self.healthcare_utilization(background_eligible).values))

        self.attend_background(background_attending, event.time)
        pop.loc[background_attending, 'last_visit_type'] = 'background'

        pop.loc[background_attending.union(pop[followup_scheduled].index), 'last_visit_date'] = event.time
        self.population_view.update(pop.loc[:, ['last_visit_type', 'last_visit_date']])

    def attend_followup(self, index, visit_date):
        sbp_measurements = self.measure_sbp(index, visit_date)
        eligible_for_tx_mask = sbp_measurements >= HYPERTENSIVE_CONTROLLED_THRESHOLD

        treatment_increase_possible = self.check_treatment_increase_possible(index[eligible_for_tx_mask])

        lost_to_ti = self.randomness['therapeutic_inertia'].filter_for_probability(treatment_increase_possible,
                                                                                   np.tile(self.ti_probability,
                                                                                           len(treatment_increase_possible)),
                                                                                   additional_key='lost_to_ti')
        self.transition_treatment(treatment_increase_possible.difference(lost_to_ti))

        self.schedule_followup(index, visit_date)  # everyone rescheduled no matter whether their tx changed or not

    def attend_background(self, index, visit_date):
        sbp_measurements = self.measure_sbp(index, visit_date)
        followup_dates = self.population_view.subview(['followup_date']).get(index).loc[:, 'followup_date']

        eligible_for_tx_mask = (sbp_measurements >= HYPERTENSIVE_CONTROLLED_THRESHOLD) & (followup_dates.isna())

        lost_to_ti = self.randomness['therapeutic_inertia'].filter_for_probability(index[eligible_for_tx_mask],
                                                                                   np.tile(self.ti_probability,
                                                                                           sum(eligible_for_tx_mask)),
                                                                                   additional_key='lost_to_ti')
        start_tx = index[eligible_for_tx_mask].difference(lost_to_ti)
        self.transition_treatment(start_tx)
        self.schedule_followup(start_tx, visit_date)  # schedule only for those who started tx

    def measure_sbp(self, index, visit_date):
        true_exp = self.sbp(index)
        assert np.all(true_exp > 0), '0 values detected for high systolic blood pressure exposure. ' \
                                     'Verify your age ranges are valid for this risk factor.'

        draw = self.randomness['sbp_measurement'].get_draw(index)

        measurement_error = self.config.high_systolic_blood_pressure_measurement.error_sd
        noise = stats.norm.ppf(draw, scale=measurement_error) if measurement_error else 0
        sbp_measurement = true_exp + noise

        updates = pd.DataFrame(sbp_measurement, columns=['high_systolic_blood_pressure_measurement'])
        updates['high_systolic_blood_pressure_last_measurement_date'] = visit_date

        self.population_view.update(updates)

        return sbp_measurement

    def transition_treatment(self, index):
        if self.config.treatment_ramp == "low_and_slow":
            new_meds = self.transition_low_and_slow(index)

        self.population_view.update(new_meds)

    def check_treatment_increase_possible(self, index):
        dosages = self.population_view.subview(DOSAGE_COLUMNS).get(index)
        if self.config.treatment_ramp == 'low_and_slow':
            no_tx_increase_mask = dosages.sum(axis=1) >= 3 * max(HYPERTENSION_DOSAGES)  # 3 drugs, each on a max dosage

        return index[~no_tx_increase_mask]

    def schedule_followup(self, index, visit_date):
        next_followup_date = pd.Series(visit_date + pd.Timedelta(days=self.config.followup_visit_interval),
                                       index=index, name='followup_date')
        self.population_view.update(next_followup_date)

    def choose_half_dose_new_drug(self, index, current_dosages):

        options = (self.med_probabilities.loc[self.med_probabilities.measure == 'individual_drug_probability']
                   .rename(columns={d: f'{d}_dosage' for d in HYPERTENSION_DRUGS}))

        # we need to customize the probabilities for each sim so the prob of any drug they're already on is 0
        probs = options[DOSAGE_COLUMNS].multiply(options.value, axis=0).sum(axis=0).to_frame().transpose()
        probs = pd.DataFrame(np.tile(probs, (len(index), 1)), columns=probs.columns, index=index)
        probs *= np.logical_not(current_dosages.mask(current_dosages < 0, 1))

        # make sure we don't end up with ACE/ARB combo
        illegal = list(ILLEGAL_DRUG_COMBINATION)
        probs.loc[(probs[f'{illegal[0]}_dosage'] == 0) | (probs[f'{illegal[1]}_dosage'] == 0),
                  [f'{d}_dosage' for d in illegal]] = 0

        probs = probs.divide(probs.sum(axis=1), axis=0)  # normalize so each sim's probs sum to 1

        drug_to_idx_map = {d: options.loc[options[d] == 1].index[0] for d in probs.columns}
        chosen_drugs = self.randomness['treatment_transition'].choice(index, probs.columns,
                                                                      p=probs.values).map(drug_to_idx_map)
        new_dosages = options.loc[chosen_drugs, DOSAGE_COLUMNS].set_index(index) / 2  # start on half dosage
        return new_dosages

    def get_minimum_dose_drug(self, index):
        meds = self.population_view.subview(DOSAGE_COLUMNS + SINGLE_PILL_COLUMNS).get(index)
        dosages = meds[DOSAGE_COLUMNS]
        in_single_pill = meds[SINGLE_PILL_COLUMNS]
        dosages = dosages.mask(dosages == 0, np.inf)  # mask 0s with inf to more easily identify min non-zero dose
        min_dosages = dosages.min(axis=1)

        min_dose_drug_mask = pd.DataFrame(0, columns=HYPERTENSION_DRUGS, index=dosages.index)
        min_dose_in_single_pill = pd.Series(0, index=dosages.index)

        for d in HYPERTENSION_DRUGS:
            mask = dosages[f'{d}_dosage'] == min_dosages
            min_dose_drug_mask.loc[mask, HYPERTENSION_DRUGS] = [0 if drug != d else 1 for drug in HYPERTENSION_DRUGS]

            min_dose_in_single_pill.loc[mask] = in_single_pill.loc[mask, f'{d}_in_single_pill']

        return (min_dose_drug_mask.rename(columns={d: f'{d}_dosage' for d in min_dose_drug_mask.columns}),
                min_dose_in_single_pill)

    def transition_low_and_slow(self, index):
        current_meds = self.population_view.subview(DOSAGE_COLUMNS + SINGLE_PILL_COLUMNS).get(index)

        no_current_tx_mask = current_meds[DOSAGE_COLUMNS].sum(axis=1) == 0

        if sum(no_current_tx_mask):
            current_meds.loc[no_current_tx_mask, DOSAGE_COLUMNS] = self.choose_half_dose_new_drug(
                index[no_current_tx_mask],
                current_meds.loc[no_current_tx_mask, DOSAGE_COLUMNS])

        increase_tx = index[~no_current_tx_mask]
        min_dose_drugs_mask, min_dose_drug_in_single_pill = self.get_minimum_dose_drug(increase_tx)

        at_max = current_meds.loc[increase_tx].mask(np.logical_not(min_dose_drugs_mask), 0).max(axis=1) == max(HYPERTENSION_DOSAGES)

        # already on treatment and maxed out on dosage of minimum dose drug - add half dose of new drug
        idx = increase_tx[at_max]
        if not idx.empty:
            current_meds.loc[idx, DOSAGE_COLUMNS] += self.choose_half_dose_new_drug(idx, current_meds.loc[idx, DOSAGE_COLUMNS])

        # already on treatment and minimum dose drug is in a single pill - double doses of all drugs in pill
        double_pill = increase_tx[~at_max & min_dose_drug_in_single_pill.loc[increase_tx[~at_max]]]
        if not double_pill.empty:
            in_pill_mask = np.logical_and(current_meds.loc[double_pill, DOSAGE_COLUMNS].mask(
                current_meds.loc[double_pill, DOSAGE_COLUMNS] > 0, 1), current_meds.loc[double_pill, SINGLE_PILL_COLUMNS])
            current_meds.loc[double_pill, DOSAGE_COLUMNS] += (current_meds.loc[double_pill, DOSAGE_COLUMNS]
                                                              .mask(~in_pill_mask, 0))

        # already on treatment and minimum dose drug is not in a single pill - double dose of min dose drug
        double_dose = increase_tx[~at_max & np.logical_not(min_dose_drug_in_single_pill.loc[increase_tx[~at_max]])]
        if not double_dose.empty:
            current_meds.loc[double_dose, DOSAGE_COLUMNS] += (current_meds.loc[double_dose, DOSAGE_COLUMNS]
                .mask(np.logical_not(min_dose_drugs_mask.loc[double_dose]), 0))

        return current_meds

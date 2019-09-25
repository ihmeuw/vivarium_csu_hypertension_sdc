from scipy import stats

import numpy as np
import pandas as pd

from vivarium_csu_hypertension_sdc.components import utilities
from vivarium_csu_hypertension_sdc.components.globals import (DOSAGE_COLUMNS, HYPERTENSIVE_CONTROLLED_THRESHOLD,
                                                              SINGLE_PILL_COLUMNS, HYPERTENSION_DOSAGES,
                                                              HYPERTENSION_DRUGS, ILLEGAL_DRUG_COMBINATION)
from vivarium_csu_hypertension_sdc.external_data.globals import FREE_CHOICE_SINGLE_PILL_PROBABILITY


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
            'followup_visit_interval': {
                'start': 90,  # days
                'end': 180,  # days
            },
            'treatment_ramp': 'low_and_slow',  # one of ["low_and_slow", "free_choice", "fixed_dose_combination", "hypothetical_baseline"]
            'probability_of_missed_appt': 0.1
        }
    }

    @property
    def name(self):
        return 'hypertension_treatment_algorithm'

    def setup(self, builder):
        self.clock = builder.time.clock()

        self.sim_start = pd.Timestamp(**builder.configuration.time.start)

        self.config = builder.configuration.hypertension_treatment

        self.med_probabilities = builder.data.load('health_technology.hypertension_medication.medication_probabilities')

        self.therapy_categories = (builder.data.load('health_technology.hypertension_medication.therapy_category')
                                   .set_index('therapy_category').value)

        columns_created = ['followup_date', 'followup_type', 'last_visit_date', 'last_visit_type',
                           'last_missed_visit_date',
                           'high_systolic_blood_pressure_measurement',
                           'high_systolic_blood_pressure_last_measurement_date',
                           'single_pill_dr', 'last_prescription_date']
        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 requires_columns=DOSAGE_COLUMNS,
                                                 creates_columns=columns_created,
                                                 requires_streams=['followup_scheduling', 'single_pill_dr'])
        self.population_view = builder.population.get_view(DOSAGE_COLUMNS + SINGLE_PILL_COLUMNS + columns_created
                                                           + ['alive'],
                                                           query='alive == "alive"')

        self.randomness = {'followup_scheduling': builder.randomness.get_stream('followup_scheduling'),
                           'background_visit_attendance': builder.randomness.get_stream('background_visit_attendance'),
                           'sbp_measurement': builder.randomness.get_stream('sbp_measurement'),
                           'therapeutic_inertia': builder.randomness.get_stream('therapeutic_inertia'),
                           'treatment_transition': builder.randomness.get_stream('treatment_transition'),
                           'single_pill_dr': builder.randomness.get_stream('single_pil_dr'),
                           'miss_appt': builder.randomness.get_stream('miss_appt'),
                           }

        self.ti_probability = utilities.get_therapeutic_inertia_probability(self.config.therapeutic_inertia.mean,
                                                                            self.config.therapeutic_inertia.sd,
                                                                            self.randomness['therapeutic_inertia'])

        self.utilization_data = builder.lookup.build_table(
            builder.data.load('healthcare_entity.outpatient_visits.utilization_rate'), key_columns=['sex'],
            parameter_columns=['age', 'year'])
        self.healthcare_utilization = builder.value.register_rate_producer('healthcare_utilization_rate',
                                                                           source=lambda index: self.utilization_data(index))

        self.sbp = builder.value.get_value('high_systolic_blood_pressure.exposure')

        builder.event.register_listener('time_step', self.on_time_step)

        transition_funcs = {'low_and_slow': self.transition_low_and_slow,
                            'fixed_dose_combination': self.transition_fdc,
                            'free_choice': self.transition_free_choice,
                            'hypothetical_baseline': self.transition_hypothetical_baseline}
        self._transition_func = transition_funcs[self.config.treatment_ramp]
        self.free_choice_single_pill_prob = FREE_CHOICE_SINGLE_PILL_PROBABILITY[builder.configuration.input_data.location]

    def on_initialize_simulants(self, pop_data):
        drug_dosages = self.population_view.subview(DOSAGE_COLUMNS).get(pop_data.index)
        sims_on_tx = drug_dosages.loc[drug_dosages.sum(axis=1) > 0].index

        initialize = pd.DataFrame({'followup_date': pd.NaT, 'followup_type': None,
                                   'last_visit_date': pd.NaT, 'last_visit_type': None,
                                   'last_missed_visit_date': pd.NaT,
                                   'high_systolic_blood_pressure_measurement': np.nan,
                                   'high_systolic_blood_pressure_last_measurement_date': pd.NaT,
                                   'last_prescription_date': pd.NaT,
                                   'single_pill_dr': False},
                                  index=pop_data.index)

        days_to_followup = utilities.get_days_in_range(self.randomness['followup_scheduling'],
                                                       low=0, high=self.config.followup_visit_interval.end,
                                                       index=sims_on_tx)
        initialize.loc[sims_on_tx, 'followup_date'] = days_to_followup + self.sim_start
        initialize.loc[sims_on_tx, 'followup_type'] = 'maintenance'

        if self.config.treatment_ramp == 'fixed_dose_combination':
            initialize.loc[:, 'single_pill_dr'] = True
        elif self.config.treatment_ramp == 'free_choice':
            single_pill_idx = (self.randomness['single_pill_dr']
                               .filter_for_probability(pop_data.index, np.tile(self.free_choice_single_pill_prob,
                                                                               len(pop_data.index))))
            initialize.loc[single_pill_idx, 'single_pill_dr'] = True
        # low and slow ramp is always single pill dr = False so no need to update
        # hypothetical baseline never changes treatment once initialized so single pill dr is irrelevant

        self.population_view.update(initialize)

    def on_time_step(self, event):
        pop = self.population_view.get(event.index)

        followup_scheduled = (self.clock() < pop.followup_date) & (pop.followup_date <= event.time)
        followup_pop = pop.index[followup_scheduled]
        miss_appt = self.randomness['miss_appt'].filter_for_probability(followup_pop,
                                                                        np.tile(self.config.probability_of_missed_appt,
                                                                                len(followup_pop)))

        self.reschedule_followup(miss_appt, pop.loc[miss_appt, 'followup_type'], event.time)

        followup_attending = pop.index[followup_scheduled].difference(miss_appt)
        self.attend_confirmatory(followup_attending[pop.loc[followup_attending, 'followup_type'] == 'confirmatory'],
                                 event.time)
        self.attend_maintenance(followup_attending[pop.loc[followup_attending, 'followup_type'] == 'maintenance'],
                                event.time)
        pop.loc[followup_attending, 'last_visit_type'] = pop.loc[followup_attending, 'followup_type']

        background_eligible = pop.index[~followup_scheduled]
        background_attending = (self.randomness['background_visit_attendance']
                                .filter_for_rate(background_eligible,
                                                 self.healthcare_utilization(background_eligible).values))

        self.attend_background(background_attending, event.time)
        pop.loc[background_attending, 'last_visit_type'] = 'background'

        pop.loc[background_attending.union(followup_attending), 'last_visit_date'] = event.time
        self.population_view.update(pop.loc[:, ['last_visit_type', 'last_visit_date']])

    def reschedule_followup(self, index, followup_types, missed_visit_date):
        self.schedule_followup(index, missed_visit_date, followup_types)
        self.population_view.update(pd.DataFrame({'last_missed_visit_date': missed_visit_date,
                                                  'last_prescription_date': missed_visit_date},  # meds ind from visits
                                                 index=index))

    def attend_confirmatory(self, index, visit_date):
        """Patients are only scheduled for confirmatory visit if they've had an
        SBP measurement above the threshold so if the measurement on this visit
        is above the threshold, they've had 2 above threshold measurements and
        should begin treatment."""
        sbp_measurements = self.measure_sbp(index, visit_date)
        eligible_for_tx_mask = sbp_measurements >= HYPERTENSIVE_CONTROLLED_THRESHOLD

        tx_possible = index[eligible_for_tx_mask]
        if not tx_possible.empty:
            lost_to_ti = self.randomness['therapeutic_inertia'].filter_for_probability(tx_possible,
                                                                                       np.tile(self.ti_probability,
                                                                                               len(tx_possible)),
                                                                                       additional_key='lost_to_ti')
            # patients who don't overcome therapeutic inertia are scheduled for another confirmatory visit
            self.schedule_followup(lost_to_ti, visit_date, 'confirmatory')

            start_tx = tx_possible.difference(lost_to_ti)
            self.transition_treatment(start_tx)
            self.schedule_followup(start_tx, visit_date, 'maintenance')
            self.population_view.update(pd.Series(visit_date, index=start_tx, name='last_prescription_date'))

        # patients who aren't hypertensive on this visit are put back into the general population
        self.population_view.update(pd.DataFrame({'followup_date': pd.NaT, 'followup_type': None},
                                                 index=index[~eligible_for_tx_mask]))

    def attend_maintenance(self, index, visit_date):
        sbp_measurements = self.measure_sbp(index, visit_date)
        eligible_for_tx_mask = sbp_measurements >= HYPERTENSIVE_CONTROLLED_THRESHOLD

        treatment_increase_possible = self.check_treatment_increase_possible(index[eligible_for_tx_mask])

        if not treatment_increase_possible.empty:
            lost_to_ti = self.randomness['therapeutic_inertia'].filter_for_probability(treatment_increase_possible,
                                                                                       np.tile(self.ti_probability,
                                                                                               len(treatment_increase_possible)),
                                                                                       additional_key='lost_to_ti')
            self.transition_treatment(treatment_increase_possible.difference(lost_to_ti))

        self.schedule_followup(index, visit_date, 'maintenance')  # everyone rescheduled whether their tx changed or not
        self.population_view.update(pd.Series(visit_date, index=index, name='last_prescription_date'))

    def attend_background(self, index, visit_date):
        sbp_measurements = self.measure_sbp(index, visit_date)
        followup_dates = self.population_view.subview(['followup_date']).get(index).loc[:, 'followup_date']

        eligible_for_confirmatory = (sbp_measurements >= HYPERTENSIVE_CONTROLLED_THRESHOLD) & (followup_dates.isna())

        lost_to_ti = self.randomness['therapeutic_inertia'].filter_for_probability(index[eligible_for_confirmatory],
                                                                                   np.tile(self.ti_probability,
                                                                                           sum(eligible_for_confirmatory)),
                                                                                   additional_key='lost_to_ti')
        schedule_confirmatory = index[eligible_for_confirmatory].difference(lost_to_ti)
        self.schedule_followup(schedule_confirmatory, visit_date, 'confirmatory')

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
        new_meds = self._transition_func(index)
        assert np.all(new_meds[DOSAGE_COLUMNS].max(axis=1) <= 2)
        self.population_view.update(new_meds)

    def check_treatment_increase_possible(self, index):
        meds = self.population_view.subview(DOSAGE_COLUMNS + SINGLE_PILL_COLUMNS).get(index)
        dosages = meds[DOSAGE_COLUMNS]
        single_pill = meds[SINGLE_PILL_COLUMNS]
        if self.config.treatment_ramp == 'low_and_slow':
            no_tx_increase_mask = dosages.sum(axis=1) >= 3 * max(HYPERTENSION_DOSAGES)  # 3 drugs, each on a max dosage
        elif self.config.treatment_ramp == 'fixed_dose_combination':
            # 3 drugs, each on a max dosage and all in single pill
            no_tx_increase_mask = (dosages.sum(axis=1) >= 3 * max(HYPERTENSION_DOSAGES)) & (single_pill.sum(axis=1) >= 3)
        elif self.config.treatment_ramp == 'free_choice':
            single_pill_dr = self.population_view.subview(['single_pill_dr']).get(index).loc[:, 'single_pill_dr']
            # 3 drugs each on a max dosage
            no_tx_increase_mask = dosages.sum(axis=1) >= 3 * max(HYPERTENSION_DOSAGES)
            # if assigned to single pill dr, must also be on a single pill of all 3 drugs
            no_tx_increase_mask.loc[single_pill_dr] &= (single_pill.loc[single_pill_dr].sum(axis=1) >= 3)
        else:  # hypothetical_baseline
            # no increases if already on treatment in hypothetical baseline scenario - only start new treatment
            no_tx_increase_mask = dosages.sum(axis=1) > 0

        return index[~no_tx_increase_mask]

    def schedule_followup(self, index, visit_date, followup_type):
        days_to_followup = utilities.get_days_in_range(self.randomness['followup_scheduling'],
                                                       low=self.config.followup_visit_interval.start,
                                                       high=self.config.followup_visit_interval.end,
                                                       index=index)
        next_followup = pd.DataFrame({'followup_date': visit_date + days_to_followup, 'followup_type': followup_type})
        self.population_view.update(next_followup)

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
        current_meds.loc[increase_tx] = self.transition_low_and_slow_fc_non_single_pill_dr(increase_tx,
                                                                                           current_meds.loc[increase_tx])
        return current_meds

    def transition_low_and_slow_fc_non_single_pill_dr(self, index, current_meds):
        min_dose_drugs_mask, min_dose_drug_in_single_pill = self.get_minimum_dose_drug(index)

        at_max = current_meds.loc[index].mask(np.logical_not(min_dose_drugs_mask), 0).max(axis=1) == max(HYPERTENSION_DOSAGES)

        # already on treatment and maxed out on dosage of minimum dose drug - add half dose of new drug
        idx = index[at_max]
        if not idx.empty:
            current_meds.loc[idx, DOSAGE_COLUMNS] += self.choose_half_dose_new_drug(idx, current_meds.loc[idx, DOSAGE_COLUMNS])

        # already on treatment and minimum dose drug is in a single pill - double doses of all drugs in pill
        double_pill = index[~at_max & min_dose_drug_in_single_pill.loc[index[~at_max]]]
        if not double_pill.empty:
            in_pill_mask = np.logical_and(current_meds.loc[double_pill, DOSAGE_COLUMNS].mask(
                current_meds.loc[double_pill, DOSAGE_COLUMNS] > 0, 1), current_meds.loc[double_pill, SINGLE_PILL_COLUMNS])
            current_meds.loc[double_pill, DOSAGE_COLUMNS] += (current_meds.loc[double_pill, DOSAGE_COLUMNS]
                                                              .mask(~in_pill_mask, 0))

        # already on treatment and minimum dose drug is not in a single pill - double dose of min dose drug
        double_dose = index[~at_max & np.logical_not(min_dose_drug_in_single_pill.loc[index[~at_max]])]
        if not double_dose.empty:
            current_meds.loc[double_dose, DOSAGE_COLUMNS] += (current_meds.loc[double_dose, DOSAGE_COLUMNS]
                .mask(np.logical_not(min_dose_drugs_mask.loc[double_dose]), 0))

        return current_meds

    def choose_half_dose_new_single_pill(self, index):
        options = utilities.get_all_legal_drug_combos(num_drugs=2)
        options = pd.concat([options.rename(columns={d: f'{d}_dosage' for d in options}) / 2,  # half dose
                             options.rename(columns={d: f'{d}_in_single_pill' for d in options})], axis=1)  # in single pill

        choices = self.randomness['treatment_transition'].choice(index, options.index)
        return options.loc[choices].set_index(index)

    def transition_fdc(self, index):
        current_meds = self.population_view.subview(DOSAGE_COLUMNS + SINGLE_PILL_COLUMNS).get(index)

        no_current_tx_mask = current_meds[DOSAGE_COLUMNS].sum(axis=1) == 0

        # not currently on any tx
        if sum(no_current_tx_mask):
            current_meds.loc[no_current_tx_mask] = self.choose_half_dose_new_single_pill(index[no_current_tx_mask])

        num_drugs = current_meds[DOSAGE_COLUMNS].mask(current_meds[DOSAGE_COLUMNS] > 0, 1).sum(axis=1)

        # currently on 1 drug
        idx = index[~no_current_tx_mask & (num_drugs == 1)]
        if not idx.empty:
            current_meds.loc[idx] = self.transition_fdc_fc_from_one_drug(idx, current_meds.loc[idx])

        # currently on 2-3 drugs
        idx = index[~no_current_tx_mask & (num_drugs > 1)]
        if not idx.empty:
            current_meds.loc[idx] = self.transition_fdc_fc_single_pill_dr_from_mult_drugs(idx, current_meds.loc[idx])

        return current_meds

    def transition_fdc_fc_from_one_drug(self, index, current_meds):
        # switch to half dose single pill combo of current drug + one other
        new_dose = self.choose_half_dose_new_drug(index, current_meds.loc[:, DOSAGE_COLUMNS])
        current_meds.loc[:, DOSAGE_COLUMNS] += new_dose
        on_meds_mask = current_meds.loc[:, DOSAGE_COLUMNS] > 0
        current_meds.loc[:, DOSAGE_COLUMNS] = current_meds.mask(on_meds_mask, 0.5)  # switch to half dose of 2 drugs
        on_meds_mask = on_meds_mask.rename(columns={d: d.replace('dosage', 'in_single_pill') for d in on_meds_mask})
        single_pill_dr = self.population_view.subview(['single_pill_dr']).get(index).loc[:, 'single_pill_dr']
        current_meds.loc[single_pill_dr, SINGLE_PILL_COLUMNS] = (current_meds.loc[:, SINGLE_PILL_COLUMNS]
                                                                 .mask(on_meds_mask, 1))
        return current_meds

    def transition_fdc_fc_single_pill_dr_from_mult_drugs(self, index, current_meds):
        non_zero_dosages = current_meds.loc[:, DOSAGE_COLUMNS].mask(current_meds.loc[:, DOSAGE_COLUMNS] == 0, np.nan)
        min_dosages = non_zero_dosages.min(axis=1)
        max_dosages = non_zero_dosages.max(axis=1)

        single_pill_eligible = max_dosages / min_dosages <= 2

        # double dosage of lower-dose drug where single pill is not possible
        dosages = current_meds.loc[~single_pill_eligible, DOSAGE_COLUMNS]
        mins = np.tile(min_dosages.loc[~single_pill_eligible], (len(DOSAGE_COLUMNS), 1)).transpose()
        min_dosage_mask = np.equal(dosages, mins)
        current_meds.loc[index[~single_pill_eligible], DOSAGE_COLUMNS] += (dosages * min_dosage_mask.astype(int))

        num_in_single = current_meds.loc[:, SINGLE_PILL_COLUMNS].sum(axis=1)
        num_drugs = current_meds[DOSAGE_COLUMNS].mask(current_meds[DOSAGE_COLUMNS] > 0, 1).sum(axis=1)
        already_in_single = num_in_single == num_drugs

        # put prescribed meds in single pill with highest currently prescribed dosage if not already in single pill
        mask = single_pill_eligible & ~already_in_single
        dosages = current_meds.loc[mask, DOSAGE_COLUMNS]
        maxes = np.tile(max_dosages.loc[mask], (len(DOSAGE_COLUMNS), 1)).transpose()

        # set the dosage of the currently prescribed meds to the highest prescribed dosage
        currently_prescribed = dosages.mask(dosages > 0, 1)
        current_meds.loc[index[mask], DOSAGE_COLUMNS] = currently_prescribed.multiply(maxes)
        # and mark the currently prescribed meds as being in a single pill
        current_meds.loc[index[mask], SINGLE_PILL_COLUMNS] = (currently_prescribed
                                                              .rename(columns={d: d.replace('dosage',
                                                                                            'in_single_pill')
                                                                               for d in dosages}))
        # double dosage of single pill where possible
        mask = single_pill_eligible & already_in_single & (max_dosages < max(HYPERTENSION_DOSAGES))
        current_meds.loc[index[mask], DOSAGE_COLUMNS] *= 2

        # add 1/2 of new drug where already at max dosage of single pill (will never hit this if on 3 drugs)
        mask = single_pill_eligible & already_in_single & (max_dosages == max(HYPERTENSION_DOSAGES))
        current_meds.loc[index[mask], DOSAGE_COLUMNS] += (
            self.choose_half_dose_new_drug(index[mask], current_meds.loc[index[mask], DOSAGE_COLUMNS]))

        return current_meds

    def transition_free_choice(self, index):
        current_meds = self.population_view.subview(DOSAGE_COLUMNS + SINGLE_PILL_COLUMNS).get(index)

        no_current_tx_mask = current_meds[DOSAGE_COLUMNS].sum(axis=1) == 0

        # not currently on any tx
        if sum(no_current_tx_mask):
            current_meds.loc[no_current_tx_mask] = self.start_free_choice(index[no_current_tx_mask])

        num_drugs = current_meds[DOSAGE_COLUMNS].mask(current_meds[DOSAGE_COLUMNS] > 0, 1).sum(axis=1)

        # currently on 1 drug
        idx = index[~no_current_tx_mask & (num_drugs == 1)]
        if not idx.empty:
            current_meds.loc[idx] = self.transition_fdc_fc_from_one_drug(idx, current_meds.loc[idx])

        # currently on 2-3 drugs
        idx = index[~no_current_tx_mask & (num_drugs > 1)]
        single_pill_dr = self.population_view.subview(['single_pill_dr']).get(idx).loc[:, 'single_pill_dr']
        single_pill_idx = idx[single_pill_dr]
        if not single_pill_idx.empty:
            current_meds.loc[single_pill_idx] = self.transition_fdc_fc_single_pill_dr_from_mult_drugs(
                single_pill_idx, current_meds.loc[single_pill_idx]
            )

        non_single_pill_idx = idx[~single_pill_dr]
        if not non_single_pill_idx.empty:
            current_meds.loc[non_single_pill_idx] = self.transition_low_and_slow_fc_non_single_pill_dr(
                non_single_pill_idx, current_meds.loc[non_single_pill_idx]
            )

        return current_meds

    def start_free_choice(self, index):
        meds = self.choose_half_dose_new_single_pill(index)

        # but we actually don't want all sims to be on a single pill
        single_pill_dr = self.population_view.subview(['single_pill_dr']).get(index).loc[:, 'single_pill_dr']
        meds.loc[~single_pill_dr, SINGLE_PILL_COLUMNS] = 0
        return meds

    def transition_hypothetical_baseline(self, index):
        current_meds = self.population_view.subview(DOSAGE_COLUMNS + SINGLE_PILL_COLUMNS).get(index)
        no_current_tx_mask = current_meds[DOSAGE_COLUMNS].sum(axis=1) == 0
        if sum(no_current_tx_mask):
            therapy_cat = self.randomness['treatment_transition'].choice(index[no_current_tx_mask],
                                                                         self.therapy_categories.index.to_list(),
                                                                         self.therapy_categories.values)
            cat_groups = therapy_cat.groupby(therapy_cat).apply(lambda g: g.index)

            # choose drug/pill combination first
            drugs = pd.DataFrame(columns=HYPERTENSION_DRUGS + SINGLE_PILL_COLUMNS, index=index[no_current_tx_mask],
                                 dtype=float)
            for cat, idx in cat_groups.iteritems():
                drugs.loc[idx] = utilities.get_initial_drugs_given_category(self.med_probabilities, cat,
                                                                            idx, self.randomness['treatment_transition'])
            # then select dosages
            drugs.loc[:, HYPERTENSION_DRUGS] = utilities.get_initial_dosages(drugs, self.randomness['treatment_transition'])

            current_meds.loc[no_current_tx_mask] = drugs.rename(columns={d: f'{d}_dosage' for d in HYPERTENSION_DRUGS})

        return current_meds


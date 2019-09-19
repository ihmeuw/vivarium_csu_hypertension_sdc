from scipy import stats

from itertools import combinations
import numpy as np
import pandas as pd

from vivarium_csu_hypertension_sdc.components import utilities
from vivarium_csu_hypertension_sdc.components.globals import (DOSAGE_COLUMNS, HYPERTENSIVE_CONTROLLED_THRESHOLD,
                                                              SINGLE_PILL_COLUMNS, HYPERTENSION_DOSAGES,
                                                              HYPERTENSION_DRUGS, ILLEGAL_DRUG_COMBINATION)
from vivarium_csu_hypertension_sdc.external_data.globals import FREE_CHOICE_TWO_PILL_PROBABILITY


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
        self.population_view = builder.population.get_view(DOSAGE_COLUMNS + SINGLE_PILL_COLUMNS + columns_created
                                                           + ['alive'],
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

        transition_funcs = {'low_and_slow': self.transition_low_and_slow,
                            'fixed_dose_combination': self.transition_fdc,
                            'free_choice': self.transition_free_choice}
        self._transition_func = transition_funcs[self.config.treatment_ramp]
        self.free_choice_two_pill_prob = FREE_CHOICE_TWO_PILL_PROBABILITY[builder.configuration.input_data.location]

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
        new_meds = self._transition_func(index)
        self.population_view.update(new_meds)

    def check_treatment_increase_possible(self, index):
        meds = self.population_view.subview(DOSAGE_COLUMNS + SINGLE_PILL_COLUMNS).get(index)
        dosages = meds[DOSAGE_COLUMNS]
        single_pill = meds[SINGLE_PILL_COLUMNS]
        if self.config.treatment_ramp == 'low_and_slow':
            no_tx_increase_mask = dosages.sum(axis=1) >= 3 * max(HYPERTENSION_DOSAGES)  # 3 drugs, each on a max dosage
        elif self.config.treatment_ramp == 'fixed_dose_combination':
            # 3 drugs, each on a max dosage and at least 2 drugs in single pill
            no_tx_increase_mask = (dosages.sum(axis=1) >= 3 * max(HYPERTENSION_DOSAGES)) & (single_pill.sum(axis=1) >= 2)

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

        num_drugs = current_meds[DOSAGE_COLUMNS].mask(current_meds[DOSAGE_COLUMNS] > 0, 1).sum(axis=1).loc[~no_current_tx_mask]

        for n in num_drugs.unique():
            idx = index[num_drugs == n]
            if not idx.empty:
                current_meds.loc[idx] = self.transition_fdc_by_num_current_drugs(idx, n, current_meds.loc[idx])

        return current_meds

    def transition_fdc_by_num_current_drugs(self, index, num_current_drugs, current_meds):
        if num_current_drugs == 1:
            # switch to half dose single pill combo of current drug + one other
            new_dose = self.choose_half_dose_new_drug(index, current_meds.loc[:, DOSAGE_COLUMNS])
            current_meds.loc[:, DOSAGE_COLUMNS] += new_dose
            on_meds_mask = current_meds.loc[:, DOSAGE_COLUMNS] > 0
            current_meds.loc[:, DOSAGE_COLUMNS] = current_meds.mask(on_meds_mask, 0.5)  # switch to half dose of 2 drugs
            on_meds_mask = on_meds_mask.rename(columns={d: d.replace('dosage', 'in_single_pill') for d in on_meds_mask})
            current_meds.loc[:, SINGLE_PILL_COLUMNS] = current_meds.loc[:, SINGLE_PILL_COLUMNS].mask(on_meds_mask, 1)

        elif num_current_drugs == 2:
            non_zero_dosages = current_meds.loc[:, DOSAGE_COLUMNS].mask(current_meds.loc[:, DOSAGE_COLUMNS] == 0, np.nan)
            min_dosages = non_zero_dosages.min(axis=1)
            max_dosages = non_zero_dosages.max(axis=1)

            single_pill_eligible = max_dosages / min_dosages <= 2

            # double dosage of lower-dose drug where single pill is not possible
            dosages = current_meds.loc[~single_pill_eligible, DOSAGE_COLUMNS]
            mins = np.tile(min_dosages.loc[~single_pill_eligible], (len(DOSAGE_COLUMNS), 1)).transpose()
            min_dosage_mask = np.equal(dosages, mins)
            current_meds.loc[index[~single_pill_eligible], DOSAGE_COLUMNS] *= (min_dosage_mask.astype(int) * 2)

            already_in_single = current_meds.loc[:, SINGLE_PILL_COLUMNS].sum(axis=1) > 0

            # put prescribed meds in single pill with highest currently prescribed dosage if not already in single pill
            mask = single_pill_eligible & ~already_in_single
            dosages = current_meds.loc[mask, DOSAGE_COLUMNS]
            maxes = np.tile(max_dosages.loc[mask], (len(DOSAGE_COLUMNS), 1)).transpose()
            max_dosage_mask = np.equal(dosages, maxes)
            # set the dosage of the 2 currently prescribed meds to the highest prescribed dosage
            current_meds.loc[index[mask], DOSAGE_COLUMNS] = max_dosage_mask.astype(int).multiply(maxes)
            # and mark the 2 currently prescribed meds as being in a single pill
            current_meds.loc[index[mask], SINGLE_PILL_COLUMNS] = (dosages.mask(dosages > 0, 1)
                                                                  .rename(columns={d: d.replace('dosage',
                                                                                                'in_single_pill')
                                                                                   for d in dosages}))
            # double dosage of single pill where possible
            mask = single_pill_eligible & already_in_single & (max_dosages < max(HYPERTENSION_DOSAGES))
            current_meds.loc[index[mask], DOSAGE_COLUMNS] *= 2

            # add 1/2 of new drug where already at max dosage of single pill
            mask = single_pill_eligible & already_in_single & (max_dosages == max(HYPERTENSION_DOSAGES))
            current_meds.loc[index[mask], DOSAGE_COLUMNS] += (
                self.choose_half_dose_new_drug(index[mask], current_meds.loc[index[mask], DOSAGE_COLUMNS]))

        else:  # 3 drugs
            on_single_pill = current_meds.loc[:, SINGLE_PILL_COLUMNS].sum(axis=1) > 0
            single_pill_dosage_mask = ((current_meds.loc[:, SINGLE_PILL_COLUMNS] == 1)
                                       .rename(columns={d: d.replace('in_single_pill', 'dosage')
                                                        for d in SINGLE_PILL_COLUMNS}))

            single_pill_dosages = current_meds.loc[:, DOSAGE_COLUMNS].mask(np.logical_not(single_pill_dosage_mask), 0)

            # double dosage of single pill where possible
            double_single_possible = single_pill_dosages.max(axis=1) < max(HYPERTENSION_DOSAGES)
            mask = on_single_pill & double_single_possible
            current_meds.loc[index[mask], DOSAGE_COLUMNS] += single_pill_dosages.loc[index[mask]]

            # double dosage of drug not in single pill where single pill is already at max dosage
            mask = on_single_pill & ~double_single_possible
            current_meds.loc[index[mask], DOSAGE_COLUMNS] += (current_meds.loc[index[mask], DOSAGE_COLUMNS]
                                                              .mask(single_pill_dosage_mask.loc[index[mask]], 0))

            # if any 2+ drugs have equal dosages, put 2 of those on a single pill
            non_zero_dosages = current_meds.loc[~on_single_pill, DOSAGE_COLUMNS].mask(
                current_meds.loc[~on_single_pill, DOSAGE_COLUMNS] == 0, np.nan)
            min_dosages = non_zero_dosages.min(axis=1)
            max_dosages = non_zero_dosages.max(axis=1)

            dosages = current_meds.loc[~on_single_pill, DOSAGE_COLUMNS]
            mins = np.tile(min_dosages, (len(DOSAGE_COLUMNS), 1)).transpose()
            maxes = np.tile(max_dosages, (len(DOSAGE_COLUMNS), 1)).transpose()
            two_at_min = np.equal(dosages, mins).sum(axis=1) == 2
            two_at_max = np.equal(dosages, maxes).sum(axis=1) == 2
            two_equal = two_at_min | two_at_max

            # if no two equal dosages, set 2 highest to double dose in single pill
            idx = index[~on_single_pill][~two_equal]
            lowest_mask = np.equal(dosages, mins)
            current_meds.loc[idx, DOSAGE_COLUMNS] = pd.DataFrame(maxes, columns=DOSAGE_COLUMNS,
                                                                 index=index[~on_single_pill]).loc[idx].mask(lowest_mask, 0)
            current_meds.loc[idx, SINGLE_PILL_COLUMNS] = (np.logical_not(lowest_mask).astype(int)
                                                          .rename(columns={d: d.replace('dosage', 'in_single_pill')
                                                                           for d in DOSAGE_COLUMNS}))

            # if 2+ have equal dosages, choose 2 and put them in a single pill
            idx = index[~on_single_pill][two_equal]
            dosages = dosages.loc[two_equal]
            equal_dosage = pd.Series(0, index=index[~on_single_pill])
            equal_dosage.loc[two_at_min] = min_dosages.loc[two_at_min]
            equal_dosage.loc[two_at_max] = max_dosages.loc[two_at_max]
            equal_dosage = equal_dosage.loc[two_equal]

            put_in_single_pill = pd.DataFrame(0, columns=SINGLE_PILL_COLUMNS, index=idx)
            # FIXME: we probably actually want to choose 2 at random but that's hard so just choosing first 2 for now
            for c in put_in_single_pill:
                still_needs_drug = put_in_single_pill.sum(axis=1) < 2
                # for everyone who doesn't have 2 drugs to put into a single pill, add the current drug if it's dosage
                # is the dosage that 2+ drugs share
                put_in_single_pill.loc[still_needs_drug, c] = ((dosages.loc[still_needs_drug,
                                                                            c.replace('in_single_pill', 'dosage')]
                                                                == equal_dosage.loc[still_needs_drug])
                                                               .astype(int))

            current_meds.loc[idx, SINGLE_PILL_COLUMNS] = put_in_single_pill

        return current_meds

    def transition_free_choice(self, index):
        current_meds = self.population_view.subview(DOSAGE_COLUMNS + SINGLE_PILL_COLUMNS).get(index)

        no_current_tx_mask = current_meds[DOSAGE_COLUMNS].sum(axis=1) == 0

        # not currently on any tx
        if sum(no_current_tx_mask):
            current_meds.loc[no_current_tx_mask] = self.start_free_choice(index[no_current_tx_mask])

        num_drugs = current_meds[DOSAGE_COLUMNS].mask(current_meds[DOSAGE_COLUMNS] > 0, 1).sum(axis=1).loc[~no_current_tx_mask]

        for n in num_drugs.unique():
            idx = index[num_drugs == n]
            if not idx.empty:
                current_meds.loc[idx] = self.transition_fdc_by_num_current_drugs(idx, n, current_meds.loc[idx])

        return current_meds

    def start_free_choice(self, index):
        meds = self.choose_half_dose_new_single_pill(index)

        # but we actually don't want all sims to be on a single pill
        on_two_pills = (self.randomness['treatment_transition']
                        .filter_for_probability(index, np.tile(self.free_choice_two_pill_prob, len(index))))
        meds.loc[on_two_pills, SINGLE_PILL_COLUMNS] = 0
        return meds

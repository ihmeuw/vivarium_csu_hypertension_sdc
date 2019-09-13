from scipy import stats

import numpy as np
import pandas as pd

from vivarium_csu_hypertension_sdc.components import utilities
from vivarium_csu_hypertension_sdc.components.globals import (DOSAGE_COLUMNS, HYPERTENSIVE_CONTROLLED_THRESHOLD)


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
            'followup_visit_interval': {
                'weeks': 12  # ~ 3 months
            }
        }
    }

    @property
    def name(self):
        return 'hypertension_treatment_algorithm'

    def setup(self, builder):
        self.clock = builder.time.clock()

        self.sim_start = pd.Timestamp(**builder.configuration.time.start)

        self.config = builder.configuration.hypertension_treatment

        self.followup_visit_interval_days = self.config.followup_visit_interval.weeks * 7

        columns_created = ['followup_date', 'last_visit_date', 'last_visit_type',
                           'high_systolic_blood_pressure_measurement',
                           'high_systolic_blood_pressure_last_measurement_date']
        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 requires_columns=DOSAGE_COLUMNS,
                                                 creates_columns=columns_created,)
                                                 #requires_streams=['followup_scheduling'])
        self.population_view = builder.population.get_view(DOSAGE_COLUMNS + columns_created + ['alive'],
                                                           query='alive == "alive"')

        self.randomness = {'followup_scheduling': builder.randomness.get_stream('followup_scheduling'),
                           'background_visit_attendance': builder.randomness.get_stream('background_visit_attendance'),
                           'sbp_measurement': builder.randomness.get_stream('sbp_measurement'),
                           'therapeutic_inertia': builder.randomness.get_stream('therapeutic_intertia')
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
                                                low=0, high=self.followup_visit_interval_days,
                                                index=sims_on_tx)
        initialize.loc[sims_on_tx, 'followup_date'] = durations + self.sim_start
        initialize.loc[sims_on_tx, 'last_visit_date'] = (self.sim_start
                                                         - pd.Timedelta(self.followup_visit_interval_days))
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
        # TODO
        pass

    def check_treatment_increase_possible(self, index):
        # TODO
        return index

    def schedule_followup(self, index, visit_date):
        next_followup_date = pd.Series(visit_date + pd.Timedelta(days=3*28), index=index, name='followup_date')
        self.population_view.update(next_followup_date)


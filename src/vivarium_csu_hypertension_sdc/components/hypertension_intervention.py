import pandas as pd

from vivarium_csu_hypertension_sdc.components import utilities
from vivarium_csu_hypertension_sdc.components.globals import DOSAGE_COLUMNS


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
            }
        }
    }

    @property
    def name(self):
        return 'hypertension_treatment_algorithm'

    def setup(self, builder):
        self.clock = builder.time.clock()

        self.sim_start = pd.Timestamp(**builder.configuration.time.start)

        columns_created = ['followup_date']
        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 requires_columns=DOSAGE_COLUMNS,
                                                 creates_columns=columns_created)
        self.population_view = builder.population.get_view(DOSAGE_COLUMNS + columns_created + ['alive'],
                                                           query='alive == "alive"')

        self.randomness = {'followup_scheduling': builder.randomness.get_stream('followup_scheduling'),
                           'background_visit_attendance': builder.randomness.get_stream('background_visit_attendance'),
                           'sbp_measurement': builder.randomness.get_stream('sbp_measured')
                           }

        self.utilization_data = builder.lookup.build_table(
            builder.data.load('healthcare_entity.outpatient_visits.utilization_rate'))
        self.healthcare_utilization = builder.value.register_rate_producer('healthcare_utilization_rate',
                                                                           source=lambda index: self.utilization_data(index))

        builder.event.register_listener('time_step', self.on_time_step)

    def on_initialize_simulants(self, pop_data):
        drug_dosages = self.population_view.subview(DOSAGE_COLUMNS).get(pop_data.index)
        sims_on_tx = drug_dosages.loc[drug_dosages.sum(axis=1) > 0].index

        initialize = pd.DataFrame({'followup_date': pd.NaT, 'last_visit_date': pd.NaT, 'last_visit_type': None},
                                      index=pop_data.index)

        durations = utilities.get_durations_in_range(self.randomness['followup_scheduling'],
                                           low=0, high=3*28,
                                           index=sims_on_tx)
        initialize.loc[sims_on_tx, 'followup_date'] = durations + self.sim_start
        initialize.loc[sims_on_tx, 'last_visit_date'] = self.sim_start - durations
        initialize.loc[sims_on_tx, 'last_visit_type'] = 'maintenance'

        self.population_view.update(initialize)

    def on_time_step(self, event):
        pop = self.population_view.get(event.index)

        followup_scheduled = (self.clock() < pop.followup_date) & (pop.followup_date <= event.time)

        pop.loc[followup_pop[followup_attendance], 'last_visit_type'] = \
            pop.loc[followup_pop[followup_attendance], 'followup_type']
        self.attend_followup(pop.index[followup_scheduled], event.time)
        self.reschedule_followup(followup_pop[~followup_attendance])
        pop.loc[followup_pop[~followup_attendance], 'last_missed_appt_date'] = event.time

        background_eligible = pop.index[~followup_scheduled]
        background_attending = (self.randomness['background_visit_attendance']
                                .filter_for_rate(background_eligible,
                                                 self.healthcare_utilization(background_eligible).values))

        self.attend_background(background_attending, event.time)

        pop.loc[background_attending.union(followup_pop[followup_attendance]), 'last_visit_date'] = event.time
        pop.loc[background_attending, 'last_visit_type'] = 'background'
        self.population_view.update(pop.loc[:, ['last_visit_type', 'last_visit_date', 'last_missed_appt_date']])

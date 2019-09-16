import pandas as pd

from vivarium_csu_hypertension_sdc.components.globals import (DOSAGE_COLUMNS, SINGLE_PILL_COLUMNS)


class SimulantTrajectoryObserver:

    configuration_defaults = {
        'metrics': {
            'sample_history_observer': {
                'sample_size': 1000,
                'path': f'/share/costeffectiveness/results/vivarium_csu_hypertension_sdc/simulant_trajectory.hdf'
            }
        }
    }

    @property
    def name(self):
        return "simulant_trajectory_observer"

    def __init__(self):
        self.history_snapshots = []
        self.sample_index = None

    def setup(self, builder):
        self.clock = builder.time.clock()
        self.sample_history_parameters = builder.configuration.metrics.sample_history_observer
        self.randomness = builder.randomness.get_stream("simulant_trajectory")

        # sets the sample index
        builder.population.initializes_simulants(self.on_initialize_simulants)

        columns_required = ['alive', 'age', 'sex', 'entrance_time', 'exit_time',
                            # 'cause_of_death', # FIXME: put this back in once the full artifact is built and we can use the Mortality component
                            # TODO: add disease event time columns
                            'followup_date',
                            'last_visit_date',
                            'last_visit_type',
                            'high_systolic_blood_pressure_measurement',
                            'high_systolic_blood_pressure_last_measurement_date'] + DOSAGE_COLUMNS + SINGLE_PILL_COLUMNS
        self.population_view = builder.population.get_view(columns_required)

        # keys will become column names in the output
        self.pipelines = {'pdc': builder.value.get_value('hypertension_meds.pdc'),
                          'medication_effect': builder.value.get_value('hypertension_meds.effect_size'),
                          'true_sbp': builder.value.get_value('high_systolic_blood_pressure.exposure'),
                          'healthcare_utilization_rate':
                              builder.value.get_value('healthcare_utilization_rate'),
                          }

        # record on time_step__prepare to make sure all pipelines + state table
        # columns are reflective of same time
        builder.event.register_listener('time_step__prepare', self.on_time_step__prepare)
        builder.event.register_listener('simulation_end', self.on_simulation_end)

    def on_initialize_simulants(self, pop_data):
        sample_size = self.sample_history_parameters.sample_size
        if sample_size is None or sample_size > len(pop_data.index):
            sample_size = len(pop_data.index)
        draw = self.randomness.get_draw(pop_data.index)
        priority_index = [i for d, i in sorted(zip(draw, pop_data.index), key=lambda x:x[0])]
        self.sample_index = pd.Index(priority_index[:sample_size])

    def on_time_step__prepare(self, event):
        pop = self.population_view.get(self.sample_index)

        pipeline_results = []
        for name, pipeline in self.pipelines.items():
            values = pipeline(pop.index)
            values = values.rename(name)
            pipeline_results.append(values)

        record = pd.concat(pipeline_results + [pop], axis=1)
        record['time'] = self.clock()
        record.index.rename("simulant", inplace=True)
        record.set_index('time', append=True, inplace=True)

        self.history_snapshots.append(record)

    def on_simulation_end(self, event):
        self.on_time_step__prepare(event)  # record once more since we were recording at the beginning of each time step
        sample_history = pd.concat(self.history_snapshots, axis=0)
        sample_history.to_hdf(self.sample_history_parameters.path, key='trajectories')
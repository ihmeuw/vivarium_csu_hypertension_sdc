from collections import Counter

import pandas as pd

from vivarium_public_health.metrics.utilities import (get_output_template, QueryString, get_age_bins,
                                                      get_age_sex_filter_and_iterables)
from vivarium_csu_hypertension_sdc.components.globals import (DOSAGE_COLUMNS, SINGLE_PILL_COLUMNS, HYPERTENSION_DRUGS)


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
        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 requires_streams=['simulant_trajectory'],
                                                 # FIXME: this creates_columns is a hack to get around the resource
                                                 #  mgr skipping initializers that don't create anything for now
                                                 creates_columns=['simulant_trajectory'])

        columns_required = ['alive', 'age', 'sex', 'entrance_time', 'exit_time',
                            'cause_of_death',
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


class MedicationObserver:
    configuration_defaults = {
        'metrics': {
            'medication': {
                'by_age': False,
                'by_year': False,
                'by_sex': False,
            }
        }
    }

    @property
    def name(self):
        return 'medication_observer'

    def setup(self, builder):
        self.config = builder.configuration['metrics']['medication']
        self.counts = Counter()

        age_sex_filter, (self.ages, self.sexes) = get_age_sex_filter_and_iterables(self.config, get_age_bins(builder))
        self.base_filter = age_sex_filter

        columns_required = ['alive', 'last_prescription_date'] + DOSAGE_COLUMNS + SINGLE_PILL_COLUMNS
        if self.config.by_age:
            columns_required += ['age']
        if self.config.by_sex:
            columns_required += ['sex']
        self.population_view = builder.population.get_view(columns_required)

        builder.value.register_value_modifier('metrics', self.metrics)
        builder.event.register_listener('collect_metrics', self.on_collect_metrics)

    def on_collect_metrics(self, event):
        # FIXME: is the timing right here? I always get confused about whether I'm looking at things at the right time
        pop = self.population_view.get(event.index).query('alive == "alive"')
        pop = pop.loc[pop.last_prescription_date == event.time]
        pop['num_in_single_pill'] = pop[SINGLE_PILL_COLUMNS].sum(axis=1)

        base_key = get_output_template(**self.config).substitute(year=event.time.year)

        med_counts = {}
        for drug in HYPERTENSION_DRUGS:
            drug_pop = pop.loc[pop[f'{drug}_dosage'] > 0]
            if not drug_pop.empty:
                med_counts.update(self.summarize_drug_by_group(drug_pop, drug, base_key))

        self.counts.update(med_counts)

    def summarize_drug_by_group(self, pop, drug, base_key):
        drug_counts = {}

        for group, age_group in self.ages:
            start, end = age_group.age_start, age_group.age_end
            for sex in self.sexes:
                filter_kwargs = {'age_start': start, 'age_end': end, 'sex': sex, 'age_group': group}
                group_key = base_key.substitute(**filter_kwargs)
                group_filter = self.base_filter.format(**filter_kwargs)
                in_group = pop.query(group_filter) if group_filter and not pop.empty else pop

                # number of prescriptions of drug (at any dose)
                key = group_key.substitute(measure=f'{drug}_prescription_count')
                drug_counts[key] = len(in_group)

                # total dosage of drug prescribed
                key = group_key.substitute(measure=f'{drug}_total_dosage_prescribed')
                drug_counts[key] = in_group[f'{drug}_dosage'].sum()

                # number of times drug prescribed in single pill with 1 other drug and 2 other drugs
                for n in [2, 3]:
                    key = group_key.substitute(measure=f'{drug}_prescribed_in_{n}_drug_single_pill_count')
                    drug_counts[key] = sum((in_group[f'{drug}_in_single_pill'] == 1)
                                           & (in_group['num_in_single_pill'] == n))

        return drug_counts

    def metrics(self, index, metrics):
        metrics.update(self.counts)
        return metrics

    def __repr__(self):
        return 'MedicationObserver()'

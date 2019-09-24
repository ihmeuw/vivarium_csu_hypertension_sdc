from collections import Counter

import pandas as pd

from vivarium_public_health.metrics import MortalityObserver
from vivarium_public_health.metrics.utilities import (get_output_template, get_age_bins, QueryString,
                                                      get_age_sex_filter_and_iterables, get_deaths,
                                                      get_years_of_life_lost, get_lived_in_span,
                                                      get_person_time_in_span, get_disease_event_counts)
from vivarium_csu_hypertension_sdc.components.globals import (DOSAGE_COLUMNS, SINGLE_PILL_COLUMNS, HYPERTENSION_DRUGS,
                                                              MIN_PDC_FOR_ADHERENT)


class SimulantTrajectoryObserver:

    configuration_defaults = {
        'metrics': {
            'simulant_trajectory_observer': {
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
        self.sample_history_parameters = builder.configuration.metrics.simulant_trajectory_observer
        self.randomness = builder.randomness.get_stream("simulant_trajectory")

        # sets the sample index
        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 requires_streams=['simulant_trajectory'],
                                                 # FIXME: this creates_columns is a hack to get around the resource
                                                 #  mgr skipping initializers that don't create anything for now
                                                 creates_columns=['simulant_trajectory'])

        columns_required = ['alive', 'age', 'sex', 'entrance_time', 'exit_time',
                            'cause_of_death',
                            'acute_myocardial_infarction_event_time',
                            'post_myocardial_infarction_event_time',
                            'acute_ischemic_stroke_event_time',
                            'post_ischemic_stroke_event_time',
                            'acute_subarachnoid_hemorrhage_event_time',
                            'post_subarachnoid_hemorrhage_event_time',
                            'acute_intracerebral_hemorrhage_event_time',
                            'post_intracerebral_hemorrhage_event_time',
                            'followup_date',
                            'followup_type',
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


class HtnMortalityObserver(MortalityObserver):
    def setup(self, builder):
        super().setup(builder)

        if self.config.by_year:
            raise ValueError('This custom mortality observer cannot be stratified by year.')

        self.step_size = pd.Timedelta(days=builder.configuration.time.step_size)
        self.tx_pop_view = builder.population.get_view(DOSAGE_COLUMNS + ['alive'])
        self.sbp = builder.value.get_value('high_systolic_blood_pressure.exposure')
        self.pdc = builder.value.get_value('hypertension_meds.pdc')

        self.person_time = Counter()

        builder.event.register_listener('time_step__prepare', self.on_time_step_prepare)
        # because we're collecting metrics on time step prepare for the previous time step, we need to do it once more
        # at simulation end for the last time step in the simulation
        builder.event.register_listener('simulation_end', self.on_time_step_prepare)

    def on_time_step_prepare(self, event):
        # I think this is right timing wise - I didn't want to do on collect metrics b/c if someone gets on tx during
        # a time step, it doesn't seem like their person time should be counted in the treated status
        base_filter = QueryString("")

        for key, index in self.get_groups(event.index).items():
            pop = self.population_view.get(index)
            pop.loc[pop.exit_time.isna(), 'exit_time'] = self.clock() + self.step_size

            t_start = self.clock()
            t_end = self.clock() + self.step_size
            lived_in_span = get_lived_in_span(pop, t_start, t_end)

            span_key = get_output_template(**self.config.to_dict()).substitute(measure=f'person_time_{key}')
            person_time_in_span = get_person_time_in_span(lived_in_span, base_filter, span_key, self.config.to_dict(),
                                                          self.age_bins)
            self.person_time.update(person_time_in_span)

    def get_groups(self, index):
        pop = self.tx_pop_view.get(index).query("alive == 'alive'")

        groups = {}

        treated = pop[DOSAGE_COLUMNS].sum(axis=1) > 0
        groups['among_untreated'] = pop.index[~treated]
        groups['among_treated'] = pop.index[treated]

        return self.get_adherence_groups(self.get_sbp_groups(groups, index), index)

    def get_sbp_groups(self, groups, index):
        sbp = self.sbp(index)
        sbp_groups = {}
        for k, idx in groups.items():
            grp_sbp = sbp.loc[idx]

            sbp_groups[f'{k}_sbp_group_<140'] = idx[grp_sbp < 140]
            sbp_groups[f'{k}_sbp_group_140_to_160'] = idx[(140 <= grp_sbp) & (grp_sbp <= 160)]
            sbp_groups[f'{k}_sbp_group_>160'] = idx[grp_sbp > 160]
        return sbp_groups

    def get_adherence_groups(self, groups, index):
        pdc = self.pdc(index)
        adherent_groups = {}
        for k, idx in groups.items():
            grp_pdc = pdc.loc[idx]

            adherent_groups[f'{k}_among_adherent'] = idx[grp_pdc >= MIN_PDC_FOR_ADHERENT]
            adherent_groups[f'{k}_among_non_adherent'] = idx[grp_pdc < MIN_PDC_FOR_ADHERENT]
        return adherent_groups

    def metrics(self, index, metrics):
        pop = self.population_view.get(index)
        pop.loc[pop.exit_time.isnull(), 'exit_time'] = self.clock()

        deaths = get_deaths(pop, self.config.to_dict(), self.start_time, self.clock(), self.age_bins, self.causes)
        ylls = get_years_of_life_lost(pop, self.config.to_dict(), self.start_time, self.clock(),
                                      self.age_bins, self.life_expectancy, self.causes)

        metrics.update(self.person_time)
        metrics.update(deaths)
        metrics.update(ylls)

        the_living = pop[(pop.alive == 'alive') & pop.tracked]
        the_dead = pop[pop.alive == 'dead']
        metrics['years_of_life_lost'] = self.life_expectancy(the_dead.index).sum()
        metrics['total_population_living'] = len(the_living)
        metrics['total_population_dead'] = len(the_dead)

        return metrics


class SBPTimeSeriesObserver:

    configuration_defaults = {
        'metrics': {
            'sbp_sample_date': {
                'month': 7,
                'day': 15,
            }
        }
    }

    @property
    def name(self):
        return 'sbp_time_series_observer'

    def setup(self, builder):
        self.config = builder.configuration.metrics['sbp_sample_date']
        self.clock = builder.time.clock()
        self.sbp_time_series = {}

        self.sbp = builder.value.get_value('high_systolic_blood_pressure.exposure')
        self.population_view = builder.population.get_view(DOSAGE_COLUMNS + ['alive'])

        # observing on time step prepare (and then once more at end of sim) because want the blood pressure, tx status
        # to be updated for the time step before checking
        builder.event.register_listener('time_step__prepare', self.on_time_step_prepare)
        builder.event.register_listener('simulation_end', self.on_time_step_prepare)
        builder.value.register_value_modifier('metrics', self.metrics)

    def on_time_step_prepare(self, event):
        if self.should_sample(event.time):
            pop = self.population_view.get(event.index).query("alive == 'alive'")
            sbp = self.sbp(pop.index)

            treated = pop[DOSAGE_COLUMNS].sum(axis=1) > 0
            self.sbp_time_series[f'average_sbp_among_treated_in_{event.time.year}'] = sbp.loc[treated].mean()
            self.sbp_time_series[f'average_sbp_among_untreated_in_{event.time.year}'] = sbp.loc[~treated].mean()

    def should_sample(self, event_time: pd.Timestamp) -> bool:
        """Returns true if we should sample on this time step."""
        sample_date = pd.Timestamp(year=event_time.year, **self.config.to_dict())
        return self.clock() <= sample_date < event_time

    def metrics(self, index, metrics):
        metrics.update(self.sbp_time_series)
        return metrics


class DiseaseCountObserver:
    configuration_defaults = {
        'metrics': {
            'disease_observer': {
                'by_age': False,
                'by_year': False,
                'by_sex': False,
            }
        }
    }

    def __init__(self, disease: str):
        self.disease = disease
        self.configuration_defaults = {
            'metrics': {f'{disease}_observer': DiseaseCountObserver.configuration_defaults['metrics']['disease_observer']}
        }

    @property
    def name(self):
        return f'disease_observer.{self.disease}'

    def setup(self, builder):
        self.config = builder.configuration['metrics'][f'{self.disease}_observer']
        self.clock = builder.time.clock()
        self.age_bins = get_age_bins(builder)
        self.counts = Counter()

        model = builder.components.get_component(f'disease_model.{self.disease}')
        self.disease_states = [s.name.split('.')[-1] for s in model.states]

        columns_required = ['alive'] + [f'{state}_event_time' for state in self.disease_states]
        if self.config.by_age:
            columns_required += ['age']
        if self.config.by_sex:
            columns_required += ['sex']
        self.population_view = builder.population.get_view(columns_required)

        builder.value.register_value_modifier('metrics', self.metrics)
        builder.event.register_listener('collect_metrics', self.on_collect_metrics)

    def on_collect_metrics(self, event):
        pop = self.population_view.get(event.index)
        for state in self.disease_states:
            events_this_step = get_disease_event_counts(pop, self.config.to_dict(), state,
                                                        event.time, self.age_bins)
            self.counts.update(events_this_step)

    def metrics(self, index, metrics):
        metrics.update(self.counts)
        return metrics

    def __repr__(self):
        return f"DiseaseCountObserver({self.disease})"


class TimeToControlObserver:

    def __init__(self, required_measurement_count: int = 3):
        self.required_measurement_count = required_measurement_count

    @property
    def name(self):
        return 'time_to_control_observer'

    def setup(self, builder):
        self.population_view = builder.population.get_view(['treatment_start_date',
                                                            'last_visit_date',
                                                            'last_visit_type',
                                                            'high_systolic_blood_pressure_measurement_date'
                                                            'high_systolic_blood_pressure_measurement'])

        builder.value.register_value_modifier('metrics', self.metrics)
        builder.event.register_listener('collect_metrics', self.on_collect_metrics)

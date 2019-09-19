import numpy as np
import pandas as pd
from vivarium.framework.state_machine import Trigger, Transition
from vivarium_public_health.disease import DiseaseState, DiseaseModel, SusceptibleState, RateTransition


class RelapseTransition(RateTransition):
    def load_transition_rate_data(self, builder):
        if 'relapse_rate' in self._get_data_functions:
            rate_data = self._get_data_functions['relapse_rate'](self.output_state.cause, builder)
            pipeline_name = f'{self.output_state.state_id}.incidence_rate'
        else:
            raise ValueError("No valid data functions supplied.")
        return rate_data, pipeline_name

class RelapseState(DiseaseState):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gated_transition = None

    def setup(self, builder):
        super().setup(builder)
        builder.event.register_listener('time_step_prepare', self.on_time_step_prepare)

    def on_time_step_prepare(self, event):
        population = self.population_view.get(event.index, query='alive == "alive"')
        if np.any(self.dwell_time(population.index)) > 0:
            dwell_time = pd.to_timedelta(self.dwell_time(population.index), unit='D')
            state_exit_time = population[self.event_time_column] + dwell_time
            can_transition = population.loc[state_exit_time <= event.time].index
            self._gated_transition.set_active(can_transition)

    # I really need to rewrite the state machine code.  It's super inflexible
    def add_transition(self, output, source_data_type=None, get_data_functions=None, **kwargs):
        # Skip the add transition stuff of DiseaseState
        if source_data_type == 'time':
            t = Transition(self, output,
                           probability_func=lambda index: np.ones(len(index), dtype=float),
                           triggered=Trigger.START_INACTIVE)
            self._gated_transition = t
        elif source_data_type == 'rate':
            if get_data_functions is None or 'relapse_rate' not in get_data_functions:
                raise ValueError('Must supply get data functions for incidence rate.')
            t = RelapseTransition(self, output, get_data_functions, **kwargs)
        else:
            raise ValueError('source_data_type must be "time" or "rate"')
        self.transition_set.append(t)
        return t

    def _filter_for_transition_eligibility(self, index, event_time):
        return index


def IschemicHeartDisease():
    susceptible = SusceptibleState('ischemic_heart_disease')
    data_funcs = {'dwell_time': lambda *args: pd.Timedelta(days=28)}
    acute_mi = DiseaseState('acute_myocardial_infarction',
                            cause_type='sequela',
                            get_data_functions=data_funcs)
    data_funcs = {'dwell_time': lambda *args: pd.Timedelta(days=365.25)}
    post_mi = RelapseState('post_myocardial_infarction',
                           cause_type='sequela',
                           get_data_functions=data_funcs)

    susceptible.allow_self_transitions()
    data_funcs = {
        'incidence_rate': lambda _, builder: builder.data.load('cause.ischemic_heart_disease.incidence_rate')
    }
    susceptible.add_transition(acute_mi,
                               source_data_type='rate',
                               get_data_functions=data_funcs)
    acute_mi.allow_self_transitions()
    acute_mi.add_transition(post_mi)
    post_mi.allow_self_transitions()
    data_funcs = {
        'relapse_rate': lambda _, builder: builder.data.load('cause.ischemic_heart_disease.incidence_rate')
    }
    post_mi.add_transition(acute_mi, source_data_type='rate', get_data_functions=data_funcs)
    post_mi.add_transition(susceptible, source_data_type='time')

    return DiseaseModel('ischemic_heart_disease', states=[susceptible, acute_mi, post_mi])


def Stroke(stroke_name):
    stroke_types = ['ischemic_stroke', 'subarachnoid_hemorrhage', 'intracerebral_hemorrhage']
    if stroke_name not in stroke_types:
        raise ValueError(f'Stroke name must be one of {stroke_types}.  You supplied {stroke_name}')

    susceptible = SusceptibleState(stroke_name)
    data_funcs = {'dwell_time': lambda *args: pd.Timedelta(days=28)}
    acute = DiseaseState(f'acute_{stroke_name}',
                         cause_type='sequela',
                         get_data_functions=data_funcs)
    data_funcs = {'dwell_time': lambda *args: pd.Timedelta(days=365.25)}
    post = RelapseState(f'post_{stroke_name}',
                        cause_type='sequela',
                        get_data_functions=data_funcs)

    susceptible.allow_self_transitions()
    data_funcs = {
        'incidence_rate': lambda _, builder: builder.data.load(f'cause.{stroke_name}.incidence_rate')
    }
    susceptible.add_transition(acute,
                               source_data_type='rate',
                               get_data_functions=data_funcs)
    acute.allow_self_transitions()
    acute.add_transition(post)
    post.allow_self_transitions()
    data_funcs = {
        'relapse_rate': lambda _, builder: builder.data.load(f'cause.{stroke_name}.incidence_rate')
    }
    post.add_transition(acute, source_data_type='rate', get_data_functions=data_funcs)
    post.add_transition(susceptible, source_data_type='time')

    return DiseaseModel(stroke_name, states=[susceptible, acute, post])


def ChronicKidneyDisease():
    raise NotImplementedError()

import math
import matplotlib.pyplot as plt
from matplotlib import colors
from ipywidgets import interact, Text
from pathlib import Path
import pandas as pd, numpy as np
import yaml
from pandas.plotting import register_matplotlib_converters

from vivarium_csu_hypertension_sdc.components.globals import (HYPERTENSIVE_CONTROLLED_THRESHOLD, HYPERTENSION_DRUGS,
                                                              DOSAGE_COLUMNS, SINGLE_PILL_COLUMNS)

DOSE_ADJUST = 10

OFFSETS = {'dr visits': -10,
          'disease events': -5,
           'meds': -50}


class SimulantTrajectoryVisualizer:

    def __init__(self, results_directory, simulant_trajectory_file: str = 'simulant_trajectory.hdf'):
        register_matplotlib_converters()
        results_path = Path(results_directory)

        self.data = load_data(results_path, simulant_trajectory_file)
        model_spec = yaml.full_load((results_path / 'model_specification.yaml').open())
        self.step_size = float(model_spec['configuration']['time']['step_size'])

    def visualize_healthcare_utilization(self):
        v = summarize_hcu(self.data, self.step_size)
        plt.scatter(v.hcu, v.total, color='navy', label='total visits')
        plt.scatter(v.hcu, v.background, color='lightblue', label='background visits only', alpha=0.5)
        upper_lim = math.ceil(max(v.total))
        plt.plot(range(0, upper_lim), range(0, upper_lim), linewidth=3, color='black', linestyle='--')
        plt.legend()
        plt.xlabel('average healthcare utilization rate (# visits/year)')
        plt.ylabel('actual # visits/year in sim')
        plt.title('Healthcare utilization')
        plt.show()

    def visualize_simulant_trajectory(self, enter_sim_id=False, extra_title_key="", starting_sim_id=None):
        data = self.data

        unique_sims = data.reset_index().simulant.drop_duplicates().sort_values()

        if not enter_sim_id:
            arg = (1, len(unique_sims), 1)
        else:
            arg = Text(value=str(unique_sims[0]), placeholder='simulant id') if not starting_sim_id else str(starting_sim_id)

        @interact(simulant=arg, include_pdc=True)
        def _visualize_simulant_trajectory(simulant, include_pdc):
            if isinstance(simulant, str):
                sim_id = int(simulant)
            else:
                sim_id = unique_sims.loc[simulant]
            simulant = data.loc[sim_id]

            sex = simulant.sex[0]
            age = round(simulant.age[0], 1)

            min_sbp, max_sbp = get_min_max_sbp(simulant)
            min_sbp = min(min_sbp, HYPERTENSIVE_CONTROLLED_THRESHOLD)

            fig=plt.figure(figsize=(16, 8))

            plot_sbp(simulant)
            plot_dr_visits(simulant, min_sbp)
            plot_disease_events(simulant, min_sbp)
            plot_dead(simulant)

            plt.ylabel('mmHg')
            sbp_ticks = [x for x in range(min_sbp, max_sbp, 20)]

            offset_ticks = []
            offset_labels = []
            for k, o in OFFSETS.items():
                if k != 'meds':
                    offset_ticks.append(min_sbp + o)
                    offset_labels.append(k)
                else:
                    offset_ticks.extend([min_sbp + o + d * DOSE_ADJUST for d in [0, 0.5, 1, 2]])
                    offset_labels.extend(['none', 'half', 'single', 'double'])

            plt.yticks(ticks=offset_ticks + sbp_ticks,
                       labels=offset_labels + sbp_ticks)
            axes = plt.gca()
            axes.set_ylim((min_sbp + min(OFFSETS.values()) - 2, max_sbp + 5))
            axes.set_xlim(simulant.index[0])

            plot_tx_changes_in_trajectory(simulant, min_sbp)

            axes.legend(loc='lower right', bbox_to_anchor=(0.9, 0), ncol=5, borderaxespad=-1,
                        bbox_transform=fig.transFigure)

            if include_pdc:
                ax2 = axes.twinx()
                ax2.set_ylabel('proportion days covered (pdc)', color='tab:orange')
                ax2.tick_params(axis='y', labelcolor='tab:orange')
                pdc = simulant.pdc

                limits = (min(pdc) * 0.8, max(pdc))
                if limits[0] != limits[1]:
                    ax2.set_ylim(limits)
                min_pdc = min(pdc)
                max_pdc = max(pdc)
                if min_pdc == max_pdc:
                    min_pdc = max_pdc / 2

                digits = 2 if min_pdc < 0.1 and max_pdc < 0.1 else 1
                min_pdc = round(min_pdc, digits)
                max_pdc = round(max_pdc, digits)
                if max_pdc < max(pdc):
                    max_pdc += 0.1

                ticks = np.linspace(min_pdc, max_pdc, 3)
                ax2.set_yticks(ticks)
                ax2.plot(pdc, label='pdc', color='tab:orange')

            plt.title(f'{extra_title_key.capitalize()}: Trajectory for simulant {sim_id}: a {age} year-old {sex}')

        return _visualize_simulant_trajectory(1, True)

    def visualize_simulant_treatments(self, enter_sim_id=False, extra_title_key="", starting_sim_id=None):
        data = self.data

        unique_sims = data.reset_index().simulant.drop_duplicates().sort_values()

        if not enter_sim_id:
            arg = (1, len(unique_sims), 1)
        else:
            arg = Text(value=str(unique_sims[0]), placeholder='simulant id') if not starting_sim_id else str(starting_sim_id)

        @interact(simulant=arg, treatment_graph_style=['bar', 'line'])
        def _visualize_simulant_treatments(simulant, treatment_graph_style):
            if isinstance(simulant, str):
                sim_id = int(simulant)
            else:
                sim_id = unique_sims.loc[simulant]
            simulant = data.loc[sim_id]

            tx_changes = track_treatment_changes(simulant)
            plt.title(f'{extra_title_key.capitalize()}: Treatment transitions for simulant {sim_id}.')
            plot_treatments(tx_changes, treatment_graph_style)

        return _visualize_simulant_treatments(arg, 'bar')


def load_data(results_path, simulant_trajectory_file) -> pd.DataFrame:
    data = pd.read_hdf(results_path / simulant_trajectory_file)
    data['untreated_sbp'] = data['true_sbp'] + data['medication_effect']
    return data


def scale_hcu(hcu, step_size):
    return hcu * pd.Timedelta(days=365.25) / pd.Timedelta(days=step_size)


def summarize_hcu(data, step_size):
    times = list(data.reset_index()['time'].drop_duplicates())
    years = (times[-1] - times[0]) / pd.Timedelta(days=365.25)

    df = data.reset_index()[['simulant', 'last_visit_date', 'last_visit_type']].drop_duplicates().dropna()
    df['last_visit_type'] = df['last_visit_type'].apply(lambda x: 'background' if x == 'background' else 'htn')
    visits = ((df[['simulant', 'last_visit_type']].groupby(['simulant', 'last_visit_type']).size() / years)
              .reset_index().pivot(index='simulant', columns='last_visit_type', values=0).fillna(0))
    visits['hcu'] = scale_hcu(data.reset_index()[['simulant', 'healthcare_utilization_rate']]
                              .groupby('simulant').mean(), step_size)

    visits = visits.sort_values('hcu')
    visits['total'] = visits.background + visits.htn

    return visits


def get_min_max_sbp(simulant):
    sbp = simulant[['true_sbp', 'untreated_sbp', 'high_systolic_blood_pressure_measurement']]
    return math.floor(sbp.min().min() // 10 * 10), math.ceil(sbp.max().max())


def get_dr_visits(simulant):
    attended = (simulant.loc[simulant.last_visit_date == simulant.index].groupby(['last_visit_type'])
                .apply(lambda g: g.last_visit_date.values))

    defaults = {'confirmatory': 'lightcoral',
                'maintenance': 'firebrick',
                'background': 'forestgreen'}
    visits = dict()
    for visit, color in defaults.items():
        if visit in attended.index:
            visits[visit] = (attended[visit], color)
    return visits


def track_treatment_changes(simulant):
    exit_time = simulant.exit_time
    simulant = simulant[DOSAGE_COLUMNS + SINGLE_PILL_COLUMNS]
    tx_changes = pd.DataFrame(columns=['start', 'end', 'drug', 'dose', 'in_single_pill'])
    curr = {'start': simulant.index[0], 'end': pd.NaT, 'tx': simulant.iloc[0]}

    for row in simulant.iterrows():
        if row[1].to_dict() != curr['tx'].to_dict() or row[0] == simulant.index[-1]:
            curr['end'] = row[0]
            for drug, dosage, s in zip(HYPERTENSION_DRUGS, DOSAGE_COLUMNS, SINGLE_PILL_COLUMNS):
                dose = curr['tx'][dosage]
                in_single = curr['tx'][s]
                tx_changes = tx_changes.append({'start': curr['start'], 'end': curr['end'], 'drug': drug,
                                                'dose': dose, 'in_single_pill': in_single}, ignore_index=True)
            curr['start'] = row[0]
            curr['tx'] = row[1]

    max_date = tx_changes.end.max()
    last = tx_changes.loc[tx_changes.end == max_date].copy()
    max_date = exit_time.max() if not np.all(exit_time.isna()) else max_date
    last.loc[:, 'start'] = max_date
    last.loc[:, 'end'] = max_date + pd.Timedelta(days=1)
    tx_changes = tx_changes.append(last)

    return tx_changes


def plot_sbp(simulant):
    sim_time = simulant.index
    sbp_measurements = simulant.loc[simulant.high_systolic_blood_pressure_last_measurement_date == sim_time]
    plt.plot(sim_time, simulant.true_sbp, label='Treated SBP',
             linewidth=3, drawstyle='steps-post', color='darkblue')
    plt.plot(sim_time, simulant.untreated_sbp, label='Untreated SBP',
             linewidth=2, drawstyle='steps-post', color='lightblue')
    plt.scatter(sbp_measurements.index, sbp_measurements.high_systolic_blood_pressure_measurement,
                label='SBP Measurement', color='slateblue', marker='x', s=200)
    plt.axhline(y=HYPERTENSIVE_CONTROLLED_THRESHOLD, color='red', label='SBP controlled threshold', linewidth=4)


def plot_dr_visits(simulant, min_sbp):
    dr_visits = get_dr_visits(simulant)
    for visit, (dates, color) in dr_visits.items():
        plt.scatter(dates, [min_sbp + OFFSETS['dr visits']] * len(dates),
                    label=f'{visit.title()} visit', marker='^', s=150,
                    color=color, edgecolors='black')


def plot_disease_events(simulant, min_sbp):
    events = {'acute_myocardial_infarction': 'navy',
              'post_myocardial_infarction': 'cornflowerblue',
              'acute_ischemic_stroke': 'darkgreen',
              'post_ischemic_stroke': 'darkseagreen',
              'acute_subarachnoid_hemorrhage': 'indigo',
              'post_subarachnoid_hemorrhage': 'mediumpurple',
              'acute_intracerebral_hemorrhage': 'darkmagenta',
              'post_intracerebral_hemorrhage': 'plum'
              }
    for e, color in events.items():
        col = f'{e}_event_time'
        disease_events = simulant.loc[simulant.index == simulant[col], col]
        if not disease_events.empty:
            plt.scatter(disease_events, [min_sbp + OFFSETS['disease events']] * len(disease_events),
                        label=e, marker='D', s=150, color=color, edgecolors='black')


def plot_dead(simulant):
    if 'dead' in simulant.alive.unique():
        death_time = sorted(simulant.exit_time.unique())[-1]
        plt.axvspan(death_time, simulant.index[-1], alpha=0.25, color='lightgrey', label='dead')


def plot_treatments(tx_changes, style='line'):
    if style == 'line':
        offsets = {d: 0.01 * i for i, d in enumerate(HYPERTENSION_DRUGS)}

        for drug in tx_changes.drug.unique():
            df = tx_changes.loc[tx_changes.drug == drug]
            plt.plot(df.start, df.dose + offsets[drug], label=drug, drawstyle='steps-post')

        in_single = tx_changes.loc[tx_changes.in_single_pill == 1]
        plt.scatter(in_single.start, in_single.dose, marker='^', s=200, label='in single pill',
                    color='white', edgecolor='black')

        not_in_single = tx_changes.loc[tx_changes.in_single_pill == 0]
        plt.scatter(not_in_single.start, not_in_single.dose, marker='o', s=200, label='not in single pill',
                    color='white', edgecolor='black')

        plt.yticks([0, 0.5, 1, 2], ['none', 'half', 'single', 'double'])

        plt.legend()
        plt.show()
    elif style == 'bar':
        # these bar charts should just show transitions so no need to have the duplicated extra rows at the end
        # that ensure the line charts show the entire treatment series to the end of the sim
        tx_changes = tx_changes.loc[tx_changes.start < tx_changes.start.max()]
        drug_colors = ['red', 'blue', 'green', 'pink', 'purple']

        num_changes = len(tx_changes.start.unique())
        base_positions = np.arange(num_changes)

        barwidth = 0.2
        plt.xlim(0, len(base_positions))

        for i, drug in enumerate(HYPERTENSION_DRUGS):
            group = tx_changes.loc[tx_changes.drug == drug].sort_values('start')
            positions = [x + barwidth * i for x in base_positions]
            color = drug_colors[i]
            group['bar_heights'] = group.dose

            if sum(group.dose) > 0:
                plt.bar(positions, group.bar_heights, width=barwidth,
                        color=color, edgecolor='white', label=drug, align='edge')

            hatch_mask = group.in_single_pill == 1
            if np.any(hatch_mask):
                group.loc[~hatch_mask, 'bar_heights'] = 0
                plt.bar(positions, group.bar_heights, width=barwidth, color=color, edgecolor='white',
                        label=f'{drug}, in single pill', hatch='x', align='edge')

        for i in range(num_changes):
            plt.axvline(base_positions[i] + (len(HYPERTENSION_DRUGS)) * barwidth, color='black',
                        linewidth=4, linestyle='dashed')

        plt.xlabel('date', fontweight='bold')
        plt.xticks([r + 0.5 for r in range(num_changes)],
                   sorted([pd.Timestamp(d).strftime("%Y-%m-%d") for d in tx_changes.start.unique()]),
                   rotation=90)
        plt.yticks([0, 0.5, 1, 2], ['none', 'half', 'single', 'double'])

        plt.legend()
        plt.show()
    else:
        raise ValueError(f'The only acceptable values for style are "line" or '
                         f'"bar". You provided {style}.')


def plot_tx_changes_in_trajectory(simulant, min_sbp):
    tx_changes = track_treatment_changes(simulant)

    baseline = min_sbp + OFFSETS['meds']
    tx_changes.dose = tx_changes.dose * DOSE_ADJUST + baseline

    offsets = {d: 0.03*DOSE_ADJUST*i for i,d in enumerate(HYPERTENSION_DRUGS)}

    for drug in tx_changes.drug.unique():
        df = tx_changes.loc[tx_changes.drug == drug]
        plt.plot(df.start, df.dose + offsets[drug], label=drug, drawstyle='steps-post')

    in_single = tx_changes.loc[tx_changes.in_single_pill == 1]
    plt.scatter(in_single.start, in_single.dose, marker='^', s=200, label='in single pill',
                color='white', edgecolor='black')

    not_in_single = tx_changes.loc[tx_changes.in_single_pill == 0]
    plt.scatter(not_in_single.start, not_in_single.dose, marker='o', s=200, label='not in single pill',
                color='white', edgecolor='black')



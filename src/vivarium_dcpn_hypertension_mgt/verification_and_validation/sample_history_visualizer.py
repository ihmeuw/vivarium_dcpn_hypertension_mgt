import math
import matplotlib.pyplot as plt
from matplotlib import colors
from ipywidgets import interact, Text
from pathlib import Path
import pandas as pd, numpy as np
import yaml
from pandas.plotting import register_matplotlib_converters

OFFSETS = {'tx profile': -17,
          'dr visits': -10,
          'disease events': -5,
          'rx filled': -14}


class SampleHistoryVisualizer:

    def __init__(self, results_directory, sample_history_file: str = 'sample_history.hdf'):
        register_matplotlib_converters()
        results_path = Path(results_directory)

        self.data = load_data(results_path, sample_history_file)
        model_spec = yaml.full_load((results_path / 'model_specification.yaml').open())
        self.step_size = float(model_spec['configuration']['time']['step_size'])

    def visualize_healthcare_utilization(self):
        v = summarize_hcu(self.data, self.step_size)
        plt.scatter(v.hcu, v.total, color='navy', label='total visits')
        plt.scatter(v.hcu, v.background, color='lightblue', label='background visits only', alpha=0.5)
        plt.plot(range(0, 15), range(0, 15), linewidth=3, color='black')
        plt.legend()
        plt.xlabel('average healthcare utilization rate (# visits/year)')
        plt.ylabel('actual # visits/year in sim')
        plt.title('Healthcare utilization')
        plt.show()

    def visualize_simulant_trajectory(self):
        data = self.data
        step_size = self.step_size

        @interact(simulant_id=Text(
            value='903',
            placeholder='Type simulant id'),
            include_hcu_rate=True)
        def _visualize_simulant_trajectory(simulant_id, include_hcu_rate):
            try:
                simulant = data.loc[int(simulant_id)]
            except:
                raise ValueError(f'Simulant {simulant_id} not exist')

            sex = simulant.sex[0]
            age = round(simulant.age[0], 1)

            min_sbp, max_sbp = get_min_max_sbp(simulant)

            plt.figure(figsize=(16, 8))

            plot_tx_profiles(simulant)
            plot_sbp(simulant)
            plot_dr_visits(simulant, min_sbp)
            plot_disease_events(simulant, min_sbp)
            plot_medication(simulant, min_sbp)
            plot_dead(simulant)

            plt.ylabel('mmHg')
            sbp_ticks = [x for x in range(min_sbp, max_sbp, 20)]
            plt.yticks(ticks=[min_sbp + o for o in OFFSETS.values()] + sbp_ticks,
                       labels=list(OFFSETS.keys()) + sbp_ticks)
            axes = plt.gca()
            axes.set_ylim((min_sbp - 20, max_sbp + 5))
            axes.set_xlim(simulant.index[0])

            plt.legend()

            if include_hcu_rate:
                ax2 = axes.twinx()
                ax2.set_ylabel('# healthcare visits/year', color='tab:orange')
                ax2.tick_params(axis='y', labelcolor='tab:orange')
                hcu = scale_hcu(simulant.healthcare_utilization_rate, step_size)

                limits = (min(hcu) * 0.8, max(hcu))
                if limits[0] != limits[1]:
                    ax2.set_ylim(limits)
                ticks = np.linspace(round(min(hcu), 1), round(max(hcu), 1), 4)
                ax2.set_yticks(ticks)
                ax2.plot(hcu, label='hcu rate', color='tab:orange')

            plt.title(f'Sample history for simulant {simulant_id}: a {age} year-old {sex}')

        return _visualize_simulant_trajectory


def load_data(results_path, sample_history_file):
    data = pd.read_hdf(results_path / sample_history_file)
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
    missed = simulant.loc[simulant.last_missed_appt_date == simulant.index]

    defaults = {'maintenance': 'orangered',
                'reassessment': 'darkorange',
                'confirmatory': 'coral',
                'background': 'forestgreen',
                'missed': 'grey'}
    visits = dict()
    for visit, color in defaults.items():
        if visit in attended.index:
            visits[visit] = (attended[visit], color)
    if not missed.empty:
        visits['missed'] = (missed.last_missed_appt_date, defaults['missed'])
    return visits


def track_profiles(simulant):
    profile_changes = pd.DataFrame(columns=['start', 'end', 'profile'])
    curr = {'start': simulant.index[0], 'end': pd.NaT, 'profile': simulant.treatment_profile[0]}

    for time, profile in simulant.treatment_profile.iteritems():
        if profile != curr['profile'] or time == simulant.index[-1]:
            curr['end'] = time
            profile_changes = profile_changes.append(curr, ignore_index=True)
            curr['start'] = time
            curr['profile'] = profile

    return profile_changes


def get_color_for_profile(p):
    if p == 'no_treatment_1':
        return 'darkgrey'
    elif 'initial' in p:
        return 'bisque'
    elif 'mono' in p:
        starting = 'lightcoral'
        delta = [0.1, 0.05, 0.01]
    elif 'combo' in p:
        starting = 'palegreen'
        delta = [0.05, 0.1, 0.05]
    elif 'elderly' in p:
        starting = 'lightblue'
        delta = [0.01, 0.02, 0.1]

    position = float(p.split('_')[-1])
    c = list(colors.to_rgb(starting))
    return [min(v - position * d, 1) for v, d in zip(c, delta)]


def plot_tx_profiles(simulant):
    profiles = track_profiles(simulant)
    for p in profiles.itertuples():
        color = get_color_for_profile(p.profile)
        plt.axvspan(p.start, p.end, ymin=0, ymax=0.1, alpha=0.35, color=color, label=p.profile)


def plot_sbp(simulant):
    sim_time = simulant.index
    sbp_measurements = simulant.loc[simulant.high_systolic_blood_pressure_last_measurement_date == sim_time]
    plt.plot(sim_time, simulant.true_sbp, label='Treated SBP',
             linewidth=3, drawstyle='steps-post', color='darkblue')
    plt.plot(sim_time, simulant.untreated_sbp, label='Untreated SBP',
             linewidth=2, drawstyle='steps-post', color='lightblue')
    plt.scatter(sbp_measurements.index, sbp_measurements.high_systolic_blood_pressure_measurement,
                label='SBP Measurement', color='slateblue', marker='x', s=200)


def plot_dr_visits(simulant, min_sbp):
    dr_visits = get_dr_visits(simulant)
    for visit, (dates, color) in dr_visits.items():
        plt.scatter(dates, [min_sbp + OFFSETS['dr visits']] * len(dates),
                    label=f'{visit.title()} visit', marker='^', s=150,
                    color=color, edgecolors='black')


def plot_disease_events(simulant, min_sbp):
    events = {'ischemic_heart_disease': 'tab:blue',
              'ischemic_stroke': 'tab:orange',
              'intracerebral_hemorrhage': 'tab:green',
              'subarachnoid_hemorrhage': 'tab_red'}
    for e, color in events.items():
        col = f'{e}_event_time'
        disease_events = simulant.loc[simulant.index == simulant[col], col]
        if not disease_events.empty:
            plt.scatter(disease_events, [min_sbp + OFFSETS['disease events']] * len(disease_events),
                        label=e, marker='D', s=150, color=color, edgecolors='black')


def plot_medication(simulant, min_sbp):
    rx = simulant.rx_filled.apply(lambda x: colors.to_rgb('g') if x else colors.to_rgb('r'))
    plt.scatter(rx.index, [min_sbp + OFFSETS['rx filled']] * len(rx), c=np.stack(rx.values), cmap=plt.get_cmap('Set1'))


def plot_dead(simulant):
    if 'dead' in simulant.alive.unique():
        death_time = sorted(simulant.exit_time.unique())[-1]
        plt.axvspan(death_time, simulant.index[-1], alpha=0.25, color='lightgrey', label='dead')
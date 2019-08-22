import pandas as pd


class SampleHistoryObserver:

    configuration_defaults = {
        'metrics': {
            'sample_history_observer': {
                'sample_size': 1000,
                'path': f'/share/costeffectiveness/results/vivarium_dcpn_hypertension_management/sample_history.hdf'
            }
        }
    }

    @property
    def name(self):
        return "sample_history_observer"

    def __init__(self):
        self.history_snapshots = []
        self.sample_index = None

    def setup(self, builder):
        self.clock = builder.time.clock()
        self.sample_history_parameters = builder.configuration.metrics.sample_history_observer
        self.randomness = builder.randomness.get_stream("sample_history")

        # sets the sample index
        builder.population.initializes_simulants(self.get_sample_index)

        columns_required = ['alive', 'age', 'sex', 'entrance_time', 'exit_time',
                            'cause_of_death',
                            'years_lived_with_disability',
                            'years_of_life_lost',
                            'ischemic_heart_disease_event_time',
                            'ischemic_stroke_event_time',
                            'intracerebral_hemorrhage_event_time',
                            'subarachnoid_hemorrhage_event_time',
                            'followup_date',
                            'followup_type',
                            'last_visit_date',
                            'last_visit_type',
                            'high_systolic_blood_pressure_measurement',
                            'high_systolic_blood_pressure_last_measurement_date',
                            'treatment_profile']
        self.population_view = builder.population.get_view(columns_required)

        # keys will become column names in the output
        self.pipelines = {'mortality_rate': builder.value.get_value('mortality_rate'),
                          'disability_weight': builder.value.get_value('disability_weight'),
                          'rx_filled': builder.value.get_value('rx_fill.currently_filled'),
                          'medication_effect': builder.value.get_value('hypertension_drugs.effect_size'),
                          'true_sbp': builder.value.get_value('high_systolic_blood_pressure.exposure'),
                          'ischemic_heart_disease_incidence_rate':
                              builder.value.get_value('ischemic_heart_disease.incidence_rate'),
                          'ischemic_stroke_incidence_rate':
                              builder.value.get_value('ischemic_stroke.incidence_rate'),
                          'intracerebral_hemorrhage_incidence_rate':
                              builder.value.get_value('intracerebral_hemorrhage_disease.incidence_rate'),
                          'subarachnoid_hemorrhage_disease_incidence_rate':
                              builder.value.get_value('subarachnoid_hemorrhage_disease.incidence_rate'),
                          }

        builder.event.register_listener('collect_metrics', self.record)
        builder.event.register_listener('simulation_end', self.dump)

    def get_sample_index(self, pop_data):
        sample_size = self.sample_history_parameters.sample_size
        if sample_size is None or sample_size > len(pop_data.index):
            sample_size = len(pop_data.index)
        draw = self.randomness.get_draw(pop_data.index)
        priority_index = [i for d, i in sorted(zip(draw, pop_data.index), key=lambda x:x[0])]
        self.sample_index = pd.Index(priority_index[:sample_size])

    def record(self, event):
        pop = self.population_view.get(self.sample_index)

        pipeline_results = []
        for name, pipeline in self.pipelines.items():
            values = pipeline(pop.index)
            if name == 'mortality_rate':
                values = values.sum(axis=1)
            values = values.rename(name)
            pipeline_results.append(values)

        record = pd.concat(pipeline_results + [pop], axis=1)
        record['time'] = self.clock()
        record.index.rename("simulant", inplace=True)
        record.set_index('time', append=True, inplace=True)

        self.history_snapshots.append(record)

    def dump(self, event):
        sample_history = pd.concat(self.history_snapshots, axis=0)
        sample_history.to_hdf(self.sample_history_parameters.path, key='histories')
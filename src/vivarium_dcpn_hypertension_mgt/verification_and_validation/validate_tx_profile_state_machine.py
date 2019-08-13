import argparse

from pathlib import Path
import matplotlib.backends.backend_pdf

from vivarium import initialize_simulation_from_model_specification


def setup_sim(country, guideline):
    model_spec = Path(__file__).parent.parent / f'model_specifications/{country}.yaml'

    sim = initialize_simulation_from_model_specification(str(model_spec))

    sim.configuration.update({'hypertension_drugs': {'guideline': guideline},
                              'population': {'population_size': 100}})

    sim.setup()
    return sim


def plot_sim_tx_profile_states(tx_profile_machine, pdf_name):
    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_name)

    for tx_profile in tx_profile_machine.states:
        pdf.savefig(tx_profile.graph_domain_filters())

    pdf.close()


def main():
    parser = argparse.ArgumentParser(description='Validate simulation treatment profile machine '
                                                 'for some country + guideline.')
    parser.add_argument('country', choices=['china', 'russian_federation', 'malaysia', 'nigeria'],
                        help='Country to run simulation for.')
    parser.add_argument('guideline', choices=['baseline', 'china', 'aha', 'who'],
                        help='Guideline to run simulation for.')

    args = parser.parse_args()

    sim = setup_sim(args.country, args.guideline)

    tx_profile_machine = sim.component_manager.get_component('machine.treatment_profile')
    tx_profile_machine.validate()

    plot_sim_tx_profile_states(tx_profile_machine, f'{args.country}_{args.guideline}.pdf')


if __name__ == "__main__":
    main()

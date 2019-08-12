from pathlib import Path

from vivarium import initialize_simulation_from_model_specification

def setup_sim(country, guideline):
    model_spec = Path(__file__).
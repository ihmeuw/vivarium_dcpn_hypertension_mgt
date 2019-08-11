import pandas as pd
import pytest

from vivarium_dcpn_hypertension_mgt.components import hypertension_medication, utilities


def test_TreatmentProfile_validate():
    domain_filters = ['a', 'b', 'c']

    tx_profile = hypertension_medication.TreatmentProfile('test', 0, {}, domain_filters)

    assert not tx_profile.is_valid()

    tx_profile.add_transition(None, probability_func=lambda index: pd.Series(0.25, index=index), domain_filter='a')
    tx_profile.add_transition(None, domain_filter='b')
    tx_profile.add_transition(None, probability_func=lambda index: pd.Series(0.5, index=index), domain_filter='a')
    tx_profile.add_transition(None, probability_func=lambda index: pd.Series(0.25, index=index), domain_filter='a')

    assert tx_profile.is_valid()


def test_TreatmentProfile_add_transition():
    domain_filters = ['a', 'b', 'c']

    tx_profile = hypertension_medication.TreatmentProfile('test', 0, {}, domain_filters)

    with pytest.raises(ValueError, match='invalid for this state'):
        tx_profile.add_transition(None, domain_filter='fail_me')

    tx_profile.add_transition(None, probability_func=lambda index: pd.Series(0.25, index=index), domain_filter='a')
    tx_profile.add_transition(None, domain_filter='b')
    tx_profile.add_transition(None, probability_func=lambda index: pd.Series(0.5, index=index), domain_filter='a')

    with pytest.raises(ValueError, match='over 1'):
        tx_profile.add_transition(None, probability_func=lambda index: pd.Series(0.35, index=index), domain_filter='a')


@pytest.mark.parametrize('efficacy, result_positions', [(0.9, [1, 2]),
                                                        (1.2, [2, 3]),
                                                        (4.5, [5, 6]),
                                                        (4.55, [6]),
                                                        (5, [6]),
                                                        (5.2, [])])
def test_get_closest_in_efficacy_in_ramp(efficacy, result_positions):
    efficacy_data = pd.Series([1, 2, 3, 4, 4.5, 5], index=[f'ramp_{n+1}' for n in range(6)])

    closest_ramp_positions = utilities.get_closest_in_efficacy_in_ramp(efficacy, efficacy_data, 'ramp')

    assert set([f'ramp_{n}' for n in result_positions]) == set(closest_ramp_positions.index)
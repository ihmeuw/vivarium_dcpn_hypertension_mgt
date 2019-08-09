import pandas as pd
import pytest

from vivarium_dcpn_hypertension_mgt.components import hypertension_medication


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

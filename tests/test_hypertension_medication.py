import pandas as pd

import pytest

from vivarium_dcpn_hypertension_mgt.components.hypertension_medication import (FilterDomain, check_subdomains_complete,
                                                                               collapse_intervals, check_interval_contains)


@pytest.mark.parametrize('intervals, match',
                         [([pd.Interval(1, 2, closed='right'), pd.Interval(2, 4, closed='left')], 'Overlapping'),
                          ([pd.Interval(1, 12, closed='left'), pd.Interval(4, 400, closed='neither')], 'Overlapping'),
                          ([pd.Interval(1, 2, closed='left'), pd.Interval(2, 4, closed='left'),
                            pd.Interval(4, 10, closed='right')], 'Gaps'),
                          ([pd.Interval(1, 2, closed='both'), pd.Interval(2.1, 100, closed='both')], 'Gaps'),
                          ([pd.Interval(1, 2, closed='both'), pd.Interval(2, 15, closed='neither'),
                            pd.Interval(15, 16, closed='neither')], 'Gaps'),
                          ([pd.Interval(1, 2, closed='neither'), pd.Interval(2, 4, closed='both'),
                            pd.Interval(4, 12, closed='both')], 'Overlapping')])
def test_collapse_intervals_fail(intervals, match):
    with pytest.raises(ValueError, match=match):
        collapse_intervals(pd.Series(intervals).sample(frac=1))


@pytest.mark.parametrize('intervals, collapsed', [([pd.Interval(1, 2, closed='both'),
                                                    pd.Interval(2, 100, closed='neither'),
                                                    pd.Interval(100, 100.01, closed='left')],
                                                   pd.Interval(1, 100.01, closed='left')),
                                                  ([pd.Interval(1, 2, closed='left'),
                                                    pd.Interval(2, 4, closed='both'),
                                                    pd.Interval(4, 20, closed='right')],
                                                   pd.Interval(1, 20, closed='both')),
                                                  ([pd.Interval(1, 3, closed='neither'),
                                                    pd.Interval(3, 5, closed='left'),
                                                    pd.Interval(5, 7, closed='both'),
                                                    pd.Interval(7, 10, closed='right'),
                                                    pd.Interval(10, 10.01, closed='right')],
                                                   pd.Interval(1, 10.01, closed='right')),
                                                  ([pd.Interval(1, 100, closed='neither')],
                                                   pd.Interval(1, 100, closed='neither'))])
def test_collapse_intervals_pass(intervals, collapsed):
    assert collapse_intervals(pd.Series(intervals).sample(frac=1)) == collapsed


@pytest.mark.parametrize("a, b, result", [(pd.Interval(1, 10), pd.Interval(0, 5), False),
                                          (pd.Interval(1, 10), pd.Interval(5, 12), False),
                                          (pd.Interval(5, 10), pd.Interval(1, 5), False),
                                          (pd.Interval(5, 10), pd.Interval(10, 12), False),
                                          (pd.Interval(5, 10, closed='right'), pd.Interval(5, 7, closed='left'), False),
                                          (pd.Interval(5, 10, closed='left'), pd.Interval(7, 10, closed='right'), False),
                                          (pd.Interval(5, 10), pd.Interval(7, 9), True),
                                          (pd.Interval(5, 10, closed='left'), pd.Interval(5, 7), True),
                                          (pd.Interval(5, 10, closed='right'), pd.Interval(7, 10, closed='both'), True),
                                          (pd.Interval(5, 10), pd.Interval(5, 10), True)])
def test_check_interval_contains(a, b, result):
    assert check_interval_contains(a, b) == result


def test_check_subdomains_complete():
    sub_domains = pd.DataFrame({'a': [pd.Interval(0, 10),
                                      pd.Interval(0, 1),
                                      pd.Interval(1, 5),
                                      pd.Interval(5, 10),
                                      pd.Interval(1, 10),

                                      pd.Interval(0, 5),
                                      pd.Interval(5, 10),
                                      pd.Interval(0, 3),
                                      pd.Interval(3, 5)],
                                'b': [pd.Interval(0, 2),
                                      pd.Interval(2, 3),
                                      pd.Interval(2, 2.5),
                                      pd.Interval(2, 2.5),
                                      pd.Interval(2.5, 3),

                                      pd.Interval(0, 2),
                                      pd.Interval(0, 3),
                                      pd.Interval(2, 3),
                                      pd.Interval(2, 3)],
                                'c': ['First'] * 5 + ['Second'] * 4})

    full_domain = {'a': pd.Interval(0, 10), 'b': pd.Interval(0, 3), 'c': {'First', 'Second'}}

    assert check_subdomains_complete(sub_domains, full_domain)


def test_contains_pass():
    fd1 = FilterDomain()

    dimensions = dict()
    for k, v in fd1.dimensions.items():
        if isinstance(v, pd.Interval):
            dimensions[k] = pd.Interval(v.left, v.mid)
        else:
            dimensions[k] = {v.pop()}
        fd2 = FilterDomain(dimensions=dimensions)
        assert fd2 in fd1


def test_contains_fail():
    fd1 = FilterDomain()

    # range dimension
    fd2 = FilterDomain(dimensions={'systolic_blood_pressure': pd.Interval(50, 300)})
    assert fd2 not in fd1

    # set dimensions
    fd2 = FilterDomain(dimensions={'sex': {'Male', 'random'}})
    assert fd2 not in fd1
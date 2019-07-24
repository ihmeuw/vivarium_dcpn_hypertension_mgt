import os
import sys

from pathlib import Path
import numpy as np
import pandas as pd
import drmaa
from loguru import logger

from risk_distributions import EnsembleDistribution
from vivarium_gbd_access.gbd import ARTIFACT_FOLDER
from vivarium_public_health.dataset_manager import Artifact
from vivarium_public_health.risks.data_transformations import pivot_categorical

HYPERTENSION_THRESHOLD = 140
HDF_KEY = 'proportion_hypertensive'


def prep_weights(art):
    weights = art.load('risk_factor.high_systolic_blood_pressure.exposure_distribution_weights')
    index_cols = weights.index.names.difference({'location', 'parameter'})
    weights = pivot_categorical(weights.reset_index())
    weights = weights.drop(columns=index_cols)
    if 'glnorm' in weights.columns:
        if np.any(weights['glnorm']):
            raise NotImplementedError('glnorm distribution is not supported')
        weights = weights.drop(columns='glnorm')
    return weights


def calc_hypertensive(location, draw):
    art_path = ARTIFACT_FOLDER / 'vivarium_dcpn_hypertension_mgt' / f'{location}.hdf'
    art = Artifact(str(art_path), filter_terms=[f'draw=={draw}'])

    # I can drop indices and know that the means/sds/weights will be aligned b/c we sort the data in vivarium_inputs
    mean = art.load('risk_factor.high_systolic_blood_pressure.exposure')
    demographic_index = mean.index  # but we'll need it later for the proportions
    mean = mean.reset_index(drop=True)
    sd = art.load('risk_factor.high_systolic_blood_pressure.exposure_standard_deviation').reset_index(drop=True)

    # these will be the same for all draws
    weights = prep_weights(art)
    threshold = pd.Series(HYPERTENSION_THRESHOLD, index=mean.index)

    dist = EnsembleDistribution(weights=weights, mean=mean[f'draw_{draw}'], sd=sd[f'draw_{draw}'])
    props = dist.cdf(threshold).fillna(0)

    props.index = demographic_index
    props.name = f'draw_{draw}'
    props = props.droplevel('parameter').fillna(0)

    return props


def calc_parallel(location):
    output_path = Path(ARTIFACT_FOLDER / f'vivarium_dcpn_hypertension_mgt/proportion_hypertensive/{location}')
    output_path.mkdir(parents=True)

    with drmaa.Session() as s:
        jt = s.createJobTemplate()
        jt.remoteCommand = sys.executable
        jt.nativeSpecification = '-l m_mem_free=1G,fthread=1,h_rt=00:30:00 -q all.q -P proj_cost_effect_dcpn'
        jt.args = [__file__, location, 'draw']
        jt.jobName = f'{location}_prop_hypertensive_draw'

        draw_jids = s.runBulkJobs(jt, 1, 1000, 1)
        draw_jid_base = draw_jids[0].split('.')

        jt.nativeSpecification = f'-l m_mem_free=10G,fthread=1,h_rt=01:30:00 ' \
            f'-q all.q -P proj_cost_effect_dcpn -hold_jid {draw_jid_base}'
        jt.args = [__file__, location, 'aggregate']
        jt.jobName = f'{location}_prop_hypertensive_aggregate'

        agg_jid = s.runJob(jt)

        logger.info(f'Draws for {location} have been submitted with jid {draw_jid_base}. '
                    f'They will be aggregated by jid {agg_jid}.')


def aggregate(out_dir, location):
    draw_dir = out_dir / location
    draws = []
    for f in draw_dir.iterdir():
        draws.append(pd.read_hdf(f, HDF_KEY))
        f.unlink()

    data = pd.concat(draws, axis=1)
    data.to_hdf(out_dir / f'{location}.hdf', HDF_KEY)
    draw_dir.rmdir()


def main():
    location = str(sys.argv[1])
    task_type = str(sys.argv[2])

    out_dir = ARTIFACT_FOLDER / f'vivarium_dcpn_hypertension_mgt/proportion_hypertensive/'

    if task_type == 'draw':
        draw = int(os.environ['SGE_TASK_ID']) - 1

        prop_hypertensive = calc_hypertensive(location, draw)

        file_path = out_dir / f'{location}/{draw}.hdf'

        prop_hypertensive.to_hdf(file_path, HDF_KEY)

    elif task_type == 'aggregate':

    else:
        raise ValueError(f'Unknown task type {task_type}.')


if __name__ == '__main__':
    main()

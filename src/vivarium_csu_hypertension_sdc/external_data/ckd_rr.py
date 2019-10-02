import os
import sys

from gbd_mapping import risk_factors
import numpy as np
import pandas as pd
from risk_distributions import EnsembleDistribution
from scipy import optimize
from vivarium import Artifact
from vivarium_gbd_access.gbd import ARTIFACT_FOLDER


RR_DATA_FOLDER = ARTIFACT_FOLDER / 'vivarium_csu_hypertension_sdc' / 'ckd_rr'


def calc_ckd_rr(location, draw):
    artifact = Artifact(RR_DATA_FOLDER.parent / f'{location}.hdf', filter_terms=[f'draw == {draw}'])

    tmred = risk_factors.high_systolic_blood_pressure.tmred
    tmrel = (tmred.min + tmred.max) / 2
    scale = float(risk_factors.high_systolic_blood_pressure.relative_risk_scalar)

    w = artifact.load('risk_factor.high_systolic_blood_pressure.exposure_distribution_weights')
    w = w.reset_index()
    # Weights are constant across all groups, so select one.
    group = (w.year_start == 2017) & (w.age_start == 40) & (w.sex == 'Female')
    w = w.loc[group, ['parameter', 'value']].set_index('parameter').drop(labels='glnorm').value

    x = artifact.load('risk_factor.high_systolic_blood_pressure.exposure')
    idx = x.xs(2017, level='year_start', drop_level=False).index
    x = x.reset_index()
    # Sim runs out into the future, so only compute the last years worth of
    # RR data.  Ages less than 30 have no exposure.
    x = x.loc[(x.year_start == 2017) & (x.age_start >= 30)]
    x = x.set_index([c for c in x.columns if 'draw' not in c])
    sub_idx = x.index
    x_v = x.values

    sd = artifact.load('risk_factor.high_systolic_blood_pressure.exposure_standard_deviation')
    sd = sd.reset_index()
    sd = sd.loc[(sd.year_start == 2017) & (sd.age_start >= 30)]
    sd = sd.set_index([c for c in sd.columns if 'draw' not in c])
    sd_v = sd.values

    paf = artifact.load('risk_factor.high_systolic_blood_pressure.population_attributable_fraction')
    paf = paf.reset_index()
    paf = paf.loc[(paf.year_start == 2017)
                  & (paf.affected_entity == 'chronic_kidney_disease')
                  & (paf.age_start >= 30)]
    paf = paf.set_index([c for c in paf.columns if 'draw' not in c])
    paf_v = paf.values

    def find_rr(weights, mean, standard_dev, attributable_fraction, sample_size=10000):
        target = 1 / (1 - attributable_fraction)

        dist = EnsembleDistribution(weights, mean=mean, sd=standard_dev)
        q = .98 * np.random.random(sample_size) + 0.01
        x_ = dist.ppf(q)

        def loss(guess):
            y = np.maximum(x_ - tmrel, 0) / scale
            mean_rr = 1 / sample_size * np.sum(guess ** y)
            return (mean_rr - target) ** 2

        return optimize.minimize(loss, 2)

    out = pd.DataFrame(data=1., columns=[f'draw_{draw}'], index=idx)

    for i in range(len(sub_idx)):
        rr = find_rr(w, x_v[i, draw], sd_v[i, draw], paf_v[i, draw])
        out.loc[sub_idx[i], f'draw_{draw}'] = rr.x[0]

    return out


def aggregate(out_dir, location):
    draw_dir = out_dir / location
    draws = []
    for f in draw_dir.iterdir():
        draws.append(pd.read_hdf(f))
        # f.unlink()

    data = pd.concat(draws, axis=1)
    data.to_hdf(out_dir / f'{location}.hdf')
    draw_dir.rmdir()


def main():
    location = str(sys.argv[1])
    task_type = str(sys.argv[2])
    out_dir = RR_DATA_FOLDER

    if task_type == 'draw':
        draw = int(os.environ['SGE_TASK_ID']) - 1
        rr_draw = calc_ckd_rr(location, draw)
        file_path = out_dir / f'{location}/{draw}.hdf'
        rr_draw.to_hdf(file_path, key='data')
    elif task_type == 'aggregate':
        aggregate(out_dir, location)
    else:
        raise ValueError(f'Unknown task type {task_type}.')


if __name__ == '__main__':
    main()

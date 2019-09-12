from pathlib import Path

import hashlib
import numpy as np
import pandas as pd
from scipy.stats import norm, beta

from gbd_mapping import risk_factors
from vivarium import Artifact
from vivarium.framework.artifact import get_location_term
from vivarium.framework.artifact.hdf import EntityKey
from vivarium_gbd_access import gbd
from vivarium_inputs.data_artifact.loaders import loader
from vivarium_inputs.data_artifact.utilities import split_interval
from vivarium_inputs import utilities, globals, utility_data, core

from vivarium_csu_hypertension_sdc.external_data.proportion_hypertensive import HYPERTENSION_DATA_FOLDER, HYPERTENSION_HDF_KEY


def build_artifact(path, location):
    artifact = create_new_artifact(path, location)

    write_demographic_data(artifact, location)

    write_ihd_data(artifact, location)
    write_stroke_data(artifact, location, 'ischemic_stroke', 9310, 10837)
    write_stroke_data(artifact, location, 'subarachnoid_hemorrhage', 18731, 18733)
    write_stroke_data(artifact, location, 'intracerebral_hemorrhage', 9311, 10836)


def build_treatment_artifact(path, location):
    artifact = create_new_artifact(path, location)

    # FIXME: merge this with the above build_artifact and remove the duplicated write_demographic_data
    write_demographic_data(artifact, location)

    write_proportion_hypertensive(artifact, location)
    write_hypertension_medication_data(artifact, location)
    write_utilization_rate(artifact, location)


def write_demographic_data(artifact, location):
    load = get_load(location)

    prefix = 'population.'
    measures = ["structure", "age_bins", "theoretical_minimum_risk_life_expectancy", "demographic_dimensions"]
    for m in measures:
        key = prefix + m
        write(artifact, key, load(key))


def write_ihd_data(artifact, location):
    load = get_load(location)

    # Metadata
    key = 'cause.ischemic_heart_disease.sequelae'
    write(artifact, key, load(key))
    key = 'cause.ischemic_heart_disease.restrictions'
    write(artifact, key, load(key))

    # Measures for Disease Model
    key = 'cause.ischemic_heart_disease.cause_specific_mortality_rate'
    write(artifact, key, load(key))

    # Measures for Disease States
    mi = ['acute_myocardial_infarction_first_2_days', 'acute_myocardial_infarction_3_to_28_days']
    p, dw = load_prev_dw(mi, location)
    write(artifact, 'sequela.acute_myocardial_infarction.prevalence', p)
    write(artifact, 'sequela.acute_myocardial_infarction.disability_weight', dw)
    write(artifact, 'sequela.acute_myocardial_infarction.excess_mortality_rate', load_em_from_meid(1814, location))

    post_mi = ['mild_angina_due_to_ischemic_heart_disease',
               'moderate_angina_due_to_ischemic_heart_disease',
               'severe_angina_due_to_ischemic_heart_disease',
               'mild_heart_failure_due_to_ischemic_heart_disease',
               'moderate_heart_failure_due_to_ischemic_heart_disease',
               'severe_heart_failure_due_to_ischemic_heart_disease',
               'asymptomatic_angina_due_to_ischemic_heart_disease',
               'asymptomatic_ischemic_heart_disease_following_myocardial_infarction',
               'controlled_medically_managed_heart_failure_due_ischemic_heart_disease']

    p, dw = load_prev_dw(post_mi, location)
    write(artifact, 'sequela.post_myocardial_infarction.prevalence', p)
    write(artifact, 'sequela.post_myocardial_infarction.disability_weight', dw)
    write(artifact, 'sequela.post_myocardial_infarction.excess_mortality_rate', load_em_from_meid(15755, location))

    # Measures for Transitions
    key = 'cause.ischemic_heart_disease.incidence_rate'
    write(artifact, key, load(key))


def write_stroke_data(artifact, stroke_name, location, acute_meid, post_meid):
    load = get_load(location)

    # Metadata
    key = f'cause.{stroke_name}.sequelae'
    write(artifact, key, load(key))
    key = f'cause.{stroke_name}.restrictions'
    write(artifact, key, load(key))
    # Measures for Disease Model
    key = f'cause.{stroke_name}.cause_specific_mortality_rate'
    write(artifact, key, load(key))

    # Measures for Disease States
    acute = [f'acute_{stroke_name}_severity_level_{i}' for i in range(1, 6)]
    p, dw = load_prev_dw(acute, location)
    write(artifact, f'sequela.acute_{stroke_name}.prevalence', p)
    write(artifact, f'sequela.acute_{stroke_name}.disability_weight', dw)
    write(artifact, f'sequela.acute_{stroke_name}.excess_mortality_rate', load_em_from_meid(acute_meid, location))

    post = [f'chronic_{stroke_name}_severity_level_{i}' for i in range(1, 6)] + [f'asymptomatic_chronic_{stroke_name}']
    p, dw = load_prev_dw(post, location)
    write(artifact, f'sequela.post_{stroke_name}.prevalence', p)
    write(artifact, f'sequela.post_{stroke_name}.disability_weight', dw)
    write(artifact, f'sequela.post_{stroke_name}.excess_mortality_rate', load_em_from_meid(post_meid, location))

    # Measures for Transitions
    key = f'cause.{stroke_name}.incidence_rate'
    write(artifact, key, load(key))


def write_ckd_data(artifact, location):
    load = get_load(location)

    # Metadata
    key = f'cause.chronic_kidney_disease.restrictions'
    write(artifact, key, load(key))

    # Measures for Disease Model
    key = f'cause.chronic_kidney_disease.cause_specific_mortality_rate'
    write(artifact, key, load(key))

    # Measures for Disease States
    key = 'cause.chronic_kidney_disease.prevalence'
    write(artifact, key, load(key))

    # TODO: Find source for YLDs at the draw level to back calc disability weight.

    key = 'cause.chronic_kidney_disease.excesss_mortality_rate'
    write(artifact, key, load(key))

    # Measures for Transitions
    key = 'cause.chronic_kidney_disease.incidence_rate'
    write(artifact, key, load(key))


def write_sbp_data(artifact, location):
    load = get_load(location)

    prefix = 'risk_factor.high_systolic_blood_pressure.'
    measures = ["restrictions", "distribution", "tmred", "exposure", "exposure_standard_deviation",
                "relative_risk_scalar",
                "exposure_distribution_weights"]
    for m in measures:
        key = prefix + m
        write(artifact, key, load(key))

    sbp = risk_factors.high_systolic_blood_pressure

    data = gbd.get_paf(sbp.gbd_id, utility_data.get_location_id(location))
    data = data[data.metric_id == globals.METRICS['Percent']]
    data = data[data.measure_id == globals.MEASURES['YLDs']]
    data = utilities.convert_affected_entity(data, 'cause_id')
    data.loc[data['measure_id'] == globals.MEASURES['YLDs'], 'affected_measure'] = 'incidence_rate'
    data = (
        data
            .groupby(['affected_entity', 'affected_measure'])
            .apply(utilities.normalize, fill_value=0)
            .reset_index(drop=True)
    )
    data = data.filter(globals.DEMOGRAPHIC_COLUMNS + ['affected_entity', 'affected_measure'] + globals.DRAW_COLUMNS)
    data = utilities.scrub_gbd_conventions(data, location)
    data = utilities.sort_hierarchical_data(data)

    key = prefix + 'population_attributable_fraction'
    write(artifact, key, data)
    ckd_paf = data[data.affected_entity == 'chronic_kidney_disease']

    data = gbd.get_relative_risk(sbp.gbd_id, utility_data.get_location_id(location))
    data = utilities.convert_affected_entity(data, 'cause_id')
    morbidity = data.morbidity == 1
    mortality = data.mortality == 1
    data.loc[morbidity & mortality, 'affected_measure'] = 'incidence_rate'
    data.loc[morbidity & ~mortality, 'affected_measure'] = 'incidence_rate'
    data.loc[~morbidity & mortality, 'affected_measure'] = 'excess_mortality'
    data = core.filter_relative_risk_to_cause_restrictions(data)
    data = data.filter(globals.DEMOGRAPHIC_COLUMNS +
                       ['affected_entity', 'affected_measure', 'parameter'] + globals.DRAW_COLUMNS)
    data = (
        data
            .groupby(['affected_entity', 'parameter'])
            .apply(utilities.normalize, fill_value=1)
            .reset_index(drop=True)
    )
    data = utilities.scrub_gbd_conventions(data, location)
    data = utilities.sort_hierarchical_data(data)
    data = append_ckd_rr(data, ckd_paf)

    key = prefix + 'relative_risk'
    write(artifact, key, data)


def write(artifact, key, data):
    data = split_interval(data, interval_column='age', split_column_prefix='age')
    data = split_interval(data, interval_column='year', split_column_prefix='year')
    artifact.write(key, data)


def get_load(location):
    return lambda key: loader(EntityKey(key), location, set())


def load_prev_dw(sequela, location):
    load = get_load(location)
    prevalence = [load(f'sequela.{s}.prevalence') for s in sequela]
    disability_weight = [load(f'sequela.{s}.disability_weight') for s in sequela]
    total_prevalence = sum(prevalence)
    total_disability_weight = sum([p * dw for p, dw in zip(prevalence, disability_weight)]) / total_prevalence
    return total_prevalence, total_disability_weight


def create_new_artifact(path: str, location: str) -> Artifact:
    if Path(path).is_file():
        Path(path).unlink()
    art = Artifact(path, filter_terms=[get_location_term(location)])
    art.write('metadata.locations', [location])
    return art


def load_em_from_meid(meid, location):
    location_id = utility_data.get_location_id(location)
    data = gbd.get_modelable_entity_draws(meid, location_id)
    data = data[data.measure_id == globals.MEASURES['excess_mortality']]
    data = utilities.normalize(data, fill_value=0)
    data = data.filter(globals.DEMOGRAPHIC_COLUMNS + globals.DRAW_COLUMNS)
    data = utilities.reshape(data)
    data = utilities.scrub_gbd_conventions(data, location)
    return utilities.sort_hierarchical_data(data)


def append_ckd_rr(data, ckd_paf):
    # TODO
    return data


def write_proportion_hypertensive(artifact, location):
    location = location.replace(' ', '_').replace("'", "-").lower()
    data = pd.read_hdf(HYPERTENSION_DATA_FOLDER / f'{location}.hdf', HYPERTENSION_HDF_KEY)
    key = f'risk_factor.high_systolic_blood_pressure.{HYPERTENSION_HDF_KEY}'
    write(artifact, key, data)


def write_hypertension_medication_data(artifact, location):
    external_data_specification = {
        'adherence': {
            'seed_columns': ['location', 'age_group_start', 'age_group_end', 'pill_category'],
            'distribution': 'beta',
        },
        'medication_probabilities': {
            'seed_columns': ['location', 'measure',  'thiazide_type_diuretics', 'beta_blockers',
                             'ace_inhibitors', 'angiotensin_ii_blockers', 'calcium_channel_blockers'],
            'distribution': 'beta',
        },
        'therapy_category': {
            'seed_columns': ['location', 'therapy_category'],
            'distribution': 'beta',
        },
        'treatment_coverage': {
            'seed_columns': ['location', 'measure'],
            'distribution': 'beta',
        },
        'drug_efficacy': {
            'seed_columns': ['location', 'drug'],  # don't include dosage so all dosages of same drug will use same seeds
            'distribution': 'normal',
        },
    }

    for k, spec in external_data_specification.items():
        data = load_external_data(k, location)
        data = generate_draws(data, spec['seed_columns'], spec['distribution'])

        if set(data.location) == {'Global'}:
            # do this post draw generation so all locations use the same draws if data is global
            data.location = location

        if k == 'medication_probabilities':  # drop ACE + ARB single pill because not used
            data = data.loc[~((data.ace_inhibitors == 1) & (data.angiotensin_ii_blockers == 1))]

        data = utilities.sort_hierarchical_data(utilities.reshape(data))

        if k == 'therapy_category':  # normalize so that sum of all categories = 1
            data = data.divide(data.sum(axis=0), axis=1)

        key = f'health_technology.hypertension_medication.{k}'
        write(artifact, key, data)


def generate_draws(data, seed_columns, distribution_type):
    draws = pd.DataFrame(data=np.transpose(np.tile(data['mean'].values, (1000, 1))),
                         columns=globals.DRAW_COLUMNS, index=data.index)
    data = pd.concat([data, draws], axis=1)

    if distribution_type is not None:
        for row in data.iterrows():
            if row[1]['sd'] != 0:
                seed = str_to_seed('_'.join([str(s) for s in row[1][seed_columns]]))
                np.random.seed(seed)
                d = np.random.random(1000)
                if distribution_type == 'normal':
                    dist = norm(loc=row[1]['mean'], scale=row[1]['sd'])
                else:  # beta
                    from risk_distributions.risk_distributions import Beta
                    params = Beta._get_parameters(mean=pd.Series(row[1]['mean']), sd=pd.Series(row[1]['sd']),
                                                  x_min=pd.Series(0), x_max=pd.Series(1))
                    dist = beta(**params)

                data.loc[row[0], globals.DRAW_COLUMNS] = dist.ppf(d)

    assert np.all(~data.isna()), "Something's wrong: NaNs were generated for draws. "
    return data.drop(columns=['mean', 'sd'])


def load_external_data(file_key, location):
    from vivarium_csu_hypertension_sdc import external_data
    data_dir = Path(external_data.__file__).parent

    data = pd.read_csv(data_dir / f'{file_key}.csv')

    # strip all string columns to prevent pesky leading/trailing spaces that may have crept in
    df_obj = data.select_dtypes(['object'])
    data[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())

    data = data.loc[(data.location == location) | (data.location == 'Global')]

    if 'uncertainty_level' in data:  # convert upper/lower bounds to sd
        ci_width_map = {99: 2.58, 95: 1.96, 90: 1.65, 68: 1}
        ci_widths = data.uncertainty_level.map(lambda l: ci_width_map.get(l, 0) * 2)
        data['sd'] = (data.upper_bound - data.lower_bound) / ci_widths
        data = data.drop(columns=['uncertainty_level', 'upper_bound', 'lower_bound'])

    return data


def str_to_seed(s):
    """ Numpy random seed requires an int between 0 and 2**32 - 1 so we have to
    do a little work to convert a string into something we can use as a seed.
    Using hashlib instead of built-in hash because the built-in is
    non-deterministic between runs.
    """
    hash_digest = hashlib.sha256(s.encode()).digest()
    seed = int.from_bytes(hash_digest, 'big') % (2**32 - 1)
    return seed


def write_utilization_rate(artifact, location):
    load = get_load(location)
    key = 'healthcare_entity.outpatient_visits.utilization_rate'
    data = load(key)
    write(artifact, key, data)

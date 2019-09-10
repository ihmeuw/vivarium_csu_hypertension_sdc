from pathlib import Path

import pandas as pd

from gbd_mapping import risk_factors
from vivarium import Artifact
from vivarium.framework.artifact import get_location_term
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

    write_proportion_hypertensive(artifact, location)


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
    return lambda key: loader(key, location, set())


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

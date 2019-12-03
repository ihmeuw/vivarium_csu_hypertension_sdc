#importing packages
from pprint import pprint
from pathlib import Path
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import re
from datetime import datetime, date, time
from matplotlib.backends.backend_pdf import PdfPages
sns.set(context = 'paper', style='whitegrid', font_scale=1.8, rc = {'axes.spines.right':False, 'axes.spines.top': False, 'figure.figsize':(12.7,8.6)}, palette='Set1')

#create a nested dictionary that takes in a specific dataframe column as a nested dictionary. Within each dictionary, the key is the column value and the value is the formatted version of the value.
mapping_dict_data = { 'scenario': {'fixed_dose_combination':'Single pill combination','low_and_slow':'Start low go slow','free_choice':'Free choice combination','hypothetical_baseline':'Baseline'},
                    'sex': {'Female':'females','Male':'males'},
                    'measure':{'yll':'YLLs','yld':'YLDs','death':'Deaths','disease_events':'Incidence','dalys':'DALYs'},
                    'cause': {'chronic_kidney_disease':'Chronic kidney disease', 'ischemic_heart_disease':'Ischemic heart disease', 'subarachnoid_hemorrhage':'Subarachnoid hemorrhage', 'intracerebral_hemorrhage':'Intracerebral hemorrhage', 'stroke':'Stroke', 'ischemic_stroke': 'Ischemic stroke','all_causes':'CKD, Stroke, and IHD'},
                    'location': {'china':'China', 'mexico':'Mexico', 'south_korea':'South Korea', 'italy':'Italy', 'russian_federation':'Russian Federation'},
                    'med': {'ace_inhibitors':'ACEi', 'angiotensin_ii_blockers':'ARB', 'beta_blockers':'Beta-blocker', 'calcium_channel_blockers':'CCB', 'thiazide_type_diuretics':'Diuretic'}}
# Only works on IHME cluster
#def get_latest_results(location):
#    """ 
#    Takes the filepath for model output .hdf files and returns a dataframe based on the latest output model results
#    """
#    path_template = Path('/share/costeffectiveness/results/vivarium_csu_hypertension_sdc')
#    latest = sorted((path_template / location).iterdir(), key=lambda path: path.name)[-1]
#    results = pd.read_hdf(str(latest / 'output.hdf'))
#    print(f'Results for location {location} contain {len(results)} rows.')
#    return results

#This is the list of locations that the model runs for
locations = ['china', 'mexico', 'south_korea', 'italy', 'russian_federation']

#takes the filepath as a string for later use in output
#TO-DO: update save path with path from repo
save_path = Path('')

# we expect 3000 rows in results, per location. for each location, create a dataframe based on the latest results
expected_results = 3000
for location in locations: 
    p = save_path / f'{location}'
    p = save_path
    p.mkdir(parents=True, exist_ok=True)
    data = get_latest_results(location)

def aggregate(data, method='sum') -> pd.DataFrame:
    """
    Takes the dataframe of latest results as an argument, and returns an aggregated dataframe based on method of 'sum' or 'mean'. 
    Renames column 'hypertension_treatment.treatment_ramp as 'scenario' and 'input_draw' as 'draw'.
    """
    scenario_column = 'hypertension_treatment.treatment_ramp'
    data = data.rename(columns={scenario_column: 'scenario', 'input_draw': 'draw'})
    if method == 'sum':
        data = data.groupby(['draw', 'scenario']).sum().sort_index()
    elif method == 'mean':
        data = data.groupby(['draw', 'scenario']).mean().sort_index()
    else:
        raise NotImplementedError()
    return data

def get_sex_from_template(template_string) -> pd.DataFrame:
    """
    takes the template_string as an argument and returns the sex of the dataset, either 'Female' or 'Male'
    """
    return 'Female' if 'female' in template_string else 'Male'


def get_age_group_from_template(template_string) -> pd.DataFrame:
    """
    Takes the template_string as an argument and returns the age group of the dataset, ranging from '30_to_34' to '95 plus'
    """
    return template_string.split('age_group_')[-1].split('_among')[0]

def get_sbp_group_from_template(template_string) -> pd.DataFrame:
    """
    Takes the template_string as an argument and returns the SBP group of the dataset, either 'normal', 'hypertensive' or 'severe'.
    """
    for k, v in {'<140': 'normal', '140_to_160': 'hypertensive', '>160': 'severe'}.items():
        if k in template_string:
            return v

def get_sbp_avg_table(data) -> pd.DataFrame:
    """
    Takes the model results values for measure 'average_sbp' and returns the table in .csv file format of average SBP per year and treatment group, by scenario. This is the function to get the 'Mean systolic blood pressure' table for the final report.
    """
    data = aggregate(data, 'mean')
    measure = 'average_sbp'
    measure_data = data.loc[:, [c for c in data.columns if measure in c]]
    measure_data = (pd.DataFrame(measure_data.stack())
                        .reset_index()
                        .rename(columns={'level_2': 'label', 0: 'value'}))
    measure_data.loc[:, 'year'] = measure_data.label.map(lambda s: int(s[-4:]))
    measure_data.loc[:, 'treatment_group'] = measure_data.label.map(get_treatment_group_from_template)
    
    measure_data = measure_data.rename(columns={'treatment_group':'Treatment group','scenario':'Scenario','year':'Year'})
    
    measure_data_m = measure_data.groupby(['Scenario','Year','Treatment group']).value.mean().reset_index()
    measure_data_m['Scenario'] = measure_data_m.Scenario.mapping_dict_data(['scenario'])
    measure_data_m['location'] = f'{location}'
    measure_data_m['location'] = measure_data_m.location.mapping_dict_data.(['location'])
    return measure_data_m

def get_adherence_group_from_template(template_string) -> pd.DataFrame:
    """
    Takes the template_string as an argument and returns the adherence group of the dataset, either 'non_adherent' or 'adherent'
    """
    return 'non_adherent' if 'non_adherent' in template_string else 'adherent'

def get_treatment_group_from_template(template_string) -> pd.DataFrame:
    """
    Takes the template_string as an argument and returns the treatment group of the dataset, either 'untreated' or 'treated'.
    """
    return 'untreated' if 'untreated' in template_string else 'treated'

def get_person_time(data) -> pd.DataFrame:
    """
    Takes the model results as an argument and returns the measure of 'person_time', which is used to calculate various rates throughout the model results analysis and visualization creation.
    """
    data = aggregate(data)
    measure = 'person_time'
    measure_data = data.loc[:, [c for c in data.columns if measure in c]]
    measure_data = (pd.DataFrame(measure_data.stack())
                        .reset_index()
                        .rename(columns={'level_2': 'label', 0: 'value'}))
    measure_data.loc[:, 'measure'] = measure
    measure_data.loc[:, 'sbp_group'] = measure_data.label.map(get_sbp_group_from_template)
    measure_data.loc[:, 'sex'] = measure_data.label.map(get_sex_from_template)
    measure_data.loc[:, 'age_group'] = measure_data.label.map(get_age_group_from_template)
    measure_data.loc[:, 'adherence_group'] = measure_data.label.map(get_adherence_group_from_template)
    measure_data.loc[:, 'treatment_group'] = measure_data.label.map(get_treatment_group_from_template)
    col_order = ['measure', 'age_group', 'sex', 'sbp_group', 'treatment_group', 'adherence_group', 'scenario', 'draw']
    measure_data = measure_data[col_order + ['value']]
    return measure_data.sort_values(col_order).reset_index(drop=True)

def get_measure_data(data, measure) -> pd.DataFrame:
    """
    Takes the model results as an argument  and returns the value of counts per measure of each cause, sex, age_group, scenario, draw.
    """
    data = aggregate(data)
    measure_data = data.loc[:, [c for c in data.columns if measure in c]]
    measure_data = measure_data.loc[:, [c for c in measure_data.columns if 'tte_' not in c]]
    measure_data = (pd.DataFrame(measure_data.stack())
                        .reset_index()
                        .rename(columns={'level_2': 'label', 0: 'value'}))
    measure_data.loc[:, 'measure'] = measure
    measure_data.loc[:, 'cause'] = measure_data.label.map(lambda s: s.split('due_to_')[1].split('_among')[0])
    measure_data.loc[:, 'sex'] = measure_data.label.map(get_sex_from_template)
    measure_data.loc[:, 'age_group'] = measure_data.label.map(get_age_group_from_template)
    col_order = ['measure', 'cause', 'age_group', 'sex', 'scenario', 'draw']
    measure_data = measure_data[col_order + ['value']]
    return measure_data.sort_values(col_order).reset_index(drop=True)

def get_disease_events(data) -> pd.DataFrame:
    """
    Takes the model results as an argument and returns the value of counts per 'disease_events' of each cause, sex, age_group, scenario, draw. 
    This is used for Incidence.
    """
    data = aggregate(data)
    measure_cols = [c for c in data.columns if 'counts' in c and 'susceptible' not in c]
    measure_data = data.loc[:, measure_cols]
    measure_data = (pd.DataFrame(measure_data.stack())
                        .reset_index()
                        .rename(columns={'level_2': 'label', 0: 'value'}))
    measure_data.loc[:, 'measure'] = 'disease_events'
    measure_data.loc[:, 'cause'] = measure_data.label.map(lambda s: s.split('_counts')[0])
    measure_data.loc[:, 'sex'] = measure_data.label.map(get_sex_from_template)
    measure_data.loc[:, 'age_group'] = measure_data.label.map(get_age_group_from_template)
    col_order = ['measure', 'cause', 'age_group', 'sex', 'scenario', 'draw']
    measure_data = measure_data[col_order + ['value']]
    return measure_data.sort_values(col_order).reset_index(drop=True)

def get_disease_events_by_cause(data) -> pd.DataFrame:
    """
    Takes the model results as an argument and returns the disease_events by cause. 
    This is used for Incidence, in get_measure_by_cause(). 
    This includes mapping of the 'acute' state for each disease to the disease.
    """
    measure_data = get_disease_events(data)
    measure_data = measure_data.set_index([c for c in measure_data.columns if c != 'value'])
    cause_map = {'ischemic_heart_disease': 'acute_myocardial_infarction'}
    for s in ['ischemic_stroke', 'subarachnoid_hemorrhage', 'intracerebral_hemorrhage']:
        cause_map[s] = f'acute_{s}'
    cause_map['chronic_kidney_disease'] = 'chronic_kidney_disease'
    
    cause_data = []
    for cause, sequela in cause_map.items():
        total = measure_data.xs(sequela, level='cause')
        total['cause'] = cause
        cause_data.append(total)
        
    return pd.concat(cause_data).reset_index()

def get_measure_by_cause(data, measure) -> pd.DataFrame:
    """
    Takes the model results as an argument and returns the value of counts per measure by cause. 
    This includes mapping between different states of diseases IHD, Strokes, and CKD.
    """
    if measure == 'disease_events':
        return get_disease_events_by_cause(data)
    else:
        measure_data = get_measure_data(data, measure)
    measure_data = measure_data.set_index([c for c in measure_data.columns if c != 'value'])
    cause_map = {'ischemic_heart_disease': ['acute_myocardial_infarction', 'post_myocardial_infarction']}
    for s in ['ischemic_stroke', 'subarachnoid_hemorrhage', 'intracerebral_hemorrhage']:
        cause_map[s] = [f'acute_{s}', f'post_{s}']
    cause_map['chronic_kidney_disease'] = ['chronic_kidney_disease']
    
    cause_data = []
    for cause, sequelae in cause_map.items():
        total = 0
        for s in sequelae:
            total = total + measure_data.xs(s, level='cause')
        total['cause'] = cause
        cause_data.append(total)
    
    return pd.concat(cause_data).reset_index()

def get_cause_specific_rates(data, measure) -> pd.DataFrame:
    """
    Takes the model results and measure as arguments and returns rates of the measure per cause, age_group, sex, scenario, and draw
    """
    shared_cols = ['age_group', 'sex', 'scenario', 'draw']
    measure_data = get_measure_by_cause(data, measure).drop(columns=['measure']).set_index(['cause'] + shared_cols)
    person_time = get_person_time(data).drop(columns=['measure']).groupby(shared_cols).sum()
    rates = measure_data / person_time * 100_000
    return rates.reset_index()

def get_proportion_adherent(data) -> pd.DataFrame:
    """
    Takes the model results as an argument and returns the proportion adherent based on person_time and adherence group.
    This is used for the 'Proportion Adherent' figure and data table.
    """
    data = get_person_time(data)
    idx_cols = ['measure', 'age_group', 'sex', 'scenario', 'draw']
    treated_mask = (data.treatment_group == 'treated')
    treated = data.loc[treated_mask].groupby(idx_cols).sum()
    adherent_mask = treated_mask & (data.adherence_group == 'adherent')
    adherent = data.loc[adherent_mask].groupby(idx_cols).sum()
    proportion_adherent = (adherent / treated)
    return proportion_adherent

def get_proportion_adherent_table(data) -> pd.DataFrame:
    """
    Takes the DataFrame from get_proportion_adherent(data) function and returns the 'Percent of patients in adherence' table from the final report 
    """
    prop_data = get_proportion_adherent(data)
    prop_data = prop_data.reset_index()
    prop_data = prop_data.groupby(['age_group','sex','scenario']).value.mean().reset_index()
    prop_data['age_group'] = prop_data.age_group.str.replace('_',' ')
    prop_data['sex'] = prop_data.sex.mapping_dict_data(['sex'])
    prop_data = prop_data.rename(columns={'scenario':'Scenario'})
    prop_data['Scenario'] = prop_data.Scenario.mapping_dict_data(['scenario'])  
    prop_data = prop_data[(prop_data.age_group != '30 to 34') & (prop_data.age_group != '35 to 39')]
    
    return prop_data

def get_gbd_pops() -> pd.DataFrame:
    """
    For this use, GBD population was extracted for the relevant 5 countries (China, Russia, Italy, Mexico, and South Korea) and saved as a .csv file. 
    This function takes the GBD population dataand returns a DataFrame to be used in the figure 'plot_country_results' and 'data output' table.
    """
    gbd_pops = pd.read_csv('gbd_pops_2019.csv')
    return gbd_pops

def get_measure_cause_diff_counts_table(data)-> pd.DataFrame:
    """ Takes the model results as an argument and outputs all outcome measures and aggregations (dalys, stroke, both_sexes_combined, all_causes, 40+ age group) that need to be first calculated in count-space. 
    Then, a calculation of the difference between the baseline scenario and intervention scenario is made for the unique value of cause, measure, sex, age_group, scenario, draw.
    This is used as the data source for the 'results', 'difference from baseline scenario', and 'country-results' figures, and the data source for the data output table.
    """
    #shared col level of hypothetical baseline - alt scenario counts / person-time * 100_000
    # either do a merge and lambda function or group the data in a way to do dataframe subtraction then divide by person years
    shared_cols = ['age_group', 'sex','draw','scenario']

    yll = get_measure_by_cause(data,'yll').set_index(['cause'] + shared_cols)
    yld = get_measure_by_cause(data,'yld').set_index(['cause'] + shared_cols)
    inc = get_measure_by_cause(data,'disease_events')
    death = get_measure_by_cause(data,'death')

    #add in daly counts, which is the sum of ylls + ylds
    daly_counts = yll + yld
    daly_counts['measure'] = 'dalys'
    daly_counts = daly_counts.reset_index()
    yll = yll.reset_index()
    yld = yld.reset_index()

    # add in stroke counts for all measures, which combines the measures for diseases in the model that include 'hemorrhage' or 'stroke'
    yll_stroke_counts = get_measure_by_cause(data,'yll')
    yll_stroke_counts = yll_stroke_counts[yll_stroke_counts['cause'].str.contains('hemorrhage|stroke')].groupby(shared_cols).sum()
    yld_stroke_counts = get_measure_by_cause(data,'yld')
    yld_stroke_counts = yld_stroke_counts[yld_stroke_counts['cause'].str.contains('hemorrhage|stroke')].groupby(shared_cols).sum()
    inc_stroke_counts = get_measure_by_cause(data,'disease_events')
    inc_stroke_counts = inc_stroke_counts[inc_stroke_counts['cause'].str.contains('hemorrhage|stroke')].groupby(shared_cols).sum().reset_index()
    inc_stroke_counts['cause'] = 'stroke'
    inc_stroke_counts['measure']='disease_events'
    death_stroke_counts = get_measure_by_cause(data,'death')
    death_stroke_counts = death_stroke_counts[death_stroke_counts['cause'].str.contains('hemorrhage|stroke')].groupby(shared_cols).sum().reset_index()
    death_stroke_counts['cause'] = 'stroke'
    death_stroke_counts['measure'] ='death'
    daly_stroke_counts = yll_stroke_counts + yld_stroke_counts
    daly_stroke_counts['cause'] = 'stroke'
    daly_stroke_counts['measure'] = 'dalys'
    yll_stroke_counts = yll_stroke_counts.reset_index()
    yll_stroke_counts['cause'] = 'stroke'
    yll_stroke_counts['measure'] = 'yll'
    yld_stroke_counts = yld_stroke_counts.reset_index()
    yld_stroke_counts['cause'] = 'stroke'
    yld_stroke_counts['measure'] = 'yld'
    daly_stroke_counts = daly_stroke_counts.reset_index()

    #concatenate all dataframes to one dataframe
    counts_frames = [yll, yld, inc, death, daly_counts, yll_stroke_counts, inc_stroke_counts, yld_stroke_counts, death_stroke_counts, daly_stroke_counts]
    counts_df = pd.concat(counts_frames)
    #remove age groups that are under 40 years old, because the client only wants ages 40+
    counts_df = counts_df[(counts_df.age_group != '30_to_34') & (counts_df.age_group != '35_to_39')]

    # add in both sexes combined, so grouping dataframe by columns other than sex to add up both sexes counts
    both_sexes_counts = counts_df.groupby(['age_group', 'cause', 'measure', 'draw', 'scenario']).sum().reset_index()
    both_sexes_counts['sex'] = 'both_sexes_combined'
    all_counts_frames = [counts_df,both_sexes_counts]
    all_counts_df = pd.concat(all_counts_frames)
    both_sexes_all_causes_counts_df = all_counts_df.groupby(['age_group', 'sex', 'measure', 'draw', 'scenario']).sum().reset_index()
    both_sexes_all_causes_counts_df['cause'] = 'all_causes'
    all_counts_frames_agg_causes_frames = [all_counts_df,both_sexes_all_causes_counts_df]
    #concatenate original dataframe with the both sexes combined dataframe
    all_counts_all_causes_df = pd.concat(all_counts_frames_agg_causes_frames)

    # add in age group of '40 +' which sums up all counts for each measure and disease across ages
    all_ages_all_causes_counts = all_counts_all_causes_df.groupby(['cause', 'sex','measure', 'draw', 'scenario']).sum().reset_index()
    all_ages_all_causes_counts['age_group'] = '40_plus'
    all_ages_causes_counts_frames=[all_counts_all_causes_df,all_ages_all_causes_counts]
    # concatenate original dataframe with the 40+ age group
    all_ages_causes_counts_df = pd.concat(all_ages_causes_counts_frames)
    # create a dataframe of only scenario = 'hypothetical_baseline' to be used for calculation of 'difference from baseline'
    counts_baseline = all_ages_causes_counts_df[all_ages_causes_counts_df['scenario']=='hypothetical_baseline']
    all_counts_df_baseline = pd.merge(all_ages_causes_counts_df,counts_baseline,left_on=['age_group', 'cause', 'measure', 'draw', 'sex'],right_on=['age_group', 'cause', 'measure', 'draw', 'sex'],how='left')
    all_counts_df_baseline['diference_from_baseline_count'] = all_counts_df_baseline.apply(lambda row: (row['value_y'] - row['value_x']),axis=1)
    all_counts_df_baseline = all_counts_df_baseline.rename(columns={'scenario_x':'scenario', 'value_x':'value_count'})
    return all_counts_df_baseline

def get_all_location_sim_counts(data) -> pd.DataFrame:
    """
    Takes the location-specific model results, analyzes the data for calculations needed, and returns one single DataFrame of all location simulation data combined. This is the function which prepares the dataset for the 'Difference from baseline' and 'model results' plots..
    """
    countries_data = []
    for location in locations: 
        data = get_latest_results(location)
        results = get_measure_cause_diff_counts_table(data)
        results['Location'] = f'{location}'
        results['Location'] = results.Location.map(mapping_dict_data['location'])
        countries_data.append(results)
    return pd.concat(countries_data)
    
def get_gbd_population_counts(data) -> pd.DataFrame: 
    """
    Takes the DataFrame from get_measure_cause_diff_counts and DataFrame from the GBD 2017 populations, calculates the country-level counts of each outcome, and returns a DataFrame.
    """    
    shared_cols = ['age_group', 'sex','draw','scenario']
    shared_ui_cols = ['age_group','sex','cause','measure','scenario']
    all_counts = get_measure_cause_diff_counts_table(data)
    #gets person-years for both sexes combined
    all_counts['location'] = f'{location}'
    all_counts['location'] = all_counts.location.str.replace('_',' ')
    all_counts['location'] = all_counts.location.str.title()
    person_time = get_person_time(data).drop(columns=['measure']).groupby(shared_cols).sum().reset_index()
    person_time = person_time[(person_time.age_group !='30_to_34') & (person_time.age_group != '35_to_39')]
    person_time_both = person_time.groupby(['age_group', 'draw', 'scenario']).sum().reset_index()
    person_time_both['sex'] = 'both_sexes_combined'
    person_time_frames = [person_time,person_time_both]
    person_time_all = pd.concat(person_time_frames)
    person_time_all_ages = person_time_all.groupby(['sex', 'draw', 'scenario']).sum().reset_index()
    person_time_all_ages['age_group'] = '40_plus'
    person_time_all_ages_frames = [person_time_all,person_time_all_ages]
    person_time_all_ages = pd.concat(person_time_all_ages_frames)
    all_counts = pd.merge(all_counts,person_time_all_ages, left_on=['age_group', 'sex', 'draw', 'scenario'],right_on=['age_group', 'sex', 'draw', 'scenario'],how='left')
    all_counts = all_counts.rename(columns={'value':'person_years'})
    all_counts['age_group'] = all_counts.age_group.str.replace('_',' ')
    all_counts['cause'] = all_counts.cause.map(mapping_dict_data['cause'])
    all_counts['measure'] = all_counts.measure.map(mapping_dict_data['measure'])
    all_counts['sex'] = all_counts.sex.map(mapping_dict_data['sex'])

    all_counts['measure_rate'] = all_counts.apply(lambda row: (row['value_count']/row['person_years'])*100_000,axis=1)
    all_counts['difference_from_baseline_rate'] = all_counts.apply(lambda row: (row['diference_from_baseline_count']/row['person_years'])*100_000,axis=1)
    
    all_counts['measure_pop_counts'] = all_counts.apply(lambda row: row['value_count']/row['person_years'],axis=1)
    all_counts['difference_from_baseline__pop_counts'] = all_counts.apply(lambda row: row['diference_from_baseline_count']/row['person_years'],axis=1)
    pops = get_gbd_pops()
    rates_pops = pd.merge(all_counts,pops,left_on=['age_group','location','sex'],right_on=['age_group_name','location_name','sex'],how='left')
    
    rates_pops['pop_measure'] = rates_pops.apply(lambda row: row['measure_pop_counts']* row['population'],axis=1)
    rates_pops['pop_diff_baseline'] = rates_pops.apply(lambda row: row['difference_from_baseline__pop_counts']* row['population'],axis=1)
    
    rates_std_baseline_diff = rates_pops.groupby(shared_ui_cols)['difference_from_baseline_rate'].agg([('std_baseline_diff_rate','std')]).reset_index()
    rates_mean_baseline_diff = rates_pops.groupby(shared_ui_cols)['difference_from_baseline_rate'].agg([('mean_baseline_diff_rate','mean')]).reset_index()
    rates_mean_measure_rate = rates_pops.groupby(shared_ui_cols)['measure_rate'].agg([('mean_measure_rate','mean')]).reset_index()
    rates_std_measure_rate = rates_pops.groupby(shared_ui_cols)['measure_rate'].agg([('std_measure_rate','std')]).reset_index()
    pop_measure_m = rates_pops.groupby(shared_ui_cols)['pop_measure'].agg([('pop_measure_mean','mean')]).reset_index()
    pop_diff_baseline_m = rates_pops.groupby(shared_ui_cols)['pop_diff_baseline'].agg([('pop_diff_baseline_mean','mean')]).reset_index()
    ui_frames = [rates_std_baseline_diff[['std_baseline_diff_rate']],rates_mean_baseline_diff[['mean_baseline_diff_rate']],rates_mean_measure_rate[['age_group','sex','cause','measure','scenario','mean_measure_rate']],rates_std_measure_rate[['std_measure_rate']],pop_measure_m[['pop_measure_mean']],pop_diff_baseline_m[['pop_diff_baseline_mean']]]
    ui_rates = pd.concat(ui_frames,axis=1)
    
    ui_rates['upper_std_baseline_diff'] = ui_rates.apply(lambda row: row['mean_baseline_diff_rate']+row['std_baseline_diff_rate'],axis=1)
    ui_rates['upper_measure_rate'] = ui_rates.apply(lambda row: row['mean_measure_rate']+row['std_measure_rate'],axis=1)

    ui_rates['lower_std_baseline_diff'] = ui_rates.apply(lambda row: row['mean_baseline_diff_rate']-row['std_baseline_diff_rate'],axis=1)
    ui_rates['lower_measure_rate'] = ui_rates.apply(lambda row: row['mean_measure_rate']-row['std_measure_rate'],axis=1)
    ui_rates['scenario'] = ui_rates.scenario.map(mapping_dict_data['scenario'])
    ui_rates = ui_rates[['age_group','sex','cause','measure','scenario','mean_measure_rate','upper_measure_rate','lower_measure_rate','mean_baseline_diff_rate','upper_std_baseline_diff','lower_std_baseline_diff','pop_measure_mean','pop_diff_baseline_mean']]
    ui_rates = ui_rates.rename(columns={'scenario':'Scenario', 'measure':'Measure', 'sex':'Sex', 'age_group':'Age group', 'cause':'Cause','mean_measure_rate':'Measure rate (per 100,000 person-years)',
        'upper_measure_rate':'Upper measure rate (per 100,000 person-years)','lower_measure_rate':'Lower measure rate (per 100,000 person-years)','mean_baseline_diff_rate':'Scenario difference from Baseline rate (per 100,000 person-years)',
        'upper_std_baseline_diff':'Upper scenario difference from Baseline rate (per 100,000 person-years)','lower_std_baseline_diff':'Lower scenario difference from Baseline rate (per 100,000 person-years)','pop_measure_mean':'Population measure counts',
        'pop_diff_baseline_mean':'Population measure difference from the Baseline scenario counts'})
    return ui_rates

def get_all_country_counts(data) -> pd.DataFrame:
    """
    Takes the location-specific model results and GBD 2017 population data, analyzes and creates calculations with the data, and returns one single DataFrame of all country data combined. This is the function to output the 'Data output table' for the final table.
    """
    countries_data = []
    for location in locations: 
        data = get_latest_results(location)
        results = get_gbd_population_counts(data)
        results['Location'] = f'{location}'
        results['Location'] = results.Location.map(mapping_dict_data['location'])
        countries_data.append(results)
    return pd.concat(countries_data)

def plot_country_results(data):
    pops_measures = get_all_country_counts(data)
  
    scenario_list = ['Baseline','Free choice combination','Start low go slow','Single pill combination']
    palette_list = ['#000000','#023eff','#ff7c00','#1ac938']
    palette_dict = {x:v for x,v in zip(scenario_list,palette_list)}

    for cause in ['CKD, Stroke, and IHD', 'Chronic kidney disease', 'Intracerebral hemorrhage','Ischemic heart disease', 'Ischemic stroke', 'Stroke','Subarachnoid hemorrhage']:
            with PdfPages(str(save_path)+f'/country_result_counts/{cause}_country_counts.pdf') as pdf:
                for sex in ['females', 'males', 'both sexes combined']:
                    for measure in ['DALYs', 'Deaths', 'Incidence','YLDs', 'YLLs']:
                        
                        for country in ['China', 'Italy', 'South Korea', 'Mexico', 'Russian Federation']:

                            data_draw = pops_measures[(pops_measures.Cause == cause) & (pops_measures.Sex == sex) & (pops_measures.Measure == measure) & (pops_measures.Location == country) & (pops_measures['Age group'] != '40 plus')].sort_values(by=['Age group','Scenario'])

                            formatted_location = data_draw.Location.unique()[0]

                            plt.figure(figsize=(20, 10))

                            sns.scatterplot(x='Age group', y='Population measure counts', 
                                            hue='Scenario', legend='brief', palette=palette_dict,
                                            s=400, marker='P', alpha=1, 
                                            ax=g.ax, data=data_draw)

                            g.ax.set_title(f'{cause} {measure} in {formatted_location}, {sex}')
                            g.ax.set_xticklabels(g.ax.get_xticklabels(), rotation=70)
                            g.ax.set_xlabel('Age group')
                            g.ax.set_ylabel(f'{measure} counts in {formatted_location}')

                            handles, labels = g.ax.get_legend_handles_labels()
                            g.ax.legend(handles[:4],labels[:4],bbox_to_anchor=(0.05, 1), loc='upper left')
                            labels = g.axes[0,0].get_yticks()
                            formatted_labels = ['{:,}'.format(int(label)) for label in labels]
                            g.set_yticklabels(formatted_labels)
                            pdf.savefig(g.fig, orientation = 'landscape', bbox_inches='tight')

                            plt.show()
                            plt.clf() 

def get_medication_start_count(data) -> pd.DataFrame:
    """
    Takes the aggregated model results and returns the medication start count of the simulants. This will be used to calculate 'Percent of patients who experience drug adverse events'.
    """
    data = aggregate(data)
    med_cols = [c for c in data.columns if 'start_count' in c and 'treatment' not in c]
    med_data = data.loc[:, med_cols]
    med_data = (pd.DataFrame(med_data.stack())
                        .reset_index()
                        .rename(columns={'level_2': 'label', 0: 'value'}))
    med_data.loc[:, 'med'] = med_data.label.map(lambda s: s.split('_start')[0])
    med_data.loc[:, 'sex'] = med_data.label.map(get_sex_from_template)
    med_data.loc[:, 'age_group'] = med_data.label.map(get_age_group_from_template)
    col_order = ['med','age_group', 'sex', 'scenario', 'draw']
    med_data = med_data[col_order + ['value']]
    return med_data.sort_values(col_order).reset_index(drop=True)

def adverse_events_rate(data) -> pd.DataFrame:    
    """
    Takes the results of get_medical_start_count function and returns the rate of adverse events for 'Percent of patients who experience drug adverse events' table.
    """
    shared_cols = ['age_group', 'sex','draw','scenario']
    med_data = get_medication_start_count(data)
    def get_adverse_events(x,y):
        if x == 'ace_inhibitors':
            return y * 0.13
        elif x == 'angiotensin_ii_blockers':
            return y * 0.14
        elif x == 'beta_blockers':
            return y * 0.13
        elif x == 'calcium_channel_blockers':
            return y * 0.2
        elif x == 'thiazide_type_diuretics':
            return y * 0.11
        else:
            return 0
    med_data['number_adverse_events'] = med_data.apply(lambda row: get_adverse_events(row['med'],row['value']),axis=1)
    med_data = med_data.groupby(['age_group', 'sex','scenario','draw']).number_adverse_events.sum().reset_index()
    med_data['med'] = 'All medications'
    med_data_both_sex = med_data.groupby(['med', 'age_group','scenario','draw']).number_adverse_events.sum().reset_index()
    med_data_both_sex['sex'] = 'both_sexes_combined'
    med_data_frames = [med_data,med_data_both_sex]
    med_data_all = pd.concat(med_data_frames)
    person_time = get_person_time(data).drop(columns=['measure']).groupby(shared_cols).sum().reset_index()
    person_time_both = person_time.groupby(['age_group','draw','scenario']).sum().reset_index()
    person_time_both['sex'] = 'both_sexes_combined'
    person_time_frames = [person_time,person_time_both]
    person_time_all = pd.concat(person_time_frames)
    adverse_events_pt = pd.merge(med_data_all,person_time_all, left_on=['age_group','sex','draw','scenario'],right_on=['age_group','sex','draw','scenario'],how='left')
    adverse_events_pt = adverse_events_pt.rename(columns={'value':'person_years'})
    adverse_events_pt['measure_rate'] = adverse_events_pt.apply(lambda row: (row['number_adverse_events']/row['person_years'])*100_000,axis=1)
    adverse_events_pt['location'] = f'{location}'
    adverse_events_pt['location'] = adverse_events_pt.location.mapping_dict_data(['location'])
    adverse_events_pt['age_group'] = adverse_events_pt.age_group.str.replace('_',' ')
    adverse_events_pt['sex'] = adverse_events_pt.sex.mapping_dict_data(['sex'])
    adverse_events_pt['scenario'] = adverse_events_pt.scenario.mapping_dict_data(['scenario'])
    adverse_events_pt['med'] = adverse_events_pt.med.mapping_dict_data(['med'])
    adverse_events_pt = adverse_events_pt[(adverse_events_pt.age_group != '30 to 34') & (adverse_events_pt.age_group != '35 to 39') & (adverse_events_pt.scenario != 'Baseline')]
    return adverse_events_pt

def get_total_days_to_control(data) -> pd.DataFrame:
    """
    Takes in the aggregated data and returns the total days to control, per scenario and draw. This will be used as an input in the 'Total days to control' table from the final report.
    """
    data = aggregate(data)
    control_cols = [c for c in data.columns if 'controlled' in c and 'days' in c]
    control_data = data.loc[:, control_cols]
    control_data = (pd.DataFrame(control_data.stack())
                        .reset_index()
                        .rename(columns={'level_2': 'label', 0: 'value'}))
    
    col_order = ['scenario', 'draw']
    control_data = control_data[col_order + ['value']]
    control_data = control_data.rename(columns={'scenario':'Scenario'})
    control_data['Scenario'] = control_data.Scenario.mapping_dict_data(['scenario'])
    control_data = control_data.groupby(['Scenario','draw']).value.mean().reset_index()
    return control_data.sort_values(['Scenario','draw']).reset_index(drop=True)

def get_controlled_among_treatment_started_count(data) -> pd.DataFrame:
    """
    Takes in the aggregated data and returns the count of number of people who started treatment and became controlled in the simulation. This will be used as an input in the 'Total days to control' table from the final report.
    """
    data = aggregate(data)
    control_cols = [c for c in data.columns if 'controlled_among_treatment_started_count' in c]
    control_data = data.loc[:, control_cols]
    control_data = (pd.DataFrame(control_data.stack())
                        .reset_index()
                        .rename(columns={'level_2': 'label', 0: 'value'}))
    
    col_order = ['scenario', 'draw']
    control_data = control_data[col_order + ['value']]
    control_data = control_data.rename(columns={'scenario':'controlled_among_treated_scenario','value':'controlled_among_treatment_started_value'})
    control_data['controlled_among_treated_scenario'] = control_data.controlled_among_treated_scenario.mapping_dict_data(['scenario'])
    control_data = control_data.groupby(['controlled_among_treated_scenario','draw']).controlled_among_treatment_started_value.mean().reset_index()
    return control_data.sort_values(['controlled_among_treated_scenario','draw']).reset_index(drop=True)

def get_total_days_to_control_div(data) -> pd.DataFrame:
    """
    This takes in the DataFrames from the functions get_total_days_to_control(data) & get_controlled_among_treatment_started_count(data) and returns 'Time to control' table by scenario.
    """
    control_days = get_total_days_to_control(data)
    controlled_treated_count = get_controlled_among_treatment_started_count(data)
    total_days_to_control_div = pd.merge(control_days,controlled_treated_count,left_on=['Scenario','draw'],right_on=['controlled_among_treated_scenario','draw'],how='left')
    total_days_to_control_div['total_days_to_control_divided_by_controlled_among_treated_pop'] = total_days_to_control_div.apply(lambda row: row['value']/ row['controlled_among_treatment_started_value'],axis=1)
    total_days_to_control_div = total_days_to_control_div.rename(columns={'value':'total_days_to_control_value'})
    total_days_to_control_div_gb = total_days_to_control_div.groupby(['Scenario'])['total_days_to_control_value','controlled_among_treatment_started_value','total_days_to_control_divided_by_controlled_among_treated_pop'].mean().reset_index()
    total_days_to_control_div_gb['location'] = f'{location}'
    total_days_to_control_div_gb['location'] = total_days_to_control_div_gb.location.mapping_dict_data(['location'])
    return total_days_to_control_div_gb

def get_treatment_start_count(data) -> pd.DataFrame:
    """
    Takes in the aggregated data and returns the number of people who started treatment in the simulation. This will be used in the 'Percent of patients with controlled blood pressure' table from the final report.
    """
    data = aggregate(data)
    treat_cols = [c for c in data.columns if 'treatment_start_count' in c]
    treat_data = data.loc[:, treat_cols]
    treat_data = (pd.DataFrame(treat_data.stack())
                        .reset_index()
                        .rename(columns={'level_2': 'label', 0: 'value'}))
    
    col_order = ['scenario', 'draw']
    treat_data = treat_data[col_order + ['value']]
    treat_data = treat_data.rename(columns={'scenario':'treatment_scenario','value':'treatment_started_value'})
    treat_data['treatment_scenario'] = treat_data.treatment_scenario.mapping_dict_data(['scenario'])
    treat_data = treat_data.groupby(['treatment_scenario','draw']).treatment_started_value.mean().reset_index()
    return treat_data.sort_values(['treatment_scenario','draw']).reset_index(drop=True)

def get_percent_patients_controlled(data) -> pd.DataFrame:
    """
    Takes in the get_treatment_start_count(data) & get_controlled_among_treatment_started_count(data) functions and. This function is used for the 'Percent of patients with controlled blood pressure' table from the final report.
    """
    treatment_start_count = get_treatment_start_count(data)
    controlled_treated_count = get_controlled_among_treatment_started_count(data)
    treatment_start_control = pd.merge(treatment_start_count,controlled_treated_count,left_on=['treatment_scenario','draw'],right_on=['controlled_among_treated_scenario','draw'],how='left')
    treatment_start_control['percent_controlled_among_started_treatment'] = treatment_start_control.apply(lambda row: row['controlled_among_treatment_started_value']/row['treatment_started_value'],axis=1)
    treatment_start_control = treatment_start_control.rename(columns={'treatment_scenario':'Scenario'})
    treatment_start_control_gb = treatment_start_control.groupby(['Scenario'])['treatment_started_value','controlled_among_treatment_started_value','percent_controlled_among_started_treatment'].mean().reset_index()
    return treatment_start_control_gb

def get_all_cause_mortality(data) -> pd.DataFrame:
    """
    Takes the aggregated model results as an argument and returns the all-cause mortality. This function is used for the 'All-cause mortality' table from the final report.
    """
    data = aggregate(data)
    death_data = data.loc[:, [c for c in data.columns if 'total_population_dead' in c]]
    death_data = (pd.DataFrame(death_data.stack())
                        .reset_index()
                        .rename(columns={'level_2': 'label', 0: 'value'}))
    col_order = ['scenario', 'draw']
    death_data = death_data[col_order + ['value']]
    person_time = get_person_time(data).drop(columns=['measure']).groupby(col_order).sum().reset_index()
    death_pt = pd.merge(death_data,person_time, left_on=['draw','scenario'],right_on=['draw','scenario'],how='left')
    death_pt = death_pt.rename(columns={'value_x':'mort_value','value_y':'person_years'})
    death_baseline = death_pt[death_pt['scenario']=='hypothetical_baseline']
    death_pt = pd.merge(death_pt,death_baseline[['draw','mort_value']],left_on=['draw'],right_on=['draw'],how='left')
    def get_diff_baseline(x,y):
        return x-y
    death_pt['diference_from_baseline_count'] = death_pt.apply(lambda row: get_diff_baseline(row['mort_value_y'],row['mort_value_x']),axis=1)
    def get_rate(x,y):
        return (x/y)*100_000
    death_pt['measure_rate'] = death_pt.apply(lambda row: get_rate(row['mort_value_x'],row['person_years']),axis=1)
    death_pt['difference_from_baseline_rate'] = death_pt.apply(lambda row: get_rate(row['diference_from_baseline_count'],row['person_years']),axis=1)
    rates_std_baseline_diff = death_pt.groupby(['scenario'])['difference_from_baseline_rate'].agg([('std_baseline_diff_rate','std')]).reset_index()
    rates_mean_baseline_diff = death_pt.groupby(['scenario'])['difference_from_baseline_rate'].agg([('mean_baseline_diff_rate','mean')]).reset_index()
    rates_mean_measure_rate = death_pt.groupby(['scenario'])['measure_rate'].agg([('mean_measure_rate','mean')]).reset_index()
    rates_std_measure_rate = death_pt.groupby(['scenario'])['measure_rate'].agg([('std_measure_rate','std')]).reset_index()
    ui_frames = [rates_std_baseline_diff[['std_baseline_diff_rate']],rates_mean_baseline_diff[['scenario','mean_baseline_diff_rate']],rates_mean_measure_rate[['mean_measure_rate']],rates_std_measure_rate[['std_measure_rate']]]
    ui_rates = pd.concat(ui_frames,axis=1)
    def get_upper_std(x,y):
        return x+y
    ui_rates['upper_std_baseline_diff'] = ui_rates.apply(lambda row: get_upper_std(row['mean_baseline_diff_rate'],row['std_baseline_diff_rate']),axis=1)
    ui_rates['upper_measure_rate'] = ui_rates.apply(lambda row: get_upper_std(row['mean_measure_rate'],row['std_measure_rate']),axis=1)
    def get_lower_std(x,y):
        return x-y
    ui_rates['lower_std_baseline_diff'] = ui_rates.apply(lambda row: get_lower_std(row['mean_baseline_diff_rate'],row['std_baseline_diff_rate']),axis=1)
    ui_rates['lower_measure_rate'] = ui_rates.apply(lambda row: get_lower_std(row['mean_measure_rate'],row['std_measure_rate']),axis=1)
    ui_rates = ui_rates.rename(columns={'scenario':'Scenario'})
    ui_rates['Scenario'] = ui_rates.Scenario.mapping_dict_data(['scenario'])
    ui_rates = ui_rates[['Scenario','mean_measure_rate','upper_measure_rate','lower_measure_rate','mean_baseline_diff_rate','upper_std_baseline_diff','lower_std_baseline_diff']]
    return ui_rates.sort_values('Scenario').reset_index(drop=True)

def plot_delta_adherence(data, save_path=None):
    """
    Takes the DataFrame of get_proportion_adherent(data) function and returns plots of the change in adherence by age group.
    """
    proportion_adherent = get_proportion_adherent(data)
    
    delta = (proportion_adherent - proportion_adherent.xs('hypothetical_baseline', level='scenario')).reset_index()
    delta['age_group'] = delta.age_group.str.replace('_',' ')
    delta['scenario'] = delta.scenario.mapping_dict_data(['scenario'])
    delta['location'] = f'{location}'
    delta['location'] = delta.location.mapping_dict_data(['location'])
    delta = delta.rename(columns={'scenario':'Scenario'})
    delta = delta[(delta.age_group != '30 to 34') & (delta.age_group != '35 to 39')]
   
    scenario_list = ['Baseline','Free choice combination','Start low go slow','Single pill combination']
    palette_list = ['#000000','#023eff','#ff7c00','#1ac938']
    palette_dict = {x:v for x,v in zip(scenario_list,palette_list)}
 
    formatted_location = delta.location.unique()[0]

    g = sns.catplot(x='age_group', y='value', 
                    hue='Scenario', palette=palette_dict, 
                    data=delta, 
                    height=10, aspect=1.3)
    g.ax.set_xticklabels(g.ax.get_xticklabels(), rotation=70)
    g.ax.set_xlabel('Age group', fontsize=18)
    g.ax.set_ylabel(f'Difference in Adherence', fontsize=18)
    if save_path is not None:
        plt.savefig(str(save_path /'difference_in_adherence_over_age_groups'/ f'adherence_delta_{formatted_location}.pdf'),orientation='landscape',bbox_inches='tight')
    plt.show()

def plot_adherence(data, save_path=None):
    """
    Takes the DataFrame of get_proportion_adherent(data) function and returns plots of adherence by age group.
    """
    proportion_adherent = get_proportion_adherent(data).reset_index()
    proportion_adherent['age_group'] = proportion_adherent.age_group.str.replace('_',' ')
    proportion_adherent['scenario'] = proportion_adherent.scenario.mapping_dict_data(['scenario'])
    proportion_adherent = proportion_adherent.rename(columns={'scenario':'Scenario'})
    proportion_adherent['location'] = f'{location}'
    proportion_adherent['location'] = proportion_adherent.location.mapping_dict_data(['location'])
    proportion_adherent = proportion_adherent[(proportion_adherent.age_group != '30 to 34') & (proportion_adherent.age_group != '35 to 39')]

    means = proportion_adherent.groupby(['measure', 'age_group', 'Scenario']).value.mean().reset_index()
    means['age_group'] = means.age_group.str.replace('_',' ')
    means['Scenario'] = means.Scenario.mapping_dict_data(['scenario'])
    
    # for each scenario, assign a specific palette color uniformally across plots
    scenario_list = ['Baseline','Free choice combination','Start low go slow','Single pill combination']
    palette_list = ['#000000','#023eff','#ff7c00','#1ac938']
    palette_dict = {x:v for x,v in zip(scenario_list,palette_list)}
 
    formatted_location = proportion_adherent.location.unique()[0]

    g = sns.catplot(x='age_group', y='value', 
                     hue='Scenario', palette=palette_dict, alpha=0.5, 
                     data=proportion_adherent, 
                     height=10, aspect=1.3)
    sns.scatterplot(x='age_group', y='value', 
                    hue='Scenario', palette=palette_dict, 
                    s=250, marker='P', 
                    ax=g.ax, legend=False, data=means)
    g.ax.set_xticklabels(g.ax.get_xticklabels(), rotation=70)
    g.ax.set_title(f'Adherence among treated individuals in {formatted_location}', fontsize=24)
    g.ax.set_xlabel('Age group', fontsize=18)
    g.ax.set_ylabel(f'Proportion adherent', fontsize=18)
    g.ax.set_ylim([.3, 1.1])
    if save_path is not None:
        plt.savefig(str(save_path / 'adherence_over_age_groups'/ 'adherence_{}.pdf'.format(location)),orientation='landscape',bbox_inches='tight')
    plt.show()
    plt.close()

def plot_sbp_time_series(data, save_path=None):
    """
    Takes the DataFrame of aggregated data and returns plots of the average SBP over time.
    """
    data = aggregate(data, 'mean')
    measure = 'average_sbp'
    measure_data = data.loc[:, [c for c in data.columns if measure in c]]
    measure_data = (pd.DataFrame(measure_data.stack())
                        .reset_index()
                        .rename(columns={'level_2': 'label', 0: 'value'}))
    measure_data.loc[:, 'year'] = measure_data.label.map(lambda s: int(s[-4:]))
    measure_data.loc[:, 'treatment_group'] = measure_data.label.map(get_treatment_group_from_template)
    measure_data['scenario'] = measure_data.scenario.str.replace('_', ' ')
    #serena added in 10/4-- take out if break
    measure_data['scenario'] = measure_data.scenario.mapping_dict_data(['scenario'])
    measure_data = measure_data.query("treatment_group == 'treated'")
    measure_data = measure_data.rename(columns={'treatment_group':'Treatment group','scenario':'Scenario'})
    measure_data['location'] = f'{location}'
    measure_data['location'] = measure_data.location.mapping_dict_data(['location'])
    
    scenario_list = ['Baseline','Free choice combination','Start low go slow','Single pill combination']
    palette_list = ['#000000','#023eff','#ff7c00','#1ac938']
    palette_dict = {x:v for x,v in zip(scenario_list,palette_list)}
 
    formatted_location = measure_data.location.unique()[0]

    plt.figure(figsize=(20, 10))
    
    g = sns.relplot(x='year', y='value', 
                        hue='Scenario', 
                        ci=None, palette=palette_dict,
                        kind='line', height=10, aspect=1.3,
                        linewidth=4,
                        data=measure_data)
    g2 = sns.relplot(x='year', y='value', 
                         kind='line', palette=palette_dict,
                         hue='Scenario', units='draw', 
                         alpha= 0.3,estimator=None, 
                         data=measure_data, ax=g.ax)
    plt.close(g2.fig)
    g.ax.set_title(f'Average systolic blood pressure among the population \n on antihypertensive treatment in {formatted_location}', fontsize=24)
    g.ax.set_xlabel('Year', fontsize=18)
    g.ax.set_ylabel('Systolic blood pressure (mmHg)', fontsize=18)
    g.ax.set_ylim(120,150)
    plt.setp(g._legend.get_texts(), fontsize=14)
    
    if save_path is not None:
        plt.savefig(str(save_path / 'sbp_time_series'/ f'sbp_time_series_{location}.pdf'),orientation='landscape',bbox_inches='tight')

def add_comma(t):
    n=1
    newt = ''
    for i in t[::-1]:
        newt+=i
        if n%3==0 and n>=3 and n<len(t):
            newt+=','
        n+=1
    return newt[::-1]

def reformat_labels(ticks):
    labels=[]
    pos = []
    for val in ticks:
        txt = str(int(val))
        pos.append(val)
        newtxt = add_comma(txt)
        labels.append(newtxt)
    return labels,pos

def get_total_days_to_event_counts(data) -> pd.DataFrame:
    data = aggregate(data)
    event_cols = [c for c in data.columns if 'tte_' in c and 'death' not in c]
    event_data = data.loc[:, event_cols]
    event_data = (pd.DataFrame(event_data.stack())
                        .reset_index()
                        .rename(columns={'level_2': 'label', 0: 'value'}))
    event_data.loc[:, 'cause'] = event_data.label.map(lambda s: s.split('_first_')[1].split('_event')[0])
    col_order = ['cause','scenario', 'draw']
    event_data = event_data[col_order + ['value']]
    event_data['cause'] = event_data.cause.str.replace('acute_myocardial_infarction','IHD')
    event_data['cause'] = event_data.cause.str.replace('acute_subarachnoid_hemorrhage','Stroke')
    event_data['cause'] = event_data.cause.str.replace('acute_intracerebral_hemorrhage','Stroke')
    event_data['cause'] = event_data.cause.str.replace('acute_ischemic_stroke','Stroke')
    event_data['cause'] = event_data.cause.str.replace('chronic_kidney_disease','CKD')
    event_data = event_data.rename(columns={'scenario':'Scenario'})
    event_data['Scenario'] = event_data.Scenario.mapping_dict_data(['scenario'])
    event_data_gb = event_data.groupby(['cause','Scenario','draw']).value.sum().reset_index()
    event_data_all = event_data_gb.groupby(['Scenario','draw']).value.sum().reset_index()
    event_data_all['cause'] = 'Total'
    event_data_all_frames = [event_data_gb,event_data_all]
    event_data_gb = pd.concat(event_data_all_frames)
    return event_data_gb.sort_values(['cause','Scenario','draw']).reset_index(drop=True)

def time_to_event(data) -> pd.DataFrame:
    """
    Takes the DataFrames from get_measure_cause_diff_counts_table & get_total_days_to_event_counts(data), creates a calculation for time to event, and returns a DataFrame. This is used for the 'Time to event' table in the final report.
    """
    shared_cols = ['age_group', 'sex','draw','scenario']
    all_counts = get_measure_cause_diff_counts_table(data)
    all_counts['location'] = f'{location}'
    all_counts['location'] = all_counts.location.mapping_dict_data(['location'])
    all_counts['age_group'] = all_counts.age_group.str.replace('_',' ')
    all_counts['cause'] = all_counts.cause.str.replace('_',' ')
    all_counts['cause'] = all_counts.cause.str.replace('all causes','Total')
    all_counts['cause'] = all_counts.cause.str.replace('chronic kidney disease','CKD')
    all_counts['cause'] = all_counts.cause.str.replace('ischemic heart disease','IHD')
    all_counts['cause'] = all_counts.cause.str.replace('stroke','Stroke')

    all_counts['measure'] = all_counts.measure.str.replace('yll','YLLs (years of life lost)')
    all_counts['measure'] = all_counts.measure.str.replace('yld','YLDs (years lived with disability)')
    all_counts['measure'] = all_counts.measure.str.replace('death','Deaths')
    all_counts['measure'] = all_counts.measure.str.replace('disease_events','Incidence')
    all_counts['measure'] = all_counts.measure.str.replace('dalys','DALYs (disability-adjusted life years)')
    all_counts['sex'] = all_counts.sex.str.mapping_dict_data(['sex'])
    all_counts = all_counts.rename(columns={'scenario':'Scenario'})
    all_counts['Scenario'] = all_counts.Scenario.mapping_dict_data(['scenario'])   
    all_counts = all_counts[(all_counts.age_group != '30 to 34') & (all_counts.age_group != '35 to 39') & (all_counts.cause != 'subarachnoid hemorrhage') & (all_counts.cause != 'ischemic Stroke') & (all_counts.cause != 'intracerebral hemorrhage')]
    events = all_counts[(all_counts.measure == 'Incidence')]
    events_sum = events[(events.age_group == 'all ages combined') & (events.sex == 'both sexes combined')][['age_group','cause','draw','measure','Scenario','sex','value_count','location']]
    tte_days = get_total_days_to_event_counts(data)
    events_tte_days_df = pd.merge(events_sum,tte_days,left_on=['cause','Scenario','draw'],right_on=['cause','Scenario','draw'],how='left')
    events_tte_days_df['time_to_event'] = events_tte_days_df.apply(lambda row: row['value']/row['value_count'],axis=1)
    events = events_tte_days_df.groupby(['age_group', 'cause','sex','Scenario','location'])[['value','value_count','time_to_event']].mean().reset_index()
    events = events.rename(columns={'value':'time_from_treatment_init_to_first_event','value_count':'event_rate'})
    return events

def get_results_plots(data):
    """
    Takes the DataFrame of all locations of counts of all measures, causes, age groups and returns plots. This function returns the 'Results' plots from the final report.
    """   
    shared_cols = ['age_group', 'sex','draw','scenario']
    all_counts_df_baseline = get_all_location_sim_counts(data)

    #gets person-years for both sexes combined
    person_time = get_person_time(data).drop(columns=['measure']).groupby(shared_cols).sum().reset_index()
    person_time_both = person_time.groupby(['age_group','draw','scenario']).sum().reset_index()
    person_time_both['sex'] = 'both_sexes_combined'
    person_time_frames = [person_time,person_time_both]
    person_time_all = pd.concat(person_time_frames)
    all_counts_baseline_pt_df = pd.merge(all_counts_df_baseline,person_time_all, left_on=['age_group','sex','draw','scenario'],right_on=['age_group','sex','draw','scenario'],how='left')
    all_counts_baseline_pt_df = all_counts_baseline_pt_df.rename(columns={'value':'person_years'})
    all_counts_baseline_pt_df['measure_rate'] = all_counts_baseline_pt_df.apply(lambda row: (row['value_count']/row['person_years'])*100_000,axis=1)
    all_counts_baseline_pt_df['difference_from_baseline_rate'] = all_counts_baseline_pt_df.apply(lambda row: (row['diference_from_baseline_count']/row['person_years'])*100_000,axis=1)
    all_counts_baseline_pt_df['age_group'] = all_counts_baseline_pt_df.age_group.str.replace('_',' ')
    all_counts_baseline_pt_df['cause'] = all_counts_baseline_pt_df.cause.mapping_dict_data(['cause'])
    
    all_counts_baseline_pt_df['measure'] = all_counts_baseline_pt_df.measure.mapping_dict_data(['measure'])
      all_counts_baseline_pt_df['sex'] = all_counts_baseline_pt_df.sex.mapping_dict_data(['sex'])
    all_counts_baseline_pt_df = all_counts_baseline_pt_df.rename(columns={'scenario':'Scenario'})
    all_counts_baseline_pt_df['Scenario'] = all_counts_baseline_pt_df.Scenario.mapping_dict_data(['scenario'])
    all_counts_baseline_pt_df = all_counts_baseline_pt_df[(all_counts_baseline_pt_df.age_group != '30 to 34') & (all_counts_baseline_pt_df.age_group != '35 to 39')]  

    clrs = sns.color_palette('muted', 4)
    bright = sns.color_palette('bright',4)
    scenario_list = ['Baseline','Free choice combination','Start low go slow','Single pill combination']
    palette_list = ['#000000','#023eff','#ff7c00','#1ac938']
    palette_dict = {x:v for x,v in zip(scenario_list,palette_list)}
       
    for cause in ['CKD, Stroke, and IHD', 'Chronic kidney disease', 'Intracerebral hemorrhage','Ischemic heart disease', 'Ischemic stroke', 'Stroke','Subarachnoid hemorrhage']:
            with PdfPages(str(save_path)+f'/results_rates/{cause}.pdf') as pdf:
                for sex in ['females', 'males', 'both sexes combined']:
                    for measure in ['DALYs', 'Deaths', 'Incidence','YLDs', 'YLLs']:
                        diff_upper = all_counts_baseline_pt_df[(all_counts_baseline_pt_df.cause == cause) & (all_counts_baseline_pt_df.sex == sex) & (all_counts_baseline_pt_df.measure == measure)].measure_rate.max()
                        diff_lower = all_counts_baseline_pt_df[(all_counts_baseline_pt_df.cause == cause) & (all_counts_baseline_pt_df.sex == sex) & (all_counts_baseline_pt_df.measure == measure)].measure_rate.min()
                        for country in ['China', 'Italy', 'South Korea', 'Mexico', 'Russian Federation']:

                            data_draw = all_counts_baseline_pt_df[(all_counts_baseline_pt_df.cause == cause) & (all_counts_baseline_pt_df.sex == sex) & (all_counts_baseline_pt_df.measure == measure) & (all_counts_baseline_pt_df.location == country)].sort_values(by=['age_group','Scenario'])


                            data_m = data_draw.groupby(['age_group', 'Scenario']).mean().reset_index()
                            formatted_location = data_draw.location.unique()[0]

                            plt.figure(figsize=(20, 10))

                            g = sns.catplot(x='age_group', y='measure_rate',
                                            hue='Scenario', palette=palette_dict,
                                            height=10, aspect=1.2, alpha=0.2,legend=False,
                                            data=data_draw)
                            sns.scatterplot(x='age_group', y='measure_rate', 
                                            hue='Scenario', legend='brief', palette=palette_dict,
                                            s=400, marker='P', alpha=1, 
                                            ax=g.ax, data=data_m)

                            g.ax.set_title(f'{cause} {measure} in {formatted_location}, {sex}')
                            g.ax.set_xticklabels(g.ax.get_xticklabels(), rotation=70)
                            g.ax.set_xlabel('Age group')
                            g.ax.set_ylabel(f'{measure} per 100k person-years')
                            g.ax.set_ylim(diff_lower,diff_upper)
                            handles, labels = g.ax.get_legend_handles_labels()
                            g.ax.legend(handles[:4],labels[:4],bbox_to_anchor=(0.05, 1), loc='upper left')
                            labels = g.axes[0,0].get_yticks()
                            formatted_labels = ['{:,}'.format(int(label)) for label in labels]
                            g.set_yticklabels(formatted_labels)

                           pdf.savefig(g.fig, orientation = 'landscape', bbox_inches='tight')

                            plt.show()
                            plt.clf() 

def plot_baseline_diff_rates(data):
    """
    Takes the DataFrame results from the get_all_country_counts(data) for all locations and returns the plots of the outcomes averted compared to the baseline scenario. 
    This function creates the 'outcome averted compared to Baseline in location, sex' plots from the final report.
    """
    ui_rates = get_all_country_counts(data)
    ui_rates_melt = ui_rates.melt(id_vars=['Age group','Sex','Cause','Measure','Scenario','Location'])
    ui_rates_melt = ui_rates_melt[(ui_rates_melt.Scenario !='Baseline') & (ui_rates_melt.age_group != '30 to 34') & (ui_rates_melt.age_group != '35 to 39') & (ui_rates_melt.age_group != 'all ages combined') &(ui_rates_melt.age_group != '40 plus') ]
    
     scenario_list = ['Free choice combination','Start low go slow','Single pill combination']
    palette_list = ['#023eff','#ff7c00','#1ac938']
    palette_dict = {x:v for x,v in zip(scenario_list,palette_list)}

    for cause in ['CKD, Stroke, and IHD', 'Chronic kidney disease', 'Intracerebral hemorrhage','Ischemic heart disease', 'Ischemic stroke', 'Stroke','Subarachnoid hemorrhage']:
        with PdfPages(str(save_path)+f'/diff_baseline_rates/{cause}_db.pdf') as pdf:
            for sex in ['females', 'males', 'both sexes combined']:
                for measure in ['DALYs', 'Deaths', 'Incidence','YLDs', 'YLLs']:
                    diff_upper = ui_rates_melt[(ui_rates_melt.cause == cause) & (ui_rates_melt.sex == sex) & (ui_rates_melt.measure == measure)&(ui_rates_melt.variable == 'Upper scenario difference from Baseline rate (per 100,000 person-years)')].value.max()
                    diff_lower = ui_rates_melt[(ui_rates_melt.cause == cause) & (ui_rates_melt.sex == sex) & (ui_rates_melt.measure == measure)& (ui_rates_melt.variable == 'Lower scenario difference from Baseline rate (per 100,000 person-years)')].value.min()
                    for country in ['China', 'Italy', 'South Korea', 'Mexico', 'Russian Federation']:

                        plt.figure(figsize=(45, 10))
                        data_r = ui_rates_melt[(ui_rates_melt.Location == country) & (ui_rates_melt.Cause == cause)& (ui_rates_melt.Sex == sex)& (ui_rates_melt.Measure == measure)]
                        formatted_location = data_r.Location.unique()[0]

                        g = sns.catplot(x='Age group',y='value',
                                    hue='Scenario',kind='bar',legend=False, aspect = 3,palette=palette_dict,data=data_r)

                        g.fig.suptitle(f'{cause} {measure} averted compared to Baseline \n in {formatted_location}, {sex}',size=18,va='center',ha='center')
                        plt.xticks(rotation=45)
                        g.set_xlabels('Age group',fontsize=18)
                        g.set_ylabels(f'{measure} per 100k person-years',fontsize=18)
                        g.ax.set_ylim(diff_lower,diff_upper)
                        plt.legend(bbox_to_anchor=(0.05, 1), loc='upper left')
                        labels = g.axes[0,0].get_yticks()
                        formatted_labels = ['{:,}'.format(int(label)) for label in labels]
                        g.set_yticklabels(formatted_labels)

                        pdf.savefig(g.fig, orientation = 'landscape', bbox_inches='tight')
                        plt.show()
                        plt.clf()

def plot_baseline_diff_rates_all_ages(data):
    """
    Takes the DataFrame results from the get_all_country_counts(data) for all locations and returns the plots of the outcomes averted compared to the baseline scenario, grouped by country. 
    This function creates the 'outcome averted compared to Baseline across countries for 40 plus, sex' plots from the final report.
    """
    ui_rates = get_all_country_counts(data)
    ui_rates_melt = ui_rates.melt(id_vars=['Age group','Sex','Cause','Measure','Scenario','Location'])
    ui_rates_melt = ui_rates_melt[(ui_rates_melt.Scenario !='Baseline') & (ui_rates_melt.age_group != '30 to 34') & (ui_rates_melt.age_group != '35 to 39') & (ui_rates_melt.age_group == '40 plus') ]
    
    scenario_list = ['Free choice combination','Start low go slow','Single pill combination']
    palette_list = ['#023eff','#ff7c00','#1ac938']
    palette_dict = {x:v for x,v in zip(scenario_list,palette_list)}

    for cause in ['CKD, Stroke, and IHD', 'Chronic kidney disease', 'Intracerebral hemorrhage','Ischemic heart disease', 'Ischemic stroke', 'Stroke','Subarachnoid hemorrhage']:
        with PdfPages(str(save_path)+f'/diff_baseline_rates/{cause}_all_ages.pdf') as pdf:
            for sex in ['females', 'males', 'both sexes combined']:
                for measure in ['DALYs', 'Deaths', 'Incidence','YLDs', 'YLLs']:

                    plt.figure(figsize=(45, 10))
                    data_r = ui_rates_melt[(ui_rates_melt.Cause == cause)& (ui_rates_melt.Sex == sex)& (ui_rates_melt.Measure == measure)]

                    g = sns.catplot(x='Location',y='value',
                                hue='Scenario',kind='bar',legend=False, aspect = 3,palette=palette_dict,data=data_r)

                    g.fig.suptitle(f'{cause} {measure} averted compared to Baseline \n across countries for 40 plus, {sex}',size=18,va='center',ha='center')
                    plt.xticks(rotation=45)
                    g.set_ylabels(f'{measure} per 100k person-years',fontsize=18)
                    plt.legend(bbox_to_anchor=(0.05, 1), loc='upper left')
                    labels = g.axes[0,0].get_yticks()
                    formatted_labels = ['{:,}'.format(int(label)) for label in labels]
                    g.set_yticklabels(formatted_labels)

                     pdf.savefig(g.fig, orientation = 'landscape', bbox_inches='tight')
                    plt.show()
                    plt.clf()

def plot_adverse_events(data):
    """
    Takes the adverse events rate data table and returns plots of adverse events by Age group. 
    This function creates the 'Treatment-related adverse events in location' plots from the final report.
    """
    adverse_events = adverse_events_rate(data)
    scenario_list = ['Free choice combination','Start low go slow','Single pill combination']
    palette_list = ['#023eff','#ff7c00','#1ac938']
    palette_dict = {x:v for x,v in zip(scenario_list,palette_list)}

    for sex in ['females', 'males','both sexes combined']:
         for med in ['All medications']:

            data_draw = adverse_events[(adverse_events.sex == sex) & (adverse_events.med==med)].sort_values(by=['age_group','scenario'])

            data_m = data_draw.groupby(['age_group', 'scenario']).measure_rate.mean().reset_index()
            formatted_location = data_draw.location.unique()[0]

            plt.figure(figsize=(20, 10))

            g = sns.catplot(x='age_group', y='measure_rate',
                            hue='scenario', palette=palette_dict,
                            height=10, aspect=1.2, alpha=0.4,legend=False,
                            data=data_draw)
            sns.scatterplot(x='age_group', y='measure_rate', 
                            hue='scenario', palette=palette_dict,
                            s=400, marker='P', 
                            ax=g.ax, legend='full',data=data_m)
            g.ax.set_title(f'Treatment-related adverse events in {formatted_location}, {sex}')
            g.ax.set_xticklabels(g.ax.get_xticklabels(), rotation=70)
            g.ax.set_xlabel('Age group')
            g.ax.set_ylabel('Adverse events per 100k person-years')
            g.ax.set(ylim=(0,1100))
            handles, labels = g.ax.get_legend_handles_labels()
            g.ax.legend(handles[:3],labels[:3],bbox_to_anchor=(0.05, 1), loc='upper left')
            labels = g.axes[0,0].get_yticks()
            formatted_labels = ['{:,}'.format(int(label)) for label in labels]
            g.set_yticklabels(formatted_labels)
            
            if path is not None:
                            plt.savefig(str(save_path)/ 'number_adverse_events' / f'{sex}_{location}.pdf'), orientation='landscape',bbox_inches='tight')                    
            plt.show()
            plt.clf() 

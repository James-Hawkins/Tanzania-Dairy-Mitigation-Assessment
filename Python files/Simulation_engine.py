import sys
import os

sys.path.append('../../../..')
from livsim import Cow
from livsim import FeedStorage
from livsim import Feed

import yaml
import datetime
import time
import math
import random
import warnings

import numpy as np
import pandas as pd
import xlrd 
import textwrap

import csv

warnings.filterwarnings("ignore")




pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)



def sim_engine(  res,
                 data_wb,
                 scenario_parameters,
                 breed,
                 cohort,
                 lps,
                 reg,
                 subsectors,
                 l,
                 r,
                 s,
                 herd,
                 herd_y0,
                 herd_baseline_yf,
                 check_feasibility,
                 timestep,
                 MC_sims,
                 reg_analysis,
                 ss_analysis,
                 consequential):
    


        ## Load data sheets from excelsubse
        ghg_sheet= data_wb.sheet_by_index(1)
        uncertainty_sheet= data_wb.sheet_by_index(2)


        yield_sheet=data_wb.sheet_by_index(5)
        ryld_data=data_wb.sheet_by_index(6)


        feed_util_effcy=data_wb.sheet_by_index(8)
        LCA_params=data_wb.sheet_by_index(9)


        ## work sheets loaded per region

        if ( reg_analysis == 0 ) :

            if (l == 'MRT'):
                sheet = data_wb.sheet_by_index(3)
                diet_data=(cwd+str('\\diet_MRT.xlsx'))
            elif (l == 'MRH'):

                sheet = data_wb.sheet_by_index(4) 
                diet_data=(cwd+str('\\diet_MRH.xlsx'))

            diets=xlrd.open_workbook(diet_data) 
            diet_wb = xlrd.open_workbook(diet_data)

        elif ( reg_analysis == 1 ) :


            reg_data=Path(cwd+str('/regional data files/'))
            regional_herds=reg_data/'regional_herd_populations.xlsx'
            regional_LUC_coeffs=reg_data/'regional_LUC_coeffs.xlsx'
            reg_herd=xlrd.open_workbook(regional_herds)


            if ( r == 'NB'):
                if (l == 'MRT'):
                    diet_data =reg_data/'diet_MRT_NB.xlsx'

                elif (l == 'MRH'):
                    diet_data =reg_data/'diet_MRH_NB.xlsx'


            elif (r == 'MF'):
                if (l == 'MRT'):
                    diet_data =reg_data/'diet_MRT_MF.xlsx'

                elif (l == 'MRH'):
                    diet_data =reg_data/'diet_MRH_MF.xlsx'

            elif (r == 'MM'):
                if (l == 'MRT'):

                    diet_data =reg_data/'diet_MRT_MM.xlsx'

                elif (l == 'MRH'):

                    diet_data =reg_data/'diet_MRH_MM.xlsx'

            elif (r == 'RW'):
                if (l == 'MRT'):

                    diet_data =reg_data/'diet_MRT_RW.xlsx'

                elif (l == 'MRH'):

                    diet_data =reg_data/'diet_MRH_RW.xlsx'


            diet_wb = xlrd.open_workbook(diet_data)
            sheet = diet_wb.sheet_by_index(7)



        # Main results from simulations are initially stored in dictionaries


        # feed intake
        grass_intake={}
        stover_intake={}
        maize_bran_intake={}
        sfsm_intake={}
        napier_intake={}
        napier_hy_intake={}
        pasture_intake={}

        # dietary and excretion (nitrogen)
        faecal_Nitrogen = {}
        urinary_Nitrogen = {}
        fraction_roughage_eaten = {}
        fraction_conc_eaten={}
        gross_energy = {}
        crude_protein_intake={}
        DMI={}
        Body_weight={}
        DMD={}
        acid_detergent_fibre={}
        neutral_detergent_fibre={}
        Diet_N={}
        ADF={}
        NDF={}
        ME={}
        Ym={}
        VS={}

        TLU_equiv={}
        utn_efy={}
        dressing_pct={}
        ensilage_fodder_frac={}

        # GHG emissions 
        enteric_CH4={}
        manure_CH4={}

        soil_N2O_cropland={}
        soil_N2O_direct_cropland={}
        soil_N2O_indirect_cropland={}
        soil_N2O_indirect_vol_cropland={}
        soil_N2O_indirect_leach_cropland={}

        soil_N2O_grassland={}
        soil_N2O_direct_grassland={}
        soil_N2O_indirect_grassland={}
        soil_N2O_indirect_vol_grassland={}
        soil_N2O_indirect_leach_grassland={}

        manure_N2O={}
        manure_N2O_direct={}
        manure_N2O_vol={}
        manure_N2O_leach={}

        soil_N2O_feed={}
        soil_N2O_feed_direct={}
        soil_N2O_feed_adn={}
        soil_N2O_feed_lch={}

        energy_use_CO2={}

        enteric_CH4_intensity={}
        manure_CH4_intensity={}
        soil_N2O_intensity={}
        manure_N2O_intensity={}
        direct_emission_intensity={}

        # variables used in uncertainty quantification
        Ym_rd={}
        B0_rd={}
        MCF_rd={}
        EF_3_FAECAL_rd={}
        EF_3_URINARY_rd={}

        enteric_CH4_rd={}
        manure_CH4_rd={}
        manure_N2O_rd={}
        manure_N2O_direct_rd={}
        manure_N2O_leach_rd={}
        manure_N2O_vol_rd={}
        energy_use_CO2_rd={}

        soil_N2O_feed_direct_rd={}
        soil_N2O_feed_adn_rd={}
        soil_N2O_feed_lch_rd={}
        soil_N2O_feed_rd={}
        soil_N2O_rd={}

        grassland_area_expansion_emissions_rd={}
        cropland_area_expansion_emissions_rd={}

        herd_rd={}
        direct_emission_intensity_rd={}
        direct_emission_absolute_rd={}
        direct_emission_absolute_breed_rd={}  
        total_emission_absolute_rd={}

        crop_primary_area_rd={}
        feed_primary_area_rd={}

        grassland_area_expansion_emissions_rd_ss={}
        cropland_area_expansion_emissions_rd_ss={}

        for MC in range(0,MC_sims):
            grassland_area_expansion_emissions_rd[MC]=0
            cropland_area_expansion_emissions_rd[MC]=0
            direct_emission_absolute_rd[MC]=0
            direct_emission_absolute_breed_rd[MC]=0
            direct_emission_intensity_rd[MC]=0
            direct_emission_absolute_breed_rd[MC,'local']=0
            direct_emission_absolute_breed_rd[MC,'improved']=0
            total_emission_absolute_rd[MC]=0

            for ss in subsectors:
                grassland_area_expansion_emissions_rd_ss[MC,ss]=0
                cropland_area_expansion_emissions_rd_ss[MC,ss]=0

        # Data frame used to store all model results for a given iteration
        out={}


        # land footprint 
        feed_area_base={}
        feed_primary_area_base={}
        feed_area={}
        feed_primary_area={}

        total_grasslands_area={}

        crop_primary_area={}
        crop_primary_area_base={}

        pasture_area={}
        pasture_area_base={}

        grass_area={}
        grass_area_base={}

        stover_area={}
        stover_area_base={}

        napier_area={}
        napier_area_base={}

        napier_hy_area={}
        napier_hy_area_base={}

        maize_as_bran_area={}
        maize_as_bran_area_base={}

        sunflower_as_meal_area={}
        sunflower_as_meal_area_base={}

        # monthly feed parameters
        mo_me_short={}
        mo_mp_short={}

        mo_dp_cp ={}
        mo_dp_me ={}

        mo_fi_nhy = {}
        mo_fi_np={}

        mo_fi_ns = {}
        mo_fi_sto={}
        mo_fi_mb = {}
        mo_fi_sc={}
        mo_fi_grs = {}
        mo_fi_pt={}

        count_calendar_month={}
        
        # Production variables (milk and meat)
        # milk yield
        milk_yield_reg = {} 
        milk_yield_lact = {}
        
        Herd_Milk_yield_kg_per_year = {}

        # dairy-beef production
        Herd_Meat_yield_Bull_kg_per_year={}
        Herd_Meat_yield_Wmale_kg_per_year={}
        Herd_Meat_yield_Adult_female_kg_per_year={}
        Herd_Meat_yield_Total_kg_per_year={}
        Meat_yield_per_head_kg_per_year={}

        final_bodyweight_kg={}
        final_age_months={}

        for ss in subsectors:
            for b in breed:
                for c in cohort:
                    Meat_yield_per_head_kg_per_year[ss,b,c]=0

                # LCA allocation parameters
                Herd_Milk_yield_kg_per_year[ss,b]=0
                Herd_Meat_yield_Total_kg_per_year[ss,b]=0
                final_bodyweight_kg[ss,b]=0
                final_age_months[ss,b]=0

                out['v1_3_Avoided_Beef_kg_per_year_'+str(ss)+'_'+str(b)]=0


        dressing_pct[('cow')]=LCA_params.cell_value(1,1)
        dressing_pct[('bull')]=LCA_params.cell_value(2,1)
        dressing_pct[('ml_calf')]=LCA_params.cell_value(3,1)

        # Populate parameters from external data files
        # GHG Emission factors (EFs)
        if (l == 'MRT'):
            col_ghg=2
        elif (l == 'MRH'):
            col_ghg=3

        # Manure efs
        B0=ghg_sheet.cell_value(2,col_ghg)
        MCF_LOC=ghg_sheet.cell_value(3,col_ghg)
        MCF_IMP=ghg_sheet.cell_value(4,col_ghg)
        EF_1=ghg_sheet.cell_value(5,col_ghg)
        EF_3_FAECAL=ghg_sheet.cell_value(6,col_ghg)
        EF_3_URINARY=ghg_sheet.cell_value(7,col_ghg)
        EF_3_PRP=ghg_sheet.cell_value(8,col_ghg)
        FRAC_MANURE_PRP_LOC=ghg_sheet.cell_value(9,col_ghg)
        FRAC_MANURE_MMT_LOC=ghg_sheet.cell_value(10,col_ghg)
        FRAC_MANURE_PRP_IMP=ghg_sheet.cell_value(11,col_ghg)
        FRAC_MANURE_MMT_IMP=ghg_sheet.cell_value(12,col_ghg)
        EF_4_LOC=ghg_sheet.cell_value(13,col_ghg)
        EF_4_IMP=ghg_sheet.cell_value(14,col_ghg)
        EF_5_LOC=ghg_sheet.cell_value(15,col_ghg)
        EF_5_IMP=ghg_sheet.cell_value(16,col_ghg)
        FRAC_vol_LOC=ghg_sheet.cell_value(17,col_ghg)
        FRAC_vol_IMP=ghg_sheet.cell_value(18,col_ghg)
        FRAC_leach_LOC=ghg_sheet.cell_value(19,col_ghg)
        FRAC_leach_IMP=ghg_sheet.cell_value(20,col_ghg)

        # Soil related efs
        EF_4_adn=ghg_sheet.cell_value(21,col_ghg)
        EF_5_lch=ghg_sheet.cell_value(22,col_ghg)
        FRAC_gasf=ghg_sheet.cell_value(23,col_ghg)
        FRAC_gasm=ghg_sheet.cell_value(24,col_ghg)
        FRAC_loss_MMS=ghg_sheet.cell_value(25,col_ghg)

        # land carbon stock densities
        mean_C_density_native_eco_Mg_C_ha = ghg_sheet.cell_value(26,col_ghg)
        Maximal_Forest_C_Density = ghg_sheet.cell_value(27,col_ghg)
        mean_C_density_grassland_Mg_C_ha = ghg_sheet.cell_value(28,col_ghg)
        luc_emission_coefficient_grass_to_crop = 11 # as described in article
        Reforestation_emission_coefficient_domestic=32.5
        Reforestation_emission_coefficient_row=45
        Rr=150
        
        # amortization period used in LUC emissions accounting
        amort_period = ghg_sheet.cell_value(29,col_ghg)

        # Set LUC coefficient based on model type
        if (reg_analysis == 0):
            luc_emission_coefficient_total_area = mean_C_density_native_eco_Mg_C_ha - mean_C_density_grassland_Mg_C_ha

        elif ( reg_analysis == 1 ) :

            LUC_regional = xlrd.open_workbook(regional_LUC_coeffs)
            luc_coeff_sheet = LUC_regional.sheet_by_index(0)

            if ( r == 'MF'):
                if (l == 'MRT'):
                    mean_C_density_native_eco_Mg_C_ha = luc_coeff_sheet.cell(1,2).value

                elif (l == 'MRH'):
                    mean_C_density_native_eco_Mg_C_ha = luc_coeff_sheet.cell(5,2).value

            elif ( r == 'MM'):
                if (l == 'MRT'):
                    mean_C_density_native_eco_Mg_C_ha = luc_coeff_sheet.cell(2,2).value

                elif (l == 'MRH'):
                    mean_C_density_native_eco_Mg_C_ha = luc_coeff_sheet.cell(6,2).value

            elif ( r == 'NB'):
                if (l == 'MRT'):
                    mean_C_density_native_eco_Mg_C_ha = luc_coeff_sheet.cell(3,2).value

                elif (l == 'MRH'):
                    mean_C_density_native_eco_Mg_C_ha = luc_coeff_sheet.cell(7,2).value

            elif ( r == 'RW'):
                if (l == 'MRT'):
                    mean_C_density_native_eco_Mg_C_ha = luc_coeff_sheet.cell(4,2).value

                elif (l == 'MRH'):
                    mean_C_density_native_eco_Mg_C_ha= luc_coeff_sheet.cell(8,2).value


        luc_emission_coefficient_total_area =  mean_C_density_native_eco_Mg_C_ha - mean_C_density_grassland_Mg_C_ha

        # allocation factors
        allocation_factor_grass = ghg_sheet.cell_value(30,col_ghg)
        allocation_factor_napier = ghg_sheet.cell_value(31,col_ghg)
        allocation_factor_maize_bran = ghg_sheet.cell_value(32,col_ghg)
        allocation_factor_sunflower_seed_meal = ghg_sheet.cell_value(33,col_ghg)
        allocation_factor_stover = ghg_sheet.cell_value(34,col_ghg)

        # Fossil energy coefficients for processed feeds (maize bran, sunflower cake) in kg co2eq per Mg 
        feed_embodied_CO2={"maize_bran": ghg_sheet.cell_value(35,col_ghg),"sunflower_seed_meal":ghg_sheet.cell_value(35,col_ghg)} 
        
        # Fossil energy coefficients for fertilizer in kg co2eq per Mg 
        fertilizer_embodied_CO2=ghg_sheet.cell_value(36,col_ghg)

        # Use efficiencies for individdual feeds
        utn_efy[('grass')]=feed_util_effcy.cell_value(1,1)
        utn_efy[('pasture')]=feed_util_effcy.cell_value(2,1)
        utn_efy[('napier')]=feed_util_effcy.cell_value(3,1)
        utn_efy[('napier_hy')]=feed_util_effcy.cell_value(4,1)
        utn_efy[('maize_bran')]=feed_util_effcy.cell_value(5,1)
        utn_efy[('sunflower_seed_meal')]=feed_util_effcy.cell_value(6,1)
        utn_efy[('stover')]=feed_util_effcy.cell_value(7,1)

        '''
        All results from each iteration of the model get stored in the 'out' dictionary.
        These dictionaries represent the results from one simulation unit of the model (which differs depending on the type of model run).
        In the 'run' method of the main model class, each dictionary forms an individual row of the final resulting 
        data file (a pandas dataframe). 
        '''
        
        # Define and initialize results variables
        out['v1_2_Milk_yield_total_kg_yr']=0
        out['v1_2_Milk_yield_local_total_kg_yr']=0
        out['v1_2_Milk_yield_improved_total_kg_yr']=0
        out['v1_2_Milk_yield_base_year_kg_yr']=0
        out['v1_3_Avoided_Beef_land_use_Total_ha']=0 # used only if model is set as consequential

        for ss in subsectors:
            for b in breed:
                out['v8_7_herd_total_'+str(b)+'_TLU_per_lps']=0
                out['v1_3_Herd_Meat_Yield_kg_per_TLU_'+str(ss)+'_'+str(b)]=0

            out['v3_4_1_Feed_intake_per_cow_grass_kg_dm_yr_ss'+str(ss)+'']=0
            out['v3_4_2_Feed_intake_per_cow_stover_kg_dm_yr_ss'+str(ss)+'']=0
            out['v3_4_3_Feed_intake_per_cow_napier_kg_dm_yr_ss'+str(ss)+'']=0
            out['v3_4_4_Feed_intake_per_cow_napier_hy_kg_dm_yr_ss'+str(ss)+'']=0
            out['v3_4_5_Feed_intake_per_cow_maize_bran_kg_dm_yr_ss'+str(ss)+'']=0
            out['v3_4_6_Feed_intake_per_cow_sfsm_kg_dm_yr_ss'+str(ss)+'']=0
            out['v3_4_7_Feed_intake_per_cow_pasture_kg_dm_yr_ss'+str(ss)+'']=0

        
        out['v3_2_1_Feed_intake_grass_%_dm']=0
        out['v3_2_2_Feed_intake_stover_%_dm']=0
        out['v3_2_3_Feed_intake_napier_%_dm']=0
        out['v3_2_4_Feed_intake_maize_bran_%_dm']=0
        out['v3_2_5_Feed_intake_sfsm_%_dm']=0

        out['v3_2_6_Feed_intake_pasture_%_dm']=0
        out['v3_2_6_Feed_intake_grass_kg_dm']=0
        out['v3_2_7_Feed_intake_stover_kg_dm']=0
        out['v3_2_8_Feed_intake_napier_kg_dm']=0
        out['v3_2_8_Feed_intake_napier_hy_kg_dm']=0
        out['v3_2_9_Feed_intake_maize_bran_kg_dm']=0
        out['v3_2_10_Feed_intake_sfsm_kg_dm']=0
        out['v3_2_11_Feed_intake_pasture_kg_dm']=0
        
        out['v5_6_Total_feed_primary_area_expansion']=0
        out['v5_6_Total_grasslands_area_expansion']=0

        out['v6_0_1_Enteric_CH4_aggregate_Mg_CO2eq_per_lps']=0
        out['v6_0_2_Manure_CH4_aggregate_Mg_CO2eq_per_lps']=0
        out['v6_0_3_Manure_N2O_aggregate_Mg_CO2eq_per_lps']=0
        out['v6_0_4_Soil_N2O_cropland_aggregate_Mg_CO2eq_per_lps']=0
        out['v6_0_4_Soil_N2O_grassland_aggregate_Mg_CO2eq_per_lps']=0
        out['v6_0_4_2_Feed_CO2_aggregate_Mg_CO2eq_per_lps']=0
        out['v6_0_5_Direct_emissions_aggregate_Mg_CO2eq_per_lps']=0
        out['v6_0_6_grassland_expansion_emissions_Mg_CO2eq_per_lps']=0
        out['v6_0_8_cropland_expansion_emissions_Mg_CO2eq_per_lps']=0
        out['v6_0_9_afforestation_emissions_Mg_CO2eq_per_lps']=0

        out['v6_1_1_Enteric_CH4_local_aggregate_Mg_CO2eq_per_lps']=0
        out['v6_1_2_Manure_CH4_local_aggregate_Mg_CO2eq_per_lps']=0
        out['v6_1_3_Manure_N2O_local_aggregate_Mg_CO2eq_per_lps']=0
        out['v6_1_4_Soil_N2O_cropland_local_aggregate_Mg_CO2eq_per_lps']=0
        out['v6_1_4_Soil_N2O_grassland_local_aggregate_Mg_CO2eq_per_lps']=0
        out['v6_1_4_2_Feed_CO2_local_aggregate_Mg_CO2eq_per_lps']=0
        out['v6_1_5_Direct_emissions_local_aggregate_Mg_CO2eq_per_lps']=0
        out['v6_1_6_grassland_expansion_direct_local_emissions_Mg_CO2eq_per_lps']=0
        out['v6_1_8_cropland_expansion_local_emissions_Mg_CO2eq_per_lps']=0
        out['v6_1_9_afforestation_local_emissions_Mg_CO2eq_per_lps']=0

        out['v6_2_1_Enteric_CH4_improved_aggregate_Mg_CO2eq_per_lps']=0
        out['v6_2_2_Manure_CH4_improved_aggregate_Mg_CO2eq_per_lps']=0
        out['v6_2_3_Manure_N2O_improved_aggregate_Mg_CO2eq_per_lps']=0
        out['v6_2_4_Soil_N2O_cropland_improved_aggregate_Mg_CO2eq_per_lps']=0
        out['v6_2_4_Soil_N2O_grassland_improved_aggregate_Mg_CO2eq_per_lps']=0
        out['v6_2_4_2_Feed_CO2_improved_aggregate_Mg_CO2eq_per_lps']=0
        out['v6_2_5_Direct_emissions_improved_aggregate_Mg_CO2eq_per_lps']=0
        out['v6_2_6_grassland_expansion_direct_improved_emissions_Mg_CO2eq_per_lps']=0
        out['v6_2_8_cropland_expansion_improved_emissions_Mg_CO2eq_per_lps']=0
        out['v6_2_9_afforestation_improved_emissions_Mg_CO2eq_per_lps']=0

        out['v8_4_herd_total_1000_head_per_lps']=0
        out['v8_6_herd_total_TLU_per_lps']=0
        out['v8_8_herd_total_local_Head_per_lps']=0
        out['v8_8_herd_total_improved_Head_per_lps']=0

        for ss in subsectors:
            for b in breed:
                for c in cohort:
                    out['v8_1_herd_'+str(ss)+'_'+str(b)+'_'+str(c)]=0
                    out['v8_2_herd_previous_year_'+str(ss)+'_'+str(b)+'_'+str(c)]=0
                    out['v8_3_herd_baseline_yf_'+str(ss)+'_'+str(b)+'_'+str(c)]=0


        out['v6_0_Units_limited']=0

        out['v9_2_1_Grasslands_area_ha_per_lps'] =0
        out['v9_2_2_Feed_area_ha_per_lps'] =0
        out['v9_2_3_Crop_primary_area_ha_per_lps']=0

        out['v9_4_1_Grass_area_ha_per_lps']=0
        out['v9_4_2_Stover_area_ha_per_lps']=0
        out['v9_4_3_Napier_area_ha_per_lps']=0
        out['v9_4_4_Napier_hy_area_ha_per_lps']=0
        out['v9_4_5_Sunflower_area_ha_per_lps']=0
        out['v9_4_6_Maize_area_ha_per_lps']=0
        out['v9_4_7_Pasture_area_ha_per_lps']=0
        
        out['v9_15_Grass_land_use_ha_per_TLU']=0
        out['v9_16_Stover_land_use_ha_per_TLU']=0
        out['v9_17_Napier_land_use_ha_per_TLU']=0
        out['v9_17_Napier_hy_land_use_ha_per_TLU']=0
        out['v9_18_Sunflower_land_use_ha_per_TLU']=0
        out['v9_19_Maize_land_use_ha_per_TLU']=0
        out['v9_20_Pasture_land_use_ha_per_TLU']=0

        out['v_11_2_error_absolute_local_Mg']=0
        out['v_11_2_error_intensity_local_kg']=0 
        out['v_11_2_error_per_animal_local_Mg_per_TLU']=0

        out['v_11_2_error_absolute_improved_Mg']=0
        out['v_11_2_error_intensity_improved_kg']=0
        out['v_11_2_error_per_animal_improved_Mg_per_TLU']=0

        total_expansion={}
        total_grslnd_expansion={}

    
        ''' Each simulation unit is disaggregated into subsectors, breeds, and cohorts. 
        The final results for a given simulation unit are then aggregated over subsectors, breeds, and cohorts
        '''
        
    # Initiate simulations

        for ss in subsectors:   
            
            for b in breed:
                
                if (herd[(ss,b,'total_herd')] == 0): # if there are no animals of a given breed, continue to next step
                    continue

                for c in cohort:

                    # initialize variables to evaluate seasonal (monthly) 
                    # nutrient availability per animal (crude protein, cp, and metabolisable energy, me)

                    for i in range(0,12):
                        mo_me_short[ss,b,c,i]  = 0
                        mo_mp_short[ss,b,c,i]  = 0
                        mo_dp_cp[ss,b,c,i]  = 0
                        mo_dp_me[ss,b,c,i]  = 0

                        mo_fi_pt[ss,b,c,i] = 0
                        mo_fi_nhy[ss,b,c,i] = 0
                        mo_fi_np[ss,b,c,i] = 0
                        mo_fi_sto[ss,b,c,i] = 0
                        mo_fi_mb[ss,b,c,i] = 0
                        mo_fi_sc[ss,b,c,i] = 0
                        mo_fi_grs[ss,b,c,i] = 0
                        mo_fi_ns[ss,b,c,i] = 0

                        
                    # Define parameters of animal (sex, breed, basic reproductive parameters) for current iteration

                    if (c == 'cow'):
                        
                        if (b == 'local'):
                            
                            bc_count=0
                            start_age=28
                            max_simu_range=12*13-20
                            
                            if (scenario_parameters[(ss,'artificial_insemination',s)]==1):
                                
                                d = Cow(age=start_age,breed='Zebu -- TZ south high')
                                d.sex='f'
                                d.parameters['potential_annual_calving_rate']['value']=1.0

                            else:
                                
                                d = Cow(age=start_age,breed='Zebu -- TZ south high') 
                                d.parameters['potential_annual_calving_rate']['value']=0.45
                                d.sex='f'

                        else:
                            
                            bc_count=5
                            start_age=20
                            max_simu_range=12*13-20
                            
                            if (scenario_parameters[(ss,'artificial_insemination',s)]==1):
                                
                                d = Cow(age=start_age,breed='Improved Tanz southern highlands')
                                d.parameters['potential_annual_calving_rate']['value']=1.0
                                
                            else:
                                d = Cow(age=start_age,breed='Improved Tanz southern highlands') 
                                d.sex='f'


                    elif (c == 'bull'):
                        
                        start_age=36
                        max_simu_range=24
                        
                        if (b == 'local'):
                            
                            bc_count=3
                            d=Cow(age=start_age,breed='Zebu -- TZ south high')
                            d.sex='m'
                            
                        else:
                            bc_count=8
                            d=Cow(age=start_age,breed='Improved Tanz southern highlands')
                            d.sex='m'

                    elif (c == 'ml_calf'):
                        
                        start_age=1
                        max_simu_range=12
                        
                        if (b == 'local'):
                            
                            bc_count=2
                            d=Cow(age=start_age,breed='Zebu -- TZ south high')
                            d.sex='m'
                            
                        else:
                            bc_count=7
                            d=Cow(age=start_age,breed='Improved Tanz southern highlands')
                            d.sex='m'

                    elif (c == 'fe_calf'):
                        
                        start_age=1
                        max_simu_range=12
                        
                        if (b == 'local'):
                            bc_count=2
                            d=Cow(age=start_age,breed='Zebu -- TZ south high')
                            d.sex='f'
                            
                        else:
                            bc_count=7
                            d=Cow(age=start_age,breed='Improved Tanz southern highlands')
                            d.sex='f'

                    elif (c == 'heifer'):
                        
                        start_age=12
                        max_simu_range=36
                        
                        if (b == 'local'):
                            bc_count=1
                            d=Cow(age=start_age,breed='Zebu -- TZ south high')
                            d.sex='f'
                        else:
                            bc_count=6
                            d=Cow(age=start_age,breed='Improved Tanz southern highlands')
                            d.sex='f'

                    elif (c == 'juv_male'):
                        start_age=12
                        max_simu_range=24
                        
                        if (b == 'local'):
                            bc_count=4
                            d=Cow(age=start_age,breed='Zebu -- TZ south high')
                            d.sex='m'
                        else:
                            bc_count=9
                            d=Cow(age=start_age,breed='Improved Tanz southern highlands')
                            d.sex='m'
                            
                    # Set basic LivSim parameters
                    d._main_variables = None 
                    d._DETERMINISTIC= True
                    d.is_bull_present = 1
                    d._MAX_ITER = 20


                    #~~~~~~~~~~~~~~~~~~~~~~~~ FEEDING SCRIPT ~~~~~~~~~~~~~~~~~~~~~~~

                    # define feeds used in simulations from library
                    feeds=['grass','stover','napier','napier_hy','maize_bran','sunflower_seed_meal','pasture']
                    concentrates=['maize_bran','sunflower_seed_meal']

                    GRASS = Feed(parameters='Native grass') 
                    PASTURE = Feed(parameters='Pasture')
                    NAPIER = Feed(parameters='napier grass') 
                    NAPIER_HY = Feed(parameters='napier-legume') 
                    NAPIER_SILAGE = Feed(parameters='napier silage') 
                    SUNFLOWER_SEED_MEAL = Feed(parameters='sunflower seed meal')
                    MAIZE_BRAN = Feed(parameters='maize bran')


                    ## specify the diets for cows in line with the scenario
                    feed_cow_base=diet_wb.sheet_by_index(0)

                    if (b=='local'):
                        feed_cow_scen=diet_wb.sheet_by_index(int(scenario_parameters[(ss,'diet_id_cow_local',s)]))

                    elif (b=='improved'):
                        feed_cow_scen=diet_wb.sheet_by_index(int(scenario_parameters[(ss,'diet_id_cow_improved',s)]))

                    # load stover nutrient properties 
                    if ((c == 'cow') and  (scenario_parameters[(ss,'urea_treatment',s)]==1)):
                        STOVER=Feed(parameters='maize stover urea treated')
                        m_stover="maize stover urea treated"

                    else:
                        STOVER=Feed(parameters='maize stover')
                        m_stover="maize stover"


                    # define feed storage object
                    feed_storage = FeedStorage()


                    # cow specific feed intake variables
                    # feed intake per day (kg dm/head/day) at specific stage of cow's production cycle (lactating, gestating, other)
                    feed_intake_cow_early_lac={}
                    feed_intake_cow_late_lac={}
                    feed_intake_cow_gest={}
                    feed_intake_cow_other={}
                    feed_intake_cow_early_lac_base={}
                    feed_intake_cow_late_lac_base={}
                    feed_intake_cow_gest_base={}
                    feed_intake_cow_other_base={}

                    feed_offer_cow={}
                    feed_offer_cow_base={}

                    fd_offer_cycle={}
                    fd_offer_cycle_base={}

                    # feed intake per year from specific production stages of cow (kg DM/head/yr)
                    cons_per_year_early_lac_base={}
                    cons_per_year_late_lac_base={}
                    cons_per_year_lac_base={}
                    cons_per_year_oth_base={}
                    cons_per_year_early_lac={}
                    cons_per_year_late_lac={}
                    cons_per_year_lac={}
                    cons_per_year_oth={}

                    # all feed intake variables
                    feed_intake={} # feed intake on an annual basis per breed/cohort (kg dm/yr/head)
                    feed_intake_base={} # reference feed intake on an annual basis per breed/cohort (kg dm/yr/head)

                    cycle = ['e_lac','l_lac','other']

                    # actual feed intake for this iteration
                    # feed for cows
                    if (c =='cow'):

                        lactation_duration = 30*d.parameters['lactation_duration']['value']
                        early_lactation_duration = 30*5
                        late_lactation_duration = lactation_duration-early_lactation_duration
                        calving_interval = 365/d.parameters['potential_annual_calving_rate']['value']
                        FRAC_time_lactation = lactation_duration/calving_interval
                        FRAC_time_early_lactation = early_lactation_duration/calving_interval
                        FRAC_time_late_lactation = late_lactation_duration/calving_interval
                        days_not_in_lac = ( 1 - FRAC_time_lactation)*365

                        # load diets from excel
                        if (b=='local'):
                            
                            breed_set = 0
                            
                        elif (b=='improved'):
                            
                            breed_set = 1

                        for f in feeds:
                                row = (subsectors.index(ss)*3)+breed_set*(9)

                                feed_intake_cow_early_lac_base[ss,b,c,f]=feed_cow_base.cell(row+1,3+feeds.index(f)).value/1000

                                feed_intake_cow_late_lac_base[ss,b,c,f]=feed_cow_base.cell(row+2,3+feeds.index(f)).value/1000
                                feed_intake_cow_gest_base[ss,b,c,f]=feed_cow_base.cell(row+2,3+feeds.index(f)).value/1000
                                feed_intake_cow_other_base[ss,b,c,f]=feed_cow_base.cell(row+3,3+feeds.index(f)).value/1000

                                # new
                                feed_offer_cow_base[ss,b,c,'e_lac',f]=feed_cow_base.cell(row+1,3+feeds.index(f)).value/1000
                                feed_offer_cow_base[ss,b,c,'l_lac',f]=feed_cow_base.cell(row+2,3+feeds.index(f)).value/1000
                                feed_offer_cow_base[ss,b,c,'other',f]=feed_cow_base.cell(row+3,3+feeds.index(f)).value/1000

                        cons_per_year_early_lac_base[ss,b,c,'sunflower_seed_meal']=(
                            feed_intake_cow_early_lac_base[ss,b,c,'sunflower_seed_meal']* 
                            early_lactation_duration*FRAC_time_early_lactation)
                        
                        cons_per_year_late_lac_base[ss,b,c,'sunflower_seed_meal']=(
                            feed_intake_cow_late_lac_base[ss,b,c,'sunflower_seed_meal']*
                            late_lactation_duration*FRAC_time_late_lactation)
                        
                        cons_per_year_oth_base[ss,b,c,'sunflower_seed_meal']= (
                            feed_intake_cow_other_base[ss,b,c,'sunflower_seed_meal']*
                            days_not_in_lac)

                        cons_per_year_early_lac_base[ss,b,c,'maize_bran'] = (
                            feed_intake_cow_early_lac_base[ss,b,c,'maize_bran']*
                            early_lactation_duration*FRAC_time_early_lactation)
                        
                        cons_per_year_late_lac_base[ss,b,c,'maize_bran'] = ( 
                            feed_intake_cow_late_lac_base[ss,b,c,'maize_bran']*
                            late_lactation_duration*FRAC_time_late_lactation)
                        
                        cons_per_year_oth_base[ss,b,c,'maize_bran'] = (
                            feed_intake_cow_other_base[ss,b,c,'maize_bran']*
                            days_not_in_lac)

                        cons_per_year_early_lac_base[ss,b,c,'napier'] = (
                            feed_intake_cow_early_lac_base[ss,b,c,'maize_bran']* 
                            early_lactation_duration*FRAC_time_early_lactation)
                        
                        cons_per_year_late_lac_base[ss,b,c,'napier'] = (
                            feed_intake_cow_late_lac_base[ss,b,c,'maize_bran']* 
                            late_lactation_duration*FRAC_time_late_lactation)
                        
                        cons_per_year_oth_base[ss,b,c,'napier'] = (
                            feed_intake_cow_other_base[ss,b,c,'maize_bran']*
                            days_not_in_lac)

                        cons_per_year_early_lac_base[ss,b,c,'napier_hy']=(
                            feed_intake_cow_early_lac_base[ss,b,c,'napier_hy']*
                            early_lactation_duration*FRAC_time_early_lactation)
                        
                        cons_per_year_late_lac_base[ss,b,c,'napier_hy']=(
                            feed_intake_cow_late_lac_base[ss,b,c,'napier_hy']*
                            late_lactation_duration*FRAC_time_late_lactation)
                        
                        cons_per_year_oth_base[ss,b,c,'napier_hy']=(
                        feed_intake_cow_other_base[ss,b,c,'napier_hy']*days_not_in_lac)

                        cons_per_year_early_lac_base[ss,b,c,'grass']=(
                            feed_intake_cow_early_lac_base[ss,b,c,'maize_bran']*
                            early_lactation_duration*FRAC_time_early_lactation)
                        
                        cons_per_year_late_lac_base[ss,b,c,'grass'] = (
                            feed_intake_cow_late_lac_base[ss,b,c,'maize_bran']*
                            late_lactation_duration*FRAC_time_late_lactation)
                        
                        cons_per_year_oth_base[ss,b,c,'grass'] = feed_intake_cow_other_base[ss,b,c,'maize_bran']*days_not_in_lac

                        cons_per_year_early_lac_base[ss,b,c,'stover'] = (
                            feed_intake_cow_early_lac_base[ss,b,c,'maize_bran']*
                            early_lactation_duration*FRAC_time_early_lactation)
                        
                        cons_per_year_late_lac_base[ss,b,c,'stover'] = (
                            feed_intake_cow_late_lac_base[ss,b,c,'maize_bran']*
                            late_lactation_duration*FRAC_time_late_lactation)
                        
                        cons_per_year_oth_base[ss,b,c,'stover'] = feed_intake_cow_other_base[ss,b,c,'maize_bran']*days_not_in_lac

                        cons_per_year_early_lac_base[ss,b,c,'pasture'] = (
                            feed_intake_cow_early_lac_base[ss,b,c,'maize_bran']*
                            early_lactation_duration*FRAC_time_early_lactation)
                        cons_per_year_late_lac_base[ss,b,c,'pasture'] = ( 
                            feed_intake_cow_late_lac_base[ss,b,c,'maize_bran']*
                            late_lactation_duration*FRAC_time_late_lactation)
                        
                        cons_per_year_oth_base[ss,b,c,'pasture'] = (
                            feed_intake_cow_other_base[ss,b,c,'maize_bran']*
                            days_not_in_lac)

                        # new

                        fd_offer_cycle_base[ss,b,c,'e_lac','sunflower_seed_meal']=feed_offer_cow_base[ss,b,c,'e_lac','sunflower_seed_meal']*early_lactation_duration*FRAC_time_early_lactation
                        fd_offer_cycle_base[ss,b,c,'l_lac','sunflower_seed_meal']=feed_offer_cow_base[ss,b,c,'l_lac','sunflower_seed_meal']*late_lactation_duration*FRAC_time_late_lactation
                        fd_offer_cycle_base[ss,b,c,'other','sunflower_seed_meal']=feed_offer_cow_base[ss,b,c,'other','sunflower_seed_meal']*days_not_in_lac

                        fd_offer_cycle_base[ss,b,c,'e_lac','maize_bran']=feed_offer_cow_base[ss,b,c,'e_lac','maize_bran']*early_lactation_duration*FRAC_time_early_lactation
                        fd_offer_cycle_base[ss,b,c,'l_lac','maize_bran']=feed_offer_cow_base[ss,b,c,'l_lac','maize_bran']*late_lactation_duration*FRAC_time_late_lactation
                        fd_offer_cycle_base[ss,b,c,'other','maize_bran']=feed_offer_cow_base[ss,b,c,'other','maize_bran']*days_not_in_lac

                        fd_offer_cycle_base[ss,b,c,'e_lac','napier']= (
                            feed_offer_cow_base[ss,b,c,'e_lac','napier']*
                            early_lactation_duration*FRAC_time_early_lactation)
                        
                        fd_offer_cycle_base[ss,b,c,'l_lac','napier']=(
                            feed_offer_cow_base[ss,b,c,'l_lac','napier']*
                            late_lactation_duration*FRAC_time_late_lactation)
                        
                        fd_offer_cycle_base[ss,b,c,'other','napier']=feed_offer_cow_base[ss,b,c,'other','napier']*days_not_in_lac

                        fd_offer_cycle_base[ss,b,c,'e_lac','napier_hy'] = (feed_offer_cow_base[ss,b,c,'e_lac','napier_hy']*
                                                                           early_lactation_duration*FRAC_time_early_lactation)
                                                                           
                        fd_offer_cycle_base[ss,b,c,'l_lac','napier_hy'] = (feed_offer_cow_base[ss,b,c,'l_lac','napier_hy']*
                                                                           late_lactation_duration*FRAC_time_late_lactation)
                        fd_offer_cycle_base[ss,b,c,'other','napier_hy']=feed_offer_cow_base[ss,b,c,'other','napier_hy']*days_not_in_lac

                        fd_offer_cycle_base[ss,b,c,'e_lac','grass']=(
                         feed_offer_cow_base[ss,b,c,'e_lac','grass']*early_lactation_duration*FRAC_time_early_lactation)
                            
                        fd_offer_cycle_base[ss,b,c,'l_lac','grass']=( 
                            feed_offer_cow_base[ss,b,c,'l_lac','grass']*
                            late_lactation_duration*FRAC_time_late_lactation)
                        
                        fd_offer_cycle_base[ss,b,c,'other','grass']=(
                        feed_offer_cow_base[ss,b,c,'other','grass']*days_not_in_lac)

                        fd_offer_cycle_base[ss,b,c,'e_lac','stover'] = ( 
                            feed_offer_cow_base[ss,b,c,'e_lac','stover']*
                            early_lactation_duration*FRAC_time_early_lactation)
                        
                        fd_offer_cycle_base[ss,b,c,'l_lac','stover'] = (
                        feed_offer_cow_base[ss,b,c,'l_lac','stover']*
                            late_lactation_duration*FRAC_time_late_lactation)
                        
                        fd_offer_cycle_base[ss,b,c,'other','stover'] = ( 
                            feed_offer_cow_base[ss,b,c,'other','stover']*days_not_in_lac)

                        fd_offer_cycle_base[ss,b,c,'e_lac','pasture'] = (
                            feed_offer_cow_base[ss,b,c,'e_lac','pasture']*
                                                                         early_lactation_duration*FRAC_time_early_lactation)
                        
                        fd_offer_cycle_base[ss,b,c,'l_lac','pasture']=(
                            feed_offer_cow_base[ss,b,c,'l_lac','pasture']*
                                                                       late_lactation_duration*FRAC_time_late_lactation)
                        
                        fd_offer_cycle_base[ss,b,c,'other','pasture']=feed_offer_cow_base[ss,b,c,'other','pasture']*days_not_in_lac



                        # the annual feed intake of a feed category is equal to the sum over all production stages
                        for f in feeds:
                            for cc in cycle:
                                feed_intake_base[ss,b,c,f] = 0

                        for f in feeds:
                            for cc in cycle:
                                feed_intake_base[ss,b,c,f] += fd_offer_cycle_base[ss,b,c,cc,f]

                       


                        # if the current scenario is a feeding scenario, diet is different from base
                        if (scenario_parameters[(ss,'diet_scenario_boolean',s)]==1):   
                            
                            # load diets from excel
                            for f in feeds:
                                row = (subsectors.index(ss)*3)+breed_set*9

                                feed_intake_cow_early_lac[ss,b,c,f]=feed_cow_scen.cell(row+1,3+feeds.index(f)).value/1000
                                feed_intake_cow_late_lac[ss,b,c,f]=feed_cow_scen.cell(row+2,3+feeds.index(f)).value/1000
                                feed_intake_cow_gest[ss,b,c,f]=feed_cow_scen.cell(row+2,3+feeds.index(f)).value/1000
                                feed_intake_cow_other[ss,b,c,f]=feed_cow_scen.cell(row+3,3+feeds.index(f)).value/1000

                                feed_offer_cow[ss,b,c,'e_lac',f]=feed_cow_scen.cell(row+1,3+feeds.index(f)).value/1000
                                feed_offer_cow[ss,b,c,'l_lac',f]=feed_cow_scen.cell(row+2,3+feeds.index(f)).value/1000
                                feed_offer_cow[ss,b,c,'other',f]=feed_cow_scen.cell(row+3,3+feeds.index(f)).value/1000

                        else: ## if base scenario 

                            for f in feeds:
                                for cc in cycle:
                                    feed_offer_cow[ss,b,c,cc,f] = feed_offer_cow_base[ss,b,c,cc,f]

                           
                        ## set annual consumption based on fraction of time in lactation

                        for f in feeds:
                            
                            fd_offer_cycle[ss,b,c,'e_lac',f] = (feed_offer_cow[ss,b,c,'e_lac',f]*
                                                                early_lactation_duration*FRAC_time_early_lactation)
                            fd_offer_cycle[ss,b,c,'l_lac',f] = (feed_offer_cow[ss,b,c,'l_lac',f]*
                                                                late_lactation_duration*FRAC_time_late_lactation)
                            fd_offer_cycle[ss,b,c,'other',f] = feed_offer_cow[ss,b,c,'other',f]*days_not_in_lac


                        ## sum intake over production stages to get the intake on an annual basis
                        for f in feeds:
                            feed_intake[ss,b,c,f] = 0

                        for f in feeds:
                            for cc in cycle:
                                 feed_intake[ss,b,c,f] += fd_offer_cycle[ss,b,c,cc,f]

                            

                    else: ## if not a cow, no life-stage specific feeding, and therefore annual intake is daily intake * 365

                        

                        feed_intake[ss,b,c,'grass']=365*sheet.cell_value(bc_count+1, 2)/1000
                        feed_intake[ss,b,c,'stover']=365*sheet.cell_value(bc_count+1, 3)/1000
                        feed_intake[ss,b,c,'napier']=365*sheet.cell_value(bc_count+1, 4)/1000 
                        feed_intake[ss,b,c,'napier_hy']=365*sheet.cell_value(bc_count+1, 5)/1000 
                        feed_intake[ss,b,c,'maize_bran']=365*sheet.cell_value(bc_count+1, 6)/1000 
                        feed_intake[ss,b,c,'sunflower_seed_meal']=365*sheet.cell_value(bc_count+1, 7)/1000 
                        feed_intake[ss,b,c,'pasture']=365*sheet.cell_value(bc_count+1, 8)/1000




                    if (c != 'cow'):
                        
                        feed_intake_base[ss,b,c,'grass']=365*sheet.cell_value(bc_count+1, 2)/1000
                        feed_intake_base[ss,b,c,'stover']=365*sheet.cell_value(bc_count+1, 3)/1000
                        feed_intake_base[ss,b,c,'napier']=365*sheet.cell_value(bc_count+1, 4)/1000
                        feed_intake_base[ss,b,c,'napier_hy']=365*sheet.cell_value(bc_count+1, 5)/1000 
                        feed_intake_base[ss,b,c,'maize_bran']=365*sheet.cell_value(bc_count+1, 6)/1000 
                        feed_intake_base[ss,b,c,'sunflower_seed_meal']=365*sheet.cell_value(bc_count+1, 7)/1000 
                        feed_intake_base[ss,b,c,'pasture']=365*sheet.cell_value(bc_count+1, 8)/1000

                    '''   
                    
                    Define LAND FOOTPRINT
                    
                    This section takes the feed intake for the current animal as an annual value and calculates area of land 
                    needed to produce this feed. This is done for both the reference (base) scenario and the current iteration,
                    which is needed in order to calculate LUC emissions between scenarios 

                    It is important to understand that in this framework, the concentrate feeds are fed according to the 
                    production cycles of the cow (lactating, gestating, etc.) whereas the other feeds are specifed as annual intakes,
                    and then the intake per month depends on the seasonality parameters (e.g the variation in yields of feeds, and 
                    rationing parameters, etc.)

                    ''' 

                    # reference yield (Mg DM/ha/yr) (corrected for efficiency of harvest/grazing)
                    grass_annual_yield = (utn_efy[('grass')]) * yield_sheet.cell_value(1, 2)
                    pasture_annual_yield = (utn_efy[('pasture')]) * yield_sheet.cell_value(2, 2)
                    napier_annual_yield = (utn_efy[('napier')]) * yield_sheet.cell_value(3, 2)
                    napier_hy_annual_yield = (utn_efy[('napier_hy')]) * yield_sheet.cell_value(4, 2)
                    stover_annual_yield = (utn_efy[('stover')]) * yield_sheet.cell_value(5, 2)
                    maize_annual_yield = ((utn_efy[('maize_bran')]) * yield_sheet.cell_value(6, 2) *(scenario_parameters[(ss,'yield_scale_factor_maize','BAU')]**timestep))
                    sunflower_annual_yield = (utn_efy[('sunflower_seed_meal')] * yield_sheet.cell_value(7, 2)*(scenario_parameters[(ss,'yield_scale_factor_sunflower','BAU')]**timestep))



                    # reference area (ha)
                    grass_area_base[ss,b,c]=(1/1000)*feed_intake_base[ss,b,c,'grass']/grass_annual_yield
                    pasture_area_base[ss,b,c]=(1/1000)*feed_intake_base[ss,b,c,'pasture']/pasture_annual_yield
                    stover_area_base[ss,b,c]=(1/1000)*(feed_intake_base[ss,b,c,'stover'])/stover_annual_yield
                    sunflower_as_meal_area_base[ss,b,c]=(1/1000)*feed_intake_base[ss,b,c,'sunflower_seed_meal']/((scenario_parameters[(ss,'yield_scale_factor_sunflower','BAU')]**timestep)*
                                                                                                                 sunflower_annual_yield)
                    maize_as_bran_area_base[ss,b,c]=(1/1000)*feed_intake_base[ss,b,c,'maize_bran']/(scenario_parameters[(ss,'yield_scale_factor_maize','BAU')]*maize_annual_yield)
                    napier_area_base[ss,b,c]=(1/1000)*feed_intake_base[ss,b,c,'napier']/napier_annual_yield       
                    napier_hy_area_base[ss,b,c]=(1/1000)*feed_intake_base[ss,b,c,'napier_hy']/napier_annual_yield       

                    # summarize area for total feed, primary feeds, and primarcy crops
                    feed_area_base[(ss,b,c)]=pasture_area_base[ss,b,c]+grass_area_base[ss,b,c]+stover_area_base[ss,b,c]+sunflower_as_meal_area_base[ss,b,c]+maize_as_bran_area_base[ss,b,c]+napier_area_base[ss,b,c]+napier_hy_area_base[ss,b,c]
                    feed_primary_area_base[(ss,b,c)]= pasture_area_base[ss,b,c]+grass_area_base[ss,b,c]+sunflower_as_meal_area_base[ss,b,c]+maize_as_bran_area_base[ss,b,c]+napier_area_base[ss,b,c]+napier_hy_area_base[ss,b,c]
                    crop_primary_area_base[(ss,b,c)]=sunflower_as_meal_area_base[ss,b,c]+maize_as_bran_area_base[ss,b,c]

                    # actual area for this simulation (ha)
                    pasture_area[ss,b,c]=(1/1000)*feed_intake[ss,b,c,'pasture']/(pasture_annual_yield)
                    grass_area[ss,b,c]=(1/1000)*feed_intake[ss,b,c,'grass']/(grass_annual_yield)
                    stover_area[ss,b,c]=(1/1000)*(feed_intake[ss,b,c,'stover'])/(stover_annual_yield)                

                    napier_area[ss,b,c]=(1/1000)*feed_intake[ss,b,c,'napier']/(napier_annual_yield)            
                    napier_hy_area[ss,b,c]=(1/1000)*feed_intake[ss,b,c,'napier_hy']/(napier_hy_annual_yield) 
                    sunflower_as_meal_area[ss,b,c]=(1/1000)*feed_intake[ss,b,c,'sunflower_seed_meal']/(sunflower_annual_yield*(scenario_parameters[(ss,'yield_scale_factor_sunflower','BAU')]**timestep))
                    maize_as_bran_area[ss,b,c]=(1/1000)*feed_intake[ss,b,c,'maize_bran']/(maize_annual_yield*(scenario_parameters[(ss,'yield_scale_factor_maize','BAU')]**timestep))


                    # summarize area for total feed, primary feeds, and primary crops
                    feed_area[(ss,b,c)]=grass_area[ss,b,c]+stover_area[ss,b,c]+sunflower_as_meal_area[ss,b,c]+maize_as_bran_area[ss,b,c]+napier_area[ss,b,c]+pasture_area[ss,b,c]
                    feed_primary_area[(ss,b,c)]=grass_area[ss,b,c]+sunflower_as_meal_area[ss,b,c]+maize_as_bran_area[ss,b,c]+napier_area[ss,b,c]+pasture_area[ss,b,c]

                    crop_primary_area[(ss,b,c)]=sunflower_as_meal_area[ss,b,c]+maize_as_bran_area[ss,b,c]
                    total_grasslands_area[(ss,b,c)]=grass_area[ss,b,c]+napier_area[ss,b,c]+pasture_area[ss,b,c]+napier_hy_area[ss,b,c]

                 
                    # Define feed available for feeding based on yield and area (defined as 'production') (kg DM/animal/d for every month)

                    pasture_production=[]## production available for feeding (kg DM/head/day)
                    napier_production=[] ## production available for feeding (kg DM/head/day)
                    napier_hy_production=[] ## production available for feeding (kg DM/head/day)
                    napier_silage_production=[] ## production available for feeding (kg DM/head/day)
                    grass_production=[]  ## production available for feeding (kg DM/head/day)
                    stover_production=[] ## production available for feeding (kg DM/head/day)
                    maize_bran_production=[] ## production available for feeding (kg DM/head/day)
                    sunflower_seed_meal_production=[] ## production available for feeding (kg DM/head/day)


                    for i in range(0,12):

                        napier_production.append(0)
                        napier_hy_production.append(0)
                        napier_silage_production.append(0)
                        grass_production.append(0)
                        stover_production.append(0)
                        pasture_production.append(0)


                    # for maize bran and sunflower cake there is no seasonal variation and therefore 
                    # monthly consumption is  annual production divided by 12
                    sunflower_yld=sunflower_annual_yield*1000/365
                    maize_bran_yld=maize_annual_yield*1000/365

                    maize_bran_production=maize_as_bran_area[ss,b,c]*maize_bran_yld
                    sunflower_seed_meal_production=sunflower_as_meal_area[ss,b,c]*sunflower_yld


                    # Feed seasonality
                    # define feed available in each month (for stover, pasture, and fodder crops) based on seasonality parameters


                    # specify months pertaining to different seasons 
                    # the first 6 months of the year are set as the dry season, the last 6 are the rainy season
                    am = [0,1,2,3,4,5,6,7,8,9,10,11]  # all months
                    ld = [0,1,2,3,4,5]  # long dry season
                    lr = [6,7,8,9,10,11] # long rainy season
                    sr = [0,1,2,3,4,5]  # months in which stover is rationed (after harvest)

                    # Load relative yield data from excel; these parameters are described in the SM of articles
                    wsgn = ryld_data.cell_value(1,1) 
                    dsdn = ryld_data.cell_value(2,1) 
                    wsg_p= ryld_data.cell_value(1,2) 
                    dsd_p= ryld_data.cell_value(2,2) 

                    # define the fraction of fodder from wet season that is ensiled for dry season feeding (only for cows)   
                    if (c == 'cow'):
                        ensilage_fodder_frac[(ss,b)] = scenario_parameters[(ss,'ensilage_fodder_frac',s)]

                    elif (c != 'cow'):
                        ensilage_fodder_frac[(ss,b)] = 0


                    for i in lr:
                        napier_production[i] = (wsgn*(1/12)*
                        (1-ensilage_fodder_frac[(ss,b)])*napier_area[ss,b,c]*1000*napier_annual_yield/(30)
                                               )

                        grass_production[i]=wsg_p*(1/30)*grass_area[ss,b,c]*1000*grass_annual_yield/12
                        pasture_production[i]=wsg_p*(1/30)*pasture_area[ss,b,c]*1000*pasture_annual_yield/12
                        napier_silage_production[i]=0
                        stover_production[i]=(1/6)*stover_area[ss,b,c]*1000*stover_annual_yield/30

                    for i in ld:
                        napier_production[i] = dsdn*(1/12)*napier_area[ss,b,c]*1000*napier_annual_yield/(30)
                        napier_hy_production[i] = dsdn*(1/12)*napier_hy_area[ss,b,c]*1000*napier_hy_annual_yield/(30)
                        grass_production[i] = dsd_p*(1/30)*grass_area[ss,b,c]*1000*grass_annual_yield/12
                        pasture_production[i] = (dsd_p*(1/30)*pasture_area[ss,b,c]*1000*pasture_annual_yield/12)
                        napier_silage_production[i] = (
                            (1/6)*(ensilage_fodder_frac[(ss,b)])*
                        (wsgn*(1/2)*napier_area[ss,b,c]*1000*napier_annual_yield/(30)))

                    # stover is rationed equally over 6 months of the year (immediately following crop harvest)    
                    for i in sr:
                        stover_production[i]=(1/6)*stover_area[ss,b,c]*1000*stover_annual_yield/30

                    ## Specify actual diets for the current animal
                    if (c =='cow'):

                        # specify feed intakes for cows
                        # for concentrates feeds are specified based on production cycle
                        # for non-concentrates, feeds are specified by month based on seasonality of these feeds as defined above

                        feed_storage.add_external_roughage(grass_production,'Native grass', 'default')
                        feed_storage.add_external_roughage(pasture_production ,'Pasture', 'default')
                        feed_storage.add_external_roughage(stover_production, m_stover, 'default')
                        feed_storage.add_external_roughage(napier_production, 'napier grass', 'default')
                        feed_storage.add_external_roughage(napier_silage_production, 'napier silage', 'default')
                        feed_storage.add_external_roughage(napier_hy_production, 'napier-legume', 'default')



                        # for concentrates:
                        feed_storage.add_external_concentrate(feed_offer_cow[ss,b,c,'e_lac','sunflower_seed_meal'],'sunflower seed meal', 'early lactation')
                        feed_storage.add_external_concentrate(feed_offer_cow[ss,b,c,'e_lac','maize_bran'],'maize bran', 'early lactation')

                        feed_storage.add_external_concentrate(feed_offer_cow[ss,b,c,'l_lac','sunflower_seed_meal'],'sunflower seed meal', 'lactating')
                        feed_storage.add_external_concentrate(feed_offer_cow[ss,b,c,'l_lac','maize_bran'],'maize bran', 'lactating')

                        feed_storage.add_external_concentrate(feed_offer_cow[ss,b,c,'other','sunflower_seed_meal'],'sunflower seed meal', 'default')
                        feed_storage.add_external_concentrate(feed_offer_cow[ss,b,c,'other','maize_bran'],'maize bran', 'default')

                    else: 
                        feed_storage.add_external_roughage(pasture_production,'Pasture', 'default')
                        feed_storage.add_external_concentrate(maize_bran_production,'maize bran', 'default')
                        feed_storage.add_external_concentrate(sunflower_seed_meal_production,'sunflower seed meal', 'default')
                        feed_storage.add_external_roughage(napier_production, 'napier grass', 'default')
                        feed_storage.add_external_roughage(napier_hy_production, 'napier-legume', 'default')
                        feed_storage.add_external_roughage(napier_silage_production, 'napier silage', 'default')
                        feed_storage.add_external_roughage(grass_production, 'Native grass', 'default')
                        feed_storage.add_external_roughage(stover_production, 'maize stover', 'default')


                    d.feed_supply = feed_storage

                    #  Rin LivSim simulation for a given breed b, cohort c

                    print('Breed is ',b)
                    print('Animal is ',c)

                    output_df = d.run(max_simu_range)

                    final_month = start_age + len(output_df)

                    #######

                    '''
                    Results analysis and summation/scaling
                    
                    Calculate the needed output parameters as averages from the entire simulation period.
                    
                    This is performed by summing all output variables over the simulation period, then dividing by time span
                    of simulation period to get an annual average.
                    
                    '''

                    start_lact = 0 # has lactation started for cow , boolean
                    start_count = 0 # has the counting period for simulation started, boolean
                    months_lactating = 0 # number of months cow has been lactating
                    simu_period = 0 # months of simulation period
                    cmc = 0 # calendar month count. used for keeping track of the season/month when calculating outputs from livsim

                    # initialize calendar month variables as zero
                    for i in range(0,12):
                        count_calendar_month[i,ss,b,c] = 0


                    # tallying variables
                    total_milk_yield = 0 ## kg/cow
                    faecal_Nitrogen_tally=0  # g N
                    urinary_Nitrogen_tally=0 # g N
                    dry_matter_intake_tally=0 # kg DM
                    body_weight_kg_tally=0 # body weight in kg
                    diet_Nitrogen_tally=0 # kg
                    frac_conc_tally=0 # fraction
                    frac_roughage_tally=0  # fraction
                    neutral_detergent_fibre_tally=0 # kg 
                    acid_detergent_fibre_tally=0 # kg
                    metabolisable_energy_tally=0 # MJ
                    feed_intake_grass_tally=0 #kg
                    feed_intake_stover_tally=0 #kg
                    feed_intake_napier_tally=0 #kg
                    feed_intake_napier_hy_tally=0 #kg
                    feed_intake_pasture_tally=0 #kg
                    feed_intake_maize_bran_tally=0  #kg
                    feed_intake_sfsm_tally=0  #kg
                    crude_protein_tally=0  #kg

                    # summation variables
                    lifetime_milk = 0  ## kg/cow
                    average_body_weight_kg = 0 # kg

                    total_diet_Nitrogen = 0 # kg


                    for m in range(len(output_df)):

                        # if cow start counting after lactation begins
                        if ((((d.sex == 'f') and output_df.get_value(m,'Cow_milk_yield') > 0 ) and (start_lact == 0)) or ((d.sex == 'f') and (start_age == m))):

                            if (c == 'cow'):
                                if ((output_df.get_value(m,'Cow_age')-start_age)<12):
                                    cmc=(output_df.get_value(m,'Cow_age') - start_age)
                                    cmc=0
                                    start_lact = 1
                                elif ((output_df.get_value(m,'Cow_age')-start_age)>12):
                                    cmc=int(12*(((output_df.get_value(m,'Cow_age') - start_age)/12)-1))
                                    cmc=0
                                    start_lact = 1
                            else:
                                cmc=0
                            start_count = m

                        # if male start counting immediately
                        elif ((d.sex == 'm') and (start_count == 0)):
                            start_count = m


                        if (start_count > 0 ):
                            simu_period += 1

                            if (output_df.get_value(m,'Cow_dead') == 'True'):
                                print('Animal ', b, ' ', c,' is dead')
                                #break


                            # sum results variables over simulation period
                            if (c == 'cow'):

                                lifetime_milk += output_df.get_value(m,'Cow_milk_yield')

                                if (output_df.get_value(m,'Cow_is_lactating')):

                                    months_lactating += 1

                                    if (output_df.get_value(m,'Cow_months_after_calving')<=5):
                                        feed_intake_maize_bran =  feed_offer_cow[ss,b,c,'e_lac','maize_bran']*MAIZE_BRAN.__getattribute__('DM')*(1/1000)*30                


                                        feed_intake_sfsm=feed_offer_cow[ss,b,c,'e_lac','sunflower_seed_meal']*SUNFLOWER_SEED_MEAL.__getattribute__('DM')*(1/1000)*30         

                                    elif (output_df.get_value(m,'Cow_months_after_calving')>5):
                                        feed_intake_maize_bran=feed_offer_cow[ss,b,c,'l_lac','maize_bran']*MAIZE_BRAN.__getattribute__('DM')*(1/1000)*30                
                                        feed_intake_sfsm=feed_offer_cow[ss,b,c,'l_lac','sunflower_seed_meal']*SUNFLOWER_SEED_MEAL.__getattribute__('DM')*(1/1000)*30         


                                    feed_intake_grass=grass_production[cmc]*GRASS.__getattribute__('DM')*(1/1000)*30                  
                                    feed_intake_pasture=pasture_production[cmc]*PASTURE.__getattribute__('DM')*(1/1000)*30                  
                                    feed_intake_stover=stover_production[cmc]*STOVER.__getattribute__('DM')*(1/1000)*30                   
                                    feed_intake_napier=napier_production[cmc]*NAPIER.__getattribute__('DM')*(1/1000)*30                  
                                    feed_intake_napier_hy=napier_hy_production[cmc]*NAPIER_HY.__getattribute__('DM')*(1/1000)*30                  
                                    feed_intake_napier_silage=napier_silage_production[cmc]*NAPIER_SILAGE.__getattribute__('DM')*(1/1000)*30                  


                                else:

                                    feed_intake_sfsm=feed_offer_cow[ss,b,c,'other','sunflower_seed_meal']*SUNFLOWER_SEED_MEAL.__getattribute__('DM')*(1/1000)*30            
                                    feed_intake_maize_bran=feed_offer_cow[ss,b,c,'other','maize_bran']*MAIZE_BRAN.__getattribute__('DM')*(1/1000)*30           


                                    feed_intake_grass=grass_production[cmc]*GRASS.__getattribute__('DM')*(1/1000)*30                  
                                    feed_intake_pasture=pasture_production[cmc]*PASTURE.__getattribute__('DM')*(1/1000)*30                  
                                    feed_intake_stover=stover_production[cmc]*STOVER.__getattribute__('DM')*(1/1000)*30 
                                    feed_intake_napier=napier_production[cmc]*NAPIER.__getattribute__('DM')*(1/1000)*30 
                                    feed_intake_napier_hy=napier_hy_production[cmc]*NAPIER_HY.__getattribute__('DM')*(1/1000)*30                  
                                    feed_intake_napier_silage=napier_silage_production[cmc]*NAPIER_SILAGE.__getattribute__('DM')*(1/1000)*30                  


                            else:

                                feed_intake_maize_bran=maize_bran_production*MAIZE_BRAN.__getattribute__('DM')*(1/1000)*30               
                                feed_intake_sfsm=sunflower_seed_meal_production*SUNFLOWER_SEED_MEAL.__getattribute__('DM')*(1/1000)*30             
                                feed_intake_grass=grass_production[cmc]*GRASS.__getattribute__('DM')*(1/1000)*30                
                                feed_intake_stover=stover_production[cmc]*STOVER.__getattribute__('DM')*(1/1000)*30                  
                                feed_intake_napier=napier_production[cmc]*NAPIER.__getattribute__('DM')*(1/1000)*30 
                                feed_intake_napier_hy=napier_hy_production[cmc]*NAPIER_HY.__getattribute__('DM')*(1/1000)*30
                                feed_intake_pasture=pasture_production[cmc]*PASTURE.__getattribute__('DM')*(1/1000)*30                  


                            total_rough_supply=(feed_intake_pasture+
                                                feed_intake_napier+
                                                feed_intake_napier_hy+
                                                feed_intake_grass+
                                                feed_intake_stover+
                                                feed_intake_napier_silage)

                            total_conc_supply=feed_intake_maize_bran+feed_intake_sfsm


                            if (total_conc_supply > 0):
                                fraction_concentrate_supply_eaten=output_df.get_value(m,'FI_concentrate_intake')/total_conc_supply
                            else:
                                fraction_concentrate_supply_eaten = 1

                            faecal_Nitrogen_tally +=output_df.get_value(m,'GrPr_faecal_N')
                            urinary_Nitrogen_tally += output_df.get_value(m,'GrPr_urinary_N')

                            fraction_roughage_supply_eaten=output_df.get_value(m,'FI_roughage_intake')/total_rough_supply
                            frac_conc_tally += fraction_concentrate_supply_eaten
                            frac_roughage_tally += fraction_roughage_supply_eaten

                            dry_matter_intake_tally += 30*output_df.get_value(m,'FI_feed_intake') 
                            body_weight_kg_tally += output_df.get_value(m,'Cow_body_weight') 
                            neutral_detergent_fibre_tally +=30*fraction_concentrate_supply_eaten*(sunflower_seed_meal_production*SUNFLOWER_SEED_MEAL.parameters['NDF']['value']+maize_bran_production*MAIZE_BRAN.parameters['NDF']['value'])+ 30*fraction_roughage_supply_eaten*(napier_silage_production[cmc]*NAPIER_SILAGE.parameters['NDF']['value']  +       pasture_production[cmc]*PASTURE.parameters['NDF']['value'][cmc]+grass_production[cmc]*GRASS.parameters['NDF']['value'][cmc]+stover_production[cmc]*STOVER.parameters['NDF']['value'][cmc]+napier_production[cmc]*NAPIER.parameters['NDF']['value'][cmc]+napier_hy_production[cmc]*NAPIER_HY.parameters['NDF']['value'][cmc])
                            acid_detergent_fibre_tally += 30*fraction_concentrate_supply_eaten*(sunflower_seed_meal_production*SUNFLOWER_SEED_MEAL.parameters['ADF']['value']+maize_bran_production*MAIZE_BRAN.parameters['ADF']['value'])+30*fraction_roughage_supply_eaten*(napier_silage_production[cmc]*NAPIER_SILAGE.parameters['ADF']['value']       + pasture_production[cmc]*PASTURE.parameters['ADF']['value'][cmc]+grass_production[cmc]*GRASS.parameters['ADF']['value'][cmc]+stover_production[cmc]*STOVER.parameters['ADF']['value'][cmc]+napier_production[cmc]*NAPIER.parameters['ADF']['value'][cmc]+napier_hy_production[cmc]*NAPIER_HY.parameters['ADF']['value'][cmc]) 
                            crude_protein_tally += 30*fraction_concentrate_supply_eaten*(sunflower_seed_meal_production*SUNFLOWER_SEED_MEAL.parameters['CP']['value']+maize_bran_production*MAIZE_BRAN.parameters['CP']['value'])+30*fraction_roughage_supply_eaten*( napier_silage_production[cmc]*NAPIER_SILAGE.parameters['CP']['value'] +        pasture_production[cmc]*PASTURE.parameters['CP']['value'][cmc]+grass_production[cmc]*GRASS.parameters['CP']['value'][cmc]+stover_production[cmc]*STOVER.parameters['CP']['value'][cmc]+napier_production[cmc]*NAPIER.parameters['CP']['value'][cmc]+napier_hy_production[cmc]*NAPIER_HY.parameters['CP']['value'][cmc]) 
                            metabolisable_energy_tally += 30*(fraction_concentrate_supply_eaten*(sunflower_seed_meal_production*SUNFLOWER_SEED_MEAL.parameters['ME']['value']+maize_bran_production*MAIZE_BRAN.parameters['ME']['value'])+fraction_roughage_supply_eaten*( napier_silage_production[cmc]*NAPIER_SILAGE.parameters['ME']['value'][cmc]   +        pasture_production[cmc]*PASTURE.parameters['ME']['value'][cmc]+grass_production[cmc]*GRASS.parameters['ME']['value'][cmc]+stover_production[cmc]*STOVER.parameters['ME']['value'][cmc]+napier_production[cmc]*NAPIER.parameters['ME']['value'][cmc]+napier_hy_production[cmc]*NAPIER_HY.parameters['ME']['value'][cmc]))


                            # MP and ME shortages for cows
                            mo_mp_short[ss,b,c,cmc] += output_df.get_value(m,'Cow_MP_shortage')
                            mo_me_short[ss,b,c,cmc] += output_df.get_value(m,'Cow_ME_shortage')

                            mo_fi_mb[ss,b,c,cmc] += fraction_concentrate_supply_eaten*maize_bran_production
                            mo_fi_sc[ss,b,c,cmc] += fraction_concentrate_supply_eaten*sunflower_seed_meal_production
                            mo_fi_sto[ss,b,c,cmc] += fraction_roughage_supply_eaten*stover_production[cmc]

                            mo_fi_pt[ss,b,c,cmc] += fraction_roughage_supply_eaten*pasture_production[cmc]
                            mo_fi_np[ss,b,c,cmc] += fraction_roughage_supply_eaten*napier_production[cmc]
                            mo_fi_nhy[ss,b,c,cmc] += fraction_roughage_supply_eaten*stover_production[cmc]
                            mo_fi_ns[ss,b,c,cmc] += fraction_roughage_supply_eaten*napier_silage_production[cmc]
                            mo_fi_grs[ss,b,c,cmc] += fraction_roughage_supply_eaten*grass_production[cmc]


                            count_calendar_month[cmc,ss,b,c] += 1

                            feed_intake_grass_tally += fraction_roughage_supply_eaten*feed_intake_grass*GRASS.__getattribute__('DM')/1000              
                            feed_intake_stover_tally += fraction_roughage_supply_eaten*feed_intake_stover*STOVER.__getattribute__('DM')/1000                 
                            feed_intake_napier_tally += fraction_roughage_supply_eaten*(feed_intake_napier+feed_intake_napier_silage)*NAPIER.__getattribute__('DM')/1000                 
                            feed_intake_napier_hy_tally += fraction_roughage_supply_eaten*(feed_intake_napier_hy)*NAPIER.__getattribute__('DM')/1000                 
                            feed_intake_maize_bran_tally += fraction_concentrate_supply_eaten*feed_intake_maize_bran*MAIZE_BRAN.__getattribute__('DM')/1000         
                            feed_intake_sfsm_tally += fraction_concentrate_supply_eaten*feed_intake_sfsm*SUNFLOWER_SEED_MEAL.__getattribute__('DM')/1000                 
                            feed_intake_pasture_tally += fraction_roughage_supply_eaten*feed_intake_pasture*PASTURE.__getattribute__('DM')/1000


                            if (m == (len(output_df)-1)):

                                final_bodyweight_kg[ss,b,c]=output_df.get_value(m,'Cow_body_weight')
                                final_age_months[ss,b,c]=output_df.get_value(m,'Cow_age')

                        if (cmc < 11):
                            cmc += 1
                        else: 
                            cmc=0


                    # Average dietary and excretion parameters over the entire simulation period of the animal
                    faecal_Nitrogen[ss,b,c] = 12*faecal_Nitrogen_tally/simu_period
                    urinary_Nitrogen[ss,b,c] = 12*urinary_Nitrogen_tally/simu_period
                    NDF[ss,b,c] = 12*neutral_detergent_fibre_tally/simu_period
                    ADF[ss,b,c] = 12*acid_detergent_fibre_tally/simu_period
                    ME[ss,b,c] = 12*metabolisable_energy_tally/simu_period
                    DMI[ss,b,c] = 12*dry_matter_intake_tally/simu_period
                    Body_weight[ss,b,c] = body_weight_kg_tally/simu_period
                    gross_energy[ss,b,c]=18.4*DMI[ss,b,c]
                    crude_protein_intake[ss,b,c]=(1/1000)*12*crude_protein_tally/simu_period

                    ## calculated dietary variables
                    DMD[ss,b,c] = 83.58 - 0.824*100*(1/1000)*ADF[ss,b,c] /DMI[ss,b,c]  + 2.626*100*(1/1000)*(crude_protein_intake[ss,b,c])/DMI[ss,b,c]
                    VS[ss,b,c]=((18.4*DMI[ss,b,c] *DMD[ss,b,c]/100)+0.04*gross_energy[ss,b,c])*(0.92/18.45)
                    Ym[ss,b,c]= 3.1 - 0.243*DMI[ss,b,c]/365 + 0.0059*NDF[ss,b,c]/DMI[ss,b,c]  + 0.0057*10*DMD[ss,b,c]/100

                    # average feed intake
                    pasture_intake[ss,b,c]=12*feed_intake_pasture_tally/simu_period
                    grass_intake[ss,b,c]=12*feed_intake_grass_tally/simu_period
                    stover_intake[ss,b,c]=12*feed_intake_stover_tally/simu_period
                    napier_intake[ss,b,c]=12*feed_intake_napier_tally/simu_period
                    napier_hy_intake[ss,b,c]=12*feed_intake_napier_hy_tally/simu_period
                    maize_bran_intake[ss,b,c]=12*feed_intake_maize_bran_tally/simu_period
                    sfsm_intake[ss,b,c]=12*feed_intake_sfsm_tally/simu_period

                    if (c == 'cow'):

                        ## convert milk production to FPCM
                        milk_yield_reg[ss,b]=12*lifetime_milk*(0.337+.116*(1/10)*d.parameters['milk_butterfat_content']['value']+0.06*(1/10)*d.parameters['milk_crude_protein_content']['value'])/simu_period
                        milk_yield_lact[ss,b]=d.parameters['lactation_duration']['value']*lifetime_milk* (0.337+.116*(1/10)*d.parameters['milk_butterfat_content']['value']+0.06*(1/10)*d.parameters['milk_crude_protein_content']['value'])/months_lactating

                        out['v3_0_simu_period_months_'+str(ss)+'_'+str(b)]=simu_period
                        out['v3_0_lactation_months_'+str(ss)+'_'+str(b)]=months_lactating

                    fraction_roughage_eaten[ss,b,c]=frac_roughage_tally/simu_period
                    fraction_conc_eaten[ss,b,c]=frac_conc_tally/simu_period


                    # Agricultural soils N2O based on IPCC 2006 methodology
                    N_MMT_Frac = {} # manure N (kg/yr) applied per land category

                    # for EFs specific to a breed, specify current iteration's EF
                    if (b=='improved'):
                        MCF=MCF_IMP
                        FRAC_MANURE_PRP=FRAC_MANURE_PRP_IMP
                        FRAC_MANURE_MMT=FRAC_MANURE_MMT_IMP
                        EF_4=EF_4_IMP
                        EF_5=EF_5_IMP
                        FRAC_vol=FRAC_vol_IMP
                        FRAC_leach=FRAC_leach_IMP

                    elif (b=='local'):
                        MCF=MCF_LOC
                        FRAC_MANURE_PRP=FRAC_MANURE_PRP_LOC
                        FRAC_MANURE_MMT=FRAC_MANURE_MMT_LOC
                        EF_4=EF_4_LOC
                        EF_5=EF_5_LOC
                        FRAC_vol=FRAC_vol_LOC
                        FRAC_leach=FRAC_leach_LOC


                    N_manure_to_Pasture=(1/1000)*(faecal_Nitrogen[ss,b,c]+urinary_Nitrogen[ss,b,c])*FRAC_MANURE_PRP
                    N_manure_to_Storage=(1/1000)*(faecal_Nitrogen[ss,b,c]*(1-EF_3_FAECAL)+urinary_Nitrogen[ss,b,c]*(1-EF_3_URINARY))*FRAC_MANURE_MMT

                    N_manure_for_Pasture=N_manure_to_Pasture
                    N_manure_for_Soil=N_manure_to_Storage*(1-FRAC_loss_MMS)

                    # allocate total n from manure management thats applied to soils to different land categories
                    # this is a management parameter; how n from management is allocated to different land uses
                    N_MMT_Frac['grass']=N_manure_for_Soil*grass_area[ss,b,c]/feed_area[(ss,b,c)]
                    N_MMT_Frac['napier']=N_manure_for_Soil*napier_area[ss,b,c]/feed_area[(ss,b,c)]
                    N_MMT_Frac['napier_hy']=N_manure_for_Soil*napier_hy_area[ss,b,c]/feed_area[(ss,b,c)]
                    N_MMT_Frac['maize_bran']=N_manure_for_Soil*maize_as_bran_area[ss,b,c]/feed_area[(ss,b,c)]
                    N_MMT_Frac['sunflower_seed_meal']=N_manure_for_Soil*sunflower_as_meal_area[ss,b,c]/feed_area[(ss,b,c)]
                    N_MMT_Frac['stover']=N_manure_for_Soil*stover_area[ss,b,c]/feed_area[(ss,b,c)]

                    # fertilizer, manure, and cr residue inputs into land categories (kg n/ha/yr)
                    Fsn={}
                    Fon={}
                    Fcr={}
                    Fprp={}

                    '''
                    This section accounts for all N inputs into different land categories
                    this includes synthetic fertilizer (sn)
                    organic fertilizer (fon) (manure)
                    crop residues (fcr)
                    manure deposited from grazing cattle
                    '''


                    # synthetic fertilizer n (kg n/ha/yr)
                    Fsn['grass']=0
                    Fsn['pasture']=0
                    Fsn['napier']=0
                    Fsn['napier_hy']=0
                    Fsn['maize_bran']=20+scenario_parameters[(ss,'nitrous_oxide_factor_maize',s)]
                    Fsn['sunflower_seed_meal']=20+scenario_parameters[(ss,'nitrous_oxide_factor_sunflower',s)]
                    Fsn['stover']=10

                    # organic fertilizer n (kg n/ha/yr)
                    if ( grass_area[ss,b,c] > 0):
                        Fon['grass']=N_MMT_Frac['grass']/grass_area[ss,b,c]

                    elif (grass_area[ss,b,c]==0):
                        Fon['grass']=0

                    if ( stover_area[ss,b,c] > 0):
                        Fon['stover']=(N_MMT_Frac['stover']+N_MMT_Frac['sunflower_seed_meal'])/stover_area[ss,b,c]


                    elif (stover_area[ss,b,c]==0):
                        Fon['stover'] = 0



                    Fon['pasture']=0
                    
                    
                    Fon['napier']=(N_MMT_Frac['napier']+N_MMT_Frac['maize_bran'])/napier_area[ss,b,c]
                    Fon['napier_hy']=(N_MMT_Frac['napier_hy'])/napier_hy_area[ss,b,c] 
                    Fon['maize_bran']=0 
                    Fon['sunflower_seed_meal']=0

                    # residue n (kg n/ha/yr) (these are calculated outside of model based on IPCC 2006 equations)
                    Fcr['grass']=2.2
                    Fcr['pasture']=4.5
                    Fcr['napier']=16.9
                    Fcr['napier_hy']=16.9
                    Fcr['maize_bran']=15.2
                    Fcr['sunflower_seed_meal']=10.6
                    Fcr['stover']=15.2

                    Synth_N_tot_kg_yr=stover_area[ss,b,c]*Fsn['stover']+sunflower_as_meal_area[ss,b,c]*Fsn['sunflower_seed_meal']+Fsn['maize_bran']*maize_as_bran_area[ss,b,c]


                    if (grass_area[ss,b,c]>0):

                        Fprp['grass']=N_manure_for_Pasture/grass_area[ss,b,c]

                    elif (grass_area[ss,b,c]==0):

                        Fprp['grass']=0



                    # direct n2o from all sources per land category (kg n2o-n/yr)
                    soil_N2O_feed_direct['pasture']=pasture_area[ss,b,c]*(EF_1*(Fsn['pasture']+Fcr['pasture']+Fon['pasture']))
                    soil_N2O_feed_direct['grass'] = grass_area[ss,b,c]*(EF_1*(Fsn['grass']+Fcr['grass']+Fon['grass']) + EF_3_PRP*Fprp['grass'])
                    soil_N2O_feed_direct['napier']=napier_area[ss,b,c]*EF_1*(Fsn['napier']+ Fon['napier']+Fcr['napier'])
                    soil_N2O_feed_direct['napier_hy']=napier_area[ss,b,c]*EF_1*(Fsn['napier_hy']+ Fon['napier_hy']+Fcr['napier_hy'])
                    soil_N2O_feed_direct['stover']=stover_area[ss,b,c]*allocation_factor_stover*EF_1*(Fsn['stover']+ Fon['stover']+Fcr['stover'])
                    soil_N2O_feed_direct['maize_bran']=maize_as_bran_area[ss,b,c]*allocation_factor_maize_bran*EF_1*(Fsn['maize_bran']+Fcr['maize_bran'])
                    soil_N2O_feed_direct['sunflower_seed_meal']=sunflower_as_meal_area[ss,b,c]*allocation_factor_sunflower_seed_meal*EF_1*(Fsn['sunflower_seed_meal']+Fcr['sunflower_seed_meal'])

                    # indirect (volatilization) (kg n2o-n/yr)
                    soil_N2O_feed_adn['pasture']=pasture_area[ss,b,c]*EF_4_adn*(Fsn['pasture']*FRAC_gasf+(Fon['pasture'])*FRAC_gasm)#
                    soil_N2O_feed_adn['grass']=grass_area[ss,b,c]*EF_4_adn*(Fsn['grass']*FRAC_gasf+(Fon['grass']+Fprp['grass'])*FRAC_gasm)
                    soil_N2O_feed_adn['napier']=napier_area[ss,b,c]*EF_4_adn*(Fsn['napier']*FRAC_gasf+Fon['napier']*FRAC_gasm)
                    soil_N2O_feed_adn['napier_hy']=napier_area[ss,b,c]*EF_4_adn*(Fsn['napier_hy']*FRAC_gasf+Fon['napier_hy']*FRAC_gasm)
                    soil_N2O_feed_adn['stover']=stover_area[ss,b,c]*allocation_factor_stover*EF_4_adn*(Fsn['stover']*FRAC_gasf+Fon['stover']*FRAC_gasm)
                    soil_N2O_feed_adn['maize_bran']=maize_as_bran_area[ss,b,c]*allocation_factor_maize_bran*EF_4_adn*(Fsn['maize_bran']*FRAC_gasf+Fon['maize_bran']*FRAC_gasm)
                    soil_N2O_feed_adn['sunflower_seed_meal']=sunflower_as_meal_area[ss,b,c]*allocation_factor_sunflower_seed_meal*EF_4_adn*(Fsn['sunflower_seed_meal']*FRAC_gasf+Fon['sunflower_seed_meal']*FRAC_gasm)

                    # indirect (leaching) (kg n2o-n/yr)
                    soil_N2O_feed_lch['pasture']=pasture_area[ss,b,c]*EF_5_lch*(Fsn['pasture']+Fcr['pasture']+Fon['pasture'])
                    soil_N2O_feed_lch['grass']=grass_area[ss,b,c]*EF_5_lch*(Fsn['grass']+Fcr['grass']+Fon['grass']+Fprp['grass'])
                    soil_N2O_feed_lch['napier']=napier_area[ss,b,c]*EF_5_lch*(Fsn['napier']+ Fon['napier']+Fcr['napier'])
                    soil_N2O_feed_lch['napier_hy']=napier_area[ss,b,c]*EF_5_lch*(Fsn['napier_hy']+ Fon['napier_hy']+Fcr['napier_hy'])
                    soil_N2O_feed_lch['stover']=stover_area[ss,b,c]*allocation_factor_stover*EF_5_lch*(Fsn['stover']+ Fon['stover']+Fcr['stover'])
                    soil_N2O_feed_lch['maize_bran']=maize_as_bran_area[ss,b,c]*allocation_factor_maize_bran*EF_5_lch*(Fsn['maize_bran']+Fcr['maize_bran'])
                    soil_N2O_feed_lch['sunflower_seed_meal']=sunflower_as_meal_area[ss,b,c]*allocation_factor_sunflower_seed_meal*EF_5_lch*(Fsn['sunflower_seed_meal']+Fcr['sunflower_seed_meal'])

                    # total 
                    soil_N2O_feed['pasture']=soil_N2O_feed_direct['pasture']+soil_N2O_feed_adn['pasture']+soil_N2O_feed_lch['pasture']
                    soil_N2O_feed['grass']=soil_N2O_feed_direct['grass']+soil_N2O_feed_adn['grass']+soil_N2O_feed_lch['grass']
                    soil_N2O_feed['napier']=soil_N2O_feed_direct['napier']+soil_N2O_feed_adn['napier']+soil_N2O_feed_lch['napier']
                    soil_N2O_feed['napier_hy']=soil_N2O_feed_direct['napier_hy']+soil_N2O_feed_adn['napier_hy']+soil_N2O_feed_lch['napier_hy']
                    soil_N2O_feed['stover']=soil_N2O_feed_direct['stover']+soil_N2O_feed_adn['stover']+soil_N2O_feed_lch['stover']
                    soil_N2O_feed['maize_bran']=soil_N2O_feed_direct['maize_bran']+soil_N2O_feed_adn['maize_bran']+soil_N2O_feed_lch['maize_bran']
                    soil_N2O_feed['sunflower_seed_meal']=soil_N2O_feed_direct['sunflower_seed_meal']+soil_N2O_feed_adn['sunflower_seed_meal']+soil_N2O_feed_lch['sunflower_seed_meal']


                    # calculate GHG emissions from all sources
                    enteric_CH4[ss,b,c]=28*gross_energy[ss,b,c]*Ym[ss,b,c]/(100*55.65)

                    manure_CH4[ss,b,c]=(28*VS[ss,b,c]*B0*MCF*0.67)

                    manure_N2O_direct[ss,b,c]=265*(44/28)*FRAC_MANURE_MMT*((1/1000)*faecal_Nitrogen[ss,b,c]*EF_3_FAECAL+(1/1000)*urinary_Nitrogen[ss,b,c]*EF_3_URINARY)
                    manure_N2O_vol[ss,b,c]=265*(44/28)*((1/1000)*FRAC_MANURE_MMT*(faecal_Nitrogen[ss,b,c]+urinary_Nitrogen[ss,b,c]))*FRAC_vol*EF_4
                    manure_N2O_leach[ss,b,c]=265*(44/28)*((1/1000)*FRAC_MANURE_MMT*(faecal_Nitrogen[ss,b,c]+urinary_Nitrogen[ss,b,c]))*FRAC_leach*EF_5
                    manure_N2O[ss,b,c]=manure_N2O_direct[ss,b,c]+manure_N2O_vol[ss,b,c]+manure_N2O_leach[ss,b,c]

                    soil_N2O_direct_grassland[ss,b,c]=265*(44/28)*(soil_N2O_feed_direct['grass']+soil_N2O_feed_direct['napier_hy']+soil_N2O_feed_direct['napier']+soil_N2O_feed_direct['pasture'])
                    soil_N2O_indirect_leach_grassland[ss,b,c]=265*(44/28)*(soil_N2O_feed_lch['grass']+soil_N2O_feed_lch['napier_hy']+soil_N2O_feed_lch['napier']+soil_N2O_feed_lch['pasture'])
                    soil_N2O_indirect_vol_grassland[ss,b,c]=265*(44/28)*(soil_N2O_feed_adn['grass']+soil_N2O_feed_adn['napier_hy']+soil_N2O_feed_adn['napier']+soil_N2O_feed_adn['pasture'])

                    soil_N2O_direct_cropland[ss,b,c]=265*(44/28)*(soil_N2O_feed_direct['maize_bran']+soil_N2O_feed_direct['sunflower_seed_meal']+soil_N2O_feed_direct['stover'])
                    soil_N2O_indirect_leach_cropland[ss,b,c]=265*(44/28)*(soil_N2O_feed_lch['maize_bran']+soil_N2O_feed_lch['sunflower_seed_meal']+soil_N2O_feed_lch['stover'])
                    soil_N2O_indirect_vol_cropland[ss,b,c]=265*(44/28)*(soil_N2O_feed_adn['maize_bran']+soil_N2O_feed_adn['sunflower_seed_meal']+soil_N2O_feed_adn['stover'])

                    soil_N2O_indirect_cropland[ss,b,c]=soil_N2O_indirect_vol_cropland[ss,b,c]+soil_N2O_indirect_leach_cropland[ss,b,c]
                    soil_N2O_cropland[ss,b,c]=soil_N2O_direct_cropland[ss,b,c]+soil_N2O_indirect_cropland[ss,b,c]

                    soil_N2O_indirect_grassland[ss,b,c]=soil_N2O_indirect_vol_grassland[ss,b,c]+soil_N2O_indirect_leach_grassland[ss,b,c]
                    soil_N2O_grassland[ss,b,c]=soil_N2O_direct_grassland[ss,b,c]+soil_N2O_indirect_grassland[ss,b,c]

                    energy_use_CO2[ss,b,c]=feed_embodied_CO2['maize_bran']*maize_bran_production*365/1000 + feed_embodied_CO2['sunflower_seed_meal']*sunflower_seed_meal_production*365/1000
                    energy_use_CO2[ss,b,c]= energy_use_CO2[ss,b,c]+Synth_N_tot_kg_yr*fertilizer_embodied_CO2

                    # Monte Carlo simulations #

                    uncertainty_sheet = data_wb.sheet_by_index(2)
                    rvars=0
                    random_var={}
                    num_vars=(uncertainty_sheet.nrows-1)

                    for MC in range(0,MC_sims):
                        for i in range(0,num_vars):
                            random_var[MC,i]=np.random.normal(100,uncertainty_sheet.cell_value(i+1, 1))/100

                        Ym_rd[ss,b,c]=Ym[ss,b,c]*random_var[MC,0]
                        B0_rd[ss,b,c]=B0*random_var[MC,1]
                        MCF_rd[ss,b,c]=MCF*random_var[MC,2]
                        EF_3_FAECAL_rd[ss,b,c]=EF_3_FAECAL*random_var[MC,3]
                        EF_3_URINARY_rd[ss,b,c]=EF_3_URINARY*random_var[MC,4]
                        EF_4_rd=EF_4*random_var[MC,5]
                        EF_5_rd=EF_5*random_var[MC,6]
                        FRAC_vol_rd= FRAC_vol*random_var[MC,7]
                        FRAC_leach_rd=FRAC_leach*random_var[MC,8]
                        EF_1_rd=EF_1*random_var[MC,9]
                        EF_3_PRP_rd=EF_3_PRP*random_var[MC,10]
                        EF_4_adn_rd=EF_4_adn*random_var[MC,11]
                        EF_5_lch_rd=EF_5_lch*random_var[MC,12]


                        # pasture
                        if ( s == 'Base' ):
                            grass_area_rd = grass_area[ss,b,c]*random_var[MC,13]
                            pasture_area_rd = pasture_area[ss,b,c]*random_var[MC,14]
                            napier_area_rd = napier_area[ss,b,c]*random_var[MC,15]
                            napier_hy_area_rd = napier_area[ss,b,c]*random_var[MC,15]
                            maize_as_bran_area_rd = maize_as_bran_area[ss,b,c]*random_var[MC,16]
                            sunflower_as_meal_area_rd = sunflower_as_meal_area[ss,b,c]*random_var[MC,17]
                            stover_area_rd = stover_area[ss,b,c]*random_var[MC,18]
                            herd_rd[(ss,b,'total_herd')] = herd[(ss,b,'total_herd')]*random_var[MC,19]
                        else:
                            grass_area_rd=grass_area[ss,b,c]
                            pasture_area_rd=pasture_area[ss,b,c]
                            napier_area_rd=napier_area[ss,b,c]
                            napier_hy_area_rd=napier_area[ss,b,c]
                            maize_as_bran_area_rd=maize_as_bran_area[ss,b,c]
                            sunflower_as_meal_area_rd=sunflower_as_meal_area[ss,b,c]
                            stover_area_rd=stover_area[ss,b,c]
                            herd_rd[(ss,b,'total_herd')]=herd[(ss,b,'total_herd')]

                        crop_primary_area_rd=crop_primary_area[ss,b,c]*random_var[MC,23]

                        feed_primary_area_rd=feed_primary_area[ss,b,c]*random_var[MC,24]



                    # direct n2o from all sources per land category (kg n2o-n/yr)
                        soil_N2O_feed_direct_rd['pasture']=pasture_area_rd*(EF_1_rd*(Fsn['pasture']+Fcr['pasture']))
                        soil_N2O_feed_direct_rd['grass']=grass_area_rd*(EF_1_rd*(Fsn['grass']+Fcr['grass']+Fon['grass'])+EF_3_PRP_rd*Fprp['grass'])
                        soil_N2O_feed_direct_rd['napier']=napier_area_rd*EF_1_rd*(Fsn['napier']+ Fon['napier']+Fcr['napier'])
                        soil_N2O_feed_direct_rd['napier_hy']=napier_hy_area_rd*EF_1_rd*(Fsn['napier_hy']+ Fon['napier_hy']+Fcr['napier_hy'])
                        soil_N2O_feed_direct_rd['stover']=stover_area_rd*allocation_factor_stover*EF_1_rd*(Fsn['stover']+ Fon['stover']+Fcr['stover'])
                        soil_N2O_feed_direct_rd['maize_bran']=maize_as_bran_area_rd*allocation_factor_maize_bran*EF_1_rd*(Fsn['maize_bran']+Fcr['maize_bran'])
                        soil_N2O_feed_direct_rd['sunflower_seed_meal']=sunflower_as_meal_area_rd*allocation_factor_sunflower_seed_meal*EF_1_rd*(Fsn['sunflower_seed_meal']+Fcr['sunflower_seed_meal'])

                    # indirect (volatilization) (kg n2o-n/yr)
                        soil_N2O_feed_adn_rd['pasture']=pasture_area_rd*EF_4_adn_rd*(Fsn['pasture']*FRAC_gasf+(Fon['pasture'])*FRAC_gasm)
                        soil_N2O_feed_adn_rd['grass']=grass_area_rd*EF_4_adn_rd*(Fsn['grass']*FRAC_gasf+(Fon['grass']+Fprp['grass'])*FRAC_gasm)
                        soil_N2O_feed_adn_rd['napier']=napier_area_rd*EF_4_adn_rd*(Fsn['napier']*FRAC_gasf+Fon['napier']*FRAC_gasm)
                        soil_N2O_feed_adn_rd['napier_hy']=napier_hy_area_rd*EF_4_adn_rd*(Fsn['napier_hy']*FRAC_gasf+Fon['napier_hy']*FRAC_gasm)
                        soil_N2O_feed_adn_rd['stover']=stover_area_rd*allocation_factor_stover*EF_4_adn_rd*(Fsn['stover']*FRAC_gasf+Fon['stover']*FRAC_gasm)
                        soil_N2O_feed_adn_rd['maize_bran']=maize_as_bran_area_rd*allocation_factor_maize_bran*EF_4_adn_rd*(Fsn['maize_bran']*FRAC_gasf+Fon['maize_bran']*FRAC_gasm)
                        soil_N2O_feed_adn_rd['sunflower_seed_meal']=sunflower_as_meal_area_rd*allocation_factor_sunflower_seed_meal*EF_4_adn_rd*(Fsn['sunflower_seed_meal']*FRAC_gasf+Fon['sunflower_seed_meal']*FRAC_gasm)

                    # indirect (leaching) (kg n2o-n/yr)
                        soil_N2O_feed_lch_rd['pasture']=pasture_area_rd*EF_5_lch_rd*(Fsn['pasture']+Fcr['pasture']+Fon['pasture'])
                        soil_N2O_feed_lch_rd['grass']=grass_area_rd*EF_5_lch_rd*(Fsn['grass']+Fcr['grass']+Fon['grass']+Fprp['grass'])
                        soil_N2O_feed_lch_rd['napier']=napier_area_rd*EF_5_lch_rd*(Fsn['napier']+ Fon['napier']+Fcr['napier'])
                        soil_N2O_feed_lch_rd['napier_hy']=napier_hy_area_rd*EF_5_lch_rd*(Fsn['napier_hy']+ Fon['napier_hy']+Fcr['napier_hy'])
                        soil_N2O_feed_lch_rd['stover']=stover_area_rd*allocation_factor_stover*EF_5_lch_rd*(Fsn['stover']+ Fon['stover']+Fcr['stover'])
                        soil_N2O_feed_lch_rd['maize_bran']=maize_as_bran_area_rd*allocation_factor_maize_bran*EF_5_lch_rd*(Fsn['maize_bran']+Fcr['maize_bran'])
                        soil_N2O_feed_lch_rd['sunflower_seed_meal']=sunflower_as_meal_area_rd*allocation_factor_sunflower_seed_meal*EF_5_lch_rd*(Fsn['sunflower_seed_meal']+Fcr['sunflower_seed_meal'])


                        soil_N2O_feed_grass_rd=(soil_N2O_feed_direct_rd['grass']+soil_N2O_feed_lch_rd['grass']+soil_N2O_feed_adn_rd['grass'])
                        soil_N2O_feed_napier_rd=(soil_N2O_feed_direct_rd['napier']+soil_N2O_feed_lch_rd['napier']+soil_N2O_feed_adn_rd['napier'])
                        soil_N2O_feed_napier_hy_rd=(soil_N2O_feed_direct_rd['napier_hy']+soil_N2O_feed_lch_rd['napier_hy']+soil_N2O_feed_adn_rd['napier_hy'])
                        soil_N2O_feed_maize_bran_rd=(soil_N2O_feed_direct_rd['maize_bran']+soil_N2O_feed_lch_rd['maize_bran']+soil_N2O_feed_adn_rd['maize_bran'])
                        soil_N2O_feed_sunflower_seed_meal_rd=(soil_N2O_feed_direct_rd['sunflower_seed_meal']+soil_N2O_feed_lch_rd['sunflower_seed_meal']+soil_N2O_feed_adn_rd['sunflower_seed_meal'])
                        soil_N2O_feed_stover_rd=(soil_N2O_feed_direct_rd['stover']+soil_N2O_feed_lch_rd['stover']+soil_N2O_feed_adn_rd['stover'])
                        soil_N2O_feed_pasture_rd=(soil_N2O_feed_direct_rd['pasture']+soil_N2O_feed_lch_rd['pasture']+soil_N2O_feed_adn_rd['pasture'])

                        enteric_CH4_rd[MC,ss,b,c]=28*gross_energy[ss,b,c]*Ym_rd[ss,b,c]*(1/(100*55.65))
                        manure_CH4_rd[MC,ss,b,c]=(28*VS[ss,b,c]*B0_rd[ss,b,c]*MCF_rd[ss,b,c]*0.67)
                        manure_N2O_direct_rd[MC,ss,b,c]=265*(44/28)*FRAC_MANURE_MMT*((1/1000)*faecal_Nitrogen[ss,b,c]*EF_3_FAECAL_rd[ss,b,c]+(1/1000)*urinary_Nitrogen[ss,b,c]*EF_3_URINARY_rd[ss,b,c])
                        manure_N2O_leach_rd[MC,ss,b,c]=265*(44/28)*FRAC_MANURE_MMT*((1/1000)*(faecal_Nitrogen[ss,b,c]+urinary_Nitrogen[ss,b,c]))*FRAC_leach_rd*EF_5_rd
                        manure_N2O_vol_rd[MC,ss,b,c]=265*(44/28)*((1/1000)*FRAC_MANURE_MMT*(faecal_Nitrogen[ss,b,c]+urinary_Nitrogen[ss,b,c]))*FRAC_vol_rd*EF_4_rd
                        manure_N2O_rd[MC,ss,b,c]=manure_N2O_direct_rd[MC,ss,b,c]+manure_N2O_leach_rd[MC,ss,b,c]+manure_N2O_vol_rd[MC,ss,b,c]
                        soil_N2O_rd[MC,ss,b,c]=265*(44/28)*(soil_N2O_feed_grass_rd*grass_area_rd+soil_N2O_feed_napier_rd*napier_area_rd+soil_N2O_feed_maize_bran_rd*maize_as_bran_area_rd+soil_N2O_feed_sunflower_seed_meal_rd*sunflower_as_meal_area_rd +soil_N2O_feed_stover_rd*stover_area_rd+soil_N2O_feed_pasture_rd*pasture_area_rd)
                        energy_use_CO2_rd[MC,ss,b,c]=energy_use_CO2[ss,b,c]*random_var[MC,24]

                        grassland_area_expansion_emissions_rd_ss[MC,ss]+=random_var[MC,20]*(44/12)*(herd[(ss,b,'total_herd')]*herd[(ss,b,c)]*feed_primary_area_rd-herd_y0[(ss,b)]*herd[(ss,b,c)]*feed_primary_area_rd)*(luc_emission_coefficient_total_area)*(timestep/amort_period)/timestep 
                        cropland_area_expansion_emissions_rd_ss[MC,ss]+=random_var[MC,21]*(44/12)*(herd[(ss,b,'total_herd')]*herd[(ss,b,c)]*(crop_primary_area_rd)-herd_y0[(ss,b)]*herd[(ss,b,c)]*(crop_primary_area_rd))*(luc_emission_coefficient_grass_to_crop)*(timestep/amort_period)/timestep 





                # report final results for this simulation
        for ss in subsectors:
            for b in breed:
                if (herd[(ss,b,'total_herd')] ==0):
                    continue

                cohorts_meat = ['cow']     

                if ('bull' in c):
                    cohorts_meat.append('bull')

                if ('ml_calf' in c):
                    cohorts_meat.append('ml_calf')


                for c in cohorts_meat:
                    Meat_yield_per_head_kg_per_year[ss,b,c] = dressing_pct[(c)]*herd[(ss,b,c)]*final_bodyweight_kg[ss,b,c]*(1/(final_age_months[ss,b,c]/12))
                    Herd_Meat_yield_Total_kg_per_year[ss,b] += Meat_yield_per_head_kg_per_year[ss,b,c] 

                #  livestock weights in tlu
                for c in cohort:
                    TLU_equiv[ss,b,c] = Body_weight[ss,b,c] / 250
                    print('Body weight for ',b,' ',c,' is', Body_weight[ss,b,c]  )

                Herd_Milk_yield_kg_per_year[ss,b]=herd[(ss,b,'cow')]*milk_yield_reg[ss,b]

                # Allocate to milk

                alloc_factor = Herd_Milk_yield_kg_per_year[ss,b]/(Herd_Milk_yield_kg_per_year[ss,b] + Herd_Meat_yield_Total_kg_per_year[ss,b])

                print('Alloc factor is ',alloc_factor)

                for c in cohort:
                    enteric_CH4_rd[MC,ss,b,c]=alloc_factor*enteric_CH4_rd[MC,ss,b,c]
                    manure_CH4_rd[MC,ss,b,c]=alloc_factor*manure_CH4_rd[MC,ss,b,c]
                    manure_N2O_direct_rd[MC,ss,b,c]=alloc_factor*manure_N2O_direct_rd[MC,ss,b,c]
                    manure_N2O_leach_rd[MC,ss,b,c]=alloc_factor*manure_N2O_leach_rd[MC,ss,b,c]
                    manure_N2O_vol_rd[MC,ss,b,c]=alloc_factor*manure_N2O_vol_rd[MC,ss,b,c]
                    manure_N2O_rd[MC,ss,b,c]=manure_N2O_direct_rd[MC,ss,b,c]+manure_N2O_leach_rd[MC,ss,b,c]+manure_N2O_vol_rd[MC,ss,b,c]
                    soil_N2O_rd[MC,ss,b,c]=alloc_factor*soil_N2O_rd[MC,ss,b,c]
                    energy_use_CO2_rd[MC,ss,b,c]=alloc_factor*energy_use_CO2_rd[MC,ss,b,c]

                grassland_area_expansion_emissions_rd_ss[MC,ss]=alloc_factor*grassland_area_expansion_emissions_rd_ss[MC,ss]
                cropland_area_expansion_emissions_rd_ss[MC,ss]=alloc_factor*cropland_area_expansion_emissions_rd_ss[MC,ss]


        for ss in subsectors:
            for b in breed:
                if (herd[(ss,b,'total_herd')] == 0):
                    continue

                Total_TLU = 0

                ## results are grouped numerically from v1 to v9, with each grouping for a specific type of result variable

                # v1 -- main production/productivity results
                # v2 -- nutrient intakes and N excretion
                # v3 -- feed intakes, simulation variables, tropical livestock unit equivalents 
                # v4 -- animal specific emissions
                # v5 -- feed emission properties
                # v6 -- aggregated emissions 
                # v7 -- aggregated emissions intensity
                # v8 -- dairy herd populations
                # v9 -- land footprint

                for c in cohort:


                    enteric_CH4[ss,b,c]=alloc_factor*enteric_CH4[ss,b,c]
                    manure_CH4[ss,b,c]=alloc_factor*manure_CH4[ss,b,c]
                    manure_N2O_direct[ss,b,c]=alloc_factor*manure_N2O_direct[ss,b,c]
                    manure_N2O_vol[ss,b,c]=alloc_factor*manure_N2O_vol[ss,b,c]
                    manure_N2O_leach[ss,b,c]=alloc_factor*manure_N2O_leach[ss,b,c]
                    manure_N2O[ss,b,c]=alloc_factor*manure_N2O[ss,b,c]
                    soil_N2O_grassland[ss,b,c]=alloc_factor*soil_N2O_grassland[ss,b,c]
                    soil_N2O_cropland[ss,b,c]=alloc_factor*soil_N2O_cropland[ss,b,c]
                    energy_use_CO2[ss,b,c]=alloc_factor*energy_use_CO2[ss,b,c]


                    if (c == 'cow'):

                        out['v1_1_Milk_yield_ss'+str(ss)+'_'+str(b)]=milk_yield_reg[ss,b]
                        out['v1_1_Milk_yield_lactation_kg_per_lactation__ss'+str(ss)+'_'+str(b)]=milk_yield_lact[ss,b]
                        out['v1_2_Milk_yield_total_kg_yr'] += (herd[(ss,b,'total_herd')]*herd[(ss,b,'cow')]*
                                                        out['v1_1_Milk_yield_ss'+str(ss)+'_'+str(b)])
                        out['v1_2_Milk_yield_'+str(b)+'_total_kg_yr'] += (herd[(ss,b,'total_herd')]*
                                                                    herd[(ss,b,'cow')]*out['v1_1_Milk_yield_ss'+str(ss)+'_'+str(b)])
                        out['v1_2_Milk_yield_base_year_kg_yr'] += (herd_y0[(ss,b)]*
                                                                    herd[(ss,b,'cow')]*
                                                                    out['v1_1_Milk_yield_ss'+str(ss)+'_'+str(b)])
                        

                        for cc in cohort:
                            Total_TLU += TLU_equiv[(ss,b,cc)]


                        out['v1_3_Herd_Meat_Yield_kg_per_TLU_'+str(ss)+'_'+str(b)]+=Herd_Meat_yield_Total_kg_per_year[ss,b]/(Total_TLU)
                        out['v1_3_Meat_yield_to_milk_g_to_kg_'+str(ss)+'_'+str(b)]=1000*out['v1_3_Herd_Meat_Yield_kg_per_TLU_'+str(ss)+'_'+str(b)]/milk_yield_reg[ss,b]                                              



                    # v2 - nutrient intakes and N excretion
                    out['v2_1_Dry_matter_intake_kg_per_head_per_yr_'+str(b)+'_'+str(c)+'_'+str(ss)+'']=DMI[ss,b,c]
                    out['v2_2_Dry_matter_intake_kg_per_head_per_day_'+str(b)+'_'+str(c)+'_'+str(ss)+'']=DMI[ss,b,c]/365
                    out['v2_3_Fraction_Roughage_'+str(b)+'_'+str(c)+'_'+str(ss)+'']=fraction_roughage_eaten[ss,b,c]
                    out['v2_3_Fraction_Concentrate_'+str(b)+'_'+str(c)+'_'+str(ss)+'']=fraction_conc_eaten[ss,b,c]
                    out['v2_3_Metabolisable_Energy_'+str(b)+'_'+str(c)+'_'+str(ss)+'']=ME[ss,b,c]
                    out['v2_4_Dry_matter_digestibility_%_'+str(b)+'_'+str(c)+'_'+str(ss)+'']=DMD[ss,b,c]
                    out['v2_5_1_Gross_energy_MJ_per_head_per_yr_'+str(b)+'_'+str(c)+'_'+str(ss)+'']=gross_energy[ss,b,c]
                    out['v2_5_2_Gross_energy_MJ_per_head_per_day_'+str(b)+'_'+str(c)+'_'+str(ss)+'']=gross_energy[ss,b,c]/365
                    out['v2_6_faecal_nitrogen_'+str(b)+'_'+str(c)+'_'+str(ss)+'']=faecal_Nitrogen[ss,b,c]
                    out['v2_6_urinary_nitrogen_'+str(b)+'_'+str(c)+'_'+str(ss)+'']=urinary_Nitrogen[ss,b,c] 
                    out['v2_7_Ym_'+str(b)+'_'+str(c)+'_'+str(ss)+'']=Ym[ss,b,c]
                    out['v2_8_VS_'+str(b)+'_'+str(c)+'_'+str(ss)+'']=VS[ss,b,c]
                    out['v2_9_Acid_detergent_fibre_g_per_head_per_year_'+str(b)+'_'+str(c)+'_'+str(ss)+'']=ADF[ss,b,c]
                    out['v2_9_Acid_detergent_fibre_%_'+str(b)+'_'+str(c)+'_'+str(ss)+'']=100*ADF[ss,b,c]*(1/1000)/DMI[ss,b,c]
                    out['v2_11_Neutral_detergent_fibre_g_per_head_per_year_'+str(b)+'_'+str(c)+'_'+str(ss)+'']=NDF[ss,b,c]
                    out['v2_11_Neutral_detergent_fibre_%_'+str(b)+'_'+str(c)+'_'+str(ss)+'']=100*NDF[ss,b,c]*(1/1000)/DMI[ss,b,c]
                    out['v2_12_Crude_protein_kg_per_head_per_year'+str(b)+'_'+str(c)+'_'+str(ss)+'']=crude_protein_intake[ss,b,c]
                    out['v2_12_Crude_protein_%_'+str(b)+'_'+str(c)+'_'+str(ss)+'']=100*crude_protein_intake[ss,b,c]/DMI[ss,b,c]

                    # v3_1 - feed intake results (% dm for specific ss, b, c)
                    out['v3_1_1_Feed_intake_grass_%_dm'+str(b)+'_'+str(c)+'_'+str(ss)+'']=100*grass_intake[ss,b,c]/DMI[ss,b,c]
                    out['v3_1_2_Feed_intake_stover_%_dm'+str(b)+'_'+str(c)+'_'+str(ss)+'']=100*stover_intake[ss,b,c]/DMI[ss,b,c]
                    out['v3_1_3_Feed_intake_napier_%_dm'+str(b)+'_'+str(c)+'_'+str(ss)+'']=100*napier_intake[ss,b,c]/DMI[ss,b,c]
                    out['v3_1_4_Feed_intake_maize_bran_%_dm'+str(b)+'_'+str(c)+'_'+str(ss)+'']=100*maize_bran_intake[ss,b,c]/DMI[ss,b,c]
                    out['v3_1_5_Feed_intake_sfsm_%_dm'+str(b)+'_'+str(c)+'_'+str(ss)+'']=100*sfsm_intake[ss,b,c]/DMI[ss,b,c]
                    out['v3_1_6_Feed_intake_pasture_%_dm'+str(b)+'_'+str(c)+'_'+str(ss)+'']=100*pasture_intake[ss,b,c]/DMI[ss,b,c]

                    # v3_2 feed intake results (kg dm for entire herd)
                    out['v3_2_6_Feed_intake_grass_kg_dm']+=herd[(ss,b,'total_herd')]*herd[(ss,b,c)]*grass_intake[ss,b,c]
                    out['v3_2_7_Feed_intake_stover_kg_dm']+=herd[(ss,b,'total_herd')]*herd[(ss,b,c)]*stover_intake[ss,b,c]
                    out['v3_2_8_Feed_intake_napier_kg_dm']+=herd[(ss,b,'total_herd')]*herd[(ss,b,c)]*napier_intake[ss,b,c]
                    out['v3_2_8_Feed_intake_napier_hy_kg_dm']+=herd[(ss,b,'total_herd')]*herd[(ss,b,c)]*napier_hy_intake[ss,b,c]
                    out['v3_2_9_Feed_intake_maize_bran_kg_dm'] += (herd[(ss,b,'total_herd')]*
                    herd[(ss,b,c)]*maize_bran_intake[ss,b,c])
                    out['v3_2_10_Feed_intake_sfsm_kg_dm']+=herd[(ss,b,'total_herd')]*herd[(ss,b,c)]*sfsm_intake[ss,b,c]
                    out['v3_2_11_Feed_intake_pasture_kg_dm']+=herd[(ss,b,'total_herd')]*herd[(ss,b,c)]*pasture_intake[ss,b,c]

                    # v3_4 feed intake results per cow (kg dm for each subsector) (this is used as an input to chapter 5 income calculations)
                    out['v3_4_1_Feed_intake_per_cow_grass_kg_dm_yr_ss'+str(ss)+'']+=herd[(ss,b,'total_herd')]*herd[(ss,b,c)]*grass_intake[ss,b,c]/(herd[(ss,b,'total_herd')]*herd[(ss,b,'cow')])
                    out['v3_4_2_Feed_intake_per_cow_stover_kg_dm_yr_ss'+str(ss)+'']+=herd[(ss,b,'total_herd')]*herd[(ss,b,c)]*stover_intake[ss,b,c]/(herd[(ss,b,'total_herd')]*herd[(ss,b,'cow')])
                    out['v3_4_3_Feed_intake_per_cow_napier_kg_dm_yr_ss'+str(ss)+'']+=herd[(ss,b,'total_herd')]*herd[(ss,b,c)]*napier_intake[ss,b,c]/(herd[(ss,b,'total_herd')]*herd[(ss,b,'cow')])
                    out['v3_4_4_Feed_intake_per_cow_napier_hy_kg_dm_yr_ss'+str(ss)+'']+=herd[(ss,b,'total_herd')]*herd[(ss,b,c)]*napier_hy_intake[ss,b,c]/(herd[(ss,b,'total_herd')]*herd[(ss,b,'cow')])
                    out['v3_4_5_Feed_intake_per_cow_maize_bran_kg_dm_yr_ss'+str(ss)+'']+=herd[(ss,b,'total_herd')]*herd[(ss,b,c)]*maize_bran_intake[ss,b,c]/(herd[(ss,b,'total_herd')]*herd[(ss,b,'cow')])
                    out['v3_4_6_Feed_intake_per_cow_sfsm_kg_dm_yr_ss'+str(ss)+'']+=herd[(ss,b,'total_herd')]*herd[(ss,b,c)]*sfsm_intake[ss,b,c]/(herd[(ss,b,'total_herd')]*herd[(ss,b,'cow')])
                    out['v3_4_7_Feed_intake_per_cow_pasture_kg_dm_yr_ss'+str(ss)+'']+=herd[(ss,b,'total_herd')]*herd[(ss,b,c)]*pasture_intake[ss,b,c]/(herd[(ss,b,'total_herd')]*herd[(ss,b,'cow')])

                    # seasonality variables (only for cows)
                    if (c == 'cow'):

                        for i in range(0,12):
                            # Met energy intake in month i
                            out['v3_2_12_M_'+str(i)+'_metenergy_Mj_d_'+str(b)+'_'+str(c)+'_'+str(ss)+'']=mo_dp_me[ss,b,c,i]/(count_calendar_month[i,ss,b,c])

                            # Crude protein intake in month i
                            out['v3_2_12_M_'+str(i)+'_cp_g_d_'+str(b)+'_'+str(c)+'_'+str(ss)+'']=mo_dp_cp[ss,b,c,i]/(count_calendar_month[i,ss,b,c])

                            # Shortage of MP in month i 
                            out['v3_2_12_M'+str(i)+'_MPS_g_d_'+str(b)+'_'+str(c)+'_'+str(ss)+'']= mo_mp_short[ss,b,c,i]/(count_calendar_month[i,ss,b,c])

                             # Shortage of ME in month i 
                            out['v3_2_12_M'+str(i)+'_MES_Mj_d_'+str(b)+'_'+str(c)+'_'+str(ss)+'']= mo_me_short[ss,b,c,i]/(count_calendar_month[i,ss,b,c])



                 # average protein or energy deficit during dry season
                        out['v3_3_mean_dryse_prot_deficit']=(
                            out['v3_2_12_M1_MPS_g_d_'+str(b)+'_'+str(c)+'_'+str(ss)+'']
                            +out['v3_2_12_M2_MPS_g_d_'+str(b)+'_'+str(c)+'_'+str(ss)+'']
                            +out['v3_2_12_M3_MPS_g_d_'+str(b)+'_'+str(c)+'_'+str(ss)+'']
                            +out['v3_2_12_M4_MPS_g_d_'+str(b)+'_'+str(c)+'_'+str(ss)+'']
                            +out['v3_2_12_M5_MPS_g_d_'+str(b)+'_'+str(c)+'_'+str(ss)+'']
                            +out['v3_2_12_M6_MPS_g_d_'+str(b)+'_'+str(c)+'_'+str(ss)+''])*(1/6)

                        out['v3_3_mean_dryse_ene_deficit']=(
                            out['v3_2_12_M1_MES_Mj_d_'+str(b)+'_'+str(c)+'_'+str(ss)+'']+
                            out['v3_2_12_M2_MES_Mj_d_'+str(b)+'_'+str(c)+'_'+str(ss)+'']+
                            out['v3_2_12_M3_MES_Mj_d_'+str(b)+'_'+str(c)+'_'+str(ss)+'']+
                            out['v3_2_12_M4_MES_Mj_d_'+str(b)+'_'+str(c)+'_'+str(ss)+'']+
                            out['v3_2_12_M5_MES_Mj_d_'+str(b)+'_'+str(c)+'_'+str(ss)+'']+
                            out['v3_2_12_M6_MES_Mj_d_'+str(b)+'_'+str(c)+'_'+str(ss)+''])*(1/6)


                    # v3 - 5 TLU equivalents
                    out['v3_5_TLU_equiv_'+str(ss)+'_'+str(b)+'_'+str(c)+''] = TLU_equiv[(ss,b,c)] 


                    # v4 - emissions per animal per year
                    out['v4_1_Enteric_CH4_kg_CO2eq_per_head_per_year_'+str(ss)+'_'+str(b)+'_'+str(c)+'']=enteric_CH4[ss,b,c]
                    out['v4_2_Manure_CH4_kg_CO2eq_per_head_per_year_'+str(ss)+'_'+str(b)+'_'+str(c)+'']=manure_CH4[ss,b,c] 
                    out['v4_3_Manure_N2O_kg_CO2eq_per_head_per_year_'+str(ss)+'_'+str(b)+'_'+str(c)+'']=manure_N2O[ss,b,c] 
                    out['v4_4_Soil_N2O_cropland_kg_CO2eq_per_head_per_year_'+str(ss)+'_'+str(b)+'_'+str(c)+''] =soil_N2O_cropland[ss,b,c] 
                    out['v4_4_Soil_N2O_grassland_kg_CO2eq_per_head_per_year_'+str(ss)+'_'+str(b)+'_'+str(c)+''] =soil_N2O_grassland[ss,b,c] 
                    out['v4_5_Feed_CO2_kg_CO2eq_per_head_per_year_'+str(ss)+'_'+str(b)+'_'+str(c)+''] =energy_use_CO2[ss,b,c]

                    # v5 - emissions from soils
                    out['v5_Grasslands_N2O_kg_per_ha_per_yr_ss_'+str(ss)+'']=(44/28)*soil_N2O_feed['grass']/grass_area[(ss,b,c)]
                    out['v5_Napier_N2O_kg_per_ha_per_yr_ss_'+str(ss)+'']=(44/28)*soil_N2O_feed['napier']/napier_area[(ss,b,c)]
                    out['v5_Maize_bran_N2O_kg_per_ha_per_yr_ss_'+str(ss)+'']=(44/28)*(1/allocation_factor_maize_bran)*soil_N2O_feed['maize_bran']/maize_as_bran_area[(ss,b,c)]
                    out['v5_Sunflower_cake_N2O_kg_per_ha_per_yr_ss_'+str(ss)+'']=(44/28)*(1/allocation_factor_sunflower_seed_meal)*soil_N2O_feed['sunflower_seed_meal']/sunflower_as_meal_area[(ss,b,c)]
                    out['v5_stover_N2O_kg_per_ha_per_yr_ss_'+str(ss)+'']=(44/28)*(1/allocation_factor_stover)*soil_N2O_feed['stover']/stover_area[(ss,b,c)]


                    if (pasture_area[(ss,b,c)] !=0):
                        out['v5_Pasture_N2O_kg_per_ha_per_yr_ss_'+str(ss)+'']=(44/28)*soil_N2O_feed['pasture']/pasture_area[(ss,b,c)]

                    # v5 - total area of feed land
                    out['v5_6_Total_feed_primary_area_expansion'] += herd[(ss,b,c)]*(herd[(ss,b,'total_herd')]*feed_primary_area[(ss,b,c)]-herd_y0[(ss,b)]*feed_primary_area[(ss,b,c)])
                    out['v5_6_Total_grasslands_area_expansion'] += herd[(ss,b,c)]*(herd[(ss,b,'total_herd')]*total_grasslands_area[(ss,b,c)]-herd_y0[(ss,b)]*total_grasslands_area[(ss,b,c)])


                    # v6 - emissions results aggregated over breed and cohort for each lps
                    out['v6_0_1_Enteric_CH4_aggregate_Mg_CO2eq_per_lps'] += herd[(ss,b,'total_herd')]*herd[(ss,b,c)]*enteric_CH4[ss,b,c]/1000 
                    out['v6_0_2_Manure_CH4_aggregate_Mg_CO2eq_per_lps'] += herd[(ss,b,'total_herd')]*herd[(ss,b,c)]*manure_CH4[ss,b,c]/1000 
                    out['v6_0_3_Manure_N2O_aggregate_Mg_CO2eq_per_lps'] += herd[(ss,b,'total_herd')]*herd[(ss,b,c)]*manure_N2O[ss,b,c]/1000 
                    out['v6_0_4_Soil_N2O_cropland_aggregate_Mg_CO2eq_per_lps'] += herd[(ss,b,'total_herd')]*herd[(ss,b,c)]*soil_N2O_cropland[ss,b,c]/1000 
                    out['v6_0_4_Soil_N2O_grassland_aggregate_Mg_CO2eq_per_lps'] += herd[(ss,b,'total_herd')]*herd[(ss,b,c)]*soil_N2O_grassland[ss,b,c]/1000 
                    out['v6_0_4_2_Feed_CO2_aggregate_Mg_CO2eq_per_lps'] += herd[(ss,b,'total_herd')]*herd[(ss,b,c)]*energy_use_CO2[ss,b,c]/1000 

                    out['v6_5_Direct_emissions_aggregate_Mg_CO2eq_per_lps']=(
                        out['v6_0_1_Enteric_CH4_aggregate_Mg_CO2eq_per_lps']
                        +out['v6_0_2_Manure_CH4_aggregate_Mg_CO2eq_per_lps']
                        +out['v6_0_3_Manure_N2O_aggregate_Mg_CO2eq_per_lps']
                        +out['v6_0_4_Soil_N2O_grassland_aggregate_Mg_CO2eq_per_lps']
                        +out['v6_0_4_Soil_N2O_cropland_aggregate_Mg_CO2eq_per_lps']
                        +out['v6_0_4_2_Feed_CO2_aggregate_Mg_CO2eq_per_lps'])


                    # aggregate emissions for local  
                    if (b == 'local'):
                        out['v6_1_1_Enteric_CH4_local_aggregate_Mg_CO2eq_per_lps']+=(herd[(ss,'local','total_herd')]*herd[(ss,'local',c)]*enteric_CH4[ss,'local',c]/1000 )
                        out['v6_1_2_Manure_CH4_local_aggregate_Mg_CO2eq_per_lps']+=(herd[(ss,'local','total_herd')]*herd[(ss,'local',c)]*manure_CH4[ss,'local',c]/1000 )
                        out['v6_1_3_Manure_N2O_local_aggregate_Mg_CO2eq_per_lps']+=(herd[(ss,'local','total_herd')]*herd[(ss,'local',c)]*manure_N2O[ss,'local',c]/1000 )
                        out['v6_1_4_Soil_N2O_cropland_local_aggregate_Mg_CO2eq_per_lps']+=(herd[(ss,'local','total_herd')]*herd[(ss,'local',c)]*soil_N2O_cropland[ss,'local',c]/1000 )
                        out['v6_1_4_Soil_N2O_grassland_local_aggregate_Mg_CO2eq_per_lps']+=(herd[(ss,'local','total_herd')]*herd[(ss,'local',c)]*soil_N2O_grassland[ss,'local',c]/1000 )
                        out['v6_1_4_2_Feed_CO2_local_aggregate_Mg_CO2eq_per_lps']+=(herd[(ss,'local','total_herd')]*herd[(ss,'local',c)]*energy_use_CO2[ss,'local',c]/1000 )

                        out['v6_1_8_cropland_expansion_local_emissions_Mg_CO2eq_per_lps'] += (alloc_factor*
                        (44/12)*herd[(ss,'local',c)]*(herd[(ss,'local','total_herd')]*
                        crop_primary_area[(ss,'local',c)]-herd_y0[(ss,b)]*
                        crop_primary_area[(ss,'local',c)])*
                        (luc_emission_coefficient_grass_to_crop)*(timestep/amort_period)/timestep)

                        out['v6_1_5_Direct_emissions_local_aggregate_Mg_CO2eq_per_lps']=(
                        out['v6_1_1_Enteric_CH4_local_aggregate_Mg_CO2eq_per_lps']
                        +out['v6_1_2_Manure_CH4_local_aggregate_Mg_CO2eq_per_lps']
                        +out['v6_1_3_Manure_N2O_local_aggregate_Mg_CO2eq_per_lps']
                        +out['v6_1_4_Soil_N2O_grassland_local_aggregate_Mg_CO2eq_per_lps']
                        +out['v6_1_4_Soil_N2O_cropland_local_aggregate_Mg_CO2eq_per_lps']
                        +out['v6_1_4_2_Feed_CO2_local_aggregate_Mg_CO2eq_per_lps'])


                    elif (b == 'improved'):
                     # aggregate emissions for improved               
                        out['v6_2_1_Enteric_CH4_improved_aggregate_Mg_CO2eq_per_lps']+=(herd[(ss,'improved','total_herd')]*herd[(ss,'improved',c)]*enteric_CH4[ss,'improved',c]/1000 )
                        out['v6_2_2_Manure_CH4_improved_aggregate_Mg_CO2eq_per_lps']+=(herd[(ss,'improved','total_herd')]*herd[(ss,'improved',c)]*manure_CH4[ss,'improved',c]/1000 )
                        out['v6_2_3_Manure_N2O_improved_aggregate_Mg_CO2eq_per_lps']+=(herd[(ss,'improved','total_herd')]*herd[(ss,'improved',c)]*manure_N2O[ss,'improved',c]/1000 )
                        out['v6_2_4_Soil_N2O_cropland_improved_aggregate_Mg_CO2eq_per_lps']+=(herd[(ss,'improved','total_herd')]*herd[(ss,'improved',c)]*soil_N2O_cropland[ss,'improved',c]/1000 )
                        out['v6_2_4_Soil_N2O_grassland_improved_aggregate_Mg_CO2eq_per_lps']+=(herd[(ss,'improved','total_herd')]*herd[(ss,'improved',c)]*soil_N2O_grassland[ss,'improved',c]/1000 )
                        out['v6_2_4_2_Feed_CO2_improved_aggregate_Mg_CO2eq_per_lps']+=(herd[(ss,'improved','total_herd')]*herd[(ss,'improved',c)]*energy_use_CO2[ss,'improved',c]/1000 )


                        out['v6_2_5_Direct_emissions_improved_aggregate_Mg_CO2eq_per_lps']=(
                        out['v6_2_1_Enteric_CH4_improved_aggregate_Mg_CO2eq_per_lps']
                        +out['v6_2_2_Manure_CH4_improved_aggregate_Mg_CO2eq_per_lps']
                        +out['v6_2_3_Manure_N2O_improved_aggregate_Mg_CO2eq_per_lps']
                        +out['v6_2_4_Soil_N2O_grassland_improved_aggregate_Mg_CO2eq_per_lps']
                        +out['v6_2_4_Soil_N2O_cropland_improved_aggregate_Mg_CO2eq_per_lps']
                        +out['v6_2_4_2_Feed_CO2_improved_aggregate_Mg_CO2eq_per_lps'])


                    # land use change as part of aLCA

                    out['v6_0_8_cropland_expansion_emissions_Mg_CO2eq_per_lps'] += alloc_factor*(44/12)*herd[(ss,b,c)]*(herd[(ss,b,'total_herd')]*crop_primary_area[(ss,b,c)]-herd_y0[(ss,b)]*crop_primary_area[(ss,b,c)])*(luc_emission_coefficient_grass_to_crop)*(timestep/amort_period)/timestep



                    # v8 - herd populations
                    out['v8_1_herd_'+str(ss)+'_'+str(b)+'_'+str(c)] = herd[(ss,b,'total_herd')]*herd[(ss,b,c)]
                    out['v8_2_herd_previous_year_'+str(ss)+'_'+str(b)+'_'+str(c)] = herd_y0[(ss,b)]*herd[(ss,b,c)]
                    out['v8_3_herd_baseline_yf_'+str(ss)+'_'+str(b)+'_'+str(c)] = herd_baseline_yf[(l,ss,b,'total_herd')]*herd_baseline_yf[(l,ss,b,c)]
                    out['v8_4_herd_total_1000_head_per_lps'] += herd[(ss,b,'total_herd')]*herd[(ss,b,c)]/1000                
                    out['v8_6_herd_total_TLU_per_lps'] += herd[(ss,b,c)]*herd[(ss,b,'total_herd')]*TLU_equiv[(ss,b,c)]
                    out['v8_5_herd_total_Million_TLU_per_lps'] = out['v8_6_herd_total_TLU_per_lps']/1000000
                    out['v8_7_herd_total_'+str(b)+'_TLU_per_lps'] += herd[(ss,b,c)]*herd[(ss,b,'total_herd')]*TLU_equiv[(ss,b,c)]
                    out['v8_8_herd_total_'+str(b)+'_Head_per_lps'] += herd[(ss,b,c)]*herd[(ss,b,'total_herd')]

                    # feed areas aggregated categories 
                    out['v9_1_1_Grasslands_area_ha_per_head_'+str(b)+'_'+str(c)+'_'+str(ss)+'']=total_grasslands_area[(ss,b,c)]
                    out['v9_1_2_Feed_area_ha_per_head_'+str(b)+'_'+str(c)+'_'+str(ss)+'']=feed_area[(ss,b,c)]
                    out['v9_2_1_Grasslands_area_ha_per_lps'] += herd[(ss,b,c)]*(herd[(ss,b,'total_herd')]*total_grasslands_area[(ss,b,c)])
                    out['v9_2_2_Feed_area_ha_per_lps'] +=  herd[(ss,b,c)]*(herd[(ss,b,'total_herd')]*(feed_area[(ss,b,c)]))
                    out['v9_2_3_Crop_primary_area_ha_per_lps']+=herd[(ss,b,c)]*(herd[(ss,b,'total_herd')]*(feed_area[(ss,b,c)]-total_grasslands_area[(ss,b,c)]))
                    
                    # feed areas individual (ha per head)
                    out['v9_3_1_Grass_area_ha_per_head_'+str(b)+'_'+str(c)+'_'+str(ss)+'']=grass_area[(ss,b,c)]
                    out['v9_3_2_Stover_area_ha_per_head_'+str(b)+'_'+str(c)+'_'+str(ss)+'']=stover_area[(ss,b,c)]
                    out['v9_3_3_Napier_area_ha_per_head_'+str(b)+'_'+str(c)+'_'+str(ss)+'']=napier_area[(ss,b,c)]
                    out['v9_3_4_Napier_hy_area_ha_per_head_'+str(b)+'_'+str(c)+'_'+str(ss)+'']=napier_hy_area[(ss,b,c)]
                    out['v9_3_5_Sunflower_area_ha_per_head_'+str(b)+'_'+str(c)+'_'+str(ss)+'']=sunflower_as_meal_area[(ss,b,c)]
                    out['v9_3_6_Maize_area_ha_per_head_'+str(b)+'_'+str(c)+'_'+str(ss)+'']=maize_as_bran_area[(ss,b,c)]
                    out['v9_3_7_Pasture_area_ha_per_head_'+str(b)+'_'+str(c)+'_'+str(ss)+'']=pasture_area[(ss,b,c)]

                    # feed areas  (ha per lps)
                    out['v9_4_1_Grass_area_ha_per_lps']+=grass_area[(ss,b,c)]*herd[(ss,b,'total_herd')]*herd[(ss,b,c)]
                    out['v9_4_2_Stover_area_ha_per_lps']+=stover_area[(ss,b,c)]*herd[(ss,b,'total_herd')]*herd[(ss,b,c)]
                    out['v9_4_3_Napier_area_ha_per_lps']+=napier_area[(ss,b,c)]*herd[(ss,b,'total_herd')]*herd[(ss,b,c)]
                    out['v9_4_4_Napier_hy_area_ha_per_lps']+=napier_hy_area[(ss,b,c)]*herd[(ss,b,'total_herd')]*herd[(ss,b,c)]
                    out['v9_4_5_Sunflower_area_ha_per_lps']+=sunflower_as_meal_area[(ss,b,c)]*herd[(ss,b,'total_herd')]*herd[(ss,b,c)]
                    out['v9_4_6_Maize_area_ha_per_lps']+=maize_as_bran_area[(ss,b,c)]*herd[(ss,b,'total_herd')]*herd[(ss,b,c)]
                    out['v9_4_7_Pasture_area_ha_per_lps']+=pasture_area[(ss,b,c)]*herd[(ss,b,'total_herd')]*herd[(ss,b,c)]
                    out['v9_4_8_Total_area_ha_per_lps']=out['v9_4_1_Grass_area_ha_per_lps']+out['v9_4_2_Stover_area_ha_per_lps']+out['v9_4_3_Napier_area_ha_per_lps']+out['v9_4_4_Napier_hy_area_ha_per_lps']+out['v9_4_5_Sunflower_area_ha_per_lps']+out['v9_4_6_Maize_area_ha_per_lps']                


        if (check_feasibility== 1):

            '''
            This code determines the extent to which expansion of the dairy land footprint results in 
            conversion of native ecosystems. The growth in demand of dairy land use is related to spatially explicit estimates of 
            grassland availability from the ESA (Bruzonne et al.) land cover data. This results in an estimate of the fraction of total
            land use change which actually converts native ecosystems. This coefficient is then multiplied by the total land area converted 
            and the LUC coefficients to calculate GHG emissions from grassland expansion.

           '''

            out['v6_0_total_native_ecosystems_converted(ha)']=0
            out['v6_0_total_grasslands_now_utilized(ha)']=0
            out['v6_0_Units_limited']=0

            out['v6_0_total_direct_native_ecosystems_converted(ha)']=0
            out['v6_0_total_direct_grasslands_now_utilized(ha)']=0
            out['v6_0_Units_direct_limited']=0

            # load data on spatial land use, lives populations
            if ( reg_analysis == 0 ) :

                system_feasibility_data=(cwd+str('\\land_feasibility.xlsx'))
                system_feasibility=xlrd.open_workbook(system_feasibility_data)

                if (l == 'MRT'):
                    feas_sheet = system_feasibility.sheet_by_index(0)
                elif (l == 'MRH'):
                    feas_sheet = system_feasibility.sheet_by_index(1)

            elif ( reg_analysis == 1):

                system_feasibility_data=reg_data/'land_feasibility_regional.xlsx'
                system_feasibility=xlrd.open_workbook(system_feasibility_data)

                if ( r == 'MF' ):
                    if (l == 'MRT'):

                        feas_sheet = system_feasibility.sheet_by_index(0)

                    elif (l == 'MRH'):

                        feas_sheet = system_feasibility.sheet_by_index(1)


                elif ( r == 'MM'):
                    if (l == 'MRT'):

                        feas_sheet = system_feasibility.sheet_by_index(2)

                    elif (l == 'MRH'):

                        feas_sheet = system_feasibility.sheet_by_index(3)

                elif (r == "NB"):
                    if (l == 'MRT'):

                        feas_sheet = system_feasibility.sheet_by_index(4)

                    elif (l == 'MRH'):

                        feas_sheet = system_feasibility.sheet_by_index(5)

                elif (r == "RW"):
                    if (l == 'MRT'):

                        feas_sheet = system_feasibility.sheet_by_index(6)

                    elif (l == 'MRH'):

                        feas_sheet = system_feasibility.sheet_by_index(7)


            # Load parameters from excel
            density={} # lives density per pixel (head/pixel)
            dgl_available={} #  grassland available as percentage of pixel (percentage)
            frac_pop={} # the fraction of livestock in a pixel relative to total in the lps (fraction)


            u  = feas_sheet.nrows # number of rows in excel sheet

            total_popn  = 0 # total population of cattle in lps (head)

            for i in range(1,u): 
                total_popn += feas_sheet.cell(i,2).value

            for i in range(1,u): 

                dgl_available[i]=feas_sheet.cell(i,1).value
                frac_pop[i]=feas_sheet.cell(i,2).value/total_popn


            # total expansion of primary feed area for this scenario (ha/lps)
            total_expansion[l]=out['v5_6_Total_feed_primary_area_expansion']
            total_grslnd_expansion[l]=out['v5_6_Total_grasslands_area_expansion']

            if (consequential == 1 and scenario_parameters[(ss,'beef_defor_Domestic',s)] == 1 ):
                total_expansion[l] = total_expansion[l]  #+ out['v1_3_Avoided_Beef_land_use_Total_ha']

            total_area_growth={} # amount of new land needed for conversion per pixel (sq km/unit)
            gl_area_growth={}

            for i in range(1,u):

                # amount of new land needed for primary feed expansion (weighted by cattle pop'n) (sq km)
                # pixel is 10km x 10km = 100 sq km 
                total_area_growth[i]=frac_pop[i]*total_expansion[l]*.01   # (sq km per pixel)
                gl_area_growth[i]=frac_pop[i]*total_grslnd_expansion[l]*.01   # (sq km per pixel)

                # if amount of new land needed (ha/100 sq km) is less than available grassland to be utilized (ha) (feas_sheet.cell_value(i,1))

                if (total_area_growth[i] <= dgl_available[i]):

                    out['v6_0_total_grasslands_now_utilized(ha)']+=total_area_growth[i]

                # if amount of new land needed is greater than available grassland to be utilized
                elif (total_area_growth[i] > dgl_available[i]):

                    out['v6_0_total_native_ecosystems_converted(ha)']+=(total_area_growth[i]-dgl_available[i])
                    out['v6_0_total_grasslands_now_utilized(ha)']+=dgl_available[i]
                    out['v6_0_Units_limited']+=1

                if (gl_area_growth[i] <= dgl_available[i]):

                    out['v6_0_total_direct_grasslands_now_utilized(ha)']+=total_area_growth[i]

                # if amount of new land needed is greater than available grassland to be utilized
                elif (gl_area_growth[i] > dgl_available[i]):

                    out['v6_0_total_direct_native_ecosystems_converted(ha)']+=(gl_area_growth[i]-dgl_available[i])
                    out['v6_0_total_direct_grasslands_now_utilized(ha)']+=dgl_available[i]
                    out['v6_0_Units_direct_limited']+=1

            fraction_feed_primary_area_expansion_converting_native_ecosystems=(
                out['v6_0_total_native_ecosystems_converted(ha)']
                /(out['v6_0_total_grasslands_now_utilized(ha)']
                  +out['v6_0_total_native_ecosystems_converted(ha)']))

            fraction_direct_grassland_area_expansion_converting_native_ecosystems=(
                out['v6_0_total_direct_native_ecosystems_converted(ha)']/
                (out['v6_0_total_direct_grasslands_now_utilized(ha)']
                 +out['v6_0_total_direct_native_ecosystems_converted(ha)']))

            out['v6_0_6_grassland_expansion_emissions_Mg_CO2eq_per_lps']=(
            alloc_factor*
            fraction_feed_primary_area_expansion_converting_native_ecosystems*(44/12)*
            out['v5_6_Total_feed_primary_area_expansion']*(
            luc_emission_coefficient_total_area)*(timestep/amort_period)/timestep)

            out['v6_0_6_grassland_expansion_direct_emissions_Mg_CO2eq_per_lps']=(
            alloc_factor*fraction_direct_grassland_area_expansion_converting_native_ecosystems*
            (44/12)*out['v5_6_Total_grasslands_area_expansion']*
            (luc_emission_coefficient_total_area)*(timestep/amort_period)/timestep)

            if (b == 'local'):
                out['v6_1_6_grassland_expansion_local_emissions_Mg_CO2eq_per_lps']=(
                alloc_factor*
                fraction_feed_primary_area_expansion_converting_native_ecosystems*(44/12)*
                out['v5_6_Total_feed_primary_area_expansion']*(
                luc_emission_coefficient_total_area)*(timestep/amort_period)/timestep)

                out['v6_1_6_grassland_expansion_direct_local_emissions_Mg_CO2eq_per_lps']=(
                alloc_factor*fraction_direct_grassland_area_expansion_converting_native_ecosystems*
                (44/12)*out['v5_6_Total_grasslands_area_expansion']*
                (luc_emission_coefficient_total_area)*(timestep/amort_period)/timestep)

                out['v6_2_6_grassland_expansion_direct_improved_emissions_Mg_CO2eq_per_lps']=0
                out['v6_2_6_grassland_expansion_improved_emissions_Mg_CO2eq_per_lps']=0

            elif (b =='improved'):

                out['v6_2_6_grassland_expansion_improved_emissions_Mg_CO2eq_per_lps']=(
                alloc_factor*
                fraction_feed_primary_area_expansion_converting_native_ecosystems*(44/12)*
                out['v5_6_Total_feed_primary_area_expansion']*(
                luc_emission_coefficient_total_area)*(timestep/amort_period)/timestep)

                out['v6_2_6_grassland_expansion_direct_improved_emissions_Mg_CO2eq_per_lps']=(
                alloc_factor*fraction_direct_grassland_area_expansion_converting_native_ecosystems*
                (44/12)*out['v5_6_Total_grasslands_area_expansion']*
                (luc_emission_coefficient_total_area)*(timestep/amort_period)/timestep)

                out['v6_1_6_grassland_expansion_direct_local_emissions_Mg_CO2eq_per_lps']=0
                out['v6_1_6_grassland_expansion_local_emissions_Mg_CO2eq_per_lps']=0


        else:
            fraction_feed_primary_area_expansion_converting_native_ecosystems = 1
            fraction_direct_grassland_area_expansion_converting_native_ecosystems = 1
            out['v6_0_6_grassland_expansion_emissions_Mg_CO2eq_per_lps']=alloc_factor*(44/12)*out['v5_6_Total_feed_primary_area_expansion']*(luc_emission_coefficient_total_area)*(timestep/amort_period)/timestep
            out['v6_0_6_grassland_expansion_direct_emissions_Mg_CO2eq_per_lps']=alloc_factor*(44/12)*out['v5_6_Total_grasslands_area_expansion']*(luc_emission_coefficient_total_area)*(timestep/amort_period)/timestep

        out['v6_0_Frac_Primary_area_expansion_converting_native_ecosystems'] = fraction_feed_primary_area_expansion_converting_native_ecosystems
        out['v6_0_Frac_Direct_Grassland_area_expansion_converting_native_ecosystems'] = fraction_direct_grassland_area_expansion_converting_native_ecosystems

        # v7 - emissions intensities
        out['v7_1_Enteric_CH4_intensity_kg_CO2eq_per_kg_milk_per_lps']=(1000*out['v6_0_1_Enteric_CH4_aggregate_Mg_CO2eq_per_lps']/out['v1_2_Milk_yield_total_kg_yr'] )
        out['v7_2_Manure_CH4_intensity_kg_CO2eq_per_kg_milk_per_lps']=(1000*out['v6_0_2_Manure_CH4_aggregate_Mg_CO2eq_per_lps']/out['v1_2_Milk_yield_total_kg_yr'])
        out['v7_3_Manure_N2O_intensity_kg_CO2eq_per_kg_milk_per_lps']=(1000*out['v6_0_3_Manure_N2O_aggregate_Mg_CO2eq_per_lps']/out['v1_2_Milk_yield_total_kg_yr'])
        out['v7_4_Soil_N2O_cropland_intensity_kg_CO2eq_per_kg_milk_per_lps']=(1000*out['v6_0_4_Soil_N2O_cropland_aggregate_Mg_CO2eq_per_lps']/out['v1_2_Milk_yield_total_kg_yr'])
        out['v7_4_Soil_N2O_grassland_intensity_kg_CO2eq_per_kg_milk_per_lps']= (1000*out['v6_0_4_Soil_N2O_grassland_aggregate_Mg_CO2eq_per_lps']/out['v1_2_Milk_yield_total_kg_yr'])
        out['v7_4_Feed_CO2_intensity_kg_CO2eq_per_kg_milk_per_lps']=(1000*out['v6_0_4_2_Feed_CO2_aggregate_Mg_CO2eq_per_lps']/out['v1_2_Milk_yield_total_kg_yr'])

        out['v7_5_Direct emissions intensity_kg_CO2eq_per_kg_milk_per_lps']=(
            out['v7_1_Enteric_CH4_intensity_kg_CO2eq_per_kg_milk_per_lps']+
            out['v7_2_Manure_CH4_intensity_kg_CO2eq_per_kg_milk_per_lps']+
            out['v7_3_Manure_N2O_intensity_kg_CO2eq_per_kg_milk_per_lps']+
            out['v7_4_Soil_N2O_cropland_intensity_kg_CO2eq_per_kg_milk_per_lps']+
            out['v7_4_Soil_N2O_grassland_intensity_kg_CO2eq_per_kg_milk_per_lps']+
            out['v7_4_Feed_CO2_intensity_kg_CO2eq_per_kg_milk_per_lps'])


        # luc emissions
        out['v7_6_grassland_direct_expansion_emissions_intensity_kg_CO2eq_per_kg_milk_per_lps']=(1000*out['v6_0_6_grassland_expansion_direct_emissions_Mg_CO2eq_per_lps']/out['v1_2_Milk_yield_total_kg_yr'])

        out['v7_6_grassland_expansion_emissions_intensity_kg_CO2eq_per_kg_milk_per_lps']=(1000*out['v6_0_6_grassland_expansion_emissions_Mg_CO2eq_per_lps']/out['v1_2_Milk_yield_total_kg_yr'])
        out['v7_8_cropland_expansion_emissions_intensity_kg_CO2eq_per_kg_milk_per_lps']=(1000*out['v6_0_8_cropland_expansion_emissions_Mg_CO2eq_per_lps']/out['v1_2_Milk_yield_total_kg_yr'])
        out['v7_9_Indirect_emissions_intensity_kg_CO2eq_per_kg_milk_per_lps']=(1000*(out['v6_0_6_grassland_expansion_emissions_Mg_CO2eq_per_lps']+out['v6_0_8_cropland_expansion_emissions_Mg_CO2eq_per_lps'])/out['v1_2_Milk_yield_total_kg_yr'])
        out['v7_5_Total emissions intensity_kg_CO2eq_per_kg_milk_per_lps']=(out['v7_5_Direct emissions intensity_kg_CO2eq_per_kg_milk_per_lps']+out['v7_9_Indirect_emissions_intensity_kg_CO2eq_per_kg_milk_per_lps'])

        # total
        out['v7_10_Emissions_total_Mg_CO2eq_per_lps']=(
            out['v6_5_Direct_emissions_aggregate_Mg_CO2eq_per_lps']+
            out['v6_0_6_grassland_expansion_emissions_Mg_CO2eq_per_lps']+
            out['v6_0_8_cropland_expansion_emissions_Mg_CO2eq_per_lps'])

        # per tlu
        out['v7_11_Enteric_CH4_intensity_kg_CO2eq_per_TLU_per_lps']=(1000*out['v6_0_1_Enteric_CH4_aggregate_Mg_CO2eq_per_lps']/out['v8_6_herd_total_TLU_per_lps'])
        out['v7_12_Manure_CH4_intensity_kg_CO2eq_per_TLU_per_lps']=(1000*out['v6_0_2_Manure_CH4_aggregate_Mg_CO2eq_per_lps']/out['v8_6_herd_total_TLU_per_lps'])
        out['v7_13_Manure_N2O_intensity_kg_CO2eq_per_TLU_per_lps']=(1000*out['v6_0_3_Manure_N2O_aggregate_Mg_CO2eq_per_lps']/out['v8_6_herd_total_TLU_per_lps'])
        out['v7_14_Soil_N2O_cropland_intensity_kg_CO2eq_per_TLU_per_lps']=(1000*out['v6_0_4_Soil_N2O_cropland_aggregate_Mg_CO2eq_per_lps']/out['v8_6_herd_total_TLU_per_lps'])
        out['v7_14_Soil_N2O_grassland_intensity_kg_CO2eq_per_TLU_per_lps']=(1000*out['v6_0_4_Soil_N2O_grassland_aggregate_Mg_CO2eq_per_lps']/out['v8_6_herd_total_TLU_per_lps'])
        out['v7_14_Feed_CO2_intensity_kg_CO2eq_per_TLU_per_lps']=(1000*out['v6_0_4_2_Feed_CO2_aggregate_Mg_CO2eq_per_lps']/out['v8_6_herd_total_TLU_per_lps'])


        out['v7_16_grassland_expansion_emissions_intensity_kg_CO2eq_per_TLU_per_lps']=(1000*out['v6_0_6_grassland_expansion_emissions_Mg_CO2eq_per_lps']/out['v8_6_herd_total_TLU_per_lps'])
        out['v7_16_grassland_expansion_direct_emissions_intensity_kg_CO2eq_per_TLU_per_lps']=(1000*out['v6_0_6_grassland_expansion_direct_emissions_Mg_CO2eq_per_lps']/out['v8_6_herd_total_TLU_per_lps'])

        out['v7_18_cropland_expansion_emissions_intensity_kg_CO2eq_per_TLU_per_lps']=(1000*out['v6_0_8_cropland_expansion_emissions_Mg_CO2eq_per_lps']/out['v8_6_herd_total_TLU_per_lps'])

        # multiply by 100 to convert to a percentage
        out['v1_5_Feed_intake_Mt_per_LPS']=(1/(1000*1000000))*(out['v3_2_6_Feed_intake_grass_kg_dm']+out['v3_2_7_Feed_intake_stover_kg_dm']+out['v3_2_8_Feed_intake_napier_kg_dm']+out['v3_2_9_Feed_intake_maize_bran_kg_dm']+out['v3_2_10_Feed_intake_sfsm_kg_dm']+out['v3_2_11_Feed_intake_pasture_kg_dm'])
        out['v3_2_1_Feed_intake_grass_(% dm)']=(100/(1000*1000000))*out['v3_2_6_Feed_intake_grass_kg_dm']/out['v1_5_Feed_intake_Mt_per_LPS']
        out['v3_2_2_Feed_intake_stover_(% dm)']=(100/(1000*1000000))*out['v3_2_7_Feed_intake_stover_kg_dm']/out['v1_5_Feed_intake_Mt_per_LPS']
        out['v3_2_3_Feed_intake_napier_(% dm)']=(100/(1000*1000000))*out['v3_2_8_Feed_intake_napier_kg_dm']/out['v1_5_Feed_intake_Mt_per_LPS']
        out['v3_2_3_Feed_intake_napier_hy_(% dm)']=(100/(1000*1000000))*out['v3_2_8_Feed_intake_napier_hy_kg_dm']/out['v1_5_Feed_intake_Mt_per_LPS']
        out['v3_2_4_Feed_intake_maize_bran_(% dm)']=(100/(1000*1000000))*out['v3_2_9_Feed_intake_maize_bran_kg_dm']/out['v1_5_Feed_intake_Mt_per_LPS']
        out['v3_2_5_Feed_intake_sfsm_(% dm)']=(100/(1000*1000000))*out['v3_2_10_Feed_intake_sfsm_kg_dm']/out['v1_5_Feed_intake_Mt_per_LPS']
        out['v3_2_6_Feed_intake_pasture_(% dm)']=(100/(1000*1000000))*out['v3_2_11_Feed_intake_pasture_kg_dm']/out['v1_5_Feed_intake_Mt_per_LPS']

        out['v3_3_1_Feed_intake_grass_kg_per_TLU']=(1/out['v8_6_herd_total_TLU_per_lps'])*out['v3_2_6_Feed_intake_grass_kg_dm']
        out['v3_3_2_Feed_intake_stover_kg_per_TLU']=(1/out['v8_6_herd_total_TLU_per_lps'])*out['v3_2_7_Feed_intake_stover_kg_dm']
        out['v3_3_3_Feed_intake_napier_kg_per_TLU']=(1/out['v8_6_herd_total_TLU_per_lps'])*out['v3_2_8_Feed_intake_napier_kg_dm']
        out['v3_3_3_Feed_intake_napier_hy_kg_per_TLU']=(1/out['v8_6_herd_total_TLU_per_lps'])*out['v3_2_8_Feed_intake_napier_hy_kg_dm']
        out['v3_3_4_Feed_intake_maize_bran_kg_per_TLU']=(1/out['v8_6_herd_total_TLU_per_lps'])*out['v3_2_9_Feed_intake_maize_bran_kg_dm']
        out['v3_3_5_Feed_intake_sfsm_kg_per_TLU']=(1/out['v8_6_herd_total_TLU_per_lps'])*out['v3_2_10_Feed_intake_sfsm_kg_dm']
        out['v3_3_6_Feed_intake_pasture_kg_per_TLU']=(1/out['v8_6_herd_total_TLU_per_lps'])*out['v3_2_11_Feed_intake_pasture_kg_dm']

        # v9 - land use
        out['v9_15_Grass_land_use_ha_per_TLU']=out['v9_4_1_Grass_area_ha_per_lps']/out['v8_6_herd_total_TLU_per_lps']
        out['v9_16_Stover_land_use_ha_per_TLU']=out['v9_4_2_Stover_area_ha_per_lps']/out['v8_6_herd_total_TLU_per_lps']
        out['v9_17_Napier_land_use_ha_per_TLU']=out['v9_4_3_Napier_area_ha_per_lps']/out['v8_6_herd_total_TLU_per_lps']
        out['v9_17_Napier_hy_land_use_ha_per_TLU']=out['v9_4_4_Napier_hy_area_ha_per_lps']/out['v8_6_herd_total_TLU_per_lps']
        out['v9_18_Sunflower_land_use_ha_per_TLU']=out['v9_4_5_Sunflower_area_ha_per_lps']/out['v8_6_herd_total_TLU_per_lps']
        out['v9_19_Maize_land_use_ha_per_TLU']=out['v9_4_6_Maize_area_ha_per_lps']/out['v8_6_herd_total_TLU_per_lps']
        out['v9_20_Pasture_land_use_ha_per_TLU']=out['v9_4_7_Pasture_area_ha_per_lps']/out['v8_6_herd_total_TLU_per_lps']

        # v1 - additional productivity results
        out['v1_3_Milk_productivity_kg_per_ha']= out['v1_2_Milk_yield_total_kg_yr']/(out['v9_4_8_Total_area_ha_per_lps'])
        out['v1_4_Land_footprint_Mha_per_LPS']=  (1/1000000)*(out['v9_4_8_Total_area_ha_per_lps'])
        out['v1_6_Feed_efficiency_kg_FPCM_per_Mg_feed_dm']=(1)*(1000/(1000*1000000))*out['v1_2_Milk_yield_total_kg_yr']/ out['v1_5_Feed_intake_Mt_per_LPS']


        # sort results alphanumerically
        sorted_list=sorted(out.keys(), key=lambda x:x.lower())

        results_ordered = {}

        for i in sorted_list:
            results_ordered[i]=out[i]

        out = {}
        out=results_ordered


        out['v_11_0_Emissions_total_Mg_CO2eq_per_lps'] = (out['v6_5_Direct_emissions_aggregate_Mg_CO2eq_per_lps'] + 
        (out['v6_0_8_cropland_expansion_emissions_Mg_CO2eq_per_lps'] +
         out['v6_0_6_grassland_expansion_emissions_Mg_CO2eq_per_lps']))

        if (('local' in breed) ):
            out['v_11_1_Direct_emissions_local_total_Mg_CO2eq_per_lps'] = out['v6_1_5_Direct_emissions_local_aggregate_Mg_CO2eq_per_lps'] 
        
        if (('improved' in breed)):
            out['v_11_2_Direct_emissions_improved_total_Mg_CO2eq_per_lps'] = out['v6_2_5_Direct_emissions_improved_aggregate_Mg_CO2eq_per_lps']       


        # Calculate aggregated values from MC simulations


        for MC in range(0,MC_sims):         
            for ss in subsectors:

                for b in breed:
                    if (herd[(ss,b,'total_herd')] ==0):
                        continue
                    for c in cohort:
                        
                        direct_emission_absolute_rd[MC] += herd[(ss,b,'total_herd')]*herd[(ss,b,c)]*(
                            enteric_CH4_rd[MC,ss,b,c]+
                            manure_CH4_rd[MC,ss,b,c]+
                            manure_N2O_rd[MC,ss,b,c]+
                            soil_N2O_rd[MC,ss,b,c]+
                            energy_use_CO2_rd[MC,ss,b,c])/1000
                        
                        direct_emission_absolute_breed_rd[MC,b] += herd[(ss,b,'total_herd')]*herd[(ss,b,c)]*(
                            enteric_CH4_rd[MC,ss,b,c]+
                            manure_CH4_rd[MC,ss,b,c]+
                            manure_N2O_rd[MC,ss,b,c]+
                            soil_N2O_rd[MC,ss,b,c]+
                            energy_use_CO2_rd[MC,ss,b,c])/1000


                grassland_area_expansion_emissions_rd[MC]+=grassland_area_expansion_emissions_rd_ss[MC,ss]*fraction_feed_primary_area_expansion_converting_native_ecosystems
                cropland_area_expansion_emissions_rd[MC]+=cropland_area_expansion_emissions_rd_ss[MC,ss]


            total_emission_absolute_rd[MC] = direct_emission_absolute_rd[MC] + (
                cropland_area_expansion_emissions_rd[MC]
                +grassland_area_expansion_emissions_rd[MC])

        total_ab_variation = 0
        direct_ab_variation_local = 0
        direct_ab_variation_improved = 0


        for MC in range(0,MC_sims):
            total_ab_variation += (total_emission_absolute_rd[MC] - out['v_11_0_Emissions_total_Mg_CO2eq_per_lps'])**2
            
            if (('local' in breed) ):
                direct_ab_variation_local += (
                    direct_emission_absolute_breed_rd[MC,'local']  - out['v_11_1_Direct_emissions_local_total_Mg_CO2eq_per_lps'])**2 
            if (('improved' in breed) ):
                direct_ab_variation_improved += (
                direct_emission_absolute_breed_rd[MC,'improved']  -out['v_11_2_Direct_emissions_improved_total_Mg_CO2eq_per_lps'])**2 

        # Standard deviation of estimated emissions 
        total_ab = math.sqrt(total_ab_variation/MC_sims)
        direct_ab_local = math.sqrt(direct_ab_variation_local/MC_sims)
        direct_ab_improved = math.sqrt(direct_ab_variation_improved /MC_sims)

        # Ninety five percent confidence interval of estimated emissions
        ninetyfivepci = 1.96*total_ab
        ninetyfivepci_loc = 1.96*direct_ab_local
        ninetyfivepci_imp = 1.96*direct_ab_improved

        # Store results in output data frame
        out['v_11_2_error_absolute_Mg']=ninetyfivepci
        out['v_11_2_error_intensity_kg']=(1000)*ninetyfivepci/out['v1_2_Milk_yield_total_kg_yr']
        out['v_11_2_error_per_animal_Mg_per_TLU']=ninetyfivepci/out['v8_6_herd_total_TLU_per_lps']

        
        if (reg_analysis == 1):
            
            if (('local' in breed)):
                out['v_11_2_error_absolute_local_Mg']=ninetyfivepci_loc
                out['v_11_2_error_intensity_local_kg']=(1000)*ninetyfivepci_loc/out['v1_2_Milk_yield_local_total_kg_yr'] 
                out['v_11_2_error_per_animal_local_Mg_per_TLU']=ninetyfivepci_loc/out['v8_7_herd_total_local_TLU_per_lps']

            if (('improved' in breed)):
                out['v_11_2_error_absolute_improved_Mg']=ninetyfivepci_imp
                out['v_11_2_error_intensity_improved_kg']=(1000)*ninetyfivepci_imp/out['v1_2_Milk_yield_improved_total_kg_yr'] 
                out['v_11_2_error_per_animal_improved_Mg_per_TLU']=ninetyfivepci_imp/out['v8_7_herd_total_improved_TLU_per_lps']

            
        '''  
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        GLOBAL MODEL RESULTS 
        
        This section sums results over all simulation units. 
        Note that this section only executes for the last simulation unit in the model run. 
        
        The results of this section are ordered alphanumerically, ranging from v_12_1 to v_12_8, which ensures these results columns 
        appear after the simulation unit specific results in the final results dataframe.  
        
        Note that:
        - the regional aggregation takes into account breeds separately (ie reports results separately for local and improved) in the case 
          that sub_sector_analysis == 1
        - the emissions accounting has functionality to consider emissions in a consequential LCA, reporting changes for a given scenario 
          relative to the base scenario
        
        
        The main sections are:
        
        v_12_1 -- Basic global parameters
        v_12_2 -- Aggregated GHG emissions
        v_12_3 -- Aggregated GHG emissions
        v_12_4 -- GHG emissions per TLU
        v_12_5 -- Herd sizes
        v_12_6 -- GHG emissions intensity   
        v_12_7 -- Feed intake 
        v_12_8 -- Land areas
        
        
        
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   
        '''
        
        # v_12_1 Basic global results variables
        out['v_12_1_Global_Milk_yield_Mg_yr']=0
        out['v_12_1_Global_Total_Emissions_Mg_yr']=0
        out['v_12_1_Global_Total_Land_Footprint_Mha'] =0
        out['v_12_1_Global_Beef_Displacement_Mg_per_year']=0
        out['v_12_1_Global_Total_Grasslands_Now_Utilized_ha']=0

        # Uncertainty 
        out['v_12_1_Error_absolute_Mg']=0
        out['v_12_1_Error_intensity'] =0
        out['v_12_1_Error_per_animal_kg_per_TLU'] =0

        out['v_12_1_Error_absolute_local_Mg']=0
        out['v_12_1_Error_intensity_local'] =0
        out['v_12_1_Error_per_animal_local_kg_per_TLU'] =0

        out['v_12_1_Error_absolute_improved_Mg']= 0
        out['v_12_1_Error_intensity_improved'] = 0
        out['v_12_1_Error_per_animal_improved_kg_per_TLU'] = 0

        out['v_12_1_Global_Milk_yield_Local_Mg_yr'] = 0
        out['v_12_1_Global_Milk_yield_Improved_Mg_yr'] = 0
        
        # v-12_2_1 Aggregated GHG emissions 
        out['v_12_2_1_1_Global_Enteric_CH4_aggregate_Mg_CO2eq_per_year'] = 0
        out['v_12_2_1_2_Global_Manure_CH4_aggregate_Mg_CO2eq_per_year'] = 0
        out['v_12_2_1_3_Global_Manure_N2O_aggregate_Mg_CO2eq_per_year'] = 0
        out['v_12_2_1_4_Global_Soil_N2O_cropland_aggregate_Mg_CO2eq_per_year'] = 0
        out['v_12_2_1_4_Global_Soil_N2O_grassland_aggregate_Mg_CO2eq_per_year'] = 0
        out['v_12_2_1_4_Global_Energy_Use_CO2_aggregate_Mg_CO2eq_per_year'] = 0
        out['v_12_2_1_5_Global_Cropland_Expansion_CO2_aggregate_Mg_CO2eq_per_year'] =0
        out['v_12_2_1_5_Global_Grassland_Expansion_CO2_aggregate_Mg_CO2eq_per_year'] =0
        out['v_12_2_1_5_Global_Grassland_Expansion_direct_CO2_aggregate_Mg_CO2eq_per_year'] =0
        out['v_12_2_1_5_Global_Total_LUC_CO2_aggregate_Mg_CO2eq_per_year'] =0
        out['v_12_2_1_5_Global_Dairy_Afforestation_CO2_aggregate_Mg_CO2eq_per_year']=0

        # v-12_2_2 Aggregated GHG emissions local herd
        out['v_12_2_2_1_Global_Enteric_CH4_local_aggregate_Mg_CO2eq_per_year'] =0
        out['v_12_2_2_2_Global_Manure_CH4_local_aggregate_Mg_CO2eq_per_year'] = 0
        out['v_12_2_2_3_Global_Manure_N2O_local_aggregate_Mg_CO2eq_per_year'] = 0
        out['v_12_2_2_4_Global_Soil_N2O_cropland_local_aggregate_Mg_CO2eq_per_year'] = 0
        out['v_12_2_2_4_Global_Soil_N2O_grassland_local_aggregate_Mg_CO2eq_per_year'] = 0
        out['v_12_2_2_4_Global_Energy_Use_CO2_local_aggregate_Mg_CO2eq_per_year'] = 0
        out['v_12_2_2_5_Global_Cropland_Expansion_CO2_local_aggregate_Mg_CO2eq_per_year'] =0
        out['v_12_2_2_5_Global_Grassland_Expansion_CO2_local_aggregate_Mg_CO2eq_per_year'] =0
        out['v_12_2_2_5_Global_Grassland_Expansion_direct_CO2_local_aggregate_Mg_CO2eq_per_year'] =0
        out['v_12_2_2_5_Global_Dairy_Afforestation_CO2_local_aggregate_Mg_CO2eq_per_year']=0

        # v-12_2_3 Aggregated GHG emissions improved herd
        out['v_12_2_3_1_Global_Enteric_CH4_improved_aggregate_Mg_CO2eq_per_year'] = 0
        out['v_12_2_3_2_Global_Manure_CH4_improved_aggregate_Mg_CO2eq_per_year'] = 0
        out['v_12_2_3_3_Global_Manure_N2O_improved_aggregate_Mg_CO2eq_per_year'] = 0
        out['v_12_2_3_4_Global_Soil_N2O_cropland_improved_aggregate_Mg_CO2eq_per_year'] = 0
        out['v_12_2_3_4_Global_Soil_N2O_grassland_improved_aggregate_Mg_CO2eq_per_year'] = 0
        out['v_12_2_3_4_Global_Energy_Use_CO2_improved_aggregate_Mg_CO2eq_per_year'] = 0
        out['v_12_2_3_5_Global_Cropland_Expansion_CO2_improved_aggregate_Mg_CO2eq_per_year'] = 0
        out['v_12_2_3_5_Global_Grassland_Expansion_CO2_improved_aggregate_Mg_CO2eq_per_year'] = 0
        out['v_12_2_3_5_Global_Grassland_Expansion_direct_CO2_improved_aggregate_Mg_CO2eq_per_year'] = 0
        out['v_12_2_3_5_Global_Dairy_Afforestation_CO2_improved_aggregate_Mg_CO2eq_per_year'] = 0
        
        # GHG emissions per TLU
        out['v_12_4_1_Global_Enteric_CH4_per_animal_Mg_CO2eq_per_TLU_per_year'] = 0
        out['v_12_4_2_Global_Manure_CH4_per_animal_Mg_CO2eq_per_TLU_per_year'] = 0
        out['v_12_4_3_Global_Manure_N2O_per_animal_Mg_CO2eq_per_TLU_per_year'] =  0
        out['v_12_4_4_Global_Soil_N2O_cropland_per_animal_Mg_CO2eq_per_TLU_per_year'] = 0
        out['v_12_4_4_Global_Soil_N2O_grassland_per_animal_Mg_CO2eq_per_TLU_per_year'] = 0
        out['v_12_4_4_Global_Energy_Use_CO2_per_animal_Mg_CO2eq_per_TLU_per_year'] = 0
        out['v_12_4_5_Global_Cropland_expansion_per_animal_Mg_CO2eq_per_TLU_per_year'] =  0
        out['v_12_4_5_Global_Grassland_expansion_per_animal_Mg_CO2eq_per_TLU_per_year'] = 0
        out['v_12_4_5_Global_Grassland_expansion_direct_per_animal_Mg_CO2eq_per_TLU_per_year'] = 0
        out['v_12_4_5_Global_Total_LUC_per_animal_Mg_CO2eq_per_TLU_per_year'] = 0

        # GHG emissions per local TLU
        out['v_12_4_1_1_Global_Enteric_CH4_local_per_animal_Mg_CO2eq_per_TLU_per_year'] = 0
        out['v_12_4_1_2_Global_Manure_CH4_local_per_animal_Mg_CO2eq_per_TLU_per_year'] = 0
        out['v_12_4_1_3_Global_Manure_N2O_local_per_animal_Mg_CO2eq_per_TLU_per_year'] =  0
        out['v_12_4_1_4_Global_Soil_N2O_cropland_local_per_animal_Mg_CO2eq_per_TLU_per_year'] = 0
        out['v_12_4_1_4_Global_Soil_N2O_grassland_local_per_animal_Mg_CO2eq_per_TLU_per_year'] = 0
        out['v_12_4_1_4_Global_Energy_Use_CO2_local_per_animal_Mg_CO2eq_per_TLU_per_year'] = 0
        out['v_12_4_1_5_Global_Cropland_expansion_local_per_animal_Mg_CO2eq_per_TLU_per_year'] =  0
        out['v_12_4_1_5_Global_Grassland_expansion_local_per_animal_Mg_CO2eq_per_TLU_per_year'] = 0
        out['v_12_4_1_5_Global_Grassland_expansion_direct_local_per_animal_Mg_CO2eq_per_TLU_per_year'] = 0
        out['v_12_4_1_5_Global_Afforestation_local_per_animal_Mg_CO2eq_per_TLU_per_year'] = 0

        # GHG emissions per improved TLU
        out['v_12_4_2_1_Global_Enteric_CH4_improved_per_animal_Mg_CO2eq_per_TLU_per_year'] = 0
        out['v_12_4_2_2_Global_Manure_CH4_improved_per_animal_Mg_CO2eq_per_TLU_per_year'] = 0
        out['v_12_4_2_3_Global_Manure_N2O_improved_per_animal_Mg_CO2eq_per_TLU_per_year'] =  0
        out['v_12_4_2_4_Global_Soil_N2O_improved_cropland_per_animal_Mg_CO2eq_per_TLU_per_year'] = 0
        out['v_12_4_2_4_Global_Soil_N2O_improved_grassland_per_animal_Mg_CO2eq_per_TLU_per_year'] = 0
        out['v_12_4_2_4_Global_Energy_Use_CO2_improved_per_animal_Mg_CO2eq_per_TLU_per_year'] = 0
        out['v_12_4_2_5_Global_Cropland_expansion_improved_per_animal_Mg_CO2eq_per_TLU_per_year'] =  0
        out['v_12_4_2_5_Global_Grassland_expansion_improved_per_animal_Mg_CO2eq_per_TLU_per_year'] = 0
        out['v_12_4_2_5_Global_Grassland_expansion_improved_direct_per_animal_Mg_CO2eq_per_TLU_per_year'] = 0
        out['v_12_4_2_5_Global_Afforestation_improved_per_animal_Mg_CO2eq_per_TLU_per_year'] = 0

        # v_12_5 Herd size
        out['v_12_5_1_herd_total_TLU'] = 0
        out['v_12_5_1_herd_total_TLU_Local'] = 0
        out['v_12_5_1_herd_total_TLU_Improved'] = 0
        out['v_12_5_2_herd_total_local'] = 0
        out['v_12_5_2_herd_total_improved'] = 0

        # v_12_6 - Emissions intensity   
        out['v_12_6_2_1_1_Global_Change_Enteric_CH4_intensity_kg_CO2eq_per_year'] = 0 
        out['v_12_6_2_1_2_Global_Change_Manure_CH4_intensity_kg_CO2eq_per_year'] = 0
        out['v_12_6_2_1_3_Global_Change_Manure_N2O_intensity_kg_CO2eq_per_year'] = 0
        out['v_12_6_2_1_4_Global_Change_Soil_N2O_cropland_intensity_kg_CO2eq_per_year'] = 0
        out['v_12_6_2_1_5_Global_Change_Soil_N2O_grassland_intensity_kg_CO2eq_per_year'] = 0
        out['v_12_6_2_1_6_Global_Change_Energy_Use_CO2_intensity_kg_CO2eq_per_year'] = 0
        out['v_12_6_2_1_7_Global_Change_Cropland_Expansion_CO2_intensity_kg_CO2eq_per_year'] = 0
        out['v_12_6_2_1_8_Global_Change_Grassland_Expansion_direct_CO2_intensity_kg_CO2eq_per_year'] = 0
        out['v_12_6_2_1_8_Global_Change_Total_LUC_CO2_intensity_kg_CO2eq_per_year'] = 0

        out['v_12_6_1_1_Global_Change_Enteric_CH4_local_intensity_kg_CO2eq_per_year'] = 0
        out['v_12_6_1_2_Global_Change_Manure_CH4_local_intensity_kg_CO2eq_per_year'] = 0
        out['v_12_6_1_3_Global_Change_Manure_N2O_local_intensity_kg_CO2eq_per_year'] = 0
        out['v_12_6_1_4_Global_Change_Soil_N2O_cropland_local_intensity_kg_CO2eq_per_year'] = 0
        out['v_12_6_1_5_Global_Change_Soil_N2O_grassland_local_intensity_kg_CO2eq_per_year'] = 0
        out['v_12_6_1_6_Global_Change_Energy_Use_CO2_local_intensity_kg_CO2eq_per_year'] = 0
        out['v_12_6_1_7_Global_Change_Cropland_Expansion_CO2_local_intensity_kg_CO2eq_per_year'] = 0
        out['v_12_6_1_8_Global_Change_Grassland_Expansion_direct_CO2_local_intensity_kg_CO2eq_per_year'] = 0

        out['v_12_6_2_1_Global_Change_Enteric_CH4_improved_intensity_kg_CO2eq_per_year'] = 0
        out['v_12_6_2_2_Global_Change_Manure_CH4_improved_intensity_kg_CO2eq_per_year'] = 0
        out['v_12_6_2_3_Global_Change_Manure_N2O_improved_intensity_kg_CO2eq_per_year'] = 0
        out['v_12_6_2_4_Global_Change_Soil_N2O_cropland_improved_intensity_kg_CO2eq_per_year'] = 0
        out['v_12_6_2_5_Global_Change_Soil_N2O_grassland_improved_intensity_kg_CO2eq_per_year'] = 0
        out['v_12_6_2_6_Global_Change_Energy_Use_CO2_improved_intensity_kg_CO2eq_per_year'] = 0
        out['v_12_6_2_7_Global_Change_Cropland_Expansion_CO2_improved_intensity_kg_CO2eq_per_year'] = 0
        out['v_12_6_2_8_Global_Change_Grassland_Expansion_direct_CO2_improved_intensity_kg_CO2eq_per_year'] = 0

        # change in emissions 

        out['v_12_6_1_1_Global_Change_Enteric_CH4_aggregate_Mg_CO2eq_per_year'] =0
        out['v_12_6_1_2_Global_Change_Manure_CH4_aggregate_Mg_CO2eq_per_year'] =0
        out['v_12_6_1_3_Global_Change_Manure_N2O_aggregate_Mg_CO2eq_per_year']=0
        out['v_12_6_1_4_Global_Change_Soil_N2O_cropland_aggregate_Mg_CO2eq_per_year'] =0
        out['v_12_6_1_4_Global_Change_Soil_N2O_grassland_aggregate_Mg_CO2eq_per_year']=0
        out['v_12_6_1_4_Global_Change_Energy_Use_CO2_aggregate_Mg_CO2eq_per_year'] =0
        out['v_12_6_1_5_Global_Change_Cropland_Expansion_CO2_aggregate_Mg_CO2eq_per_year']=0
        out['v_12_6_1_5_Global_Change_Grassland_Expansion_CO2_aggregate_Mg_CO2eq_per_year'] =0
        out['v_12_6_1_5_Global_Change_Grassland_Expansion_direct_CO2_aggregate_Mg_CO2eq_per_year'] =0
        out['v_12_6_1_7_Global_Change_Total_aggregate_Mg_CO2eq_per_year']=0


        # change in emissions as part of alca
        out['v_12_6_2_1_1_Global_Change_Enteric_CH4_intensity_kg_CO2eq_per_year'] =0
        out['v_12_6_2_1_2_Global_Change_Manure_CH4_intensity_kg_CO2eq_per_year'] =0
        out['v_12_6_2_1_3_Global_Change_Manure_N2O_intensity_kg_CO2eq_per_year'] =0
        out['v_12_6_2_1_4_Global_Change_Soil_N2O_cropland_intensity_kg_CO2eq_per_year'] =0
        out['v_12_6_2_1_5_Global_Change_Soil_N2O_grassland_intensity_kg_CO2eq_per_year'] =0
        out['v_12_6_2_1_6_Global_Change_Energy_Use_CO2_intensity_kg_CO2eq_per_year'] =0
        out['v_12_6_2_1_7_Global_Change_Cropland_Expansion_CO2_intensity_kg_CO2eq_per_year'] =0
        out['v_12_6_2_1_8_Global_Change_Grassland_Expansion_CO2_intensity_kg_CO2eq_per_year'] =0
        out['v_12_6_2_1_8_Global_Change_Grassland_Expansion_direct_CO2_intensity_kg_CO2eq_per_year'] =0


        # v_12_7_1 - feed intake Mg per year 
        out['v_12_7_1_1_Feed_intake_grass_Mg_per_year']  =0
        out['v_12_7_1_2_Feed_intake_stover_Mg_per_year']  =0
        out['v_12_7_1_3_Feed_intake_napier_hy_Mg_per_year']  =0
        out['v_12_7_1_4_Feed_intake_maize_bran_Mg_per_year']  =0
        out['v_12_7_1_5_Feed_intake_sfsm_Mg_per_year']  =0
        out['v_12_7_1_6_Feed_intake_pasture_Mg_per_year']  =0
        out['v_12_7_1_7_Feed_intake_napier_Mg_per_year'] =0

        # v_12_7_1 - feed intake kg DM per TLU per year
        out['v_12_7_2_1_Feed_intake_grass_kg_dm_per_TLU']  = 0
        out['v_12_7_2_2_Feed_intake_stover_kg_dm_per_TLU']  = 0
        out['v_12_7_2_3_Feed_intake_napier_hy_kg_dm_per_TLU']  = 0
        out['v_12_7_2_4_Feed_intake_maize_bran_kg_dm_per_TLU']  = 0
        out['v_12_7_2_5_Feed_intake_sfsm_kg_dm_per_TLU']  = 0
        out['v_12_7_2_6_Feed_intake_pasture_kg_dm_per_TLU']  = 0
        out['v_12_7_2_7_Feed_intake_napier_kg_dm_per_TLU']  = 0

        # v_12_8_1 land area total (ha)
        out['v_12_8_1_Grass_area_Total_ha']  =0
        out['v_12_8_1_Feed_area_Total_ha']  =0
        out['v_12_8_1_Crop_primary_Total_ha'] =0 

        # v_12_8_2 land area per feed category (ha)
        out['v_12_8_2_1_Grass_area_ha']  =0
        out['v_12_8_2_2_Stover_area_ha']  =0
        out['v_12_8_2_3_Napier_area_ha']  =0
        out['v_12_8_2_4_Napier_hy_area_ha']  =0
        out['v_12_8_2_5_Sunflower_area_ha']  =0
        out['v_12_8_2_6_Maize_area_ha']  =0
        out['v_12_8_2_7_Pasture_area_ha']  =0
        
        # v_12_8_3 land area per feed category (ha per tlu)
        out['v_12_8_3_1_Grass_area_ha_per_TLU']  =0
        out['v_12_8_3_2_Stover_area_ha_per_TLU']  =0
        out['v_12_8_3_3_Napier_area_ha_per_TLU']  =0 
        out['v_12_8_3_4_Napier_hy_area_ha_per_TLU']  =0 
        out['v_12_8_3_5_Sunflower_area_ha_per_TLU']  =0 
        out['v_12_8_3_6_Maize_area_ha_per_TLU']  = 0 
        out['v_12_8_3_7_Pasture_area_ha_per_TLU']  =0 

        frac_prod={}

        # If the current simulation unit is non-final, add results from results dictionary to aggregated variables
        
        if ( (reg_analysis  == 1)  & (l == lps[len(lps)-1]) & (r == reg[len(reg)-1]) ):
            
            for systems in lps:
                for regions in reg:
                  
                        if (  ((systems == lps[0]) or (regions < reg[len(reg)-1])) and ((len(lps) != 1))  ):

                            out['v_12_1_Global_Milk_yield_Mg_yr'] += res[(s,systems,regions)]['v1_2_Milk_yield_total_kg_yr']/1000
                            out['v_12_1_Global_Total_Emissions_Mg_yr'] += (
                                res[(s,systems,regions)]['v_11_0_Emissions_total_Mg_CO2eq_per_lps'])
                            out['v_12_1_Global_Total_Land_Footprint_Mha'] += (res[(s,systems,regions)]['v1_4_Land_footprint_Mha_per_LPS'])

                            # milk yield local, improved
                            out['v_12_1_Global_Milk_yield_Local_Mg_yr'] += res[(s,systems,regions)]['v1_2_Milk_yield_local_total_kg_yr']/1000
                            out['v_12_1_Global_Milk_yield_Improved_Mg_yr'] += (
                                res[(s,systems,regions)]['v1_2_Milk_yield_improved_total_kg_yr'] 
                                / 1000)

                            if (check_feasibility == 1)  :
                                out['v_12_1_Global_Total_Grasslands_Now_Utilized_ha'] += (
                                100*res[(s,systems,regions)]['v6_0_total_grasslands_now_utilized(ha)'] )


                            for ss in subsectors:    
                                for b in breed:
                                    out['v_12_1_Global_Beef_Displacement_Mg_per_year'] +=  (1/1000)*(
                                        res[(s,systems,regions)]['v1_3_Avoided_Beef_kg_per_year_'+str(ss)+'_'+str(b)])
                                    

                            out['v_12_2_1_1_Global_Enteric_CH4_aggregate_Mg_CO2eq_per_year'] += (
                                res[(s,systems,regions)]['v6_0_1_Enteric_CH4_aggregate_Mg_CO2eq_per_lps'])
                            out['v_12_2_1_2_Global_Manure_CH4_aggregate_Mg_CO2eq_per_year'] += (
                                res[(s,systems,regions)]['v6_0_2_Manure_CH4_aggregate_Mg_CO2eq_per_lps'])
                            out['v_12_2_1_3_Global_Manure_N2O_aggregate_Mg_CO2eq_per_year'] += (
                                res[(s,systems,regions)]['v6_0_3_Manure_N2O_aggregate_Mg_CO2eq_per_lps'])
                            out['v_12_2_1_4_Global_Soil_N2O_cropland_aggregate_Mg_CO2eq_per_year'] += (
                                res[(s,systems,regions)]['v6_0_4_Soil_N2O_cropland_aggregate_Mg_CO2eq_per_lps'])
                            out['v_12_2_1_4_Global_Soil_N2O_grassland_aggregate_Mg_CO2eq_per_year'] += (
                                res[(s,systems,regions)]['v6_0_4_Soil_N2O_grassland_aggregate_Mg_CO2eq_per_lps'])
                            out['v_12_2_1_4_Global_Energy_Use_CO2_aggregate_Mg_CO2eq_per_year'] += (
                                res[(s,systems,regions)]['v6_0_4_2_Feed_CO2_aggregate_Mg_CO2eq_per_lps'])
                            out['v_12_2_1_5_Global_Cropland_Expansion_CO2_aggregate_Mg_CO2eq_per_year'] += (
                                res[(s,systems,regions)]['v6_0_8_cropland_expansion_emissions_Mg_CO2eq_per_lps'])
                            out['v_12_2_1_5_Global_Grassland_Expansion_CO2_aggregate_Mg_CO2eq_per_year'] += (
                                res[(s,systems,regions)]['v6_0_6_grassland_expansion_emissions_Mg_CO2eq_per_lps'])
                            out['v_12_2_1_5_Global_Grassland_Expansion_direct_CO2_aggregate_Mg_CO2eq_per_year'] += (
                                res[(s,systems,regions)]['v6_0_6_grassland_expansion_direct_emissions_Mg_CO2eq_per_lps'])


                            # local
                            out['v_12_2_2_1_Global_Enteric_CH4_local_aggregate_Mg_CO2eq_per_year'] += (
                                res[(s,systems,regions)]['v6_1_1_Enteric_CH4_local_aggregate_Mg_CO2eq_per_lps'] )
                            out['v_12_2_2_2_Global_Manure_CH4_local_aggregate_Mg_CO2eq_per_year'] += (
                                res[(s,systems,regions)]['v6_1_2_Manure_CH4_local_aggregate_Mg_CO2eq_per_lps'] )
                            out['v_12_2_2_3_Global_Manure_N2O_local_aggregate_Mg_CO2eq_per_year'] += (
                                res[(s,systems,regions)]['v6_1_3_Manure_N2O_local_aggregate_Mg_CO2eq_per_lps'])
                            out['v_12_2_2_4_Global_Soil_N2O_cropland_local_aggregate_Mg_CO2eq_per_year'] += (
                                res[(s,systems,regions)]['v6_1_4_Soil_N2O_cropland_local_aggregate_Mg_CO2eq_per_lps'])
                            out['v_12_2_2_4_Global_Soil_N2O_grassland_local_aggregate_Mg_CO2eq_per_year'] += (
                                res[(s,systems,regions)]['v6_1_4_Soil_N2O_grassland_local_aggregate_Mg_CO2eq_per_lps'])
                            out['v_12_2_2_4_Global_Energy_Use_CO2_local_aggregate_Mg_CO2eq_per_year'] += (
                                res[(s,systems,regions)]['v6_1_4_2_Feed_CO2_local_aggregate_Mg_CO2eq_per_lps'])
                            out['v_12_2_2_5_Global_Cropland_Expansion_CO2_local_aggregate_Mg_CO2eq_per_year'] += (   
                                res[(s,systems,regions)]['v6_1_8_cropland_expansion_local_emissions_Mg_CO2eq_per_lps'])
                            out['v_12_2_2_5_Global_Grassland_Expansion_CO2_local_aggregate_Mg_CO2eq_per_year'] += ( 
                                res[(s,systems,regions)]['v6_1_6_grassland_expansion_local_emissions_Mg_CO2eq_per_lps'] )


                            # improved
                            out['v_12_2_3_1_Global_Enteric_CH4_improved_aggregate_Mg_CO2eq_per_year'] += (
                                res[(s,systems,regions)]['v6_2_1_Enteric_CH4_improved_aggregate_Mg_CO2eq_per_lps'] )
                            out['v_12_2_3_2_Global_Manure_CH4_improved_aggregate_Mg_CO2eq_per_year'] += (
                                res[(s,systems,regions)]['v6_2_2_Manure_CH4_improved_aggregate_Mg_CO2eq_per_lps'])
                            out['v_12_2_3_3_Global_Manure_N2O_improved_aggregate_Mg_CO2eq_per_year'] += (
                                res[(s,systems,regions)]['v6_2_3_Manure_N2O_improved_aggregate_Mg_CO2eq_per_lps'])
                            out['v_12_2_3_4_Global_Soil_N2O_cropland_improved_aggregate_Mg_CO2eq_per_year'] += (
                                res[(s,systems,regions)]['v6_2_4_Soil_N2O_cropland_improved_aggregate_Mg_CO2eq_per_lps'])
                            out['v_12_2_3_4_Global_Soil_N2O_grassland_improved_aggregate_Mg_CO2eq_per_year'] += (
                                res[(s,systems,regions)]['v6_2_4_Soil_N2O_grassland_improved_aggregate_Mg_CO2eq_per_lps'])
                            out['v_12_2_3_4_Global_Energy_Use_CO2_improved_aggregate_Mg_CO2eq_per_year'] += (
                                res[(s,systems,regions)]['v6_2_4_2_Feed_CO2_improved_aggregate_Mg_CO2eq_per_lps'])
                            out['v_12_2_3_5_Global_Cropland_Expansion_CO2_improved_aggregate_Mg_CO2eq_per_year'] += (  
                                res[(s,systems,regions)]['v6_2_8_cropland_expansion_improved_emissions_Mg_CO2eq_per_lps'])
                            out['v_12_2_3_5_Global_Grassland_Expansion_CO2_improved_aggregate_Mg_CO2eq_per_year'] += ( 
                                res[(s,systems,regions)]['v6_2_6_grassland_expansion_improved_emissions_Mg_CO2eq_per_lps'])
                            out['v_12_2_3_5_Global_Dairy_Afforestation_CO2_improved_aggregate_Mg_CO2eq_per_year'] += (
                                res[(s,systems,regions)]['v6_2_9_afforestation_improved_emissions_Mg_CO2eq_per_lps'] )

                            out['v_12_1_Error_absolute_Mg'] +=  res[(s,systems,regions)]['v_11_2_error_absolute_Mg']
                            
                            if (ss_analysis  == 0):
                                out['v_12_1_Error_absolute_local_Mg'] +=  res[(s,systems,regions)]['v_11_2_error_absolute_local_Mg']
                                out['v_12_1_Error_absolute_improved_Mg'] +=  res[(s,systems,regions)]['v_11_2_error_absolute_improved_Mg']

                            out['v_12_5_1_herd_total_TLU']  += res[(s,systems,regions)]['v8_6_herd_total_TLU_per_lps']

                            if ('improved' in breed):
                                out['v_12_5_1_herd_total_TLU_Improved']  += (
                                    res[(s,systems,regions)]['v8_7_herd_total_improved_TLU_per_lps'] )

                            if ('local' in breed):
                                out['v_12_5_1_herd_total_TLU_Local']  += (
                                    res[(s,systems,regions)]['v8_7_herd_total_local_TLU_per_lps'] )

                          
                            for ss in subsectors:
                                for c in cohort:
                                    out['v_12_5_2_herd_total_local']  += ( 
                                        res[(s,systems,regions)]['v8_1_herd_'+str(ss)+'_local_'+str(c)] )
                                    out['v_12_5_2_herd_total_improved']  += (
                                        res[(s,systems,regions)]['v8_1_herd_'+str(ss)+'_improved_'+str(c)] )


                            out['v_12_8_1_Grass_area_Total_ha']  += ( res[(s,systems,regions)]['v9_2_1_Grasslands_area_ha_per_lps'] )
                            out['v_12_8_1_Feed_area_Total_ha'] += ( res[(s,systems,regions)]['v9_2_2_Feed_area_ha_per_lps'] )
                            out['v_12_8_1_Crop_primary_Total_ha'] += ( res[(s,systems,regions)]['v9_2_3_Crop_primary_area_ha_per_lps'] )


                            out['v_12_7_1_1_Feed_intake_grass_Mg_per_year']  += res[(s,systems,regions)]['v3_2_6_Feed_intake_grass_kg_dm']
                            
                            out['v_12_7_1_2_Feed_intake_stover_Mg_per_year']  += (
                                res[(s,systems,regions)]['v3_2_7_Feed_intake_stover_kg_dm'])
                            out['v_12_7_1_3_Feed_intake_napier_hy_Mg_per_year']  += (
                                res[(s,systems,regions)]['v3_2_8_Feed_intake_napier_hy_kg_dm'])
                            out['v_12_7_1_4_Feed_intake_maize_bran_Mg_per_year']  += (
                                res[(s,systems,regions)]['v3_2_9_Feed_intake_maize_bran_kg_dm'] )
                            out['v_12_7_1_5_Feed_intake_sfsm_Mg_per_year']  += (
                                res[(s,systems,regions)]['v3_2_10_Feed_intake_sfsm_kg_dm'])
                            out['v_12_7_1_6_Feed_intake_pasture_Mg_per_year']  += (
                                res[(s,systems,regions)]['v3_2_11_Feed_intake_pasture_kg_dm'])
                            out['v_12_7_1_7_Feed_intake_napier_Mg_per_year']  += (
                                res[(s,systems,regions)]['v3_2_8_Feed_intake_napier_kg_dm'])

                            out['v_12_8_2_1_Grass_area_ha']  += res[(s,systems,regions)]['v9_4_1_Grass_area_ha_per_lps']
                            out['v_12_8_2_2_Stover_area_ha']  += res[(s,systems,regions)]['v9_4_2_Stover_area_ha_per_lps']
                            out['v_12_8_2_3_Napier_area_ha']  += res[(s,systems,regions)]['v9_4_3_Napier_area_ha_per_lps']
                            out['v_12_8_2_4_Napier_hy_area_ha']  += res[(s,systems,regions)]['v9_4_4_Napier_hy_area_ha_per_lps']
                            out['v_12_8_2_5_Sunflower_area_ha']  += res[(s,systems,regions)]['v9_4_5_Sunflower_area_ha_per_lps']
                            out['v_12_8_2_6_Maize_area_ha']  += res[(s,systems,regions)]['v9_4_6_Maize_area_ha_per_lps']
                            out['v_12_8_2_7_Pasture_area_ha']  += res[(s,systems,regions)]['v9_4_7_Pasture_area_ha_per_lps']


                        # If the current simulation unit is final, add results from current simulation unit to aggregate results variables
                        elif (   ((systems == lps[(len(lps)-1)])  and (regions == reg[len(reg)-1])) or (len(lps) == 1) ):
                            
                            out['v_12_1_Global_Milk_yield_Mg_yr'] += out['v1_2_Milk_yield_total_kg_yr']/1000
                            out['v_12_1_Global_Total_Emissions_Mg_yr'] += out['v_11_0_Emissions_total_Mg_CO2eq_per_lps']
                            out['v_12_1_Global_Total_Land_Footprint_Mha'] += out['v1_4_Land_footprint_Mha_per_LPS']


                             # milk yield local, improved
                            out['v_12_1_Global_Milk_yield_Local_Mg_yr'] += out['v1_2_Milk_yield_local_total_kg_yr']/1000
                            out['v_12_1_Global_Milk_yield_Improved_Mg_yr'] += out['v1_2_Milk_yield_improved_total_kg_yr']/1000

                            if (check_feasibility == 1):
                                out['v_12_1_Global_Total_Grasslands_Now_Utilized_ha'] += 100*out['v6_0_total_grasslands_now_utilized(ha)']
                                                                                                              
                            # Both breeds
                            out['v_12_2_1_1_Global_Enteric_CH4_aggregate_Mg_CO2eq_per_year'] += (
                                out['v6_0_1_Enteric_CH4_aggregate_Mg_CO2eq_per_lps'] )
                            out['v_12_2_1_2_Global_Manure_CH4_aggregate_Mg_CO2eq_per_year'] += (
                                out['v6_0_2_Manure_CH4_aggregate_Mg_CO2eq_per_lps'] )
                            out['v_12_2_1_3_Global_Manure_N2O_aggregate_Mg_CO2eq_per_year'] += (
                                out['v6_0_3_Manure_N2O_aggregate_Mg_CO2eq_per_lps'])
                            out['v_12_2_1_4_Global_Soil_N2O_cropland_aggregate_Mg_CO2eq_per_year'] += (
                                out['v6_0_4_Soil_N2O_cropland_aggregate_Mg_CO2eq_per_lps'])
                            out['v_12_2_1_4_Global_Soil_N2O_grassland_aggregate_Mg_CO2eq_per_year'] += (
                                out['v6_0_4_Soil_N2O_grassland_aggregate_Mg_CO2eq_per_lps'])
                            out['v_12_2_1_4_Global_Energy_Use_CO2_aggregate_Mg_CO2eq_per_year'] += (
                                out['v6_0_4_2_Feed_CO2_aggregate_Mg_CO2eq_per_lps'])
                            out['v_12_2_1_5_Global_Cropland_Expansion_CO2_aggregate_Mg_CO2eq_per_year'] += (
                                out['v6_0_8_cropland_expansion_emissions_Mg_CO2eq_per_lps'] )
                            out['v_12_2_1_5_Global_Grassland_Expansion_CO2_aggregate_Mg_CO2eq_per_year'] += (
                                out['v6_0_6_grassland_expansion_emissions_Mg_CO2eq_per_lps'] )
                            out['v_12_2_1_5_Global_Grassland_Expansion_direct_CO2_aggregate_Mg_CO2eq_per_year'] += (
                                out['v6_0_6_grassland_expansion_direct_emissions_Mg_CO2eq_per_lps'])

                            out['v_12_2_1_5_Global_Total_LUC_CO2_aggregate_Mg_CO2eq_per_year'] = (
                                out['v_12_2_1_5_Global_Cropland_Expansion_CO2_aggregate_Mg_CO2eq_per_year']+
                            out['v_12_2_1_5_Global_Grassland_Expansion_CO2_aggregate_Mg_CO2eq_per_year'] )

                            # Local
                            out['v_12_2_2_1_Global_Enteric_CH4_local_aggregate_Mg_CO2eq_per_year'] += ( 
                                out['v6_1_1_Enteric_CH4_local_aggregate_Mg_CO2eq_per_lps'])
                            out['v_12_2_2_2_Global_Manure_CH4_local_aggregate_Mg_CO2eq_per_year'] += (
                                out['v6_1_2_Manure_CH4_local_aggregate_Mg_CO2eq_per_lps'] )
                            out['v_12_2_2_3_Global_Manure_N2O_local_aggregate_Mg_CO2eq_per_year'] += (
                                out['v6_1_3_Manure_N2O_local_aggregate_Mg_CO2eq_per_lps'])
                            out['v_12_2_2_4_Global_Soil_N2O_cropland_local_aggregate_Mg_CO2eq_per_year'] += (
                                out['v6_1_4_Soil_N2O_cropland_local_aggregate_Mg_CO2eq_per_lps'])
                            out['v_12_2_2_4_Global_Soil_N2O_grassland_local_aggregate_Mg_CO2eq_per_year'] += (
                                out['v6_1_4_Soil_N2O_grassland_local_aggregate_Mg_CO2eq_per_lps'])
                            out['v_12_2_2_4_Global_Energy_Use_CO2_local_aggregate_Mg_CO2eq_per_year'] += (
                                out['v6_1_4_2_Feed_CO2_local_aggregate_Mg_CO2eq_per_lps'])
                            out['v_12_2_2_5_Global_Cropland_Expansion_CO2_local_aggregate_Mg_CO2eq_per_year'] += (
                                out['v6_1_8_cropland_expansion_local_emissions_Mg_CO2eq_per_lps'])
                            out['v_12_2_2_5_Global_Grassland_Expansion_CO2_local_aggregate_Mg_CO2eq_per_year'] += (
                                out['v6_1_6_grassland_expansion_local_emissions_Mg_CO2eq_per_lps'])
                            out['v_12_2_2_5_Global_Grassland_Expansion_direct_CO2_local_aggregate_Mg_CO2eq_per_year'] += (
                                out['v6_1_6_grassland_expansion_direct_local_emissions_Mg_CO2eq_per_lps'])

                            # improved
                            out['v_12_2_3_1_Global_Enteric_CH4_improved_aggregate_Mg_CO2eq_per_year'] += (
                                out['v6_2_1_Enteric_CH4_improved_aggregate_Mg_CO2eq_per_lps'])
                            out['v_12_2_3_2_Global_Manure_CH4_improved_aggregate_Mg_CO2eq_per_year'] += (
                                out['v6_2_2_Manure_CH4_improved_aggregate_Mg_CO2eq_per_lps'] )
                            out['v_12_2_3_3_Global_Manure_N2O_improved_aggregate_Mg_CO2eq_per_year'] += (
                                out['v6_2_3_Manure_N2O_improved_aggregate_Mg_CO2eq_per_lps'])
                            out['v_12_2_3_4_Global_Soil_N2O_cropland_improved_aggregate_Mg_CO2eq_per_year'] += (
                                out['v6_2_4_Soil_N2O_cropland_improved_aggregate_Mg_CO2eq_per_lps'])
                            out['v_12_2_3_4_Global_Soil_N2O_grassland_improved_aggregate_Mg_CO2eq_per_year'] += (
                                out['v6_2_4_Soil_N2O_grassland_improved_aggregate_Mg_CO2eq_per_lps'])
                            out['v_12_2_3_4_Global_Energy_Use_CO2_improved_aggregate_Mg_CO2eq_per_year'] += (
                                out['v6_2_4_2_Feed_CO2_improved_aggregate_Mg_CO2eq_per_lps'])
                            out['v_12_2_3_5_Global_Cropland_Expansion_CO2_improved_aggregate_Mg_CO2eq_per_year'] += ( 
                                 out['v6_2_8_cropland_expansion_improved_emissions_Mg_CO2eq_per_lps'])
                            out['v_12_2_3_5_Global_Grassland_Expansion_CO2_improved_aggregate_Mg_CO2eq_per_year'] += (
                                out['v6_2_6_grassland_expansion_improved_emissions_Mg_CO2eq_per_lps'])
                            out['v_12_2_3_5_Global_Grassland_Expansion_direct_CO2_improved_aggregate_Mg_CO2eq_per_year'] += (
                                out['v6_2_6_grassland_expansion_direct_improved_emissions_Mg_CO2eq_per_lps'])
                            out['v_12_2_1_5_Global_Dairy_Afforestation_CO2_aggregate_Mg_CO2eq_per_year']+=(
                                out['v6_0_9_afforestation_emissions_Mg_CO2eq_per_lps'])

                            if (ss_analysis  == 0):
                                out['v_12_1_Error_absolute_Mg'] +=  out['v_11_2_error_absolute_Mg']
                                out['v_12_5_1_herd_total_TLU']  += out['v8_6_herd_total_TLU_per_lps']
                            
                            if ( ('local' in breed) ):
                                out['v_12_1_Error_absolute_local_Mg'] +=  out['v_11_2_error_absolute_local_Mg']
                                out['v_12_5_1_herd_total_TLU_Local']  += out['v8_7_herd_total_local_TLU_per_lps'] 
                                
                            if ( ('improved' in breed) ):
                                out['v_12_1_Error_absolute_improved_Mg'] +=  out['v_11_2_error_absolute_improved_Mg']
                                out['v_12_5_1_herd_total_TLU_Improved']  += out['v8_7_herd_total_improved_TLU_per_lps'] 


                            for b in breed:
                                for ss in subsectors:
                                    for c in cohort:
                                        out['v_12_5_2_herd_total_'+str(b)]  += out['v8_1_herd_'+str(ss)+'_'+str(b)+'_'+str(c)]

                            out['v_12_8_1_Grass_area_Total_ha']  += out['v9_2_1_Grasslands_area_ha_per_lps']
                            out['v_12_8_1_Feed_area_Total_ha'] += out['v9_2_2_Feed_area_ha_per_lps']
                            out['v_12_8_1_Crop_primary_Total_ha'] += out['v9_2_3_Crop_primary_area_ha_per_lps']


                            out['v_12_7_1_1_Feed_intake_grass_Mg_per_year']  += (1/1000)*out['v3_2_6_Feed_intake_grass_kg_dm']
                            out['v_12_7_1_2_Feed_intake_stover_Mg_per_year']  += (1/1000)*out['v3_2_7_Feed_intake_stover_kg_dm']
                            out['v_12_7_1_3_Feed_intake_napier_hy_Mg_per_year']  += (1/1000)*out['v3_2_8_Feed_intake_napier_hy_kg_dm']
                            out['v_12_7_1_4_Feed_intake_maize_bran_Mg_per_year']  += (1/1000)*out['v3_2_9_Feed_intake_maize_bran_kg_dm']
                            out['v_12_7_1_5_Feed_intake_sfsm_Mg_per_year']  += (1/1000)*out['v3_2_10_Feed_intake_sfsm_kg_dm']
                            out['v_12_7_1_6_Feed_intake_pasture_Mg_per_year']  += (1/1000)*out['v3_2_11_Feed_intake_pasture_kg_dm']
                            out['v_12_7_1_7_Feed_intake_napier_Mg_per_year']  += (1/1000)*out['v3_2_8_Feed_intake_napier_kg_dm']

                            out['v_12_8_2_1_Grass_area_ha']  += out['v9_4_1_Grass_area_ha_per_lps']
                            out['v_12_8_2_2_Stover_area_ha']  += out['v9_4_2_Stover_area_ha_per_lps']
                            out['v_12_8_2_3_Napier_area_ha']  += out['v9_4_3_Napier_area_ha_per_lps']
                            out['v_12_8_2_4_Napier_hy_area_ha']  += out['v9_4_4_Napier_hy_area_ha_per_lps']
                            out['v_12_8_2_5_Sunflower_area_ha']  += out['v9_4_5_Sunflower_area_ha_per_lps']
                            out['v_12_8_2_6_Maize_area_ha']  += out['v9_4_6_Maize_area_ha_per_lps']
                            out['v_12_8_2_7_Pasture_area_ha']  += out['v9_4_7_Pasture_area_ha_per_lps']


            # Aggregate total emissions intensity
            out['v_12_3_1_Global_Enteric_CH4_intensity_kg_CO2eq_per_year'] = (  
                out['v_12_2_1_1_Global_Enteric_CH4_aggregate_Mg_CO2eq_per_year'] /out['v_12_1_Global_Milk_yield_Mg_yr'] )
            out['v_12_3_2_Global_Manure_CH4_intensity_kg_CO2eq_per_year'] = (
                out['v_12_2_1_2_Global_Manure_CH4_aggregate_Mg_CO2eq_per_year'] /out['v_12_1_Global_Milk_yield_Mg_yr'] )
            out['v_12_3_3_Global_Manure_N2O_intensity_kg_CO2eq_per_year'] = (
                out['v_12_2_1_3_Global_Manure_N2O_aggregate_Mg_CO2eq_per_year'] /out['v_12_1_Global_Milk_yield_Mg_yr'] )
            out['v_12_3_4_Global_Soil_N2O_cropland_intensity_kg_CO2eq_per_year'] = (
                out['v_12_2_1_4_Global_Soil_N2O_cropland_aggregate_Mg_CO2eq_per_year']/out['v_12_1_Global_Milk_yield_Mg_yr'] )
            out['v_12_3_4_Global_Soil_N2O_grassland_intensity_kg_CO2eq_per_year'] =(
            out['v_12_2_1_4_Global_Soil_N2O_grassland_aggregate_Mg_CO2eq_per_year']/out['v_12_1_Global_Milk_yield_Mg_yr'] )
            out['v_12_3_4_Global_Energy_Use_CO2_intensity_kg_CO2eq_per_year'] = (
            out['v_12_2_1_4_Global_Energy_Use_CO2_aggregate_Mg_CO2eq_per_year']/out['v_12_1_Global_Milk_yield_Mg_yr'] )
            out['v_12_3_5_Global_Cropland_expansion_CO2_intensity_kg_CO2eq_per_year'] = (
                out['v_12_2_1_5_Global_Cropland_Expansion_CO2_aggregate_Mg_CO2eq_per_year']/out['v_12_1_Global_Milk_yield_Mg_yr'] )
            out['v_12_3_5_Global_Grassland_expansion_CO2_intensity_kg_CO2eq_per_year'] =(
            out['v_12_2_1_5_Global_Grassland_Expansion_CO2_aggregate_Mg_CO2eq_per_year']/out['v_12_1_Global_Milk_yield_Mg_yr'] )
            out['v_12_3_5_Global_Grassland_expansion_direct_CO2_intensity_kg_CO2eq_per_year'] = (
            out['v_12_2_1_5_Global_Grassland_Expansion_direct_CO2_aggregate_Mg_CO2eq_per_year']/out['v_12_1_Global_Milk_yield_Mg_yr'] )
            out['v_12_3_5_Global_Total_LUC_CO2_intensity_kg_CO2eq_per_year'] = (
            out['v_12_3_5_Global_Cropland_expansion_CO2_intensity_kg_CO2eq_per_year']+
            out['v_12_3_5_Global_Grassland_expansion_CO2_intensity_kg_CO2eq_per_year'])

            # intensity local
            if ('local' in breed):
                out['v_12_3_1_1_Global_Enteric_CH4_local_intensity_kg_CO2eq_per_year'] =  (
                    out['v_12_2_2_1_Global_Enteric_CH4_local_aggregate_Mg_CO2eq_per_year'] /out['v_12_1_Global_Milk_yield_Local_Mg_yr'] )
                out['v_12_3_1_2_Global_Manure_CH4_local_intensity_kg_CO2eq_per_year'] = (
                    out['v_12_2_2_2_Global_Manure_CH4_local_aggregate_Mg_CO2eq_per_year'] /out['v_12_1_Global_Milk_yield_Local_Mg_yr'] )
                out['v_12_3_1_3_Global_Manure_N2O_local_intensity_kg_CO2eq_per_year'] = (
                    out['v_12_2_2_3_Global_Manure_N2O_local_aggregate_Mg_CO2eq_per_year'] /out['v_12_1_Global_Milk_yield_Local_Mg_yr'] )
                out['v_12_3_1_4_Global_Soil_N2O_cropland_local_intensity_kg_CO2eq_per_year'] = (
                  out['v_12_2_2_4_Global_Soil_N2O_cropland_local_aggregate_Mg_CO2eq_per_year']/
                    out['v_12_1_Global_Milk_yield_Local_Mg_yr'])
                out['v_12_3_1_4_Global_Soil_N2O_grassland_local_intensity_kg_CO2eq_per_year'] = (
                    out['v_12_2_2_4_Global_Soil_N2O_grassland_local_aggregate_Mg_CO2eq_per_year']/
                    out['v_12_1_Global_Milk_yield_Local_Mg_yr'] )
                out['v_12_3_1_4_Global_Energy_Use_CO2_local_intensity_kg_CO2eq_per_year'] = (
                    out['v_12_2_2_4_Global_Energy_Use_CO2_local_aggregate_Mg_CO2eq_per_year']/
                    out['v_12_1_Global_Milk_yield_Local_Mg_yr'] )
                out['v_12_3_1_5_Global_Cropland_expansion_CO2_local_intensity_kg_CO2eq_per_year'] = (
                    out['v_12_2_2_5_Global_Cropland_Expansion_CO2_local_aggregate_Mg_CO2eq_per_year']/
                    out['v_12_1_Global_Milk_yield_Local_Mg_yr'] )
                out['v_12_3_1_5_Global_Grassland_expansion_CO2_local_intensity_kg_CO2eq_per_year'] = (
                    out['v_12_2_2_5_Global_Grassland_Expansion_CO2_local_aggregate_Mg_CO2eq_per_year']/
                    out['v_12_1_Global_Milk_yield_Local_Mg_yr'] )
                out['v_12_3_1_5_Global_Grassland_expansion_direct_CO2_local_intensity_kg_CO2eq_per_year'] = (
                  out['v_12_2_2_5_Global_Grassland_Expansion_direct_CO2_local_aggregate_Mg_CO2eq_per_year']/
                    out['v_12_1_Global_Milk_yield_Local_Mg_yr'] )
 

            # intensity improved
            #if (('improved' in breed) & (herd[(ss,'improved','herd_pop')] != 0)):
            if ('improved' in breed):
                out['v_12_3_2_1_Global_Enteric_CH4_improved_intensity_kg_CO2eq_per_year'] =  ( 
                out['v_12_2_3_1_Global_Enteric_CH4_improved_aggregate_Mg_CO2eq_per_year'] /
                    out['v_12_1_Global_Milk_yield_Improved_Mg_yr']  )                                                                        
                out['v_12_3_2_2_Global_Manure_CH4_improved_intensity_kg_CO2eq_per_year'] = (
                    out['v_12_2_3_2_Global_Manure_CH4_improved_aggregate_Mg_CO2eq_per_year'] / 
                    out['v_12_1_Global_Milk_yield_Improved_Mg_yr']  )
                out['v_12_3_2_3_Global_Manure_N2O_improved_intensity_kg_CO2eq_per_year'] = (
                    out['v_12_2_3_3_Global_Manure_N2O_improved_aggregate_Mg_CO2eq_per_year'] / 
                     out['v_12_1_Global_Milk_yield_Improved_Mg_yr']   )                                                                  
                out['v_12_3_2_4_Global_Soil_N2O_cropland_improved_intensity_kg_CO2eq_per_year'] = (
                    out['v_12_2_3_4_Global_Soil_N2O_cropland_improved_aggregate_Mg_CO2eq_per_year']/
                    out['v_12_1_Global_Milk_yield_Improved_Mg_yr']   )
                out['v_12_3_2_4_Global_Soil_N2O_grassland_improved_intensity_kg_CO2eq_per_year'] = (
                  out['v_12_2_3_4_Global_Soil_N2O_grassland_improved_aggregate_Mg_CO2eq_per_year']/
                    out['v_12_1_Global_Milk_yield_Improved_Mg_yr']  )
                out['v_12_3_2_4_Global_Energy_Use_CO2_improved_intensity_kg_CO2eq_per_year'] = (
                    out['v_12_2_3_4_Global_Energy_Use_CO2_improved_aggregate_Mg_CO2eq_per_year']/
                    out['v_12_1_Global_Milk_yield_Improved_Mg_yr']  )
                out['v_12_3_2_5_Global_Cropland_expansion_CO2_improved_intensity_kg_CO2eq_per_year'] = (
                 out['v_12_2_3_5_Global_Cropland_Expansion_CO2_improved_aggregate_Mg_CO2eq_per_year']/ 
                    out['v_12_1_Global_Milk_yield_Improved_Mg_yr'] )                                                                       
                out['v_12_3_2_5_Global_Grassland_expansion_CO2_improved_intensity_kg_CO2eq_per_year'] = (
                    out['v_12_2_3_5_Global_Grassland_Expansion_CO2_improved_aggregate_Mg_CO2eq_per_year']/
                    out['v_12_1_Global_Milk_yield_Improved_Mg_yr']   )
                out['v_12_3_2_5_Global_Grassland_expansion_direct_CO2_improved_intensity_kg_CO2eq_per_year'] = (
                    out['v_12_2_3_5_Global_Grassland_Expansion_direct_CO2_improved_aggregate_Mg_CO2eq_per_year']/
                    out['v_12_1_Global_Milk_yield_Improved_Mg_yr']  )


            # total per animal unit
            out['v_12_4_1_Global_Enteric_CH4_per_animal_Mg_CO2eq_per_TLU_per_year'] = (     
                out['v_12_2_1_1_Global_Enteric_CH4_aggregate_Mg_CO2eq_per_year'] 
                /out['v_12_5_1_herd_total_TLU'] )
            out['v_12_4_2_Global_Manure_CH4_per_animal_Mg_CO2eq_per_TLU_per_year'] = (
                out['v_12_2_1_2_Global_Manure_CH4_aggregate_Mg_CO2eq_per_year'] /
                out['v_12_5_1_herd_total_TLU'] )
            out['v_12_4_3_Global_Manure_N2O_per_animal_Mg_CO2eq_per_TLU_per_year'] = (
                out['v_12_2_1_3_Global_Manure_N2O_aggregate_Mg_CO2eq_per_year'] /
                out['v_12_5_1_herd_total_TLU'] )
            out['v_12_4_4_Global_Soil_N2O_cropland_per_animal_Mg_CO2eq_per_TLU_per_year'] = (
                out['v_12_2_1_4_Global_Soil_N2O_cropland_aggregate_Mg_CO2eq_per_year']/
                out['v_12_5_1_herd_total_TLU'] )
            out['v_12_4_4_Global_Soil_N2O_grassland_per_animal_Mg_CO2eq_per_TLU_per_year'] = (
                out['v_12_2_1_4_Global_Soil_N2O_grassland_aggregate_Mg_CO2eq_per_year']/out['v_12_5_1_herd_total_TLU'] )
            out['v_12_4_4_Global_Energy_Use_CO2_per_animal_Mg_CO2eq_per_TLU_per_year'] = (
                out['v_12_2_1_4_Global_Energy_Use_CO2_aggregate_Mg_CO2eq_per_year']/out['v_12_5_1_herd_total_TLU'] )
            out['v_12_4_5_Global_Cropland_expansion_per_animal_Mg_CO2eq_per_TLU_per_year'] = ( 
                out['v_12_2_1_5_Global_Cropland_Expansion_CO2_aggregate_Mg_CO2eq_per_year']/out['v_12_5_1_herd_total_TLU']  )
            out['v_12_4_5_Global_Grassland_expansion_per_animal_Mg_CO2eq_per_TLU_per_year'] = (
                out['v_12_2_1_5_Global_Grassland_Expansion_CO2_aggregate_Mg_CO2eq_per_year']/out['v_12_5_1_herd_total_TLU'] )
            out['v_12_4_5_Global_Total_LUC_per_animal_Mg_CO2eq_per_TLU_per_year'] = (
                out['v_12_4_5_Global_Grassland_expansion_per_animal_Mg_CO2eq_per_TLU_per_year']+
            out['v_12_4_5_Global_Cropland_expansion_per_animal_Mg_CO2eq_per_TLU_per_year'] )
            out['v_12_4_5_Global_Grassland_expansion_direct_per_animal_Mg_CO2eq_per_TLU_per_year'] = (
                out['v_12_2_1_5_Global_Grassland_Expansion_direct_CO2_aggregate_Mg_CO2eq_per_year']/out['v_12_5_1_herd_total_TLU'] )


            # per unit local
            #if (('local' in breed) & (herd[(ss,'local','herd_pop')] != 0)):
            if (('local' in breed)):
                out['v_12_4_1_1_Global_Enteric_CH4_local_per_animal_Mg_CO2eq_per_TLU_per_year'] =   (
                    out['v_12_2_2_1_Global_Enteric_CH4_local_aggregate_Mg_CO2eq_per_year'] /out['v_12_5_1_herd_total_TLU_Local'] )
                out['v_12_4_1_2_Global_Manure_CH4_local_per_animal_Mg_CO2eq_per_TLU_per_year'] = (
                    out['v_12_2_2_2_Global_Manure_CH4_local_aggregate_Mg_CO2eq_per_year'] / out['v_12_5_1_herd_total_TLU_Local'] )
                out['v_12_4_1_3_Global_Manure_N2O_local_per_animal_Mg_CO2eq_per_TLU_per_year'] = (  
                    out['v_12_2_2_3_Global_Manure_N2O_local_aggregate_Mg_CO2eq_per_year'] / 
                    out['v_12_5_1_herd_total_TLU_Local'] )                                                                           
                out['v_12_4_1_4_Global_Soil_N2O_local_cropland_per_animal_Mg_CO2eq_per_TLU_per_year'] = (
                out['v_12_2_2_4_Global_Soil_N2O_cropland_local_aggregate_Mg_CO2eq_per_year'] / 
                    out['v_12_5_1_herd_total_TLU_Local'] )                                                                
                out['v_12_4_1_4_Global_Soil_N2O_local_grassland_per_animal_Mg_CO2eq_per_TLU_per_year'] = (
                    out['v_12_2_2_4_Global_Soil_N2O_grassland_local_aggregate_Mg_CO2eq_per_year']/
                    out['v_12_5_1_herd_total_TLU_Local'] ) 
                out['v_12_4_1_4_Global_Energy_Use_CO2_local_per_animal_Mg_CO2eq_per_TLU_per_year'] = (
                    out['v_12_2_2_4_Global_Energy_Use_CO2_local_aggregate_Mg_CO2eq_per_year']/
                    out['v_12_5_1_herd_total_TLU_Local'] )
                out['v_12_4_1_5_Global_Cropland_expansion_local_per_animal_Mg_CO2eq_per_TLU_per_year'] = (
                        out['v_12_2_2_5_Global_Cropland_Expansion_CO2_local_aggregate_Mg_CO2eq_per_year'] / 
                     out['v_12_5_1_herd_total_TLU_Local']  )                                                                         
                out['v_12_4_1_5_Global_Grassland_expansion_local_per_animal_Mg_CO2eq_per_TLU_per_year'] = (
                 out['v_12_2_2_5_Global_Grassland_Expansion_CO2_local_aggregate_Mg_CO2eq_per_year'] /
                    out['v_12_5_1_herd_total_TLU_Local'] )
                out['v_12_4_1_5_Global_Grassland_expansion_direct_local_per_animal_Mg_CO2eq_per_TLU_per_year'] = (
                  out['v_12_2_2_5_Global_Grassland_Expansion_direct_CO2_local_aggregate_Mg_CO2eq_per_year']/
                    out['v_12_5_1_herd_total_TLU_Local'] )

            # per unit improved
            #if (('improved' in breed) & (herd[(ss,'improved','herd_pop')] != 0)):
            if (('improved' in breed)):
                out['v_12_4_2_1_Global_Enteric_CH4_improved_per_animal_Mg_CO2eq_per_TLU_per_year'] = (
                    out['v_12_2_3_1_Global_Enteric_CH4_improved_aggregate_Mg_CO2eq_per_year'] /  
                    out['v_12_5_1_herd_total_TLU_Improved'] )                                                                                    
                out['v_12_4_2_2_Global_Manure_CH4_improved_per_animal_Mg_CO2eq_per_TLU_per_year'] = (
                    out['v_12_2_3_2_Global_Manure_CH4_improved_aggregate_Mg_CO2eq_per_year'] /
                    out['v_12_5_1_herd_total_TLU_Improved']) 
                out['v_12_4_2_3_Global_Manure_N2O_improved_per_animal_Mg_CO2eq_per_TLU_per_year'] = (
                    out['v_12_2_3_3_Global_Manure_N2O_improved_aggregate_Mg_CO2eq_per_year'] /
                    out['v_12_5_1_herd_total_TLU_Improved'] )
                out['v_12_4_2_4_Global_Soil_N2O_improved_cropland_per_animal_Mg_CO2eq_per_TLU_per_year'] = (
                  out['v_12_2_3_4_Global_Soil_N2O_cropland_improved_aggregate_Mg_CO2eq_per_year']/
                    out['v_12_5_1_herd_total_TLU_Improved'] )
                out['v_12_4_2_4_Global_Soil_N2O_improved_grassland_per_animal_Mg_CO2eq_per_TLU_per_year'] =(
                    out['v_12_2_3_4_Global_Soil_N2O_grassland_improved_aggregate_Mg_CO2eq_per_year']/
                    out['v_12_5_1_herd_total_TLU_Improved'] )
                out['v_12_4_2_4_Global_Energy_Use_CO2_improved_per_animal_Mg_CO2eq_per_TLU_per_year'] = (
                    out['v_12_2_3_4_Global_Energy_Use_CO2_improved_aggregate_Mg_CO2eq_per_year']/
                    out['v_12_5_1_herd_total_TLU_Improved'] )
                out['v_12_4_2_5_Global_Cropland_expansion_improved_per_animal_Mg_CO2eq_per_TLU_per_year'] = (
                    out['v_12_2_3_5_Global_Cropland_Expansion_CO2_improved_aggregate_Mg_CO2eq_per_year']/                                                     out['v_12_5_1_herd_total_TLU_Improved'] )
                out['v_12_4_2_5_Global_Grassland_expansion_improved_per_animal_Mg_CO2eq_per_TLU_per_year'] = (
                    out['v_12_2_3_5_Global_Grassland_Expansion_CO2_improved_aggregate_Mg_CO2eq_per_year']/
                    out['v_12_5_1_herd_total_TLU_Improved'] )
                out['v_12_4_2_5_Global_Grassland_expansion_direct_improved_per_animal_Mg_CO2eq_per_TLU_per_year'] = (
                    out['v_12_2_3_5_Global_Grassland_Expansion_direct_CO2_improved_aggregate_Mg_CO2eq_per_year']/
                    out['v_12_5_1_herd_total_TLU_Improved'] )                                                                         
                                                                        

            if (s != 'Base'):
                out['v_12_6_1_1_Global_Change_Enteric_CH4_aggregate_Mg_CO2eq_per_year'] =  (
                    out['v_12_2_1_1_Global_Enteric_CH4_aggregate_Mg_CO2eq_per_year'] - 
                    res[('Base',systems,regions)]['v_12_2_1_1_Global_Enteric_CH4_aggregate_Mg_CO2eq_per_year'])
                out['v_12_6_1_2_Global_Change_Manure_CH4_aggregate_Mg_CO2eq_per_year'] =  (
                    out['v_12_2_1_2_Global_Manure_CH4_aggregate_Mg_CO2eq_per_year'] - 
                    res[('Base',systems,regions)]['v_12_2_1_2_Global_Manure_CH4_aggregate_Mg_CO2eq_per_year'])
                out['v_12_6_1_3_Global_Change_Manure_N2O_aggregate_Mg_CO2eq_per_year'] =  (
                    out['v_12_2_1_3_Global_Manure_N2O_aggregate_Mg_CO2eq_per_year'] - 
                    res[('Base',systems,regions)]['v_12_2_1_3_Global_Manure_N2O_aggregate_Mg_CO2eq_per_year'])
                out['v_12_6_1_4_Global_Change_Soil_N2O_cropland_aggregate_Mg_CO2eq_per_year'] =  (
                    out['v_12_2_1_4_Global_Soil_N2O_cropland_aggregate_Mg_CO2eq_per_year'] - 
                    res[('Base',systems,regions)]['v_12_2_1_4_Global_Soil_N2O_cropland_aggregate_Mg_CO2eq_per_year'])
                out['v_12_6_1_4_Global_Change_Soil_N2O_grassland_aggregate_Mg_CO2eq_per_year'] =  (
                    out['v_12_2_1_4_Global_Soil_N2O_grassland_aggregate_Mg_CO2eq_per_year'] - 
                    res[('Base',systems,regions)]['v_12_2_1_4_Global_Soil_N2O_grassland_aggregate_Mg_CO2eq_per_year'])
                out['v_12_6_1_4_Global_Change_Energy_Use_CO2_aggregate_Mg_CO2eq_per_year'] =  (
                    out['v_12_2_1_4_Global_Energy_Use_CO2_aggregate_Mg_CO2eq_per_year'] - 
                    res[('Base',systems,regions)]['v_12_2_1_4_Global_Energy_Use_CO2_aggregate_Mg_CO2eq_per_year'])



                out['v_12_6_2_1_1_Global_Change_Enteric_CH4_intensity_kg_CO2eq_per_year'] =  (
                    out['v_12_2_1_1_Global_Enteric_CH4_aggregate_Mg_CO2eq_per_year']
                    /out['v_12_1_Global_Milk_yield_Mg_yr']-
                    res[('Base',systems,regions)]['v_12_3_1_Global_Enteric_CH4_intensity_kg_CO2eq_per_year'])
                out['v_12_6_2_1_2_Global_Change_Manure_CH4_intensity_kg_CO2eq_per_year'] =  (
                    out['v_12_2_1_2_Global_Manure_CH4_aggregate_Mg_CO2eq_per_year']
                    /out['v_12_1_Global_Milk_yield_Mg_yr'] 
                    -res[('Base',systems,regions)]['v_12_3_2_Global_Manure_CH4_intensity_kg_CO2eq_per_year'])
                out['v_12_6_2_1_3_Global_Change_Manure_N2O_intensity_kg_CO2eq_per_year'] =  (
                    out['v_12_2_1_3_Global_Manure_N2O_aggregate_Mg_CO2eq_per_year']
                    /out['v_12_1_Global_Milk_yield_Mg_yr']
                    - res[('Base',systems,regions)]['v_12_3_3_Global_Manure_N2O_intensity_kg_CO2eq_per_year'])
                out['v_12_6_2_1_4_Global_Change_Soil_N2O_cropland_intensity_kg_CO2eq_per_year'] =  (
                    out['v_12_2_1_4_Global_Soil_N2O_cropland_aggregate_Mg_CO2eq_per_year'] 
                    /out['v_12_1_Global_Milk_yield_Mg_yr'] 
                    - res[('Base',systems,regions)]['v_12_3_4_Global_Soil_N2O_cropland_intensity_kg_CO2eq_per_year'])
                out['v_12_6_2_1_5_Global_Change_Soil_N2O_grassland_intensity_kg_CO2eq_per_year'] =  (
                    out['v_12_2_1_4_Global_Soil_N2O_grassland_aggregate_Mg_CO2eq_per_year']
                    /out['v_12_1_Global_Milk_yield_Mg_yr'] 
                    - res[('Base',systems,regions)]['v_12_3_4_Global_Soil_N2O_grassland_intensity_kg_CO2eq_per_year'])
                out['v_12_6_2_1_6_Global_Change_Energy_Use_CO2_intensity_kg_CO2eq_per_year'] =  (
                    out['v_12_2_1_4_Global_Energy_Use_CO2_aggregate_Mg_CO2eq_per_year'] 
                    /out['v_12_1_Global_Milk_yield_Mg_yr'] 
                    - res[('Base',systems,regions)]['v_12_3_4_Global_Energy_Use_CO2_intensity_kg_CO2eq_per_year'])
                out['v_12_6_2_1_7_Global_Change_Cropland_Expansion_CO2_intensity_kg_CO2eq_per_year'] =   (
                    out['v_12_2_1_5_Global_Cropland_Expansion_CO2_aggregate_Mg_CO2eq_per_year'] 
                    /out['v_12_1_Global_Milk_yield_Mg_yr'] 
                    -  res[('Base',systems,regions)]['v_12_3_5_Global_Cropland_expansion_CO2_intensity_kg_CO2eq_per_year'])
                out['v_12_6_2_1_8_Global_Change_Grassland_Expansion_CO2_intensity_kg_CO2eq_per_year'] =  (
                    out['v_12_2_1_5_Global_Grassland_Expansion_CO2_aggregate_Mg_CO2eq_per_year']
                    /out['v_12_1_Global_Milk_yield_Mg_yr']
                    - res[('Base',systems,regions)]['v_12_3_5_Global_Grassland_expansion_CO2_intensity_kg_CO2eq_per_year'])
                out['v_12_6_2_1_8_Global_Change_Grassland_Expansion_direct_CO2_intensity_kg_CO2eq_per_year'] =  (
                    out['v_12_2_1_5_Global_Grassland_Expansion_direct_CO2_aggregate_Mg_CO2eq_per_year'] /
                    out['v_12_1_Global_Milk_yield_Mg_yr'] - 
                    res[('Base',systems,regions)]['v_12_3_5_Global_Grassland_expansion_CO2_intensity_kg_CO2eq_per_year'])

            out['v_12_6_1_7_Global_Change_Total_aggregate_Mg_CO2eq_per_year'] =  (
            out['v_12_6_1_1_Global_Change_Enteric_CH4_aggregate_Mg_CO2eq_per_year'] 
            +out['v_12_6_1_2_Global_Change_Manure_CH4_aggregate_Mg_CO2eq_per_year']
            +out['v_12_6_1_3_Global_Change_Manure_N2O_aggregate_Mg_CO2eq_per_year']
            +out['v_12_6_1_4_Global_Change_Soil_N2O_cropland_aggregate_Mg_CO2eq_per_year']
            +out['v_12_6_1_4_Global_Change_Soil_N2O_grassland_aggregate_Mg_CO2eq_per_year']
            +out['v_12_6_1_4_Global_Change_Energy_Use_CO2_aggregate_Mg_CO2eq_per_year']
            +out['v_12_6_1_5_Global_Change_Cropland_Expansion_CO2_aggregate_Mg_CO2eq_per_year']
            +out['v_12_6_1_5_Global_Change_Grassland_Expansion_direct_CO2_aggregate_Mg_CO2eq_per_year']
            )

            # Feed intake per tlu
            out['v_12_7_2_1_Feed_intake_grass_kg_dm_per_TLU']  = ( 
                out['v_12_7_1_1_Feed_intake_grass_Mg_per_year']   /  out['v_12_5_1_herd_total_TLU'] )
            out['v_12_7_2_2_Feed_intake_stover_kg_dm_per_TLU']  = (      
                out['v_12_7_1_2_Feed_intake_stover_Mg_per_year'] /out['v_12_5_1_herd_total_TLU'] )
            out['v_12_7_2_3_Feed_intake_napier_hy_kg_dm_per_TLU']  = ( out['v_12_7_1_3_Feed_intake_napier_hy_Mg_per_year']/
            out['v_12_5_1_herd_total_TLU'] )
            out['v_12_7_2_4_Feed_intake_maize_bran_kg_dm_per_TLU']  = ( 
            out['v_12_7_1_4_Feed_intake_maize_bran_Mg_per_year'] / out['v_12_5_1_herd_total_TLU'] )
            out['v_12_7_2_5_Feed_intake_sfsm_kg_dm_per_TLU']  = ( out['v_12_7_1_5_Feed_intake_sfsm_Mg_per_year']/
                                                                 out['v_12_5_1_herd_total_TLU'] )
            out['v_12_7_2_6_Feed_intake_pasture_kg_dm_per_TLU']  = (
                out['v_12_7_1_6_Feed_intake_pasture_Mg_per_year']/
                out['v_12_5_1_herd_total_TLU'] )
            out['v_12_7_2_7_Feed_intake_napier_kg_dm_per_TLU']  = (  
                out['v_12_7_1_7_Feed_intake_napier_Mg_per_year']/out['v_12_5_1_herd_total_TLU'] )

            # Land footprint
            out['v_12_8_3_1_Grass_area_ha_per_TLU']  =out['v_12_8_2_1_Grass_area_ha'] /out['v_12_5_1_herd_total_TLU'] 
            out['v_12_8_3_2_Stover_area_ha_per_TLU']  =out['v_12_8_2_2_Stover_area_ha']/out['v_12_5_1_herd_total_TLU'] 
            out['v_12_8_3_3_Napier_area_ha_per_TLU']  =out['v_12_8_2_3_Napier_area_ha']/out['v_12_5_1_herd_total_TLU'] 
            out['v_12_8_3_4_Napier_hy_area_ha_per_TLU']  =out['v_12_8_2_4_Napier_hy_area_ha']/out['v_12_5_1_herd_total_TLU'] 
            out['v_12_8_3_5_Sunflower_area_ha_per_TLU']  =out['v_12_8_2_5_Sunflower_area_ha'] /out['v_12_5_1_herd_total_TLU'] 
            out['v_12_8_3_6_Maize_area_ha_per_TLU']  = out['v_12_8_2_6_Maize_area_ha'] /out['v_12_5_1_herd_total_TLU'] 
            out['v_12_8_3_7_Pasture_area_ha_per_TLU']  =out['v_12_8_2_7_Pasture_area_ha'] /out['v_12_5_1_herd_total_TLU'] 


            out['v_12_1_Error_intensity'] =  out['v_12_1_Error_absolute_Mg']  / out['v_12_1_Global_Milk_yield_Mg_yr']  
            out['v_12_1_Error_per_animal_kg_per_TLU'] = out['v_12_1_Error_absolute_Mg']  /out['v_12_5_1_herd_total_TLU'] 

            if (ss_analysis  == 0):
                if (('local' in breed) ):
                    out['v_12_1_Error_intensity_local'] =  (
                        out['v_12_1_Error_absolute_local_Mg']  / out['v_12_1_Global_Milk_yield_Local_Mg_yr'] ) 

                    out['v_12_1_Error_per_animal_local_kg_per_TLU'] = (
                        out['v_12_1_Error_absolute_local_Mg']  /  out['v_12_5_1_herd_total_TLU_Local'])

                if (('improved' in breed) ):
                    out['v_12_1_Error_intensity_improved'] = (
                        out['v_12_1_Error_absolute_improved_Mg']  / out['v_12_1_Global_Milk_yield_Improved_Mg_yr'] )
                    out['v_12_1_Error_per_animal_improved_kg_per_TLU'] = ( 
                        out['v_12_1_Error_absolute_improved_Mg']  /out['v_12_5_1_herd_total_TLU_Improved'] )



        return out


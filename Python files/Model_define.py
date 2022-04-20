import sys
import os

sys.path.append('../../../..')

# Import basic livsim modules
from livsim import Cow
from livsim import FeedStorage
from livsim import Feed

from data_visual import data_visual as vis
from simulation_engine import sim_engine as sim

# Matplot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.pyplot import Axes
from pathlib import Path
from scipy.stats import norm
from matplotlib import rcParams
from matplotlib.patches import Patch
from matplotlib import lines
from matplotlib.lines import Line2D

# Other miscelleneous packages
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

#%load_ext autoreload
#%autoreload 2

cwd = os.getcwd()
cwd

# mean days per month for livsim timestep
DPM = 30


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


'''
Define 'Model' which is a generic class to run LED scenarios for ruminant (dairy) production systems.

 The model conducts livestock sector simulations at the production system level, calculating GHG emissions inventories taking into
 account the land footprint/land use of production. Basic data sources are derived from a series of external excel sheets,
 combined with spatial data used to define land availability (to determine GHG impacts of model derived demand for 
 grasslands).
 
 The model consists of two main python files -- python notebook (.ipynb) or .py files:
 
    Run_model.ipynb (the current file):
    
        Defines a 'model' class which is a generic framework to run LED scenarios for dairy. The model version and 
        scenarios are defined, and the model sequentially iterates simulations representing each simulation unit in a loop. The 
        Results for each iteration are initially stored as a dictionary, and these are then sequentially converted into 
        a  dataframe. Each row of the resulting dataframe thus represents the output for a given simulation unit.
    
    Simulator.py
    
      Runs the simulations for a given simulation unit. Returns all output in the locally defined 'out' dictionary.


# ~~~~~~~~~~~~~~~~~

 Description of core aspects of model (to help understand code):

  1.  Subsector terminology

    'Subsectors' are used to differentiate production systems based on variability in productivity characteristics within a 
    given locality. This can be thought of as different farms with different management or technology (and therefore environmental
    footprints) within a livestock production system. In the current version, this is used to differentiate breeds,
    and subsector 1 is comprised exclusively of improved, subsector2 exclusively of local. 
     
     
  2. Production terminology
  
     The quantity of (milk) production in the base model run is used as a reference to define production targets,
     defined as a specific level of growth in production over the base scenario. These form the basis for the herd scaling
     factors as described in the text. Further, the model provides functionality for production to remain fixed at the base
     scenario level. These are included by first running inidividual scenarios to determine the productivity impacts of 
     the interventions, then re-running the scenario after the herd size has been re-scaled. The herd scaled scenarios have
     suffixes to represent that production has been calibrated with a given milk target. These values are 'pt' (production
     target) 1,2, ... and 'base prod' (base production).
     
     
  3. Regional disaggregation
 
     The model has two methods of spatial dissaggregation: disaggregation based only on livestock production systems (LPS)
     (Robinson et al. 2011), and disaggregation based on both LPS and administrative units for the local region (which in
     Tanzania are either 'districts' or 'regions'). If parameter 'regional_analysis' == 1 then the model accounts for 
     administrative units, and otherwise defaults to disaggregation solely by LPS.
     
     
  4. Life cycle assessment
    
     The model allows for two different LCA methodologies -- attributional and consequential. If the user specifies
     cons = 1 then consequential LCA is used, otherwise the model defaults to attributional.
     
     
# ~~~~~~~~~~~~~~~~~


'''


class Model:
    
    

    def __init__(self, sub_sectoral, years, scenario_set, systems):
        
        # User defined parameters
        self.years = years 
        self.scenario_set = scenario_set
        self.systems = systems
        self.sub_sectoral = sub_sectoral
        
        # Define default parameters
        #self.cohorts=['cow','heifer','bull','ml_calf','fe_calf','juv_male']
        self.cohorts=['cow']
        self.breed=['local','improved']
        self.consq = 0 # by default LCA is attributional
        
        self.regional_analysis = 0 # by default model is not disaggregated by (administrative) region
        self.production_target = 1  #  by default model considers production targets
        self.fixed_prod = 0   # by default model does not consider fixed production scenarios (fixed at baseline)
        self.num_production_targets = 2  # by default model has two production targets
      
        
    ''' 
    Model versionining
   
    '''
    
    def sub_sector(self,ss):
        self.sub_sectoral = ss
        
    def get_sub_sectoral(self):
        return self.sub_sectoral
            
    def regional(self,regional):
        
        if (regional == 1):
            self.regional_analysis = 1
            
        elif (regional == 0):
            self.regional_analysis = 0
            
        else:
            print('Error')
            
    def life_cycle_category(self,t):
        
        if (t == 'attr'):
            self.consq = 0
            
        elif (t == 'consq'):
            self.consq = 1
            
        else:
            print('Error, must be either consq or attr')
        
        

    # Production terminology      
    def prod_target(self,prodtarget):
        
        #if (self.prodtarget == 1):
        #self.production_target == prodtarget 
            
        #elif (self.prodtarget == 0):
        self.production_target == prodtarget 
            
    def prod_target_bool(self,r):
        self.prod_target= r
    
    def fixed_production_bool(self,p):
        self.fixed_prod = p
    
    def number_prod_targets(self,n):
        self.num_production_targets = n
        
    # Number of iterations of monte carlo    
    def mc_sims(self,mc_sims):
        
        self.mc_sims = mc_sims
        
    # Specify whether model checks for land feasibility of scenarios
    def land_constraint(self,land_constr):
        
        if (land_constr == 0):
            self.land_constr = 0
            
        elif (land_constr == 1):
            self.land_constr = 1
            
    def list_breeds(self,b):
        
        self.breed = b
        
    
    def list_cohorts(self,c):
        
        self.cohorts = c
        
    def region(self,r):
        
        self.regions = r
            
            
    
    '''
    Run model
    
    
    '''
    
    def run(self):
        
        
        
        start_time = time.time()



        # MODEL CONDITIONS
        production_target = self.production_target # are targets included in model yes/NO  
        fixed_prod = self.fixed_prod # are simulations with fixed production included in model yes no
        num_production_targets = self.num_production_targets # number of rebounds if rebounds included
        check_feasibility = self.land_constr # if the feasibility of the scenario wrt land use is assessed
        
        
        ss_analysis = self.sub_sectoral # does model run disaggregate by subsectors
        reg_analysis = self.regional_analysis # does model disaggregate by region? yes/no
        consequential= self.consq # consequential LCA
        
        
        if (self.regional_analysis == 1):
            reg = self.regions
            
        else:
            reg = 'none'
            
        #prod_target=1 # run model with respect to production target
        

        timestep = self.years # time period model runs over in years
        MC_iterations= self.mc_sims   # number of monte carlo simulations

        
        # MODEL SETS
        lps = self.systems
        breed = self.breed
        cohort = self.cohorts
        
        subsectors = [1,2]
        items = ['herd_pop','cow','heifer','bull','ml_calf','fe_calf','juv_male']

        # Load data files
        loc = (cwd+str('\\all_data.xlsx'))
        reg_data = Path(cwd+str('/regional data files/'))


        data_wb = xlrd.open_workbook(loc)
        
        
        reg_data=Path(cwd+str('/regional data files/'))
        regional_herds=reg_data/'regional_herd_populations.xlsx'
        regional_LUC_coeffs=reg_data/'regional_LUC_coeffs.xlsx'
        reg_herd=xlrd.open_workbook(regional_herds)


        herd_pop = reg_herd.sheet_by_index(0) 

        

        subsec_frac={}
       


        # Scenarios
        load_scenarios = (cwd+str('\\scenarios.xlsx'))
        scenos = xlrd.open_workbook(load_scenarios)
        scenos_ss1 = scenos.sheet_by_index(0)
        scenos_ss2 = scenos.sheet_by_index(1)
        
        
   
        
        # define index for scenario parameters
        scenario_parameters={}
        scen_index=[]
        all_scenarios = []
        
        # Load scenarios from excel sheet
        for i in range(1, scenos_ss1.ncols):
            all_scenarios.append(scenos_ss1.cell_value(1,i))
            
        
         
        # Load scenario parameter labels from excel sheet 
        for i in range(2,scenos_ss1.nrows):
            scen_index.append(scenos_ss1.cell_value(i,0))
            
        
        print('The scenarios are ',all_scenarios)
        print('The scenario indices are ',scen_index)
        
        ## define sets for which scenarios are implemented in current model run

        if ( ss_analysis == 0 ):

            if (self.scenario_set =='thesis chapt 5'):

                scenarios_core=['Baseline',
                                'Status quo',
                                'Inequitable',
                                'Middle road']
                
                scenarios=scenarios_core
                scenarios.insert(0,'Base')

                set1=['Base',
                      'Status quo',
                      'Inequitable',
                      'Middle road',
                      'Status quo-pt 1',
                      'Inequitable-pt 1',
                      'Middle road-pt 2']


            scen_sets=[set1]
            subsector_scenarios=[scen_sets]


        elif ( ss_analysis == 1 ):
            
            # Scenarios for subsectoral analysis

            scenarios_core=['BAU',
                            'L-Cn',
                            'L-CnFo',
                            'L-CnFoCo',
                            'L-FoCo',
                            'L-Fo',
                            'L-Co',
                            'L-Cn+Cyg',
                            'L-CnFo+Cyg',
                            'L-CnFoCo+Cyg',
                            'L-FoCo+Cyg',
                            'L-Fo+Cyg',
                            'L-Co+Cyg',
                            'I-Cn',
                            'I-CnFo',
                            'I-CnFoCo',
                            'I-Cn+Cyg',
                            'I-CnFo+Cyg',
                            'I-CnFoCo+Cyg']

            scenarios=scenarios_core
            scenarios.insert(0,'Base')

            set1_ss2=['Base','L-Cn', 'L-Cn+Cyg','L-CnFo','L-CnFo+Cyg','L-CnFoCo','L-CnFoCo+Cyg'] 
            subsec2=[set1_ss2]


            set1_ss1 = ['Base','I-Cn', 'I-Cn+Cyg','I-CnFo','I-CnFo+Cyg','I-CnFoCo','I-CnFoCo+Cyg'] 
            subsec1=[set1_ss1]

            subsector_scenarios=[subsec1,subsec2]
            
            
            self.subsec1 = subsec1
            self.subsec2 = subsec2

        subsec = 0
        
        for s in all_scenarios:
            for si in scen_index:
                scenario_parameters[(1),(si),(s)] = (scenos_ss1.cell(scen_index.index(si)+2,
                        1+(all_scenarios.index(s))).value
                    )
                scenario_parameters[(2),(si),(s)] = (scenos_ss2.cell(scen_index.index(si)+2,
                        1+(all_scenarios.index(s))).value
                    )
                    

        # Add the production terminology and specify parameters
        for ss in subsectors:
            for s in scenarios_core:
                for t in scen_index:
                    scenario_parameters[(ss,t,s+str('-pt 1'))]=scenario_parameters[(ss,t,s)]
                    scenario_parameters[(ss,t,s+str('-pt 2'))]=scenario_parameters[(ss,t,s)]
                    scenario_parameters[(ss,t,s+str('-base prod'))]=scenario_parameters[(ss,t,s)]

        #            
        for t in scenarios_core:
            if (fixed_prod == 1) :
                scenarios.insert(len(scenarios),s+str('-base prod'))
       



        # dictionaries for storing results
        res = {}

        # Herd related variables (total herd sizes, and cohort fractions)
        herd_raw = {} # herd data at LPS level drawn from external spatial data
        herd = {}  # herd data disaggregated into subsectors for model iterations
        herd_y0 = {}  # herd data specified for base year of model 
        herd_baseline_yf = {} # herd data specified for final year of baseline simulation
        herd_ref = {} # herd data used in scaling equation
        herd_scale = [] # herd scaling factor for production calibration
        

        i0=0
        i = 0 
        count=0
        all_count=0

        the_index=[]
        the_index_all_sets=[]
        scenarios_index=[]
        lps_index=[]
        reg_index=[]



        data_raw =[]

        # subsector indices
        index_ss1=[]
        index_ss2=[]

        lps_index_ss1=[]
        lps_index_ss2=[]

        r_index_ss1=[]
        r_index_ss2=[]

        data_ss1=[]
        data_ss2=[]

        ss1_count=0

        data2_all=[]
        scenarios_all_index=[]
        lps_all_index=[]
        reg_all_index=[]


        scenarios_ss2_index=[]
        scenarios_ss1_index=[]
        ss2_count=0

        print('Scenarios are ', scenarios)


        # ~~ Begin iterations

        for r in reg:   
            for sector in subsector_scenarios:
                for l in lps:
                    for sc_s in sector:
                        for s in sc_s:
                            
                            print('Current scenario is ',s)
                            
                            '''
                            Implement production calibration
                            
                            
                            '''
                            if ('-base prod' in s):   

                                s_b = s.replace('-base prod','')

                                herdscale = 1 / (res[(s_b,l,r)]['v1_2_Milk_yield_total_kg_yr']/
                                                 res[('Base',l,r)]['v1_2_Milk_yield_total_kg_yr']) 

                            elif ('-pt' in s): 

                                if ('-pt 1' in s):
                                    
                                    s_b = s.replace('-pt 1','')

                                    herdscale=(
                                        scenario_parameters[(ss,'production target 1 relative to base',s)]/
                                    (res[(s_b,l,r)]['v1_2_Milk_yield_total_kg_yr']/res[('Base',l,r)]['v1_2_Milk_yield_total_kg_yr'])  
                                    )
                                        
                                elif ('-pt 2' in s):
                                    
                                    s_b = s.replace('-pt 2','')
                                    
                                    herdscale =  scenario_parameters[(ss,'production target 2 relative to base',s)]/(
                                    res[(s_b,l,r)]['v1_2_Milk_yield_total_kg_yr']/res[('Base',l,r)]['v1_2_Milk_yield_total_kg_yr'])      

                            else:

                                herdscale = 1
                                
                            '''
                            Specify cattle numbers based on model settings
                            
                            
                            '''

                            if ( reg_analysis == 0 ):

                                herd_pop = data_wb.sheet_by_index(7) 
                                
                            elif ( reg_analysis == 1 ):

                                if ('MF' == r):
                                    herd_pop = reg_herd.sheet_by_index(0) 
                                elif ('MM' == r):
                                    herd_pop = reg_herd.sheet_by_index(1) 
                                elif ('NB' == r):
                                    herd_pop =reg_herd.sheet_by_index(2) 
                                elif ('RW' == r):
                                    herd_pop = reg_herd.sheet_by_index(3) 
                               

                            # Actual herd sizes by breed and cohort and lps in base year     
                            herd_raw={
                            "MRT":{"improved":{"total_herd":herd_pop.cell(8,2).value,
                                              "cow":herd_pop.cell(9,2).value,
                                            "heifer":herd_pop.cell(10,2).value,
                                            "fe_calf":herd_pop.cell(12,2).value,
                                            "ml_calf":herd_pop.cell(13,2).value,
                                            "bull":herd_pop.cell(11,2).value,
                                            "juv_male":herd_pop.cell(14,2).value},
                                   "local":{"total_herd":herd_pop.cell(1,2).value,
                                            "cow":herd_pop.cell(2,2).value,
                                            "heifer":herd_pop.cell(3,2).value,
                                            "fe_calf":herd_pop.cell(5,2).value,
                                            "ml_calf":herd_pop.cell(6,2).value,
                                            "bull":herd_pop.cell(4,2).value,
                                            "juv_male":herd_pop.cell(7,2).value}

                            },
                            "MRH":{"improved":{"total_herd":herd_pop.cell(8,3).value,
                                               "cow":herd_pop.cell(9,3).value,
                                                "heifer":herd_pop.cell(10,3).value,
                                            "fe_calf":herd_pop.cell(12,3).value,
                                            "ml_calf":herd_pop.cell(13,3).value,
                                        "bull":herd_pop.cell(11,3).value,
                                        "juv_male":herd_pop.cell(14,3).value},
                               "local":{"total_herd":herd_pop.cell(1,3).value,
                                        "cow":herd_pop.cell(2,3).value,
                                        "heifer":herd_pop.cell(3,3).value,
                                        "fe_calf":herd_pop.cell(5,3).value,
                                        "ml_calf":herd_pop.cell(6,3).value,
                                        "bull":herd_pop.cell(4,3).value,
                                        "juv_male":herd_pop.cell(7,3).value}
                            }}

                        # Specify sub-sector herd sizes (base year)
                            if (ss_analysis == 0):

                                herd[('MRT','local',1)]=0
                                herd[('MRT','local',2)]=herd_raw['MRT']['local']['total_herd']
                                herd[('MRT','local',3)]=0
                                herd[('MRH','local',1)]=0
                                herd[('MRH','local',2)]=herd_raw['MRH']['local']['total_herd']
                                herd[('MRH','local',3)]=0

                                herd[('MRT','improved',1)]=herd_raw['MRT']['improved']['total_herd']
                                herd[('MRT','improved',2)]=0
                                herd[('MRT','improved',3)]=0
                                herd[('MRH','improved',1)]=herd_raw['MRH']['improved']['total_herd']
                                herd[('MRH','improved',2)]=0
                                herd[('MRH','improved',3)]=0


                            elif ( ss_analysis == 1 ):

                                if (subsector_scenarios.index(sector)==0):
                                    herd[('MRT','local',1)]=0
                                    herd[('MRT','local',2)]=0
                                    herd[('MRT','local',3)]=0
                                    herd[('MRH','local',1)]=0
                                    herd[('MRH','local',2)]=0
                                    herd[('MRH','local',3)]=0

                                    herd[('MRT','improved',1)]=herd_raw['MRT']['improved']['total_herd']
                                    herd[('MRT','improved',2)]=0
                                    herd[('MRT','improved',3)]=0
                                    herd[('MRH','improved',1)]=herd_raw['MRH']['improved']['total_herd']
                                    herd[('MRH','improved',2)]=0
                                    herd[('MRH','improved',3)]=0


                                elif (subsector_scenarios.index(sector)==1):

                                    herd[('MRT','local',1)] = 0
                                    herd[('MRT','local',2)] = herd_raw['MRT']['local']['total_herd']
                                    herd[('MRT','local',3)] = 0
                                    herd[('MRH','local',1)] = 0
                                    herd[('MRH','local',2)] = herd_raw['MRH']['local']['total_herd']
                                    herd[('MRH','local',3)] = 0

                                    herd[('MRT','improved',1)] = 0
                                    herd[('MRT','improved',2)] = 0
                                    herd[('MRT','improved',3)] = 0
                                    herd[('MRH','improved',1)] = 0
                                    herd[('MRH','improved',2)] = 0
                                    herd[('MRH','improved',3)] = 0

                # Specify aggregated and cohort specific herd sizes based on scenario 

                            for ss in subsectors:
                                for b in breed:

                                    herd_ref[(ss,b,'total_herd')] = herdscale * herd[(l,b,ss)]

                                    herd_ref[(ss,b,'cow')] = herd_raw[l][b]['cow']
                                    herd_ref[(ss,b,'heifer')] = herd_raw[l][b]['heifer']
                                    herd_ref[(ss,b,'fe_calf')] = herd_raw[l][b]['fe_calf']
                                    herd_ref[(ss,b,'bull')] = herd_raw[l][b]['bull']
                                    herd_ref[(ss,b,'ml_calf')] = herd_raw[l][b]['ml_calf']
                                    herd_ref[(ss,b,'juv_male')] = herd_raw[l][b]['juv_male']

                                    
                            # Specify baseline herd growth rate (Annual %)
                            scenario_parameters[(1,str('herd_growth_rate_local'),s)] = 0
                            scenario_parameters[(2,str('herd_growth_rate_local'),s)] = herd_pop.cell(18,3).value
                            scenario_parameters[(1,str('herd_growth_rate_improved'),s)] = herd_pop.cell(19,3).value
                            scenario_parameters[(2,str('herd_growth_rate_improved'),s)] = 0


                            # If current scenario involves dairy genetics targets, respecify growth rates 
                            # to meet final year breed percentage targets (% improved to total cattle)
                            if (scenario_parameters[(ss,str('boolean_genetic_scenario'),s)]): 


                                Total_final_year=(
                                   ( herd[(l,'local',1)]+
                                    herd[(l,'local',2)])*
                                    scenario_parameters[(2,str('herd_growth_rate_local'),s)]**timestep  
                               + (herd[(l,'improved',1)]+
                                 herd[(l,'improved',2)])*
                                    scenario_parameters[(2,str('herd_growth_rate_improved'),s)]**timestep)

                                Total_y0=(herd[(l,'local',1)]+
                                          herd[(l,'local',2)]+
                                (herd[(l,'improved',1)]+
                                 herd[(l,'improved',2)]))

                                # percent of each breed to total population in end model period
                                if ( scenario_parameters[(ss,str('genetic_target_scenario'),s)] == 1 ):
                                    pct_imp = 100*(herd_pop.cell(23,2).value)
                                    pct_loc = 100*(1-herd_pop.cell(23,2).value)

                                elif ( scenario_parameters[(ss,str('genetic_target_scenario'),s)] == 2 ):
                                    pct_imp = 100*(herd_pop.cell(24,2).value)
                                    pct_loc = 100*(1-herd_pop.cell(24,2).value)


                                ro=pct_loc/pct_imp
                                B=Total_y0*pct_imp*.01
                                A=Total_y0*pct_loc*.01

                                B_over_A = B/A
                                y =  (Total_final_year/(ro*B+B))**(1/timestep)
                                x = y*(ro*B_over_A)**(1/timestep)


                                scenario_parameters[(1,str('herd_growth_rate_local'),s)]=0
                                scenario_parameters[(2,str('herd_growth_rate_local'),s)]=x
                                scenario_parameters[(1,str('herd_growth_rate_improved'),s)]=y
                                scenario_parameters[(2,str('herd_growth_rate_improved'),s)]=0

      
                                print('Herd growth rate local is ', scenario_parameters[(2,str('herd_growth_rate_local'),s)])
                                print('Herd growth rate improved is ',scenario_parameters[(1,str('herd_growth_rate_improved'),s)])

                            # ~~~~~~~~~~~~~
                            # ~~~ Specify herd sizes based on respective annual growth rates throughout simulation period
                            # ~~~~~~~~~~~~~
                        
                            for ss in subsectors:
                                for b in breed:
                                    herd_y0[(ss,b)] = herd_ref[(ss,b,'total_herd')]
                                    
                                    # Final year herd size is equal to reference (base) year value
                                    # multiplied by (growth rate) ^ (# years in simulation)
            
                                    herd[(ss,b,'total_herd')] = (
                                        herd_ref[(ss,b,'total_herd')]*
                                        float(scenario_parameters[(ss,str('herd_growth_rate_'+b),s)])**(timestep))

                                    for c in cohort:
                                        herd[(ss,b,c)] = herd_ref[(ss,b,c)]


                        # set base scenario herd size equal to that specified in the first year 
                                if (s == 'Base'):
                                    for b in breed:
                                        herd_baseline_yf[(l,ss,b,'total_herd')]=herd[(ss,b,'total_herd')]
                                        for c in cohort: 
                                            herd_baseline_yf[(l,ss,b,c)]=herd[(ss,b,c)] 

                                            
                            print('Current LPS is ', l)
                            print('Current region is ',r)

                            res[(s,l,r)] = sim(      res, # Model data
                                                  data_wb,
                                                  scenario_parameters,
                                                  breed, # model sets
                                                  cohort,
                                                  lps, # current model iteration
                                                  reg,
                                                  subsectors,
                                                  l, 
                                                  r,
                                                  s,
                                                  herd, # herd data
                                                  herd_y0,
                                                  herd_baseline_yf,
                                                # Model settings
                                                  check_feasibility,
                                                  timestep,
                                                  MC_iterations,
                                                  reg_analysis,
                                                  ss_analysis,
                                                  consequential)
                            
                    
                            '''
                    End of production system simulations
                    
                    The dictionary of results for the previous iteration now gets stored
                    in a (pandas) dataframe, representing one row of the resultant table. 
                    
                            '''

                            # If model does not disaggregate by region, the only indices are scenario and LPS
                            if ( reg_analysis == 0 ):
                                data_raw.insert(i,res[(s,l,r)])
                                scenarios_index.insert(i,s)
                                lps_index.insert(i,l)
                                i += 1


                                if ( ss_analysis == 1 ):
                                        data_raw.insert(i,res[(s,l,r)])
                                        scenarios_index.insert(i,s)
                                        lps_index.insert(i,l)    
                                        count+=1


                            elif ( reg_analysis == 1 ):
                                
                                data2_all.insert(all_count,res[(s,l,r)])
                                scenarios_all_index.insert(all_count,s)
                                lps_all_index.insert(all_count,l)
                                reg_all_index.insert(all_count,r)
                                all_count += 1

                            i0 += 1
                            
                            
                            
        if (reg_analysis == 0):
            
            the_index.insert(0,scenarios_index)
            the_index.insert(1,lps_index)

            # define dataframe based on concatenated dictionaries
            data=pd.DataFrame(data_raw, index=the_index)
            data.index.names = ['Scenario','System']


        if (reg_analysis == 1) :

            the_index_all_sets.insert(0,scenarios_all_index)
            the_index_all_sets.insert(1,lps_all_index)
            the_index_all_sets.insert(2,reg_all_index)

            # Define dataframes based on concatenated dictionaries
            data2 = pd.DataFrame(data2_all, index = the_index_all_sets)
            data2.index.names = ['Scenario','System','Region']


        self.data = data
        self.scenarios = scenarios
        
        print("Entire model completed  in ", (time.time() - start_time)/60, "minutes")

        if ( self.regional_analysis == 1 ):
            return data2
        else:
            return data


    
    # run data visualization code to automatically generate results figures
    
    def scatter(self): # create scatter plot
        vis.scatter(self.data,self.data)
        
    def out_regional_prodtarget(self): # create emissions plots for production target scenarios
        vis.prod_target_out(self.data,self.data,self.scenarios)
        
    def out_regional_prodtarget_landfeed(self): # create land & feed figures
        vis.vis_land_feed(self.data,self.data,self.scenarios)
        
    def out_regional_subsector(self):  # create emissions plots for subsector analysis scenarios
                            
        vis.vis_subsector(self.data,
            2,
         self.data_ss2,
         self.scenarios_ss2_index,
         self.subsec2,
         self.systems,
         self.scenario_set,
         self.fixed_prod,
         self.rebound,
         self.num_rebounds,
        'improved',
         self.data_ss1,
        'local',
         self.scenarios_ss1_index,
         self.subsec1)
        

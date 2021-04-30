    # -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 10:29:01 2019

@author: Simon.Hermes
"""

import os

import cloudpickle
import numpy as np
import pandas as pd

from datetime import datetime
from logManager import LogManager

import pyomo.environ as pyo
from pyomo.opt import SolverFactory

# Setup two loggers
logManager = LogManager()
logger_main = logManager.setup_logger()
logger_run =  logManager.setup_logger(name='logger_run', log_file='log_run.txt', level=None)

"""
OptSys: Linear Optimization
"""

class OptSys:

    def __init__(self, scenario='', fixed=False, pickle_model=False, write_lp=False, unitConv=None, start_disc=None):
        # Initiation of class
        LogManager()

        # Naming convention
        self.pre_input = '0_'
        self.pre_output = '1_'

        # Scenario manager:
        # If scenario arg. is empty than files in working directory are used
        # else files in specified scenario directory.
        # Results are written in same directory.
        if scenario == '':
            self.scenario = 'none'
            self.repository = ''
        else:
            self.scenario = str(scenario)
            #self.repository = '{}\\'.format(scenario)
            self.repository = str(scenario) + os.path.sep

        logger_main.info('Scenario: ' + self.scenario)

        # Enable import of fixed variables
        if fixed:
            self.fixed = True
            logger_main.info('Load fixed configuration')
        else:
            self.fixed = False

        # Enable/disable save model to pickle file for separate postprocessing
        if pickle_model:
            self.pickle_model = True
            logger_main.info('Enable save model to pickle file')
        else:
            self.pickle_model = False

        # Enable/disable writting the LP file (for debugging)
        if write_lp:
            self.write_lp = True
            logger_main.info('Write LP file')
        else:
            self.write_lp = False

        # Unit conversion
        if unitConv == None:
            self.unitConv = {'kW':       {'newUnit': 'MW',       'conv': 1/1000},
                             'kW_th':    {'newUnit': 'MW_th',    'conv': 1/1000},
                             'kWh':      {'newUnit': 'MWh',      'conv': 1/1000},
                             'kWh_th':   {'newUnit': 'MWh_th',   'conv': 1/1000},
                             'kg':       {'newUnit': 't',        'conv': 1/1000},
                             'h':        {'newUnit': 'd',        'conv': 1/24},
                             'd':        {'newUnit': 'a',        'conv': 1/365},
                             'm3':       {'newUnit': 'Ml',       'conv': 1/1000}}
        else:
            self.unitConv = unitConv

        # Discount setting
        if start_disc == None:
            self.start_disc = {'capex':  0,
                               'opex':   1}
        else:
            self.start_disc = start_disc


    def build_model(self):
        """
        Build pyomo model for Hybrid Wind, Solar, Gas Engine and Electrolysis
        Power System.

        Input or fixed parameters are Capitalised.
        Variable parameters, available for optimization, are in lower case.

        """

        logger_main.info('Start building pyomo model')
        total_steps = 8 # only for logging

        # Define infinity for default value of params
        infinity = float('inf')

        # Abstract model
        step = 1
        logger_main.info('{}/{}: Declare abstract model'.format(step, total_steps))
        m = pyo.AbstractModel()

        # ---------------------------------------------------------------------
		# Initialise sets -----------------------------------------------------
        # ---------------------------------------------------------------------
        step += 1
        logger_main.info('{}/{}: Initialise sets'.format(step, total_steps))

        # Temporal sets
        m.SubHour = pyo.Set()

        m.Hour_All = pyo.Set()
        m.Hour = pyo.Set(within=m.Hour_All)

        m.Day_All = pyo.Set()
        m.Day = pyo.Set(within=m.Day_All) # Subset of Day_All

        m.Weekday = pyo.Set()

        m.Year_All = pyo.Set()
        m.Year = pyo.Set(within=m.Year_All) # Subset of Year_All

        # Technology sets
        m.Tech = pyo.Set()
        m.StorageTech = pyo.Set(within=m.Tech)
        m.PartLoadTech = pyo.Set(within=m.Tech)
        m.ExternalTech = pyo.Set()

        # Nodes
        m.Node = pyo.Set()

        # Fuels
        m.Fuel = pyo.Set()
        m.Fuel1 = pyo.SetOf(m.Fuel) # Alias of set Fuel
        m.VreFuel = pyo.Set(within=m.Fuel) # Subset of Fuel
        #m.StorageFuel = pyo.Set(within=m.Fuel) # Subset of Fuel
        #m.ImportFuel1 = pyo.Set(within=m.Fuel1) # Subset of Fuel
        #m.ExportFuel = pyo.Set(within=m.Fuel) # Subset of Fuel

        # Reference fuels
        m.RefFuel = pyo.Set(within=m.Fuel)

        # Auxiliary medium
        m.AuxMedium = pyo.Set()


        # ---------------------------------------------------------------------
        # Initialise parameters -----------------------------------------------
        # ---------------------------------------------------------------------
        step += 1
        logger_main.info('{}/{}: Initialise parameters'.format(step, total_steps))

        # Slack switch [-] -> include slack variables = 1
        m.Slack_switch = pyo.Param(within=pyo.Binary, default=1)

        # Slack variable costs [EUR/kWh]
        m.F_slack_costs = pyo.Param(m.Node, m.Fuel1, within=pyo.NonNegativeReals, default=1e4)

        # Subsidy of CAPEX [EUR]
        m.Subsidy_CAPEX = pyo.Param(within=pyo.NonPositiveReals, default=-0.0)

        # Project margin specific [EUR/kW]
        m.Project_margin_spec = pyo.Param(within=pyo.NonNegativeReals, default=0.0)

        # Taxes for OPEX [-]
        m.Taxes = pyo.Param(within=pyo.NonNegativeReals, default=0.0)

        # Specific energy [kWh/kg]
        m.Spec_energy = pyo.Param(m.Fuel1, within=pyo.NonNegativeReals, default=0)

        # Unit of fuel
        m.F_unit = pyo.Param(m.Fuel1)

        # Unit of capacity of technology
        m.T_unit = pyo.Param(m.Tech)

        # Unit of capacity of storage technology
        m.St_unit = pyo.Param(m.StorageTech)

        # Energy input [-]
        m.E_input = pyo.Param(m.Tech, m.Fuel, within=pyo.Binary, default=0)

        # Energy output [-]
        m.E_output = pyo.Param(m.Tech, m.Fuel1, within=pyo.Binary, default=0)

        # Refer installed capacity to input fuel, else to output fuel
        m.Cap_of_input = pyo.Param(m.Tech - m.StorageTech, within=pyo.Binary, default=0)

        # Fuel substitutes [-]
        m.F_subst = pyo.Param(m.Fuel, m.Fuel1, within=pyo.Binary, default=0)

        # Year steps [y]
        m.Delta_Y = pyo.Param(within=pyo.NonNegativeIntegers, default=1)

        # Day steps [d]
        m.Delta_D = pyo.Param(within=pyo.NonNegativeIntegers, default=1)

        # Hour steps [h]
        m.Delta_H = pyo.Param(within=pyo.NonNegativeIntegers, default=1)

        # SubHour steps []
        m.Delta_sH = pyo.Param(within=pyo.NonNegativeIntegers, default=1)

        # Year start [y]
        m.Y_start = pyo.Param(within=pyo.NonNegativeIntegers, default=1)

        # Scale simulation time range to project time range (wrt. opex, capex)
        m.Scale_Y_to = pyo.Param(within=pyo.NonNegativeIntegers, default=1)
        m.Scale_D_to = pyo.Param(within=pyo.NonNegativeIntegers, default=365)
        m.Scale_H_to = pyo.Param(within=pyo.NonNegativeIntegers, default=24)

        # Fuel demand [kWh]
        m.F_demand = pyo.Param(m.Node, m.Fuel, m.Year_All, m.Day_All, m.Hour_All, m.SubHour, within=pyo.NonNegativeReals, default=0.0)

        # Fixed electricity demand profiles (eg. constant self-consumption of technologies) [kWh/h]
        m.F_edp = pyo.Param(m.Node, m.Tech, m.Year_All, m.Day_All, m.Hour_All, m.SubHour, within=pyo.NonNegativeReals, default=0.0)

        # Variable electricity demand profiles (eg. self-consumption of wind turbines) [kWh/kW/h]
        m.V_edp = pyo.Param(m.Node, m.Tech, m.Year_All, m.Day_All, m.Hour_All, m.SubHour, within=pyo.NonNegativeReals, default=0.0)

        # Aux. electricity consumption [kW_el/kW_fp] (per kW fuel production)
        m.Aux_ed = pyo.Param(m.Tech, within=pyo.NonNegativeReals, default=0.0)
        # Unit capacity switch: 0=disabled / 1=enabled

        # Unit capacity switch [-] -> constraint capacity to integer multiples of unit = 1
        m.Unit_cap_switch = pyo.Param(within=pyo.Binary, default=0)

        # Unit capacity of technology [kW]
        m.Unit_cap = pyo.Param(m.Node, m.Tech, m.Year_All, within=pyo.NonNegativeReals, default=1)

        # Unit volume of storage technology [kWh]
        m.Unit_volume = pyo.Param(m.Node, m.StorageTech, m.Year_All, within=pyo.NonNegativeReals, default=1)

        # Max and min  added capacity of technologies [kW]
        m.Max_cap_add = pyo.Param(m.Node, m.Tech, m.Year_All, within=pyo.NonNegativeReals, default=infinity)
        m.Min_cap_add = pyo.Param(m.Node, m.Tech, m.Year_All, within=pyo.NonNegativeReals, default=0.0)

        # Max and min  sub capacity of technologies [kW]
        m.Max_cap_sub= pyo.Param(m.Node, m.Tech, m.Year_All, within=pyo.NonNegativeReals, default=infinity)
        m.Min_cap_sub = pyo.Param(m.Node, m.Tech, m.Year_All, within=pyo.NonNegativeReals, default=0.0)

        # Max and min installable capacity of technologies [kW]
        m.Max_inst_cap = pyo.Param(m.Node, m.Tech, m.Year_All, within=pyo.NonNegativeReals, default=infinity)
        m.Min_inst_cap = pyo.Param(m.Node, m.Tech, m.Year_All, within=pyo.NonNegativeReals, default=0.0)

        # Max and min installable capacity combined of certain technologies [kW]
        m.Max_inst_cap_combined = pyo.Param(m.Node, m.Year_All, within=pyo.NonNegativeReals, default=infinity)
        m.Min_inst_cap_combined = pyo.Param(m.Node, m.Year_All, within=pyo.NonNegativeReals, default=0.0)

        # Technologies for combined capacity constraint [1]
        m.Capacity_constraint_tech = pyo.Param(m.Tech, within=pyo.Binary, default=0)

        # Available area [m2]
        m.Max_area = pyo.Param(m.Node, m.Year_All, within=pyo.NonNegativeReals, default=infinity)

        # Land use of tech [m2/kW] / [m2/kWh]
        m.Land_use = pyo.Param(m.Tech, within=pyo.NonNegativeReals, default=0.0)

        # Max and min added volume of storage technologies [kWh]
        m.Max_storage_vol_add = pyo.Param(m.Node, m.StorageTech, m.Year_All, within=pyo.NonNegativeReals, default=infinity)
        m.Min_storage_vol_add = pyo.Param(m.Node, m.StorageTech, m.Year_All, within=pyo.NonNegativeReals, default=0.0)

        # Max and min  sub capacity of technologies [kWh]
        m.Max_storage_vol_sub= pyo.Param(m.Node, m.StorageTech, m.Year_All, within=pyo.NonNegativeReals, default=infinity)
        m.Min_storage_vol_sub = pyo.Param(m.Node, m.StorageTech, m.Year_All, within=pyo.NonNegativeReals, default=0.0)

        # Max and min installable storage volume of storage technologies [kWh]
        m.Max_inst_storage_vol = pyo.Param(m.Node, m.StorageTech, m.Year_All, within=pyo.NonNegativeReals, default=infinity)
        m.Min_inst_storage_vol = pyo.Param(m.Node, m.StorageTech, m.Year_All, within=pyo.NonNegativeReals, default=0.0)

        # Max and min storage level [-]
        m.Max_storage_level = pyo.Param(m.Node, m.StorageTech, within=pyo.NonNegativeReals, default=1.0)
        m.Min_storage_level = pyo.Param(m.Node, m.StorageTech, within=pyo.NonNegativeReals, default=0.0)

        # Usable storage volume ratio [-]
        m.Availability_storage_vol = pyo.Param(m.StorageTech, within=pyo.NonNegativeReals, default=1.0)

        # Ratio of storage power capacity to energy capacity [kW/kWh]
        #m.Storage_capacity_ratio = pyo.Param(m.StorageTech, within=pyo.NonNegativeReals, default=1.0)

# NEW
        # Energy-to-power ratio [h] ('0' disables contraint, default = 0) {NonNegativeReals}
        m.Energy_power_ratio = pyo.Param(m.StorageTech, within=pyo.NonNegativeReals, default=0)

        # Incentive for high storage level, will be substracted in the end [EUR/ F_unit]
        m.High_storage_level_incentive = pyo.Param(m.StorageTech, within=pyo.NonNegativeReals, default=0)

        # Min reserve storage volume [kWh]
        m.Min_energy_reserve = pyo.Param(m.Node, m.StorageTech, m.Fuel, within=pyo.NonNegativeReals, default=0.0)

        # Window of rolling reserve capacity [h]
        m.Window_rolling_reserve = pyo.Param(m.Node, m.StorageTech, m.Fuel, within=pyo.NonNegativeIntegers, default=0)

        # Security factor for rolling reserve [1]
        m.F_rolling_reserve = pyo.Param(m.Node, m.StorageTech, m.Fuel, within=pyo.NonNegativeReals, default=1.0)


# TODO ...
# set up start_storage_level as variable and equal to finish storage level
        m.Start_storage_level = pyo.Param(m.Node, m.StorageTech,  within=pyo.NonNegativeReals, default=0.5)

        # Efficiencies [-]
        m.Eff = pyo.Param(m.Tech, within=pyo.NonNegativeReals, default=1.0)

        # Part load at max. efficiency of technology [-]
        m.Part_load_max_eff = pyo.Param(m.Tech, within=pyo.NonNegativeReals, default=1.0)

        # Part load at bend point [-]
        m.Part_load_bend = pyo.Param(m.Tech, within=pyo.NonNegativeReals, default=1.0)

        # Factor efficiency reduction part load at max efficiency [1]
        m.K_part_load_max_eff = pyo.Param(m.Tech, within=pyo.NonNegativeReals, default=0.0)

        # Factor efficiency reduction part load at bend point [1]
        m.K_part_load_bend = pyo.Param(m.Tech, within=pyo.NonNegativeReals, default=0.0)

        # Specific aux medium ratio [Nm3/kg_H2]
        m.Spec_medium_ratio = pyo.Param(m.AuxMedium, within=pyo.NonNegativeReals, default=0.0)

        # Firm capacity switch [-] -> apply requirement of firm capacity = 1
        m.Cap_switch = pyo.Param(within=pyo.Binary, default=1)

    	# Upper limit for fuel production from technology as share of annual f_demand [-]
        m.K_f_prod_upperlimit = pyo.Param(m.Tech, within=pyo.NonNegativeReals, default=infinity)

        # Capacity value of technology [-]
        m.Cap_value = pyo.Param(m.Tech, within=pyo.NonNegativeReals, default=0)


        # Discount rate (to discount future to present) [y]
        m.Discount_rate = pyo.Param(within=pyo.NonNegativeReals, default=0)

        # Availability [-]
        m.Availability = pyo.Param(m.Node, m.Tech, m.Year_All, m.Day_All, m.Hour_All, m.SubHour, within=pyo.NonNegativeReals, default=1.0)

        # Fixed operational costs [EUR/kW/a]
        m.Fo_costs = pyo.Param(m.Tech, m.Year_All, within=pyo.NonNegativeReals, default=0.0)

        # Variable operational costs [EUR/kWh]
        m.Vo_costs = pyo.Param(m.Tech, m.Year_All, within=pyo.NonNegativeReals, default=0.0)

        # Auxiliary medium costs [EUR/Nm3]
        m.AM_costs = pyo.Param(m.AuxMedium, m.Year_All, within=pyo.Reals, default=0.0)

        # Fuel costs [EUR/kWh]
        m.F_costs = pyo.Param(m.Fuel1, m.Year_All, within=pyo.NonNegativeReals, default=0.0)

        # Fuel export prices [Eur/MWh]
        m.F_export_price = pyo.Param(m.Fuel, m.Year_All, within=pyo.NonNegativeReals, default=0.0)

        # Spot market prices [Eur/MWh]
        m.F_export_timeseries_price = pyo.Param(m.Fuel, m.Year_All, m.Day_All, m.Hour_All, m.SubHour, within=pyo.Reals, default=0.0)
        m.F_import_timeseries_price = pyo.Param(m.Fuel, m.Year_All, m.Day_All, m.Hour_All, m.SubHour, within=pyo.Reals, default=0.0)

        # Spot market export [kWh/h]
        m.Max_f_export_timeseries = pyo.Param(m.Node, m.Fuel, m.Year_All, within=pyo.NonNegativeReals, default=0.0)
        m.Min_f_export_timeseries = pyo.Param(m.Node, m.Fuel, m.Year_All, within=pyo.NonNegativeReals, default=0.0)

        # Spot market import [kWh/h]
        m.Max_f_import_timeseries = pyo.Param(m.Node, m.Fuel, m.Year_All, within=pyo.NonNegativeReals, default=0.0)
        m.Min_f_import_timeseries = pyo.Param(m.Node, m.Fuel, m.Year_All, within=pyo.NonNegativeReals, default=0.0)

        # F_export_timeseries and F_import_timeseries fee [EUR/MWh]
        m.F_export_timeseries_fee = pyo.Param(m.Fuel, m.Year_All, within=pyo.NonNegativeReals, default=0.0)
        m.F_import_timeseries_fee = pyo.Param(m.Fuel, m.Year_All, within=pyo.NonNegativeReals, default=0.0)

        # Max fuel import [kWh/h]
        m.Max_f_import = pyo.Param(m.Node, m.Fuel1, within=pyo.NonNegativeReals, default=0.0)

        # Min fuel import [kWh/h]
        m.Min_f_import = pyo.Param(m.Node, m.Fuel1, within=pyo.NonNegativeReals, default=0.0)

        # Max fuel export  [kWh/h]
        m.Max_f_export = pyo.Param(m.Node, m.Fuel, within=pyo.NonNegativeReals, default=0.0)

		# Min fuel export  [kWh/h]
        m.Min_f_export = pyo.Param(m.Node, m.Fuel, within=pyo.NonNegativeReals, default=0.0)

        # Network capacity charge [EUR/kW]
        m.F_network_capacity_charge = pyo.Param(m.Node, m.Fuel, m.Year, within=pyo.NonNegativeReals, default=0.0)

        # Max/Min fuel injection into fuel network flow [1]
        m.Max_f_injection = pyo.Param(m.Node, m.Fuel, within=pyo.NonNegativeReals, default=0.0)
        m.Min_f_injection = pyo.Param(m.Node, m.Fuel, within=pyo.NonNegativeReals, default=0.0)

        # Fuel network flow [kW]
        m.F_network_flow = pyo.Param(m.Node, m.Fuel, m.Year_All, m.Day_All, m.Hour_All, m.SubHour, within=pyo.Reals, default=0)

        # Fuel fix quantity import size [kWh/h]
        m.F_fix_quant_import_size = pyo.Param(m.Node, m.Fuel1, within=pyo.NonNegativeReals, default=0.0)

        # Max fix quantity import [1/h]
        m.Max_f_fix_quant_import = pyo.Param(m.Node, m.Fuel1, within=pyo.NonNegativeIntegers, default=0)

        # Fix quantity supply window: day of week and hour of day [h]
        m.F_fix_quant_supply_day = pyo.Param(m.Node, m.Fuel1, m.Weekday, within=pyo.Binary, default=0)
        m.F_fix_quant_supply_hour = pyo.Param(m.Node, m.Fuel1, m.Hour_All, within=pyo.Binary, default=0)

        # Fuel fix quantity costs [EUR/MWh]
        m.F_fix_quant_costs =pyo.Param(m.Fuel1, m.Year_All, within=pyo.NonNegativeReals, default=0.0)

        # Lifetime of technologies [a]
        m.Tech_lifetime = pyo.Param(m.Tech, within=pyo.NonNegativeReals, default=1.0)

        # Economic lifetime / depreciation period of technologies [a]
        m.Econ_lifetime = pyo.Param(m.Tech, within=pyo.NonNegativeReals, default=1.0)

        # Investment costs [EUR/kW]
        m.Invest_costs = pyo.Param(m.Tech, m.Year_All, within=pyo.NonNegativeReals, default=0.0)

        # CAPEX external technologies (outside system boundary), e.g. compressors, pipeline, etc [EUR/y]
        m.capex_ExternalTech = pyo.Param(m.ExternalTech, m.Year_All, within=pyo.Reals, default=0.0)

        # OPEX external technologies (outside system boundary), e.g. compressors, pipeline, etc ****excl. OPEX from electricity consumption accounted for in F_demand_exo_El****   [EUR/y]
        m.opex_ExternalTech = pyo.Param(m.ExternalTech, m.Year_All, within=pyo.Reals, default=0.0)

        # CAPEX System costs as share of total CAPEX [1]
        m.Capex_system_share = pyo.Param(within=pyo.NonNegativeReals, default=0.0)

        # OPEX System costs as share of CAPEX System [1]
        m.Opex_system_share = pyo.Param(within=pyo.NonNegativeReals, default=0.0)

        # Hydrogen system components [1]
        m.Hydrogen_system = pyo.Param(m.Tech, within=pyo.Binary, default=0)

        # Technology accountable for system costs and external technology costs (relevant for LCOE)
        m.System_tech = pyo.Param(m.Tech, within=pyo.Binary, default=0)

        # Share for constant system electricity consumption of certains techs [1]
        m.Share_const_cons_system = pyo.Param(m.Fuel1, within=pyo.NonNegativeReals, default=0.0)

        # Constant elcons system components [1]
        m.Tech_const_cons_system = pyo.Param(m.Tech, within=pyo.Binary, default=0)

        # Subsidy share of certain techs [1]
        m.Subsidy_share = pyo.Param(within=pyo.NonPositiveReals, default=0.0)

        # Subsidized technologies [1]
        m.Subsidy_tech = pyo.Param(m.Tech, within=pyo.Binary, default=0)

        # ---------------------------------------------------------------------
        # Initialise variables ------------------------------------------------
        # ---------------------------------------------------------------------
        step += 1
        logger_main.info('{}/{}: Initialise variables'.format(step, total_steps))

        # Slack variable [kW]
        m.f_slack_pos = pyo.Var(m.Node, m.Fuel1, m.Year, m.Day, m.Hour, m.SubHour, within=pyo.NonNegativeReals, initialize=0.0)
        m.f_slack_neg = pyo.Var(m.Node, m.Fuel1, m.Year, m.Day, m.Hour, m.SubHour, within=pyo.NonPositiveReals, initialize=-0.0)


        # Add units of technology [-]
        m.unit_add = pyo.Var(m.Node, m.Tech, m.Year, within=pyo.NonNegativeIntegers, initialize=0)
        m.unit_sub = pyo.Var(m.Node, m.Tech, m.Year, within=pyo.NonNegativeIntegers, initialize=0)

        # Add units of storage technology [-]
        m.storage_unit_add = pyo.Var(m.Node, m.StorageTech, m.Year, within=pyo.NonNegativeIntegers, initialize=0)
        m.storage_unit_sub = pyo.Var(m.Node, m.StorageTech, m.Year, within=pyo.NonNegativeIntegers, initialize=0)

        # Added capacity of technologies [kW]
        m.cap_add = pyo.Var(m.Node, m.Tech, m.Year, within=pyo.NonNegativeReals, initialize=0.0)

        # Added storage volume of storage technologies [kWh]
        m.storage_vol_add = pyo.Var(m.Node, m.StorageTech, m.Year, within=pyo.NonNegativeReals, initialize=0.0)

        # Decommissioned capacity of technologies [kW]
        m.cap_sub = pyo.Var(m.Node, m.Tech, m.Year, within=pyo.NonNegativeReals, initialize=0.0)

        # Decommissioned storage volume of storage technologies [kWh]
        m.storage_vol_sub = pyo.Var(m.Node, m.StorageTech, m.Year, within=pyo.NonNegativeReals, initialize=0.0)

        # Storage energy level of F1 [kWh]
        m.storage_energy_level = pyo.Var(m.Node, m.StorageTech, m.Fuel1, m.Year, m.Day, m.Hour, m.SubHour, within=pyo.NonNegativeReals, initialize=0.0)

        # Start storage energy level of F1 [kWh]
        m.start_storage_energy_level = pyo.Var(m.Node, m.StorageTech, m.Fuel1, within=pyo.NonNegativeReals, initialize=0.0)

        # Fuel consumption of each technology and fuel [kW]
        m.f_cons = pyo.Var(m.Node, m.Tech, m.Fuel, m.Fuel1, m.Year, m.Day, m.Hour, m.SubHour, within=pyo.NonNegativeReals, initialize=0.0)

        # Fuel production of each technology and fuel (Fuel1) [kW]
        m.f_prod = pyo.Var(m.Node, m.Tech, m.Fuel1, m.Year, m.Day, m.Hour, m.SubHour, within=pyo.NonNegativeReals, initialize=0.0)

        # Auxiliary variables for part load adjustments [kW]
        m.f_prod_lin = pyo.Var(m.Node, m.Tech, m.Fuel1, m.Year, m.Day, m.Hour, m.SubHour, within=pyo.NonNegativeReals, initialize=0.0)
        m.f_prod_over_max_eff = pyo.Var(m.Node, m.Tech, m.Fuel1, m.Year, m.Day, m.Hour, m.SubHour, within=pyo.NonNegativeReals, initialize=0.0)
        m.f_prod_over_bend = pyo.Var(m.Node, m.Tech, m.Fuel1, m.Year, m.Day, m.Hour, m.SubHour, within=pyo.NonNegativeReals, initialize=0.0)

        # Import of fuel [kW]
        m.f_import = pyo.Var(m.Node, m.Fuel1, m.Year, m.Day, m.Hour, m.SubHour, within=pyo.NonNegativeReals, initialize=0.0)

        # Export of fuel [kW]
        m.f_export = pyo.Var(m.Node, m.Fuel, m.Fuel1, m.Year, m.Day, m.Hour, m.SubHour, within=pyo.NonNegativeReals, initialize=0.0)

        # Delivery of fuel demand [kW]
        m.f_delivery = pyo.Var(m.Node, m.Fuel, m.Fuel1, m.Year, m.Day, m.Hour, m.SubHour, within=pyo.NonNegativeReals, initialize=0.0)

        # Supply of constant system consumption [kW]
        m.f_supply_cons_system = pyo.Var(m.Node, m.Fuel, m.Fuel1, m.Year, m.Day, m.Hour, m.SubHour, within=pyo.NonNegativeReals, initialize=0.0)

        # Spot market export and import of fuel [kW]
        m.f_export_timeseries = pyo.Var(m.Node, m.Fuel, m.Fuel1, m.Year, m.Day, m.Hour, m.SubHour, within=pyo.NonNegativeReals, initialize=0.0)
        m.f_import_timeseries = pyo.Var(m.Node, m.Fuel1, m.Year, m.Day, m.Hour, m.SubHour, within=pyo.NonNegativeReals, initialize=0.0)

        # Network capacity for import, export, import_timeseries, export_timeseries [kW]
        m.f_network_capacity = pyo.Var(m.Node, m.Fuel, m.Year, within=pyo.NonNegativeReals, initialize=0.0)

        # Import fix quantity fuel [-]
        m.f_fix_quant_import = pyo.Var(m.Node, m.Fuel1, m.Year, m.Day, m.Hour, m.SubHour, within=pyo.NonNegativeIntegers, initialize=0)

        # Peak electricity demand [kW]
        m.peak_el_demand = pyo.Var(m.Node, m.Year, within=pyo.NonNegativeReals, initialize=0.0)
        m.peak_V_edp = pyo.Var(m.Node, m.Year, within=pyo.NonNegativeReals, initialize=0.0)
        m.peak_F_edp = pyo.Var(m.Node, m.Year, within=pyo.NonNegativeReals, initialize=0.0)
        m.peak_F_demand_el = pyo.Var(m.Node, m.Year, within=pyo.NonNegativeReals, initialize=0.0)
        m.peak_Aux_demand_el = pyo.Var(m.Node, m.Year, within=pyo.NonNegativeReals, initialize=0.0)


        # Expressions ---------------------------------------------------------
        step += 1
        logger_main.info('{}/{}: Formulate expressions'.format(step, total_steps))

        def calc_scale_y(m):
            """[1]"""
            return m.Scale_Y_to / len(m.Year)
        m.Scale_Y = pyo.Expression(rule=calc_scale_y)

        def calc_scale_d(m):
            """[1]"""
            return m.Scale_D_to / len(m.Day)
        m.Scale_D = pyo.Expression(rule=calc_scale_d)

        def calc_scale_h(m):
            """[1]"""
            return m.Scale_H_to / len(m.Hour)
        m.Scale_H = pyo.Expression(rule=calc_scale_h)

        def calc_delta_t(m):
            """[h]"""
            return 1 / len(m.SubHour)
        m.Delta_T = pyo.Expression(rule=calc_delta_t)

        # Capacity motion
        def cap_motion_rule(m, N, T, Y):
            """[kW]"""
            if Y == m.Y_start:
                return m.cap_add[N, T, Y]
            else:
                return m.inst_cap[N, T, Y-m.Delta_Y] + m.cap_add[N, T, Y] - m.cap_sub[N, T, Y]
        m.inst_cap = pyo.Expression(m.Node, m.Tech, m.Year, rule=cap_motion_rule)

        # Constant electricity consumption of system [kW]
        def const_cons_system_rule(m, N, F1, Y):
            return m.Share_const_cons_system[F1] * sum(m.inst_cap[N,T,Y] for T in m.Tech if m.Tech_const_cons_system[T])
        m.Const_cons_system = pyo.Expression(m.Node, m.Fuel1, m.Year, rule=const_cons_system_rule)

        # Storage volume motion
        def storage_vol_motion_rule(m, N, StoreT, Y):
            """[kWh]"""
            if Y == m.Y_start:
                return m.storage_vol_add[N, StoreT, Y]
            else:
                return m.inst_storage_vol[N, StoreT, Y-m.Delta_Y] + m.storage_vol_add[N, StoreT, Y] - m.storage_vol_sub[N, StoreT, Y]

        m.inst_storage_vol = pyo.Expression(m.Node, m.StorageTech, m.Year, rule=storage_vol_motion_rule)

        # Fuel production at max efficiency [kW]
        def fuel_production_part_load_max_eff_rule(m, N, T, Y):
            """[kW]"""
            if m.Max_inst_cap[N,T,Y]>0:
                if m.Cap_of_input[T]:
                    # eg: Electrolysis
                    Eff_temp = m.Eff[T]
                else:
                    # eg: GenSet, Wind
                    Eff_temp = 1
                return m.inst_cap[N,T,Y] * Eff_temp * m.Part_load_max_eff[T]
            else:
                return 0
        m.F_prod_part_load_max_eff = pyo.Expression(m.Node, m.Tech-m.StorageTech, m.Year, rule=fuel_production_part_load_max_eff_rule)

        def fuel_production_part_load_bend_rule(m, N, T, Y):
            """[kW]"""
            if m.Max_inst_cap[N,T,Y]>0:
                if m.Cap_of_input[T]:
                    # eg: Electrolysis
                    Eff_temp = m.Eff[T]
                else:
                    # eg: GenSet, Wind
                    Eff_temp = 1
                return m.inst_cap[N,T,Y] * Eff_temp * m.Part_load_bend[T]
            else:
                return 0
        m.F_prod_part_load_bend = pyo.Expression(m.Node, m.Tech-m.StorageTech, m.Year, rule=fuel_production_part_load_bend_rule)



        # Min and max storage volume [kWh]
        def max_storage_energy_level_rule(m, N, StoreT, Y):
            """[kWh]"""
            if m.Max_inst_storage_vol[N,StoreT,Y]>0:
                return m.Max_storage_level[N, StoreT] * m.inst_storage_vol[N, StoreT, Y]
            else:
                return 0
        m.max_storage_energy_level = pyo.Expression(m.Node, m.StorageTech, m.Year, rule=max_storage_energy_level_rule)

        def min_storage_energy_level_rule(m, N, StoreT, Y):
            """[kWh]"""
            if m.Max_inst_storage_vol[N,StoreT,Y]>0:
                return (m.Min_storage_level[N, StoreT] + 1-m.Availability_storage_vol[StoreT]) * m.inst_storage_vol[N, StoreT, Y]
            else:
                return 0
        m.min_storage_energy_level = pyo.Expression(m.Node, m.StorageTech, m.Year, rule=min_storage_energy_level_rule)


        # Rolling reserve capacity [kWh]
        def storage_energy_level_rolling_reserve_rule(m, N, StoreT, F, Y, D, H, sH):
            """Rolling reserve capacity to secure supply of F_demand for defined time period"""
            if m.Window_rolling_reserve[N, StoreT, F] > 0 and m.F_rolling_reserve[N, StoreT, F] > 0:

                # Sum up F_demand of following time steps [kWh]
                rolling_sum = 0

                for i in range(m.Window_rolling_reserve[N, StoreT, F] * len(m.SubHour)):

                    # Next time step
                    sH = sH + m.Delta_sH

                    if sH > max(m.SubHour):
                        sH = min(m.SubHour)
                        H += m.Delta_H

                        if H > max(m.Hour):
                            H = min(m.Hour)
                            D += m.Delta_D

                            if D > max(m.Day):
                                D = min(m.Day)
                                Y += m.Delta_Y

                                if Y > max(m.Year):
                                    Y = min(m.Day)

                    rolling_sum += m.F_demand[N,F,Y,D,H,sH]

                return rolling_sum * m.F_rolling_reserve[N, StoreT, F] * m.Delta_T
            else:
                return 0

        m.Rolling_energy_reserve = pyo.Expression(m.Node, m.StorageTech, m.Fuel, m.Year, m.Day, m.Hour, m.SubHour, rule=storage_energy_level_rolling_reserve_rule)


        # Auxiliary medium [Nm3/h] ... equivalent unit to [kW]
        def aux_medium_flow_rule(m, N, T, A, Y, D, H, sH):
        # TODO...
        # for Tech with output H2
            """[Nm3/h]"""
            F1 = 'PtX_Hydrogen_LP'
            Tech = 'Electrolysis'   # aux_medium_flow is calculated only for T containing Tech, small and caps are ignored
            if Tech.lower() in T.lower() and m.E_output[T, F1]:
                return m.f_prod[N, T, F1, Y, D, H, sH] / m.Spec_energy[F1] * m.Spec_medium_ratio[A]
            else:
                return 0
        m.Aux_medium_flow = pyo.Expression(m.Node, m.Tech, m.AuxMedium, m.Year, m.Day, m.Hour, m.SubHour, rule=aux_medium_flow_rule)


        # Revenues
        def revenue_rule(m, F, Y):
            """[EUR]"""
            conv = self.unitConv[m.F_unit[F]]['conv']
            return - m.F_export_price[F,Y] * conv * sum(m.f_export[N,F,F1,Y,D,H,sH] for N in m.Node if (m.Max_f_export[N,F] > 0 or m.Max_f_injection[N,F] > 0) for F1 in m.Fuel1 if m.F_subst[F,F1] and sum(m.F_subst[F1,F2] for F2 in m.Fuel1) == 1 for D in m.Day for H in m.Hour for sH in m.SubHour) * m.Delta_T * m.Scale_H * m.Scale_D
        m.revenue = pyo.Expression(m.Fuel, m.Year, rule=revenue_rule)

        def revenue_timeseries_rule(m, F, Y):
            """[EUR]"""
            conv = self.unitConv[m.F_unit[F]]['conv']
            return - conv * sum((m.F_export_timeseries_price[F,Y,D,H,sH] - m.F_export_timeseries_fee[F,Y]) * m.f_export_timeseries[N,F,F1,Y,D,H,sH] for N in m.Node if m.Max_f_export_timeseries[N,F,Y] > 0 for F1 in m.Fuel1 if m.F_subst[F,F1] and sum(m.F_subst[F1,F2] for F2 in m.Fuel1) == 1 for D in m.Day for H in m.Hour for sH in m.SubHour) * m.Delta_T * m.Scale_H * m.Scale_D
        m.revenue_timeseries = pyo.Expression(m.Fuel, m.Year, rule=revenue_timeseries_rule)



        # Slack costs rule
        def slack_rule(m):
            """Sum all slack variable vaules over time and fuel [EUR]"""
            if m.Slack_switch == 1:
                return sum(sum(m.f_slack_pos[N, F1, Y, D, H, sH] - m.f_slack_neg[N, F1, Y, D, H, sH] for Y in m.Year for D in m.Day for H in m.Hour for sH in m.SubHour) * m.F_slack_costs[N, F1] for N in m.Node for F1 in m.Fuel1) * m.Delta_T * m.Scale_H * m.Scale_D * m.Scale_Y
            else:
                return 0
        m.slack_costs = pyo.Expression(rule=slack_rule)


        # OPEX
        def opex_rule(m, N, T, Y):
            """
            Determine OPEX for each year, node and tech.
            Fixed operational costs multiplied by installed capacity or installed storage volume
            and variable operational costs multiplied by production or consumption of tech
            over days and hours.
            [EUR]
            """
            if T in m.StorageTech:
                # OPEX StorageTech
                opex_temp = m.Fo_costs[T,Y] * m.inst_storage_vol[N,T,Y] if m.Fo_costs[T,Y] != 0 else 0 \
                          + m.Vo_costs[T,Y] * sum(m.f_prod[N, T, F1, Y, D, H, sH] for F1 in m.Fuel1 for D in m.Day for H in m.Hour for sH in m.SubHour) * m.Delta_T * m.Scale_H * m.Scale_D if m.Vo_costs[T,Y] != 0 else 0

            elif T in m.ExternalTech:
                # OPEX ExternalTech
                opex_temp = m.opex_ExternalTech[T,Y]

            else:
                # OPEX Tech - StorageTech
                if m.Cap_of_input[T]:
                    # eg: Electrolysis
                    opex_temp = m.Fo_costs[T,Y] * m.inst_cap[N,T,Y] if m.Fo_costs[T,Y] != 0 else 0 \
                              + m.Vo_costs[T,Y] * sum(m.f_cons[N, T, F, F1, Y, D, H, sH] for F in m.Fuel if m.E_input[T,F] for F1 in m.Fuel1 if m.F_subst[F,F1] and sum(m.F_subst[F1,F2] for F2 in m.Fuel1) == 1 for D in m.Day for H in m.Hour for sH in m.SubHour) * m.Delta_T * m.Scale_H * m.Scale_D if m.Vo_costs[T,Y] != 0 else 0
                else:
                    # eg: GenSet, Wind
                    opex_temp = m.Fo_costs[T,Y] * m.inst_cap[N,T,Y] if m.Fo_costs[T,Y] != 0 else 0 \
                              + m.Vo_costs[T,Y] * sum(m.f_prod[N,T,F1,Y,D,H,sH] for F1 in m.Fuel1 if m.E_output[T,F1] for D in m.Day for H in m.Hour for sH in m.SubHour) * m.Delta_T * m.Scale_H * m.Scale_D if m.Vo_costs[T,Y] != 0 else 0

            return opex_temp

        m.opex = pyo.Expression(m.Node, m.Tech | m.ExternalTech, m.Year, rule=opex_rule)


        # OPEX Fuel
        def opex_fuel_rule(m, N, F1, Y):
            """[EUR]"""
            conv = self.unitConv[m.F_unit[F1]]['conv']
            opex_import = 1 * conv * m.F_costs[F1,Y] * sum(m.f_import[N,F1,Y,D,H,sH] for D in m.Day for H in m.Hour for sH in m.SubHour) * m.Delta_T * m.Scale_H * m.Scale_D if m.Max_f_import[N,F1] > 0 else 0
            opex_fix_quant_import = 1 * conv * m.F_fix_quant_costs[F1,Y] * m.F_fix_quant_import_size[N, F1] * sum(m.f_fix_quant_import[N, F1, Y, D, H, sH]  for D in m.Day for H in m.Hour for sH in m.SubHour) * m.Scale_H * m.Scale_D if m.Max_f_fix_quant_import[N,F1] > 0 else 0

            return opex_import + opex_fix_quant_import

        m.opex_fuel = pyo.Expression(m.Node, m.Fuel1, m.Year, rule=opex_fuel_rule)

        # OPEX spot market fuel import
        def opex_timeseries_rule(m,F1,Y):
            """[EUR]"""
            conv = self.unitConv[m.F_unit[F1]]['conv']
            return conv * sum((m.F_import_timeseries_price[F1,Y,D,H,sH] + m.F_import_timeseries_fee[F1,Y]) * m.f_import_timeseries[N,F1,Y,D,H,sH] for N in m.Node if m.Max_f_import_timeseries[N,F1,Y] > 0 for D in m.Day for H in m.Hour for sH in m.SubHour) * m.Delta_T * m.Scale_H * m.Scale_D

        m.opex_timeseries = pyo.Expression(m.Fuel1, m.Year, rule=opex_timeseries_rule)

        # OPEX Auxiliary medium
        def opex_auxmedium_rule(m, N, T, A, Y):
            """[EUR]"""
            return sum(m.Aux_medium_flow[N, T, A, Y, D, H, sH] for D in m.Day for H in m.Hour for sH in m.SubHour) * m.AM_costs[A, Y] * m.Delta_T * m.Scale_H * m.Scale_D
        m.opex_auxmedium = pyo.Expression(m.Node, m.Tech, m.AuxMedium, m.Year, rule=opex_auxmedium_rule)

        # OPEX Network capacity charges
        def opex_network_capacity_rule(m, N, F, Y):
            """[EUR]"""
            return m.F_network_capacity_charge[N,F,Y] * m.f_network_capacity[N,F,Y] if m.F_network_capacity_charge[N,F,Y] > 0 else 0
        m.opex_network_capacity = pyo.Expression(m.Node, m.Fuel, m.Year, rule=opex_network_capacity_rule)

        # CAPEX
        def capex_rule(m, N, T, Y):
            """
            Determine CAPEX for each year and technology.
            Resolution on technology level is necessary for LCOE calculation.
            Sum of investment costs multiplied by additionally installed capacity over nodes.
            Cost for storages depend on additionally installed storage volume.
            [EUR]
            """
            if T in m.StorageTech:
                capex_temp = m.Invest_costs[T,Y] * m.storage_vol_add[N, T, Y]

            elif T in m.ExternalTech:
                capex_temp = m.capex_ExternalTech[T,Y]

            else:
                capex_temp = m.Invest_costs[T,Y] * m.cap_add[N, T, Y]

            return capex_temp

        m.capex = pyo.Expression(m.Node, m.Tech | m.ExternalTech , m.Year, rule=capex_rule)


        # CAPEX subsidy share for certains technologies
        def subsidy_tech_rule(m, N, Y):
            if m.Subsidy_share == 0:
                return 0
            else:
                return m.Subsidy_share * sum(m.capex[N,T,Y] for T in m.Tech if m.Subsidy_tech[T])
        m.capex_subsidy = pyo.Expression(m.Node, m.Year, rule=subsidy_tech_rule)

        # CAPEX System
        def capex_system_rule(m, N, Y):
            return m.Capex_system_share * sum(m.capex[N,T,Y] for T in m.Tech if m.Hydrogen_system[T])
        m.capex_system = pyo.Expression(m.Node, m.Year, rule=capex_system_rule)

        # OPEX System
        def opex_system_rule(m, N, Y):
            return m.Opex_system_share * m.capex_system[N,Y]
        m.opex_system = pyo.Expression(m.Node, m.Year, rule=opex_system_rule)


        # Discounting costs
        m.Capex_disc = pyo.Expression(m.Tech | m.ExternalTech, rule=self.calc_disc_capex)
        m.Capex_subsidy_disc = pyo.Expression(rule=self.calc_disc_capex_subsidy)
        m.Capex_system_disc = pyo.Expression(rule=self.calc_disc_capex_system)
        m.Opex_disc = pyo.Expression(m.Tech | m.ExternalTech, rule=self.calc_disc_opex)
        m.Opex_system_disc = pyo.Expression(rule=self.calc_disc_opex_system)
        m.Opex_fuel_disc = pyo.Expression(m.Fuel1, rule=self.calc_disc_opex_fuel)
        m.Opex_timeseries_disc = pyo.Expression(m.Fuel1, rule=self.calc_disc_opex_timeseries)
        m.Opex_auxmedium_disc = pyo.Expression(m.AuxMedium, rule=self.calc_disc_opex_auxmedium)
        m.Opex_network_capacity_disc = pyo.Expression(m.Fuel, rule=self.calc_disc_opex_network_capacity)
        m.Revenue_disc = pyo.Expression(m.Fuel, rule=self.calc_disc_revenue)
        m.Revenue_timeseries_disc = pyo.Expression(m.Fuel, rule=self.calc_disc_revenue_timeseries)

        def calc_project_margin(m, Y):
            return sum(m.Project_margin_spec * m.cap_add[N, T, Y] for N in m.Node for T in m.Tech - m.StorageTech)

        m.Project_margin = pyo.Expression(m.Year, rule=calc_project_margin)
        m.Project_Margin_disc = pyo.Expression(rule=self.calc_disc_project_margin)

        def calc_opex_taxes(m, Y):
            return (sum(m.opex[N,T,Y] for N in m.Node for T in m.Tech | m.ExternalTech) \
                  + sum(m.opex_fuel[N,F1,Y] for N in m.Node for F1 in m.Fuel1) \
                  + sum(m.opex_auxmedium[N,T,A,Y] for N in m.Node for T in m.Tech-m.StorageTech for A in m.AuxMedium)
                  + sum(m.opex_network_capacity[N,F,Y] for N in m.Node for F in m.Fuel)) * m.Taxes

        m.Opex_taxes = pyo.Expression(m.Year, rule=calc_opex_taxes)
        m.Opex_taxes_disc = pyo.Expression(rule=self.calc_disc_opex_taxes)


        # Incentive for high storage level
        def high_storage_level_incentive_rule(m, StoreT):
            if m.High_storage_level_incentive[StoreT] > 0:
                F = [F for F in m.Fuel if m.E_input[StoreT, F]][0]
                return sum(m.storage_energy_level[N,StoreT,F1,Y,D,H,sH] for F1 in m.Fuel1 if m.F_subst[F,F1] for N in m.Node for Y in m.Year for D in m.Day for H in m.Hour for sH in m.SubHour) * m.Delta_T * m.Scale_H * m.Scale_D * m.High_storage_level_incentive[StoreT]
            else:
                return 0
        m.high_storage_level_incentive = pyo.Expression(m.StorageTech, rule=high_storage_level_incentive_rule)




        # Objective -----------------------------------------------------------
        step += 1
        logger_main.info('{}/{}: Formulate objective'.format(step, total_steps))


        # Minimise total cost
        def total_cost_rule(m):
            """[EUR]"""

            return sum(m.Capex_disc[T] for T in m.Tech | m.ExternalTech) \
                  + m.Capex_system_disc \
                  +sum(m.Opex_disc[T] for T in m.Tech | m.ExternalTech) \
                  + m.Opex_system_disc \
                  +sum(m.Opex_fuel_disc[F1] for F1 in m.Fuel1) \
                  +sum(m.Opex_timeseries_disc[F1] for F1 in m.Fuel1) \
                  +sum(m.Opex_auxmedium_disc[A] for A in m.AuxMedium) \
                  +sum(m.Opex_network_capacity_disc[F] for F in m.Fuel) \
                  +sum(m.Revenue_disc[F] for F in m.Fuel) \
                  +sum(m.Revenue_timeseries_disc[F] for F in m.Fuel) \
                  + m.Subsidy_CAPEX \
                  + m.Capex_subsidy_disc \
                  + m.Project_Margin_disc \
                  + m.Opex_taxes_disc \
                  + m.slack_costs \
                  + sum(m.high_storage_level_incentive[StoreT] for StoreT in m.StorageTech)

        m.tc_obj = pyo.Objective(rule=total_cost_rule, sense=pyo.minimize)



        # Constraints ---------------------------------------------------------
        step += 1
        logger_main.info('{}/{}: Formulate constraints'.format(step, total_steps))


        # Peak demand constraint
        """
        Obtain max possible electricity demand, which has to be satisfied by secure energy supply.
        Analyse each demand separately and sum up (worst case).
        """
        def peak_V_edp_constraint_rule(m, N, Y, D, H, sH):
            """[kW]"""
            if sum(m.V_edp[N,T,Y,D,H,sH] for T in m.Tech) == 0:
                # Constraint is not required!
                return pyo.Constraint.Skip
            else:
                return m.peak_V_edp[N,Y] >= sum(m.V_edp[N,T,Y,D,H,sH] * m.inst_cap[N,T,Y] for T in m.Tech)
        m.peak_V_edp_constraint = pyo.Constraint(m.Node, m.Year, m.Day, m.Hour, m.SubHour, rule=peak_V_edp_constraint_rule)

        def peak_F_edp_constraint_rule(m, N, Y, D, H, sH):
            """[kW]"""
            if sum(m.F_edp[N,T,Y,D,H,sH] for T in m.Tech) == 0:
                # Constraint is not required!
                return pyo.Constraint.Skip
            else:
                return m.peak_F_edp[N,Y] >= sum(m.F_edp[N,T,Y,D,H,sH] for T in m.Tech)
        m.peak_F_edp_constraint = pyo.Constraint(m.Node, m.Year, m.Day, m.Hour, m.SubHour, rule=peak_F_edp_constraint_rule)

        def peak_F_demand_el_constraint_rule(m, N, Y, D, H, sH):
            """[kW]"""
            if m.F_demand[N,'Electricity',Y,D,H,sH] == 0:
                # Constraint is not required!
                return pyo.Constraint.Skip
            else:
                return m.peak_F_demand_el[N,Y] >= m.F_demand[N,'Electricity',Y,D,H,sH]
        m.peak_F_demand_el_constraint = pyo.Constraint(m.Node, m.Year, m.Day, m.Hour, m.SubHour, rule=peak_F_demand_el_constraint_rule)

        def peak_Aux_demand_el_constraint_rule(m, N, Y):
            """[kW]"""
            if sum(m.Aux_ed[T] for T in m.Tech) == 0:
                # Constraint is not required!
                return pyo.Constraint.Skip
            else:
                return m.peak_Aux_demand_el[N,Y] >= sum(m.Aux_ed[T] * m.inst_cap[N,T,Y] for T in m.Tech if m.Aux_ed[T] > 0)
        m.peak_Aux_demand_el_constraint = pyo.Constraint(m.Node, m.Year, rule=peak_Aux_demand_el_constraint_rule)

        def peak_el_demand_constraint_rule(m, N, Y):
            """[kW]"""
            return m.peak_el_demand[N,Y] >= m.peak_V_edp[N,Y] + m.peak_F_edp[N,Y] +  m.peak_F_demand_el[N,Y] + m.peak_Aux_demand_el[N,Y]
        m.peak_el_demand_constraint = pyo.Constraint(m.Node, m.Year, rule=peak_el_demand_constraint_rule)

        # Peak capacity constraint
        def peak_cap_constraint_rule(m, N, Y):
            """Define firm capacity to be able to satisfy peak demand [kW]"""
            if m.Cap_switch == 1:
                firm_capacity = sum(m.inst_cap[N, T, Y] * m.Cap_value[T] for T in m.Tech)
                return m.peak_el_demand[N, Y] <= firm_capacity
            else:
                return pyo.Constraint.Skip

        m.peak_cap_constraint = pyo.Constraint(m.Node, m.Year, rule=peak_cap_constraint_rule)


        # Peak network capacity
        def f_network_capacity_constraint_rule(m, N, F, Y, D, H, sH):
            """Determine peak network capacity and calculate network charges"""
            if m.F_network_capacity_charge[N,F,Y] > 0:

                f_import_temp = m.f_import[N,F,Y,D,H,sH] if m.Max_f_import[N,F] > 0 else 0
                f_import_timeseries_temp = m.f_import_timeseries[N,F,Y,D,H,sH] if m.Max_f_import_timeseries[N,F,Y] > 0 else 0

                f_export_temp = sum(m.f_export[N,F,F1,Y,D,H,sH] for F1 in m.Fuel1 if m.F_subst[F,F1] if m.Max_f_export[N,F] > 0)
                f_export_timeseries_temp = sum(m.f_export_timeseries[N,F,F1,Y,D,H,sH] for F1 in m.Fuel1 if m.F_subst[F,F1] if m.Max_f_export_timeseries[N,F,Y] > 0)

                return m.f_network_capacity[N,F,Y] >= f_import_temp + f_import_timeseries_temp + f_export_temp + f_export_timeseries_temp
            else:
                return pyo.Constraint.Skip

        m.f_network_capacity_constraint = pyo.Constraint(m.Node, m.Fuel, m.Year, m.Day, m.Hour, m.SubHour, rule=f_network_capacity_constraint_rule)

        # Installed capacity min/max constraint
        def inst_cap_constraint_rule(m, N, T, Y):
            """[kW]"""
            if m.Min_inst_cap[N, T, Y] > 0 or m.Max_inst_cap[N, T, Y] < infinity:
                return pyo.inequality(m.Min_inst_cap[N,T,Y], m.inst_cap[N,T,Y], m.Max_inst_cap[N,T,Y])
            else:
                return pyo.Constraint.Skip
        m.inst_cap_constraint = pyo.Constraint(m.Node, m.Tech, m.Year, rule=inst_cap_constraint_rule)

        # **NEW (JP)** Installed capacity combined min/max constraint
        def inst_cap_combined_constraint_rule(m, N, Y):
            """[kW]"""
            if m.Min_inst_cap_combined[N,Y] > 0 or m.Max_inst_cap_combined[N,Y] < infinity:
                combined_cap = sum(m.inst_cap[N,T,Y] for T in m.Tech if m.Capacity_constraint_tech[T])
                return pyo.inequality(m.Min_inst_cap_combined[N, Y], combined_cap, m.Max_inst_cap_combined[N, Y])
            else:
                return pyo.Constraint.Skip
        m.inst_cap_combined_constraint = pyo.Constraint(m.Node, m.Year, rule=inst_cap_combined_constraint_rule)

        # Area constraint
        def inst_cap_area_constraint_rule(m, N, Y):
            """[m2]"""
            if m.Max_area[N,Y] < infinity:
                area_used = sum(m.inst_cap[N,T,Y] * m.Land_use[T] for T in m.Tech - m.StorageTech) + sum(m.inst_storage_vol[N,StoreT,Y] * m.Land_use[StoreT] for StoreT in m.StorageTech)
                return area_used <= m.Max_area[N,Y]
            else:
                return pyo.Constraint.Skip
        m.inst_cap_area_constraint = pyo.Constraint(m.Node, m.Year, rule=inst_cap_area_constraint_rule)

        # Max min capacity addition constraint
        def cap_add_constraint_rule(m, N, T, Y):
            """[kW]"""
            if m.Min_cap_add[N, T, Y] > 0 or m.Max_cap_add[N, T, Y] < infinity:
                return pyo.inequality(m.Min_cap_add[N, T, Y], m.cap_add[N, T, Y] , m.Max_cap_add[N, T, Y])
            else:
                return pyo.Constraint.Skip
        m.cap_add_constraint = pyo.Constraint(m.Node, m.Tech, m.Year, rule=cap_add_constraint_rule)


        # Max min capacity subtraction constraint
        def cap_sub_constraint_rule(m, N, T, Y):
            if m.Min_cap_sub[N, T, Y] > 0 or m.Max_cap_sub[N, T, Y] < infinity:
                return pyo.inequality(m.Min_cap_sub[N, T, Y], m.cap_sub[N, T, Y] , m.Max_cap_sub[N, T, Y])
            else:
                return pyo.Constraint.Skip
        m.cap_sub_constraint = pyo.Constraint(m.Node, m.Tech, m.Year, rule=cap_sub_constraint_rule)


        # Set cap add and cap sub to integer multiple of cap unit
        def unit_add_constraint_rule(m, N, T, Y):
            """[kW]"""
            if m.Unit_cap_switch == 1:
                return m.cap_add[N, T, Y] == m.unit_add[N, T, Y] * m.Unit_cap[N, T, Y]
            else:
                return pyo.Constraint.Skip

        m.unit_add_constraint = pyo.Constraint(m.Node, m.Tech, m.Year, rule=unit_add_constraint_rule)


        def unit_sub_constraint_rule(m, N, T, Y):
            """[kW]"""
            if m.Unit_cap_switch == 1:
                return m.cap_sub[N, T, Y] == m.unit_sub[N, T, Y] * m.Unit_cap[N, T, Y]
            else:
                return pyo.Constraint.Skip

        m.unit_sub_constraint = pyo.Constraint(m.Node, m.Tech, m.Year, rule=unit_sub_constraint_rule)


        # Max min storage volume addition constraint
        def storage_vol_add_constraint_rule(m, N, T, Y):
            """[kWh]"""
            if m.Min_storage_vol_add[N, T, Y] > 0 or m.Max_storage_vol_add[N, T, Y] < infinity:
                return pyo.inequality(m.Min_storage_vol_add[N, T, Y], m.storage_vol_add[N, T, Y], m.Max_storage_vol_add[N, T, Y])
            else:
                return pyo.Constraint.Skip
        m.storage_vol_add_constraint = pyo.Constraint(m.Node, m.StorageTech, m.Year, rule=storage_vol_add_constraint_rule)


        # Max min capacity subtraction constraint
        def storage_vol_sub_constraint_rule(m, N, T, Y):
            """[kWh]"""
            if m.Min_storage_vol_sub[N, T, Y] > 0 or m.Max_storage_vol_sub[N, T, Y] < infinity:
                return pyo.inequality(m.Min_storage_vol_sub[N, T, Y], m.storage_vol_sub[N, T, Y], m.Max_storage_vol_sub[N, T, Y])
            else:
                return pyo.Constraint.Skip
        m.storage_vol_sub_constraint = pyo.Constraint(m.Node, m.StorageTech, m.Year, rule=storage_vol_sub_constraint_rule)


        # Set storage vol add and storage vol sub to integer multiple of storage unit
        def storage_unit_add_constraint_rule(m, N, T, Y):
            """[kWh]"""
            if m.Unit_cap_switch == 1:
                return m.storage_vol_add[N, T, Y] == m.storage_unit_add[N, T, Y] * m.Unit_volume[N, T, Y]
            else:
                return pyo.Constraint.Skip

        m.storage_unit_add_constraint = pyo.Constraint(m.Node, m.StorageTech, m.Year, rule=storage_unit_add_constraint_rule)


        def storage_unit_sub_constraint_rule(m, N, T, Y):
            """[kW]"""
            if m.Unit_cap_switch == 1:
                return m.storage_vol_sub[N, T, Y] == m.storage_unit_sub[N, T, Y] * m.Unit_volume[N, T, Y]
            else:
                return pyo.Constraint.Skip

        m.storage_unit_sub_constraint = pyo.Constraint(m.Node, m.StorageTech, m.Year, rule=storage_unit_sub_constraint_rule)


        # Fuel balance constraint
        def fuel_balance_constraint_rule(m, N, F1, Y, D, H, sH):
            """[kW]"""

            if sum(m.F_subst[F1,F2] for F2 in m.Fuel1) == 1 and not F1 in m.VreFuel:
                # Filter fuels that are superordinated to more than one sub fuels.
                # Those fuels represent only the superset of a fuel type and are not used in particular
                # skip.constraint for superset fuels

                # Sum consumption over all technologies
                f_cons_temp = sum(m.f_cons[N,T,F,F1,Y,D,H,sH] for T in m.Tech - m.StorageTech if m.Max_inst_cap[N,T,Y] > 0 for F in m.Fuel if m.F_subst[F,F1] and (m.E_input[T,F] or m.F_subst['Electricity',F1] and (m.F_edp[N,T,Y,D,H,sH]>0 or m.V_edp[N,T,Y,D,H,sH]>0 or m.Aux_ed[T]>0)))

                # Sum production over technologies except storages
                f_prod_temp = sum(m.f_prod[N,T,F1,Y,D,H,sH] for T in m.Tech - m.StorageTech if m.Max_inst_cap[N,T,Y] > 0 if m.E_output[T,F1])

                # Sum charge of storages
                f_charge_temp = sum(m.f_cons[N,T,F,F1,Y,D,H,sH] for T in m.StorageTech if (m.Max_inst_cap[N,T,Y] > 0 and m.Max_inst_storage_vol[N,T,Y] > 0) for F in m.Fuel if m.E_input[T,F] and m.F_subst[F,F1] and sum(m.F_subst[F1,F2] for F2 in m.Fuel1) == 1)

                # Sum discharge of storages
                f_discharge_temp = sum(m.f_prod[N,T,F1,Y,D,H,sH] for T in m.StorageTech if (m.Max_inst_cap[N,T,Y] > 0 and m.Max_inst_storage_vol[N,T,Y] > 0) for F in m.Fuel if m.E_output[T,F] and m.F_subst[F, F1] and sum(m.F_subst[F1,F2] for F2 in m.Fuel1) == 1)

                f_import_temp = m.f_import[N,F1,Y,D,H,sH] if m.Max_f_import[N,F1] > 0 else 0

                f_fix_quant_import_temp = (m.f_fix_quant_import[N,F1,Y,D,H,sH] * m.F_fix_quant_import_size[N,F1] / m.Delta_T) if m.Max_f_fix_quant_import[N,F1] > 0 else 0

                f_import_timeseries_temp = m.f_import_timeseries[N,F1,Y,D,H,sH] if m.Max_f_import_timeseries[N,F1,Y] > 0 else 0

                f_delivery_temp = sum(m.f_delivery[N,F,F1,Y,D,H,sH] for F in m.Fuel if m.F_subst[F,F1] and m.F_demand[N,F,Y,D,H,sH] > 0 and sum(m.F_subst[F1,F2] for F2 in m.Fuel1) == 1)

                f_supply_cons_system_temp = sum(m.f_supply_cons_system[N,F,F1,Y,D,H,sH] for F in m.Fuel if m.F_subst[F,F1] and m.Share_const_cons_system[F] > 0 and sum(m.F_subst[F1,F2] for F2 in m.Fuel1) == 1)

                f_export_temp = sum(m.f_export[N,F,F1,Y,D,H,sH] for F in m.Fuel if m.F_subst[F, F1] and (m.Max_f_export[N,F] > 0 or m.Max_f_injection[N,F] > 0) and sum(m.F_subst[F1,F2] for F2 in m.Fuel1) == 1)
                f_export_timeseries_temp = sum(m.f_export_timeseries[N,F,F1,Y,D,H,sH] for F in m.Fuel if m.F_subst[F,F1] and m.Max_f_export_timeseries[N,F,Y] > 0 and sum(m.F_subst[F1,F2] for F2 in m.Fuel1) == 1)

                f_slack_pos = m.f_slack_pos[N,F1,Y,D,H,sH] if m.Slack_switch == 1 else 0
                f_slack_neg = m.f_slack_neg[N,F1,Y,D,H,sH] if m.Slack_switch == 1 else 0

                balance = f_prod_temp + f_discharge_temp + f_import_temp + f_fix_quant_import_temp + f_import_timeseries_temp + f_slack_pos + f_slack_neg == f_cons_temp + f_supply_cons_system_temp + f_charge_temp + f_delivery_temp + f_export_temp + f_export_timeseries_temp

                # Filter constraints, if balance is 0 == 0 (True) skip constraint
                if not isinstance(balance, bool) :
                    return balance
                elif balance:
                    return pyo.Constraint.Skip
                else:
                    return None

            else:
                return pyo.Constraint.Skip

        m.fuel_balance_constraint = pyo.Constraint(m.Node, m.Fuel1, m.Year, m.Day, m.Hour, m.SubHour, rule=fuel_balance_constraint_rule)


        def fuel_delivery_constraint_rule(m, N, F, Y, D, H, sH):
            """[kW]"""
            if m.F_demand[N,F,Y,D,H,sH] > 0:
                return sum(m.f_delivery[N,F,F1,Y,D,H,sH] for F1 in m.Fuel1 if m.F_subst[F,F1] and sum(m.F_subst[F1,F2] for F2 in m.Fuel1) == 1) == m.F_demand[N,F,Y,D,H,sH]
            else:
                return pyo.Constraint.Skip
        m.fuel_delivery_constraint = pyo.Constraint(m.Node, m.Fuel, m.Year, m.Day, m.Hour, m.SubHour, rule=fuel_delivery_constraint_rule)


        def fuel_supply_cons_system_constraint_rule(m, N, F, Y, D, H, sH):
            """[kW]"""
            if m.Share_const_cons_system[F] > 0:
                return sum(m.f_supply_cons_system[N,F,F1,Y,D,H,sH] for F1 in m.Fuel1 if m.F_subst[F,F1] and sum(m.F_subst[F1,F2] for F2 in m.Fuel1) == 1) == m.Const_cons_system[N,F,Y]
            else:
                return pyo.Constraint.Skip
        m.f_supply_cons_system_constraint = pyo.Constraint(m.Node, m.Fuel, m.Year, m.Day, m.Hour, m.SubHour, rule=fuel_supply_cons_system_constraint_rule)



        # Fuel production for technologies except storages, storage consumption and production (charge/discharge) are treated in storage balance
        def fuel_production_linear_constraint_rule(m, N, T, F1, Y, D, H, sH):
            """[kW]"""
            if m.E_output[T,F1] and m.Max_inst_cap[N,T,Y] > 0:
                if m.Cap_of_input[T]:
                    # eg: Electrolysis
                    Eff_temp = m.Eff[T]
                else:
                    # eg: GenSet, Wind
                    Eff_temp = 1

                avail_f_prod_temp = m.Availability[N,T,Y,D,H,sH] * m.inst_cap[N,T,Y] * Eff_temp
                return m.f_prod_lin[N,T,F1,Y,D,H,sH] <= avail_f_prod_temp
            else:
                return pyo.Constraint.Skip

        m.fuel_production_linear_constraint = pyo.Constraint(m.Node, m.Tech - m.StorageTech, m.Fuel1, m.Year, m.Day, m.Hour, m.SubHour, rule=fuel_production_linear_constraint_rule)


        # Fuel production over fuel production at max. efficiency [kW]
        def fuel_production_over_part_load_max_eff_rule(m,N,T,F1,Y,D,H,sH):
            """ [kW] """
            if m.E_output[T, F1] and m.K_part_load_max_eff[T] > 0 and m.Max_inst_cap[N,T,Y] > 0:
                return m.f_prod_over_max_eff[N,T,F1,Y,D,H,sH] >= m.f_prod_lin[N,T,F1,Y,D,H,sH] - m.F_prod_part_load_max_eff[N,T,Y]
            else:
                return pyo.Constraint.Skip
        m.fuel_production_over_partload_max_eff_constraint = pyo.Constraint(m.Node, m.Tech-m.StorageTech, m.Fuel1, m.Year, m.Day, m.Hour, m.SubHour, rule=fuel_production_over_part_load_max_eff_rule)

        def fuel_production_over_part_load_bend_rule(m,N,T,F1,Y,D,H,sH):
            """ [kW] """
            if m.E_output[T, F1] and m.K_part_load_bend[T] > 0 and m.Max_inst_cap[N,T,Y] > 0:
                return m.f_prod_over_bend[N,T,F1,Y,D,H,sH] >= m.f_prod_lin[N,T,F1,Y,D,H,sH] - m.F_prod_part_load_bend[N,T,Y]
            else:
                return pyo.Constraint.Skip
        m.fuel_production_over_partload_bend_constraint = pyo.Constraint(m.Node, m.Tech-m.StorageTech, m.Fuel1, m.Year, m.Day, m.Hour, m.SubHour, rule=fuel_production_over_part_load_bend_rule)


        def fuel_production_constraint_rule(m, N, T, F1, Y, D, H, sH):
            """[kW]"""
            if m.E_output[T,F1] and m.Max_inst_cap[N,T,Y] > 0:
                f_prod_temp = m.f_prod_lin[N,T,F1,Y,D,H,sH] - m.f_prod_over_max_eff[N,T,F1,Y,D,H,sH] * m.K_part_load_max_eff[T] - m.f_prod_over_bend[N,T,F1,Y,D,H,sH] * m.K_part_load_bend[T]
                return m.f_prod[N,T,F1,Y,D,H,sH] <= f_prod_temp
            else:
                return pyo.Constraint.Skip

        m.fuel_production_constraint = pyo.Constraint(m.Node, m.Tech - m.StorageTech, m.Fuel1, m.Year, m.Day, m.Hour, m.SubHour, rule=fuel_production_constraint_rule)


        def fuel_production_upperlimit_constraint_rule(m, N, T, F1, Y):
            """[kW]"""
            if m.E_output[T,F1] and m.Max_inst_cap[N,T,Y] > 0 and m.K_f_prod_upperlimit[T] < infinity:
                if sum(m.F_demand[N,F,Y,D,H,sH] for F in m.Fuel if m.F_subst[F,F1] for D in m.Day for H in m.Hour for sH in m.SubHour) == 0:
                    f_demand_upperlimit_temp = 0
                else:
                    f_demand_upperlimit_temp  = m.K_f_prod_upperlimit[T] * sum(m.f_delivery[N,F,F1,Y,D,H,sH] for F in m.Fuel if m.F_subst[F,F1] for D in m.Day for H in m.Hour for sH in m.SubHour) * m.Delta_T * m.Scale_H * m.Scale_D
                return sum(m.f_prod[N,T,F1,Y,D,H,sH] for D in m.Day for H in m.Hour for sH in m.SubHour) * m.Delta_T * m.Scale_H * m.Scale_D  <= f_demand_upperlimit_temp
            else:
                return pyo.Constraint.Skip

        m.fuel_production_upperlimit_constraint = pyo.Constraint(m.Node, m.Tech - m.StorageTech, m.Fuel1, m.Year, rule=fuel_production_upperlimit_constraint_rule)


        # Fuel consumption for technologies not in storage technologies, storage cons&prod are treated in storage balances
        def fuel_consumption_constraint_rule(m, N, T, F, Y, D, H, sH):
            """[kW]"""
            if m.Max_inst_cap[N,T,Y] > 0 and not F in m.VreFuel and (m.E_input[T,F] or (F == 'Electricity' and (m.F_edp[N,T,Y,D,H,sH]>0 or m.V_edp[N,T,Y,D,H,sH]>0 or m.Aux_ed[T]>0))):
                f_cons_total = sum(m.f_prod_lin[N,T,F1,Y,D,H,sH] for F1 in m.Fuel1 if m.E_output[T,F1]) / m.Eff[T] if m.E_input[T,F] else 0

                if F == 'Electricity':
                    # Add electricity demand profils
                    f_cons_total += m.F_edp[N,T,Y,D,H,sH] + m.V_edp[N,T,Y,D,H,sH] * m.inst_cap[N,T,Y] \
                                  + (sum(m.f_prod_lin[N,T,F1,Y,D,H,sH] for F1 in m.Fuel1 if m.E_output[T,F1]) * m.Aux_ed[T] if m.Aux_ed[T] > 0 else 0)
# SHE: parenthesis are necessary around ( ... if m.Aux_ed[T] != 0 else 0 ) otherwise whole eq is set to zero

                return sum(m.f_cons[N,T,F,F1,Y,D,H,sH] for F1 in m.Fuel1 if m.F_subst[F,F1] and sum(m.F_subst[F1,F2] for F2 in m.Fuel1) == 1) == f_cons_total
            else:
                return pyo.Constraint.Skip

        m.fuel_consumption_constraint = pyo.Constraint(m.Node, m.Tech - m.StorageTech, m.Fuel, m.Year, m.Day, m.Hour, m.SubHour, rule=fuel_consumption_constraint_rule)


        # Fuel import constraint
        def fuel_import_constraint_rule(m, N, F1, Y, D, H, sH):
            """[kW]"""
            if not F1 in m.VreFuel and (m.Min_f_import[N,F1] > 0 or m.Max_f_import[N,F1] < infinity) and not m.Max_f_import[N,F1] == 0:
                return pyo.inequality(m.Min_f_import[N,F1], m.f_import[N,F1,Y,D,H,sH], m.Max_f_import[N,F1])
            else:
                return pyo.Constraint.Skip

        m.fuel_import_constraint = pyo.Constraint(m.Node, m.Fuel1, m.Year, m.Day, m.Hour, m.SubHour, rule=fuel_import_constraint_rule)



        def fuel_fix_quant_import_constraint_rule(m, N, F1, Y, D, H, sH):
            """[1/h]"""
            if m.Max_f_fix_quant_import[N,F1] == 0:
                return pyo.Constraint.Skip
            else:
            	# Weekday from 1 to 7; for D=5 -> (5-1)%7 + 1 = 5 and (12-1)%7 + 1 = 5
                if m.F_fix_quant_supply_day[N,F1, (D-1) % 7 + 1] and m.F_fix_quant_supply_hour[N, F1, H] and sH == 0:
                    return m.f_fix_quant_import[N,F1,Y,D,H,sH] <= m.Max_f_fix_quant_import[N, F1]
                else:
                    return m.f_fix_quant_import[N, F1, Y, D, H, sH] == 0
        m.fuel_fix_quant_import_constraint = pyo.Constraint(m.Node, m.Fuel1, m.Year, m.Day, m.Hour, m.SubHour, rule=fuel_fix_quant_import_constraint_rule)

        # Fuel export constraint
        def fuel_export_constraint_rule(m, N, F, Y, D, H, sH):
            """[kW]"""
            if not F in m.VreFuel and (m.Min_f_export[N,F] > 0 or m.Max_f_export[N,F] < infinity) and not m.Max_f_export[N,F] == 0:
                return pyo.inequality(m.Min_f_export[N, F], sum(m.f_export[N,F,F1,Y,D,H,sH] for F1 in m.Fuel1 if m.F_subst[F,F1] and sum(m.F_subst[F1,F2] for F2 in m.Fuel1) == 1), m.Max_f_export[N, F])
            else:
                return pyo.Constraint.Skip
        m.fuel_export_constraint = pyo.Constraint(m.Node, m.Fuel, m.Year, m.Day, m.Hour, m.SubHour, rule=fuel_export_constraint_rule)

        # Fuel injection constraint
        def fuel_injection_constraint_rule(m, N, F, Y, D, H, sH):
            """[kW]"""
            if not F in m.VreFuel and (m.Min_f_injection[N,F] > 0 or m.Max_f_injection[N,F] < infinity) and not m.Max_f_injection[N,F] == 0:
                return pyo.inequality(m.Min_f_injection[N,F] * m.F_network_flow[N,F,Y,D,H,sH], sum(m.f_export[N,F,F1,Y,D,H,sH] for F1 in m.Fuel1 if m.F_subst[F,F1] and sum(m.F_subst[F1,F2] for F2 in m.Fuel1) == 1), m.Max_f_injection[N,F] * m.F_network_flow[N,F,Y,D,H,sH])
            else:
                return pyo.Constraint.Skip
        m.fuel_injection_constraint = pyo.Constraint(m.Node, m.Fuel, m.Year, m.Day, m.Hour, m.SubHour, rule=fuel_injection_constraint_rule)

        # Fuel spot market export constraint
        def fuel_timeseries_export_constraint_rule(m, N, F, Y, D, H, sH):
            """[kW]"""
            if m.Min_f_export_timeseries[N,F,Y] > 0 or m.Max_f_export_timeseries[N,F,Y] < infinity and not m.Max_f_export_timeseries[N,F,Y] == 0:
                return pyo.inequality(m.Min_f_export_timeseries[N,F,Y], sum(m.f_export_timeseries[N,F,F1,Y,D,H,sH] for F1 in m.Fuel1 if m.F_subst[F,F1] and sum(m.F_subst[F1,F2] for F2 in m.Fuel1) == 1), m.Max_f_export_timeseries[N,F,Y])
            else:
                return pyo.Constraint.Skip
        m.fuel_timeseries_export_constraint = pyo.Constraint(m.Node, m.Fuel, m.Year, m.Day, m.Hour, m.SubHour, rule=fuel_timeseries_export_constraint_rule)

        # Fuel spot market import constraint
        def fuel_timeseries_import_constraint_rule(m, N, F1, Y, D, H, sH):
            """[kW]"""
            if m.Min_f_import_timeseries[N,F1,Y] > 0 or m.Max_f_import_timeseries[N,F1,Y] < infinity and not m.Max_f_import_timeseries[N,F1,Y] == 0:
                return pyo.inequality(m.Min_f_import_timeseries[N,F1,Y], m.f_import_timeseries[N,F1,Y,D,H,sH], m.Max_f_import_timeseries[N,F1,Y])
            else:
                return pyo.Constraint.Skip
        m.fuel_import_timeseries_constraint = pyo.Constraint(m.Node, m.Fuel1, m.Year, m.Day, m.Hour, m.SubHour, rule=fuel_timeseries_import_constraint_rule)


        # Define start storage energy levels for F1, sum of F1 == Start_storage_level[F]
        def start_storage_energy_level_constraint_rule(m, N, StoreT):
            """[kWh]"""
            if m.Max_inst_cap[N,StoreT,max(m.Year)] > 0 and m.Max_inst_storage_vol[N,StoreT,max(m.Year)] > 0:
                F = [F for F in m.Fuel if m.E_input[StoreT, F]][0]
                return sum(m.start_storage_energy_level[N, StoreT, F1] for F1 in m.Fuel1 if m.F_subst[F,F1] and sum(m.F_subst[F1,F2] for F2 in m.Fuel1) == 1) == m.Start_storage_level[N, StoreT] * m.inst_storage_vol[N, StoreT, min(m.Year)]
            else:
                return pyo.Constraint.Skip

        m.start_storage_energy_level_constraint = pyo.Constraint(m.Node, m.StorageTech, rule=start_storage_energy_level_constraint_rule)


        # End storage energy level for each F1
        def end_storage_energy_level_constraint_rule(m, N, StoreT, F1):
            """[kWh]"""
            # Fuel of tech
            F = [F for F in m.Fuel if m.E_input[StoreT, F]][0]
            if m.Max_inst_cap[N,StoreT,max(m.Year)] > 0 and m.Max_inst_storage_vol[N,StoreT,max(m.Year)] > 0 and m.F_subst[F,F1] and sum(m.F_subst[F1,F2] for F2 in m.Fuel1) == 1:
                return m.storage_energy_level[N, StoreT, F1, max(m.Year), max(m.Day), max(m.Hour), max(m.SubHour)] == m.start_storage_energy_level[N, StoreT, F1]
            else:
                return pyo.Constraint.Skip

        m.end_storage_energy_level_constraint = pyo.Constraint(m.Node, m.StorageTech, m.Fuel1, rule=end_storage_energy_level_constraint_rule)


        # Storage balance discharge, charge and change in storage level
        def storage_energy_balance_constraint_rule(m, N, StoreT, F1, Y, D, H, sH):
            """[kWh]"""
            if m.Max_inst_cap[N,StoreT,Y] > 0 and m.Max_inst_storage_vol[N,StoreT,Y] > 0 and sum(m.F_subst[F1,F2] for F2 in m.Fuel1) == 1:
                # Filter fuels that are superordinated to more than one sub fuels.
        		# Those fuels represent only the superset of a fuel type and are not used in particular
        		# skip.constraint for superset fuels

                # Fuel type of storage
                F = [F for F in m.Fuel if m.E_input[StoreT, F]][0]

                if not m.F_subst[F,F1]:
                    return pyo.Constraint.Skip

                # Determine charged and discharged energy compromised by efficiency: [kW] * dT[h]
                charge = m.f_cons[N, StoreT, F, F1, Y, D, H, sH] * m.Eff[StoreT] * m.Delta_T
                discharge = m.f_prod[N, StoreT, F1, Y, D, H, sH] / m.Eff[StoreT] * m.Delta_T

                # Previous time stamp
                preSubH = sH - m.Delta_sH
                preH = H; preD = D; preY = Y

                if preSubH < min(m.SubHour):
                    preSubH = max(m.SubHour)
                    preH -= m.Delta_H

                    if preH < min(m.Hour):
                        preH = max(m.Hour)
                        preD -= m.Delta_D

                        if preD < min(m.Day):
                            preD = max(m.Day)
                            preY -= m.Delta_Y

                # Balance for first time step
                if preY < m.Y_start or preY < min(m.Year):
                    preLevel = m.start_storage_energy_level[N, StoreT, F1]
                else:
                    preLevel = m.storage_energy_level[N, StoreT, F1, preY, preD, preH, preSubH]

                return m.storage_energy_level[N, StoreT, F1, Y, D, H, sH] == preLevel + charge - discharge
            else:
                # Skip constraint if no storage is allowed
                return pyo.Constraint.Skip

        m.storage_energy_balance_constraint = pyo.Constraint(m.Node, m.StorageTech, m.Fuel1, m.Year, m.Day, m.Hour, m.SubHour, rule=storage_energy_balance_constraint_rule)


        # Storage charge constraint
        def storage_charge_constraint_rule(m, N, StoreT, F, Y, D, H, sH):
            """[kW]"""
            if m.Max_inst_cap[N,StoreT,Y] > 0 and m.Max_inst_storage_vol[N,StoreT,Y] > 0 and m.E_input[StoreT, F]:
                return sum(m.f_cons[N, StoreT, F, F1, Y, D, H, sH] for F1 in m.Fuel1 if m.F_subst[F,F1] and sum(m.F_subst[F1,F2] for F2 in m.Fuel1) == 1) <= m.inst_cap[N, StoreT, Y]
            else:
                return pyo.Constraint.Skip

        m.storage_charge_constraint = pyo.Constraint(m.Node, m.StorageTech, m.Fuel, m.Year, m.Day, m.Hour, m.SubHour, rule=storage_charge_constraint_rule)

        # Storage discharge constraint
        def storage_discharge_constraint_rule(m, N, StoreT, F, Y, D, H, sH):
            """[kW]"""
            if m.Max_inst_cap[N,StoreT,Y] > 0 and m.Max_inst_storage_vol[N,StoreT,Y] > 0 and m.E_output[StoreT, F]:
                return sum(m.f_prod[N, StoreT, F1, Y, D, H, sH] for F1 in m.Fuel1 if m.F_subst[F,F1] and sum(m.F_subst[F1,F2] for F2 in m.Fuel1) == 1) <= m.inst_cap[N, StoreT, Y]
            else:
                return pyo.Constraint.Skip

        m.storage_discharge_constraint = pyo.Constraint(m.Node, m.StorageTech, m.Fuel, m.Year, m.Day, m.Hour, m.SubHour, rule=storage_discharge_constraint_rule)



        def storage_energy_level_reserve_constraint_rule(m, N, StoreT, F, Y, D, H, sH):
            """Storage level greater than reserve capacity [kWh]"""
            if m.Min_energy_reserve[N, StoreT, F] > 0:
                return m.Min_energy_reserve[N, StoreT, F] + (1-m.Availability_storage_vol[StoreT]) * m.inst_storage_vol[N, StoreT, Y] <= sum(m.storage_energy_level[N, StoreT, F1, Y, D, H, sH] for F1 in m.Fuel1 if m.F_subst[F, F1])
            else:
                return pyo.Constraint.Skip

        m.storage_energy_level_reserve_constraint = pyo.Constraint(m.Node, m.StorageTech, m.Fuel, m.Year, m.Day, m.Hour, m.SubHour, rule=storage_energy_level_reserve_constraint_rule)


        def storage_energy_level_rolling_reserve_constraint_rule(m, N, StoreT, F, Y, D, H, sH):
            """Storage level greater than rolling reserve capacity [kWh]"""
            if m.Window_rolling_reserve[N, StoreT, F] > 0 and m.F_rolling_reserve[N, StoreT, F] > 0:
                return m.Rolling_energy_reserve[N,StoreT,F,Y,D,H,sH] + (1-m.Availability_storage_vol[StoreT]) * m.inst_storage_vol[N,StoreT,Y] <= sum(m.storage_energy_level[N, StoreT, F1, Y, D, H, sH] for F1 in m.Fuel1 if m.F_subst[F, F1])
            else:
                return pyo.Constraint.Skip

        m.storage_energy_level_rolling_reserve_constraint = pyo.Constraint(m.Node, m.StorageTech, m.Fuel, m.Year, m.Day, m.Hour, m.SubHour, rule=storage_energy_level_rolling_reserve_constraint_rule)


        # Min and max storage level constraints [kWh]
        def storage_energy_level_min_constraint_rule(m, N, StoreT, Y, D, H, sH):
            """Storage level between min and max allowed storage levels. [kWh]"""
            if m.Max_inst_cap[N,StoreT,Y] > 0 and m.Max_inst_storage_vol[N,StoreT,Y] > 0:
                # Fuel type of storage
                F = [F for F in m.Fuel if m.E_input[StoreT, F]][0]
                return m.min_storage_energy_level[N, StoreT, Y] <= sum(m.storage_energy_level[N, StoreT, F1, Y, D, H, sH] for F1 in m.Fuel1 if m.F_subst[F, F1] and sum(m.F_subst[F1,F2] for F2 in m.Fuel1) == 1)
            else:
                return pyo.Constraint.Skip

        m.storage_energy_level_min_constraint = pyo.Constraint(m.Node, m.StorageTech, m.Year, m.Day, m.Hour, m.SubHour, rule=storage_energy_level_min_constraint_rule)

        def storage_energy_level_max_constraint_rule(m, N, StoreT, Y, D, H, sH):
            """Storage level between min and max allowed storage levels. [kWh]"""
            if m.Max_inst_cap[N,StoreT,Y] > 0 and m.Max_inst_storage_vol[N,StoreT,Y] > 0:
                # Fuel type of storage
                F = [F for F in m.Fuel if m.E_input[StoreT, F]][0]
                return sum(m.storage_energy_level[N, StoreT, F1, Y, D, H, sH] for F1 in m.Fuel1 if m.F_subst[F, F1] and sum(m.F_subst[F1,F2] for F2 in m.Fuel1) == 1) <= m.max_storage_energy_level[N, StoreT, Y]
            else:
                return pyo.Constraint.Skip

        m.storage_energy_level_max_constraint = pyo.Constraint(m.Node, m.StorageTech, m.Year, m.Day, m.Hour, m.SubHour, rule=storage_energy_level_max_constraint_rule)


        # Installed storage volume min/max constraint [kWh]
        def inst_storage_vol_constraint_rule(m, N, StoreT, Y):
            """[kWh]"""
            return pyo.inequality(m.Min_inst_storage_vol[N, StoreT, Y], m.inst_storage_vol[N, StoreT, Y], m.Max_inst_storage_vol[N, StoreT, Y])

        m.inst_storage_vol_constraint = pyo.Constraint(m.Node, m.StorageTech, m.Year, rule=inst_storage_vol_constraint_rule)


        # Energy-to-power ratio storage constraint
        def inst_storage_power_constraint_rule(m, N, StoreT, Y):
            """[h]"""
            if m.Max_inst_cap[N,StoreT,Y] > 0 and m.Max_inst_storage_vol[N,StoreT,Y] > 0 and m.Energy_power_ratio[StoreT] > 0:
                return m.inst_cap[N,StoreT,Y] <= m.inst_storage_vol[N,StoreT,Y] / m.Energy_power_ratio[StoreT]
            else:
                return pyo.Constraint.Skip

        m.inst_storage_power_constraint = pyo.Constraint(m.Node, m.StorageTech, m.Year, rule=inst_storage_power_constraint_rule)


        # Write model as class attribute
        step += 1
        logger_main.info('{}/{}: Write model to class'.format(step, total_steps))
        logger_main.info('Finished building pyomo model')

        return m



    def calc_disc_capex(self, model, T):
        """
    	Discount investment costs as sum over Years and Nodes.
    	Index1 and index2 provided as argument will be returned.

    	Provided model contains following attributes:
    		Set: Year
    		Param: Discount_rate, Scale_Y_to
    	"""
        r_frac = 1/(1+model.Discount_rate)
        y_offset = self.start_disc['capex']
        res = sum(model.capex[N,T,Y] * r_frac**((Y-1)*model.Scale_Y+y_offset) for Y in model.Year for N in model.Node)
        return res

    def calc_disc_capex_subsidy(self, model):
        """
    	Discount subsidues of investment costs as sum over Years and Nodes.
    	"""
        r_frac = 1/(1+model.Discount_rate)
        y_offset = self.start_disc['capex']
        res = sum(model.capex_subsidy[N,Y] * r_frac**((Y-1)*model.Scale_Y+y_offset) for Y in model.Year for N in model.Node)
        return res


    def calc_disc_capex_system(self, model):
        """
    	Discount system investment costs as sum over Years and Nodes.
    	"""
        r_frac = 1/(1+model.Discount_rate)
        y_offset = self.start_disc['capex']
        res = sum(model.capex_system[N,Y] * r_frac**((Y-1)*model.Scale_Y+y_offset) for Y in model.Year for N in model.Node)
        return res


    def calc_disc_project_margin(self, model):
        r_frac = 1/(1+model.Discount_rate)
        y_offset = self.start_disc['capex']
        res = sum(model.Project_margin[Y] * r_frac**((Y-1)*model.Scale_Y+y_offset) for Y in model.Year)
        return res


    def calc_disc_opex(self, model, T):
        """
        Discount operational costs as sum Years and Nodes.
        T provided as argument will be returned.

        Provided model contains following attributes:
            Set: Year, Day, Hour
            Param: Discount_rate, Scale_Y_to
        """
        Scale_Y_int = int(model.Scale_Y_to / len(model.Year))
        r_frac = 1/(1+model.Discount_rate)
        y_offset = self.start_disc['opex']
        considered_years = len(model.Year) * Scale_Y_int
        #missing_years = [model.Scale_Y_to - (i+1) for i in range(model.Scale_Y_to - considered_years)]
        missing_years = [model.Scale_Y_to - i-(1-y_offset) for i in range(model.Scale_Y_to - considered_years)]


        return sum(model.opex[N,T,Y] * r_frac**(Y2+y_offset + (Y-1)*Scale_Y_int) for Y in model.Year for Y2 in range(Scale_Y_int) for N in model.Node) \
           	 + sum(model.opex[N,T,max(model.Year)] * r_frac**Y for Y in missing_years for N in model.Node)


    def calc_disc_opex_system(self, model):
        """
        Discount system operational costs as sum Years and Nodes.
        """
        Scale_Y_int = int(model.Scale_Y_to / len(model.Year))
        r_frac = 1/(1+model.Discount_rate)
        y_offset = self.start_disc['opex']
        considered_years = len(model.Year) * Scale_Y_int
        #missing_years = [model.Scale_Y_to - (i+1) for i in range(model.Scale_Y_to - considered_years)]
        missing_years = [model.Scale_Y_to - i-(1-y_offset) for i in range(model.Scale_Y_to - considered_years)]

        return sum(model.opex_system[N,Y] * r_frac**(Y2+y_offset + (Y-1)*Scale_Y_int) for Y in model.Year for Y2 in range(Scale_Y_int) for N in model.Node) \
           	 + sum(model.opex_system[N,max(model.Year)] * r_frac**Y for Y in missing_years for N in model.Node)



    def calc_disc_opex_fuel(self, model, F1):
        """
        Discount operational costs as sum Years and Nodes.
        T provided as argument will be returned.

        Provided model contains following attributes:
            Set: Year, Day, Hour
            Param: Discount_rate, Scale_Y_to
        """
        Scale_Y_int = int(model.Scale_Y_to / len(model.Year))
        r_frac = 1/(1+model.Discount_rate)
        y_offset = self.start_disc['opex']
        considered_years = len(model.Year) * Scale_Y_int
        #missing_years = [model.Scale_Y_to - (i+1) for i in range(model.Scale_Y_to - considered_years)]
        missing_years = [model.Scale_Y_to - i-(1-y_offset) for i in range(model.Scale_Y_to - considered_years)]

        return sum(model.opex_fuel[N,F1,Y] * r_frac**(Y2+y_offset + (Y-1)*Scale_Y_int) for Y in model.Year for Y2 in range(Scale_Y_int) for N in model.Node) \
           	 + sum(model.opex_fuel[N,F1,max(model.Year)] * r_frac**Y for Y in missing_years for N in model.Node)

    def calc_disc_opex_network_capacity(self, model, F):
        """
        """
        Scale_Y_int = int(model.Scale_Y_to / len(model.Year))
        r_frac = 1/(1+model.Discount_rate)
        y_offset = self.start_disc['opex']
        considered_years = len(model.Year) * Scale_Y_int
        #missing_years = [model.Scale_Y_to - (i+1) for i in range(model.Scale_Y_to - considered_years)]
        missing_years = [model.Scale_Y_to - i-(1-y_offset) for i in range(model.Scale_Y_to - considered_years)]

        return sum(model.opex_network_capacity[N,F,Y] * r_frac**(Y2+y_offset + (Y-1)*Scale_Y_int) for Y in model.Year for Y2 in range(Scale_Y_int) for N in model.Node) \
           	 + sum(model.opex_network_capacity[N,F,max(model.Year)] * r_frac**Y for Y in missing_years for N in model.Node)

    def calc_disc_opex_auxmedium(self, model, A):
        """
        Discount operational costs as sum Years and Nodes.
        T provided as argument will be returned.

        Provided model contains following attributes:
            Set: Year, Day, Hour
            Param: Discount_rate, Scale_Y_to
        """
        Scale_Y_int = int(model.Scale_Y_to / len(model.Year))
        r_frac = 1/(1+model.Discount_rate)
        y_offset = self.start_disc['opex']
        considered_years = len(model.Year) * Scale_Y_int
        #missing_years = [model.Scale_Y_to - (i+1) for i in range(model.Scale_Y_to - considered_years)]
        missing_years = [model.Scale_Y_to - i-(1-y_offset) for i in range(model.Scale_Y_to - considered_years)]

        return sum(model.opex_auxmedium[N,T,A,Y] * r_frac**(Y2+y_offset + (Y-1)*Scale_Y_int) for Y in model.Year for Y2 in range(Scale_Y_int) for N in model.Node for T in model.Tech-model.StorageTech) \
           	 + sum(model.opex_auxmedium[N,T,A,max(model.Year)] * r_frac**Y for Y in missing_years for N in model.Node for T in model.Tech-model.StorageTech)

    def calc_disc_revenue(self, model, F):
        """
        Discount operational costs as sum Years and Nodes.
        T provided as argument will be returned.

        Provided model contains following attributes:
            Set: Year, Day, Hour
            Param: Discount_rate, Scale_Y_to
        """
        Scale_Y_int = int(model.Scale_Y_to / len(model.Year))
        r_frac = 1/(1+model.Discount_rate)
        y_offset = self.start_disc['opex']
        considered_years = len(model.Year) * Scale_Y_int
        #missing_years = [model.Scale_Y_to - (i+1) for i in range(model.Scale_Y_to - considered_years)]
        missing_years = [model.Scale_Y_to - i-(1-y_offset) for i in range(model.Scale_Y_to - considered_years)]

        return sum(model.revenue[F,Y] * r_frac**(Y2+y_offset + (Y-1)*Scale_Y_int) for Y in model.Year for Y2 in range(Scale_Y_int)) \
           	 + sum(model.revenue[F,max(model.Year)] * r_frac**Y for Y in missing_years)


    def calc_disc_revenue_timeseries(self, model, F):
        """
        Discount operational costs as sum Years and Nodes.
        T provided as argument will be returned.

        Provided model contains following attributes:
            Set: Year, Day, Hour
            Param: Discount_rate, Scale_Y_to
        """
        Scale_Y_int = int(model.Scale_Y_to / len(model.Year))
        r_frac = 1/(1+model.Discount_rate)
        y_offset = self.start_disc['opex']
        considered_years = len(model.Year) * Scale_Y_int
        #missing_years = [model.Scale_Y_to - (i+1) for i in range(model.Scale_Y_to - considered_years)]
        missing_years = [model.Scale_Y_to - i-(1-y_offset) for i in range(model.Scale_Y_to - considered_years)]

        return sum(model.revenue_timeseries[F,Y] * r_frac**(Y2+y_offset + (Y-1)*Scale_Y_int) for Y in model.Year for Y2 in range(Scale_Y_int)) \
           	 + sum(model.revenue_timeseries[F,max(model.Year)] * r_frac**Y for Y in missing_years)


    def calc_disc_opex_timeseries(self, model,F1):
        """
        Discount operational costs as sum Years and Nodes.
        T provided as argument will be returned.

        Provided model contains following attributes:
            Set: Year, Day, Hour
            Param: Discount_rate, Scale_Y_to
        """
        Scale_Y_int = int(model.Scale_Y_to / len(model.Year))
        r_frac = 1/(1+model.Discount_rate)
        y_offset = self.start_disc['opex']
        considered_years = len(model.Year) * Scale_Y_int
        #missing_years = [model.Scale_Y_to - (i+1) for i in range(model.Scale_Y_to - considered_years)]
        missing_years = [model.Scale_Y_to - i-(1-y_offset) for i in range(model.Scale_Y_to - considered_years)]

        return sum(model.opex_timeseries[F1,Y] * r_frac**(Y2+y_offset + (Y-1)*Scale_Y_int) for Y in model.Year for Y2 in range(Scale_Y_int)) \
           	 + sum(model.opex_timeseries[F1,max(model.Year)] * r_frac**Y for Y in missing_years)


    def calc_disc_opex_taxes(self, model):
        """
        Discount operational costs as sum Years and Nodes.

        Provided model contains following attributes:
            Set: Year, Day, Hour
            Param: Discount_rate, Scale_Y_to
        """
        Scale_Y_int = int(model.Scale_Y_to / len(model.Year))
        r_frac = 1/(1+model.Discount_rate)
        y_offset = self.start_disc['opex']
        considered_years = len(model.Year) * Scale_Y_int
        #missing_years = [model.Scale_Y_to - (i+1) for i in range(model.Scale_Y_to - considered_years)]
        missing_years = [model.Scale_Y_to - i-(1-y_offset) for i in range(model.Scale_Y_to - considered_years)]

        return sum(model.Opex_taxes[Y] * r_frac**(Y2+y_offset + (Y-1)*Scale_Y_int) for Y in model.Year for Y2 in range(Scale_Y_int)) \
           	 + sum(model.Opex_taxes[max(model.Year)] * r_frac**Y for Y in missing_years)


    def calc_disc_opex_postprocessing(self, model, component, index1=None, index2=None, index3=None):
        """
        Discount operational costs as sum Years.
        Indexes provided as argument will be returned.

        Provided model contains following attributes:
            Set: Year, Day, Hour
            Param: Discount_rate, Scale_Y_to
        """
        Scale_Y_int = int(model.Scale_Y_to / len(model.Year))
        r_frac = 1/(1+model.Discount_rate)
        y_offset = self.start_disc['opex']
        considered_years = len(model.Year) * Scale_Y_int
        #missing_years = [model.Scale_Y_to - (i+1) for i in range(model.Scale_Y_to - considered_years)]
        missing_years = [model.Scale_Y_to - i-(1-y_offset) for i in range(model.Scale_Y_to - considered_years)]

        if index1 == None and index2 == None and index3==None:
            # Component of domain [Year]
            res = sum(component[Y] * r_frac**(Y2+y_offset + (Y-1)*Scale_Y_int) for Y in model.Year for Y2 in range(Scale_Y_int)) \
        		+ sum(component[max(model.Year)] * r_frac**Y for Y in missing_years)

        elif index2 == None and index3==None:
            # Component of domain [index1, Year]
            res = sum(component[index1, Y] * r_frac**(Y2+y_offset + (Y-1)*Scale_Y_int) for Y in model.Year for Y2 in range(Scale_Y_int)) \
        		+ sum(component[index1, max(model.Year)] * r_frac**Y for Y in missing_years)

        elif index3==None:
            # Component of domain [index1, index2, Year]
            res = sum(component[index1, index2, Y] * r_frac**(Y2+y_offset + (Y-1)*Scale_Y_int) for Y in model.Year for Y2 in range(Scale_Y_int)) \
        		+ sum(component[index1, index2, max(model.Year)] * r_frac**Y for Y in missing_years)

        else:
            # Component of domain [index1, index2, index3, Year]
        	res = sum(component[index1, index2, index3, Y] * r_frac**(Y2+y_offset + (Y-1)*Scale_Y_int) for Y in model.Year for Y2 in range(Scale_Y_int)) \
        		+ sum(component[index1, index2, index3, max(model.Year)] * r_frac**Y for Y in missing_years)

        return res

    def calc_disc_capex_postprocessing(self, model, component, index1=None, index2=None):
        """
    	Discount investment costs as sum over Years and Nodes.
    	Index1 and index2 provided as argument will be returned.

    	Provided model contains following attributes:
    		Set: Year
    		Param: Discount_rate, Scale_Y_to
    	"""
        r_frac = 1/(1+model.Discount_rate)
        y_offset = self.start_disc['capex']

        if index1 == None and index2 == None:
            # Component of domain [Year]
            res = sum(component[Y] * r_frac**((Y-1)*model.Scale_Y+y_offset) for Y in model.Year)

        elif index2 == None:
            # Component of domain [index1, Year]
            res = sum(component[index1,Y] * r_frac**((Y-1)*model.Scale_Y+y_offset) for Y in model.Year)

        else:
            # Component of domain [index1, index2, Year]
            res = sum(component[index1,index2,Y] * r_frac**((Y-1)*model.Scale_Y+y_offset) for Y in model.Year)

        return res





    def calc_slack_balance(self, model):
        """Calc slack variable sum for each node and fuel [kWh]"""
        model.slack_pos_balance = pyo.Param(model.Fuel, default=0.0, mutable=True)
        model.slack_neg_balance = pyo.Param(model.Fuel, default=0.0, mutable=True)

        for F in model.Fuel:
            model.slack_pos_balance[F] = pyo.value(sum(model.f_slack_pos[N, F, Y, D, H,sH] for N in model.Node for Y in model.Year for D in model.Day for H in model.Hour for sH in model.SubHour) * model.Delta_T * model.Scale_H * model.Scale_D * model.Scale_Y)
            model.slack_neg_balance[F] = pyo.value(sum(model.f_slack_neg[N, F, Y, D, H,sH] for N in model.Node for Y in model.Year for D in model.Day for H in model.Hour for sH in model.SubHour) * model.Delta_T * model.Scale_H * model.Scale_D * model.Scale_Y)


    def calc_energy_balance(self, model):
        """Validate energy balance over all years and nodes [kWh]"""
        logger_main.info('Calc energy balance')

        def energy_balance_rule(m, F1):
            """
            Balance over F1.
            Do not use formulations like if F_subst[F,F1] or E_input[T,F] to check if any fuels are used besides constraints
            [kWh]
            """
            return sum(m.f_prod_y[N,T,F1,Y] for N in m.Node for T in m.Tech for Y in m.Year) \
                 + sum(m.f_import_y[F1,Y] for Y in m.Year) \
                 + sum(m.f_import_timeseries_y[F1,Y] for Y in m.Year) \
                 + sum(m.f_fix_quant_import_y[F1,Y] for Y in m.Year) \
                 - sum(m.f_cons_y[N,T,F,F1,Y] for N in m.Node for T in m.Tech for F in m.Fuel for Y in m.Year) \
                 - sum(m.f_export_y[F,F1,Y] for F in m.Fuel for Y in m.Year) \
                 - sum(m.f_export_timeseries_y[F,F1,Y] for F in m.Fuel for Y in m.Year) \
                 - sum(m.f_delivery_y[F,F1,Y] for F in m.Fuel for Y in m.Year) \
                 - sum(m.f_supply_cons_system_y[F,F1,Y] for F in m.Fuel for Y in m.Year)

        model.energy_balance = pyo.Expression(model.Fuel1, rule=energy_balance_rule)


    def calc_land_use(self, model):
        """Calculate used land for installed capacity [m2]"""
        logger_main.info('Calc land use')

        def land_use_rule(m, N, Y):
            """Sum over tech and storage tech"""
            return sum(m.inst_cap[N,T,Y] * m.Land_use[T] for T in m.Tech - m.StorageTech) + sum(m.inst_storage_vol[N,StoreT,Y] * m.Land_use[StoreT] for StoreT in m.StorageTech)

        model.land_use = pyo.Expression(model.Node, model.Year, rule=land_use_rule)


    def calc_ldc(self, model, Node, Tech, Year):
        """ """
        if type(Tech) is not list:
            Tech = [Tech]

        #inst_cap = dict()
        values = []

        for T in Tech:
            if model.Cap_of_input[T]:
                # Electrolysis: f_cons and E_input are relevant for load
                F = [F for F in model.Fuel if model.E_input[T,F]][0]
                values.append([pyo.value(sum(model.f_cons[Node,T,F,F1,Year,D,H,sH] for F1 in model.Fuel1 if model.F_subst[F,F1])) for D in model.Day for H in model.Hour for _ in range(pyo.value(model.Delta_H)) for sH in model.SubHour])

            else:
                # GenSet: f_prod and E_output are relevant for load
                F1 = [F1 for F1 in model.Fuel1 if model.E_output[T,F1]][0]
                values.append([pyo.value(model.f_prod[Node, T, F1, Year, D, H, sH]) for D in model.Day for H in model.Hour for _ in range(pyo.value(model.Delta_H)) for sH in model.SubHour])

        # Sum of rows, unsorted
        ldc = np.sum(values, 0)

        # Sort
        ldc = - np.sort(-ldc)

        # Normalize
        #inst_cap_sum = sum([inst_cap[k] for k in inst_cap.keys()])
        #if inst_cap_sum == 0:
        #    ldc = [0 for v in ldc]
        #else:
        #    ldc = [v/inst_cap_sum for v in ldc]

        header = '{}_{}_Y{}'.format('+'.join(Tech), Node, Year)

        return (header, ldc)


    def calc_kpi(self, model):
        """Calculate Key Performance Indicators and store them as attribute to model"""
        logger_main.info('Calculating KPIs.')
        start = datetime.now()

        def flh_rule(m, N, T, Y):
            """FLH of simulated operation [h]"""

            inst_cap = m.inst_cap[N, T, Y]
            if inst_cap == 0:
                return 0
            else:

                if model.Cap_of_input[T]:
                    F = [F for F in model.Fuel if model.E_input[T,F]][0]
                    load = sum(m.f_cons[N,T,F,F1,Y,D,H,sH] for F1 in model.Fuel1 if model.F_subst[F,F1] for D in m.Day for H in m.Hour for sH in m.SubHour) * m.Delta_T * m.Scale_H * m.Scale_D
                else:
                    F1 = [F for F in model.Fuel if model.E_output[T,F]][0]
                    load = sum(m.f_prod[N,T,F1,Y,D,H,sH] for D in m.Day for H in m.Hour for sH in m.SubHour) * m.Delta_T * m.Scale_H * m.Scale_D

                return load / inst_cap

        model.flh = pyo.Expression(model.Node, model.Tech - model.StorageTech, model.Year, rule=flh_rule)

        def flh_possible_rule(m, N, T, Y):
            """Possible FLH of technology ignoring constraints of simulated system [h]"""

            return sum(m.Availability[N, T, Y, D, H, sH] for D in m.Day for H in m.Hour for sH in m.SubHour) * m.Delta_T * m.Scale_H * m.Scale_D

        model.flh_possible = pyo.Expression(model.Node, model.Tech, model.Year, rule=flh_possible_rule)


        def storage_ratio_rule(m, N, StoreT, F, Y):
            """Storeable energy volume in relation to average daily consumption"""

            if m.E_output[StoreT, F]:
                inst_storage_vol = model.inst_storage_vol[N,StoreT,Y]

                # Daily consumption + exogenous demand
                daily_cons = [sum(sum(m.f_cons[N,T,F,F1,Y,D,H,sH] for T in (m.Tech - m.StorageTech)) \
                                  + m.f_delivery[N,F,F1,Y,D,H,sH] for F1 in m.Fuel1 if m.F_subst[F, F1] for H in m.Hour for sH in m.SubHour) * m.Delta_T * m.Scale_H for D in m.Day]

                avg_cons = sum(daily_cons) / len(daily_cons)

                if pyo.value(avg_cons) == 0:
                    return 0
                else:
                    return inst_storage_vol / avg_cons
            else:
                return 0

        model.storage_ratio = pyo.Expression(model.Node, model.StorageTech, model.Fuel, model.Year, rule=storage_ratio_rule)

        dt = datetime.now() - start
        logger_main.info('\tDuration: {}min {}sec'.format(int(dt.total_seconds()/60), round(dt.total_seconds()%60)))



    def calc_energy_share(self, model):
        """
        Calculate electricity production-consumption share for each time step, then sum up.
        """
        logger_main.info('Calculating energy share.')
        start = datetime.now()

        # Total availability of electricity [kW]
        F = 'Electricity'
        power_total = [pyo.value(sum(model.f_prod[N,T,F1,Y,D,H,sH] for N in model.Node for T in model.Tech for F1 in model.Fuel1 if model.F_subst[F, F1]) \
                                 + sum(model.f_import[N,F1,Y,D,H,sH] +  model.f_import_timeseries[N,F1,Y,D,H,sH] + model.f_fix_quant_import[N,F1,Y,D,H,sH] for N in model.Node for F1 in model.Fuel1 if model.F_subst[F, F1])) \
                       for Y in model.Year for D in model.Day for H in model.Hour for sH in model.SubHour]

        # Create matrix with producers in rows and consumers in columns
        add_producer = ['f_import', 'f_import_timeseries']
        add_consumer = ['f_export', 'f_export_timeseries', 'f_delivery']
        manual_consumer = ['const_cons_system']
        dim = (len(model.Tech)+len(add_producer), len(model.Tech)+len(add_consumer)+len(manual_consumer))
        M = pd.DataFrame(np.zeros(dim), index=[p for p in model.Tech]+add_producer, columns=[c for c in model.Tech]+add_consumer+manual_consumer)

        # Loop through producer
        for T in model.Tech:
            # Power of considered tech [kW]
            power_tech = [pyo.value(sum(model.f_prod[N,T,F1,Y,D,H,sH] for N in model.Node for F1 in model.Fuel1 if model.F_subst[F, F1])) for Y in model.Year for D in model.Day for H in model.Hour for sH in model.SubHour]
            share_tech = [x/y if y != 0 else 0 for x,y in zip(power_tech, power_total)]

            # Loop through consumer
            for C in model.Tech:
                # load of considered tech [kW]
                power_consumer = [pyo.value(sum(model.f_cons[N,C,F,F1,Y,D,H,sH] for N in model.Node for F1 in model.Fuel1 if model.F_subst[F, F1])) for Y in model.Year for D in model.Day for H in model.Hour for sH in model.SubHour]

                # Energy share over modelled time range [kWh]
                M.loc[T,C] = sum([x*y for x,y in zip(share_tech, power_consumer)]) * pyo.value(model.Delta_T * model.Scale_H * model.Scale_D * model.Scale_Y)

            # Loop through added consumers
            for C in add_consumer:
                power_consumer = [pyo.value(sum(model.component(C)[N,F,F1,Y,D,H,sH] for N in model.Node for F1 in model.Fuel1 if model.F_subst[F, F1])) for Y in model.Year for D in model.Day for H in model.Hour for sH in model.SubHour]

                # Energy share over modelled time range [kWh]
                M.loc[T,C] = sum([x*y for x,y in zip(share_tech, power_consumer)]) * pyo.value(model.Delta_T * model.Scale_H * model.Scale_D * model.Scale_Y)

            # For const_cons_system as 'manual_consumer'
            C = 'Const_cons_system'
            power_consumer = [pyo.value(sum(model.component(C)[N,F1,Y] for F1 in model.Fuel1 if model.F_subst[F,F1] for N in model.Node)) for Y in model.Year for D in model.Day for H in model.Hour for sH in model.SubHour]
            M.loc[T,C] = sum([x*y for x,y in zip(share_tech, power_consumer)]) * pyo.value(model.Delta_T * model.Scale_H * model.Scale_D * model.Scale_Y)

        # Again for P = f_import & f_import_timeseries
        for P in add_producer:
            power_tech = [pyo.value(sum(model.component(P)[N,F1,Y,D,H,sH] for N in model.Node for F1 in model.Fuel1 if model.F_subst[F, F1])) for Y in model.Year for D in model.Day for H in model.Hour for sH in model.SubHour]
            share_tech = [x/y if y != 0 else 0 for x,y in zip(power_tech, power_total)]

            # Loop through consumer
            for C in model.Tech:
                # load of considered tech [kW]
                power_consumer = [pyo.value(sum(model.f_cons[N,C,F,F1,Y,D,H,sH] for N in model.Node for F1 in model.Fuel1 if model.F_subst[F, F1])) for Y in model.Year for D in model.Day for H in model.Hour for sH in model.SubHour]

                # Energy share over modelled time range [kWh]
                M.loc[P,C] = sum([x*y for x,y in zip(share_tech, power_consumer)]) * pyo.value(model.Delta_T * model.Scale_H * model.Scale_D * model.Scale_Y)

            # Loop through added consumers
            for C in add_consumer:
                power_consumer = [pyo.value(sum(model.component(C)[N,F,F1,Y,D,H,sH] for N in model.Node for F1 in model.Fuel1 if model.F_subst[F, F1])) for Y in model.Year for D in model.Day for H in model.Hour for sH in model.SubHour]

                # Energy share over modelled time range [kWh]
                M.loc[P,C] = sum([x*y for x,y in zip(share_tech, power_consumer)]) * pyo.value(model.Delta_T * model.Scale_H * model.Scale_D * model.Scale_Y)

            # For const_cons_system as 'manual_consumer'
            C = 'Const_cons_system'
            power_consumer = [pyo.value(sum(model.component(C)[N,F1,Y] for F1 in model.Fuel1 if model.F_subst[F,F1] for N in model.Node)) for Y in model.Year for D in model.Day for H in model.Hour for sH in model.SubHour]
            M.loc[P,C] = sum([x*y for x,y in zip(share_tech, power_consumer)]) * pyo.value(model.Delta_T * model.Scale_H * model.Scale_D * model.Scale_Y)

        # Store matrix to model
        model.energy_share = M

        dt = datetime.now() - start
        logger_main.info('\tDuration: {}min {}sec'.format(int(dt.total_seconds()/60), round(dt.total_seconds()%60)))




    def calc_LCOEnergy(self, model):
        """
        Levelized Costs of Energy:
            Calculate LCOE, LCOH (Low and high pressure)
            [EUR/MWh]
        """
        logger_main.info('Calculating LCOEnergy.')
        start = datetime.now()
        # Fuel production at nodes over time [kW]
        model.f_prod_node = pyo.Param(model.Node, model.Fuel1, model.Year, model.Day, model.Hour, model.SubHour, default=0.0, mutable=True)

        # Parameter tracking internal fuel costs at nodes over time [EUR/MWh]
        # Set equal to 'F_costs' for first iteration
        model.lcoEnergy = pyo.Param(model.Node, model.Fuel1, model.Year, model.Day, model.Hour, model.SubHour, default=0.0, mutable=True)

        # Loop through sets
        for N in model.Node:
            for F1 in model.Fuel1:
                for Y in model.Year:
                    for D in model.Day:
                        for H in model.Hour:
                            for sH in model.SubHour:
                                model.f_prod_node[N,F1,Y,D,H,sH] = pyo.value(sum(model.f_prod[N,T,F1,Y,D,H,sH] for T in model.Tech) + model.f_import[N,F1,Y,D,H,sH] + model.f_import_timeseries[N,F1,Y,D,H,sH] + model.f_fix_quant_import[N,F1,Y,D,H,sH] * model.F_fix_quant_import_size[N,F1] / model.Delta_T)

        # Calculate constant numbers per technology and fuel
        def ideal_prod_rule(m, N, T, F1, Y):
            """Possible annual energy production [kWh]"""

            if m.E_output[T,F1]:
                if T in m.StorageTech:
                    Eff_temp = m.Eff[T]
                elif m.Cap_of_input[T]:   # eg: Electrolysis
                    Eff_temp = m.Eff[T]
                else:                   # eg: GenSet, Wind
                    Eff_temp = 1

                return m.inst_cap[N,T,Y] * sum(m.Availability[N,T,Y,D,H,sH] for D in m.Day for H in m.Hour for sH in m.SubHour) * Eff_temp * m.Delta_T * m.Scale_H * m.Scale_D
            else:
                return 0

        model.ideal_prod = pyo.Expression(model.Node, model.Tech, model.Fuel1, model.Year, rule=ideal_prod_rule)

        def ideal_cons_rule(m, N, T, F, Y):
            """Possible annual energy consumption [kWh]"""
            if m.E_input[T, F]:
                if T in m.StorageTech:
                    Eff_temp = m.Eff[T]
                elif m.Cap_of_input[T]:   # eg: Electrolysis
                    Eff_temp = 1
                else:                   # eg: GenSet, Wind
                    Eff_temp = m.Eff[T]

                return m.inst_cap[N,T,Y] * sum([m.Availability[N,T,Y,D,H,sH] for D in m.Day for H in m.Hour for sH in m.SubHour]) / Eff_temp * m.Delta_T * m.Scale_H * m.Scale_D
            else:
                return 0

        model.ideal_cons = pyo.Expression(model.Node, model.Tech, model.Fuel, model.Year, rule=ideal_cons_rule)


        def fixed_costs_rule(m, N, T, Y):
            """[EUR]"""
            return m.inst_cap[N,T,Y] * m.Fo_costs[T,Y]
        model.fixed_costs = pyo.Expression(model.Node, model.Tech, model.Year, rule=fixed_costs_rule)

        # Define parameters for following calculation
        model.lcoEnergy_mean = pyo.Param(model.Node, model.Fuel, model.Year, default=0.0, mutable=True)

        model.fuel_costs_ideal = pyo.Param(model.Node, model.Tech, model.Year, default=0.0, mutable=True)
        model.fuel_costs = pyo.Param(model.Node, model.Tech, model.Year, default=0.0, mutable=True)

        model.var_costs_ideal = pyo.Param(model.Node, model.Tech, model.Year, default=0.0, mutable=True)
        model.var_costs = pyo.Param(model.Node, model.Tech, model.Year, default=0.0, mutable=True)

        model.tech_LCOEnergy_ideal = pyo.Param(model.Node, model.Tech, model.Fuel, model.Year, default=0.0, mutable=True)
        model.tech_LCOEnergy = pyo.Param(model.Node, model.Tech, model.Fuel, model.Year, default=0.0, mutable=True)
        model.tech_LCOEnergy_cons = pyo.Param(model.Node, model.Tech, model.Fuel, model.Year, default=0.0, mutable=True)
        model.export_LCOEnergy = pyo.Param(model.Node, model.Fuel, model.Year, default=0.0, mutable=True)
        model.demand_LCOEnergy = pyo.Param(model.Node, model.Fuel, model.Year, default=0.0, mutable=True)

		# Tracking LCOE while interation
        model.pre_lcoe = dict(zip(model.RefFuel, [0]*len(model.RefFuel)))

        model.delta_lcoe = float('inf')

        # Begin iteration
        i = 1
        min_inter = 4; max_iter = 20
        while (model.delta_lcoe > self.zero_threshold or i < min_inter)  and i < max_iter:
            print('Iteration = {} | sum(delta) = {}'.format(i, model.delta_lcoe))
            for N in model.Node:
                for F1 in model.Fuel:
                    for Y in model.Year:

                        w_lcoe = [pyo.value(model.lcoEnergy[N,F1,Y,D,H,sH] * (sum(model.f_prod[N,T,F1,Y,D,H,sH] for T in model.Tech) + model.f_import[N,F1,Y,D,H,sH] + model.f_import_timeseries[N,F1,Y,D,H,sH] + model.f_fix_quant_import[N,F1,Y,D,H,sH] * model.F_fix_quant_import_size[N,F1] / model.Delta_T)) \
                                for D in model.Day for H in model.Hour for sH in model.SubHour]

                        nominator = pyo.value(sum(sum(model.f_prod[N,T,F1,Y,D,H,sH] for T in model.Tech) + model.f_import[N,F1,Y,D,H,sH] + model.f_import_timeseries[N,F1,Y,D,H,sH] + model.f_fix_quant_import[N,F1,Y,D,H,sH] * model.F_fix_quant_import_size[N,F1] / model.Delta_T for D in model.Day for H in model.Hour for sH in model.SubHour))

                        if nominator == 0:
                            model.lcoEnergy_mean[N,F1,Y] = 0
                        else:
                            model.lcoEnergy_mean[N,F1,Y] = sum(w_lcoe) / nominator

            self.update_tech_LCOEnergy(model)

            self.update_LCOEnergy(model)
            i += 1

        if model.delta_lcoe <= self.zero_threshold:
            print('Iteration reached convergence criteria')
        elif i >= max_iter:
            print('Iteration reached max interation number')

        dt = datetime.now() - start
        logger_main.info('\tDuration: {}min {}sec'.format(int(dt.total_seconds()/60), round(dt.total_seconds()%60)))



    def update_LCOEnergy(self, m):
        """ Update Levelized Costs of Energy at nodes over time [EUR/MWh]"""
        #logger_main.info('Updating LCOEnergy at nodes.')
        #start = datetime.now()

        for N in m.Node:
            for F1 in m.Fuel1:
                conv = self.unitConv[m.F_unit[F1]]['conv']

                for Y in m.Year:
                    for D in m.Day:
                        for H in m.Hour:
                            for sH in m.SubHour:

                                # Join streams of available energy
                                f_sum = pyo.value(sum(m.f_prod[N,T,F1,Y,D,H,sH] for T in m.Tech) + m.f_import[N,F1,Y,D,H,sH] + m.f_import_timeseries[N,F1,Y,D,H,sH] + m.f_fix_quant_import[N,F1,Y,D,H,sH] * m.F_fix_quant_import_size[N,F1] / m.Delta_T)
                                if f_sum < self.zero_threshold:
                                    m.lcoEnergy[N,F1,Y,D,H,sH] = 0
                                else:
                                    m.lcoEnergy[N,F1,Y,D,H,sH] = 1/f_sum * pyo.value(sum(m.f_prod[N,T,F1,Y,D,H,sH] * m.tech_LCOEnergy[N,T,F1,Y] for T in m.Tech) \
                                                                           + m.f_import[N,F1,Y,D,H,sH] * m.F_costs[F1,Y] * conv \
                                                                           + m.f_import_timeseries[N,F1,Y,D,H,sH] * (m.F_import_timeseries_price[F1,Y,D,H,sH] + m.F_import_timeseries_fee[F1,Y]) * conv \
                                                                           + m.f_fix_quant_import[N,F1,Y,D,H,sH] * m.F_fix_quant_import_size[N,F1] / m.Delta_T * m.F_fix_quant_costs[F1,Y] * conv)

        #dt = datetime.now() - start
        #logger_main.info('\tDuration: {}min {}sec'.format(int(dt.total_seconds()/60), round(dt.total_seconds()%60)))


    def update_tech_LCOEnergy(self, m):
        """ Update Levelized Costs of Energy of technology at nodes over time [ERU/MWh]"""
        #start = datetime.now()
        print_switch = True
        for N in m.Node:
            for Y in m.Year:

                # Print interation results
                if print_switch:
                    print_switch = False
                    m.delta_lcoe = 0 # reset delta tracker

                    for F in m.pre_lcoe.keys():
                        lcoe = pyo.value(m.lcoEnergy_mean[N,F,Y])
                        m.delta_lcoe += abs(lcoe - m.pre_lcoe[F])
                        unit = m.F_unit[F]

                        print('\tLCOE mean {} [EUR/{}] \t= {}  (delta = {})'.format(F, unit, round(lcoe, 4), lcoe - m.pre_lcoe[F]))
                        m.pre_lcoe.update({F: lcoe})

                # export_LCOEnergy and demand_LCOEnergy
                for F in m.Fuel:

                    # Export of energy
                    sum_f_export = pyo.value(sum(m.f_export[N,F,F1,Y,D,H,sH] + m.f_export_timeseries[N,F,F1,Y,D,H,sH] for F1 in m.Fuel1 if m.F_subst[F,F1] for D in m.Day for H in m.Hour for sH in m.SubHour))
                    if  sum_f_export < self.zero_threshold:
                        m.export_LCOEnergy[N,F,Y] = 0
                    else:
                        m.export_LCOEnergy[N,F,Y] = pyo.value(sum((m.f_export[N,F,F1,Y,D,H,sH] + m.f_export_timeseries[N,F,F1,Y,D,H,sH]) * m.lcoEnergy[N,F1,Y,D,H,sH] for F1 in m.Fuel1 if m.F_subst[F,F1] for D in m.Day for H in m.Hour for sH in m.SubHour)) / sum_f_export

                    # Supply of demand of energy
                    sum_f_demand = pyo.value(sum(m.F_demand[N,F,Y,D,H,sH] for D in m.Day for H in m.Hour for sH in m.SubHour))
                    if  sum_f_demand < self.zero_threshold:
                        m.demand_LCOEnergy[N,F,Y] = 0
                    else:
                        m.demand_LCOEnergy[N,F,Y] = pyo.value(sum(m.f_delivery[N,F,F1,Y,D,H,sH] * m.lcoEnergy[N,F1,Y,D,H,sH] for F1 in m.Fuel1 if m.F_subst[F,F1] for D in m.Day for H in m.Hour for sH in m.SubHour)) / sum_f_demand


                # tech_LCOEnergy_cons, tech_LCOEnergy_ideal and tech_LCOEnergy

                # Costs of system components and external technology
                capex_system_disc = pyo.value(m.Capex_system_disc + sum(m.Capex_disc[T] for T in m.ExternalTech))
                opex_system_disc = pyo.value(m.Opex_system_disc + sum(m.Opex_disc[T] for T in m.ExternalTech))

                # Sum capex of technologies as reference
                capex_tech_disc = pyo.value(sum(m.Capex_disc[T] for T in m.Tech if m.System_tech[T]))

                for T in m.Tech:

                    if pyo.value(m.inst_cap[N,T,Y]) < self.zero_threshold:
                        continue
                    for F in m.Fuel:

                        # Energy consumption of technology
                        sum_f_cons = pyo.value(sum(m.f_cons[N,T,F,F1,Y,D,H,sH] for F1 in m.Fuel1 if m.F_subst[F,F1] for D in m.Day for H in m.Hour for sH in m.SubHour))
                        if  sum_f_cons < self.zero_threshold:
                            m.tech_LCOEnergy_cons[N,T,F,Y] = 0
                        else:
                            m.tech_LCOEnergy_cons[N,T,F,Y] = pyo.value(sum(m.f_cons[N,T,F,F1,Y,D,H,sH] * m.lcoEnergy[N,F1,Y,D,H,sH] for F1 in m.Fuel1 if m.F_subst[F,F1] for D in m.Day for H in m.Hour for sH in m.SubHour)) / sum_f_cons

                        m.fuel_costs_ideal[N,T,Y] = pyo.value(sum((m.ideal_cons[N,T,F,Y] + m.Share_const_cons_system[F] * m.inst_cap[N,T,Y] * m.Tech_const_cons_system[T] * len(m.Day)*len(m.Hour)*len(m.SubHour)*m.Delta_T*m.Scale_H*m.Scale_D) * m.lcoEnergy_mean[N,F,Y] for F in m.Fuel) \
                                                  + sum(m.ideal_prod[N,T,F1,Y] for F1 in m.Fuel1) * m.Aux_ed[T] * m.lcoEnergy_mean[N,'Electricity',Y] \
                                                  + sum((m.F_edp[N,T,Y,D,H,sH] + m.inst_cap[N,T,Y] * m.V_edp[N,T,Y,D,H,sH]) * m.lcoEnergy[N,'Electricity',Y,D,H,sH] for D in m.Day for H in m.Hour for sH in m.SubHour) * m.Delta_T * m.Scale_H * m.Scale_D)
                                                    # missing AuxMedium costs

                        m.fuel_costs[N,T,Y] = pyo.value(sum((m.f_cons[N,T,F,F1,Y,D,H,sH] + m.Share_const_cons_system[F1] * m.inst_cap[N,T,Y] * m.Tech_const_cons_system[T]) * m.lcoEnergy[N,F1,Y,D,H,sH] for F in m.Fuel if m.E_input[T,F] or F=='Electricity' for F1 in m.Fuel1 if m.F_subst[F,F1] for D in m.Day for H in m.Hour for sH in m.SubHour) * m.Delta_T * m.Scale_H * m.Scale_D \
                                                      + sum(m.Aux_medium_flow[N,T,A,Y,D,H,sH] * m.AM_costs[A,Y] for A in m.AuxMedium if m.AM_costs[A,Y] > 0 for D in m.Day for H in m.Hour for sH in m.SubHour) * m.Delta_T * m.Scale_H * m.Scale_D)

                        m.var_costs_ideal[N,T,Y] = pyo.value(sum(m.ideal_prod[N,T,F1,Y] for F1 in m.Fuel1) * m.Vo_costs[T, Y])
                        m.var_costs[N,T,Y] = pyo.value(sum(m.f_prod_y[N,T,F1,Y] for F1 in m.Fuel1) * m.Vo_costs[T, Y])


                        # tech_LCOEnergy_ideal
                        if m.E_output[T,F]:

                            # Energy
                            prod_disc = self.calc_disc_opex_postprocessing(m, m.ideal_prod, N,T,F)

                            # Costs
                            fuel_costs_disc = self.calc_disc_opex_postprocessing(m, m.fuel_costs_ideal, N, T)
                            fixed_costs_disc = self.calc_disc_opex_postprocessing(m, m.fixed_costs, N, T)
                            var_costs_disc = self.calc_disc_opex_postprocessing(m, m.var_costs_ideal, N, T)
                            capex_disc = self.calc_disc_capex_postprocessing(m, m.capex, N, T)

                            # Share at system costs and external costs
                            system_share_disc = (capex_disc/capex_tech_disc) * (capex_system_disc + opex_system_disc) if m.System_tech[T] else 0

                            # LCOE = 0 if electricity consumption and production are equal
                            if pyo.value(prod_disc) < self.zero_threshold:
                                m.tech_LCOEnergy_ideal[N,T,F,Y] = 0
                            else:
                                m.tech_LCOEnergy_ideal[N,T,F,Y] = pyo.value((capex_disc + fuel_costs_disc + fixed_costs_disc + var_costs_disc + system_share_disc) / prod_disc)

                        else:
                            m.tech_LCOEnergy_ideal[N,T,F,Y] = 0

                        # tech_LCOEnergy
                        if m.E_output[T,F]:

                            # Energy
                            if T in m.StorageTech:
                                # Storages can discharge several fuels, as they charge several fuels
                                prod_disc = sum(self.calc_disc_opex_postprocessing(m, m.f_prod_y, N,T,F1) for F1 in m.Fuel1 if m.F_subst[F,F1])
                            else:
                                prod_disc = self.calc_disc_opex_postprocessing(m, m.f_prod_y, N,T,F)

                            # Costs
                            fuel_costs_disc = self.calc_disc_opex_postprocessing(m, m.fuel_costs, N, T)
                            fixed_costs_disc = self.calc_disc_opex_postprocessing(m, m.fixed_costs, N, T)
                            var_costs_disc = self.calc_disc_opex_postprocessing(m, m.var_costs, N, T)
                            capex_disc = self.calc_disc_capex_postprocessing(m, m.capex, N, T)

                            # Share at system costs and external costs
                            system_share_disc = (capex_disc/capex_tech_disc) * (capex_system_disc + opex_system_disc) if m.System_tech[T] else 0

                            # LCOE = 0 if electricity consumption and production are equal
                            if pyo.value(prod_disc) < self.zero_threshold:
                                continue
                                #m.tech_LCOEnergy[N,T,F,Y] = 0
                            else:
                                lcoe_temp = pyo.value((capex_disc + fuel_costs_disc + fixed_costs_disc + var_costs_disc + system_share_disc) / prod_disc)

                                if T in m.StorageTech:
                                    # Define LCOE for storage output for all possible fuels
                                    for F1 in [F1 for F1 in m.Fuel1 if m.F_subst[F,F1]]:
                                        m.tech_LCOEnergy[N,T,F1,Y] = lcoe_temp
                                else:
                                    m.tech_LCOEnergy[N,T,F,Y] = lcoe_temp

                        else:
                            continue


    def calc_annual_values(self, model):
        """
        Calculate annual values and store as attribute to model.
        """
        start = datetime.now()
        logger_main.info('Calculating annual values')

        # Yearly fuel demand [kWh]
        def sum_F_demand(m, F, Y):
            """[kWh]"""
            return sum(m.F_demand[N,F,Y,D,H,sH] for N in m.Node for D in m.Day for H in m.Hour for sH in m.SubHour) * m.Delta_T * m.Scale_H * m.Scale_D
        model.F_demand_y = pyo.Expression(model.Fuel, model.Year, rule=sum_F_demand)

        # Yearly fuel delivery [kWh]
        def sum_f_delivery(m, F, F1, Y):
            """[kWh]"""
            return sum(m.f_delivery[N,F,F1,Y,D,H,sH] for N in m.Node for D in m.Day for H in m.Hour for sH in m.SubHour) * m.Delta_T * m.Scale_H * m.Scale_D
        model.f_delivery_y = pyo.Expression(model.Fuel, model.Fuel1, model.Year, rule=sum_f_delivery)

        # Yearly fuel consumption and production
        def sum_f_cons(m, N, T, F, F1, Y):
            """[kWh]"""
            return sum(m.f_cons[N,T,F,F1,Y,D,H,sH] for D in m.Day for H in m.Hour for sH in m.SubHour) * m.Delta_T * m.Scale_H * m.Scale_D
        model.f_cons_y = pyo.Expression(model.Node, model.Tech, model.Fuel, model.Fuel1, model.Year, rule=sum_f_cons)

        def sum_f_prod(m, N, T, F1, Y):
            """[kWh]"""
            return sum(m.f_prod[N,T,F1,Y,D,H,sH] for D in m.Day for H in m.Hour for sH in m.SubHour) * m.Delta_T * m.Scale_H * m.Scale_D
        model.f_prod_y = pyo.Expression(model.Node, model.Tech, model.Fuel1, model.Year, rule=sum_f_prod)

        # Yearly sum of auxiliary medium flow
        def sum_Aux_medium_flow(m, N, A, Y):
            """[Nm3]"""
            return sum(m.Aux_medium_flow[N,T,A,Y,D,H,sH] for T in model.Tech for D in m.Day for H in m.Hour for sH in m.SubHour) * m.Delta_T * m.Scale_H * m.Scale_D
        model.Aux_medium_flow_y = pyo.Expression(model.Node, model.AuxMedium, model.Year, rule=sum_Aux_medium_flow)

        # Yearly fuel import and export
        def sum_f_import(m, F1, Y):
            """[kWh]"""
            return sum(m.f_import[N,F1,Y,D,H,sH] for N in m.Node for D in m.Day for H in m.Hour for sH in m.SubHour) * m.Delta_T * m.Scale_H * m.Scale_D
        model.f_import_y = pyo.Expression(model.Fuel1, model.Year, rule=sum_f_import)

        # Yearly sum of constant fuel consumption of system
        def sum_f_supply_cons_system(m, F, F1, Y):
            """[kWh]"""
            return sum(m.f_supply_cons_system[N,F,F1,Y,D,H,sH] for N in m.Node for D in m.Day for H in m.Hour for sH in m.SubHour) * m.Delta_T * m.Scale_H * m.Scale_D
        model.f_supply_cons_system_y = pyo.Expression(model.Fuel, model.Fuel1, model.Year, rule=sum_f_supply_cons_system)

        def sum_f_import_timeseries(m,F1,Y):
            """[kWh]"""
            return sum(m.f_import_timeseries[N,F1,Y,D,H,sH] for N in m.Node for D in m.Day for H in m.Hour for sH in m.SubHour) * m.Delta_T * m.Scale_H * m.Scale_D
        model.f_import_timeseries_y = pyo.Expression(model.Fuel1, model.Year, rule=sum_f_import_timeseries)


        def sum_f_fix_quant_import(m, F1, Y):
            """ ! Fixed quantity fuel import is not scaled by resolution ! [kWh] """
            return sum(m.f_fix_quant_import[N,F1,Y,D,H,sH] * m.F_fix_quant_import_size[N, F1] for N in m.Node for D in m.Day for H in m.Hour for sH in m.SubHour) * m.Scale_D
        model.f_fix_quant_import_y = pyo.Expression(model.Fuel1, model.Year, rule=sum_f_fix_quant_import)

        def sum_f_export(m, F, F1, Y):
            """[kWh]"""
            return sum(m.f_export[N,F,F1,Y,D,H,sH] for N in m.Node for D in m.Day for H in m.Hour for sH in m.SubHour) * m.Delta_T * m.Scale_H * m.Scale_D
        model.f_export_y = pyo.Expression(model.Fuel, model.Fuel1, model.Year, rule=sum_f_export)

        def sum_f_export_timeseries(m, F, F1, Y):
            """[kWh]"""
            return sum(m.f_export_timeseries[N,F,F1,Y,D,H,sH] for N in m.Node for D in m.Day for H in m.Hour for sH in m.SubHour) * m.Delta_T * m.Scale_H * m.Scale_D
        model.f_export_timeseries_y = pyo.Expression(model.Fuel, model.Fuel1, model.Year, rule=sum_f_export_timeseries)

        dt = datetime.now() - start
        logger_main.info('\tDuration: {}min {}sec'.format(int(dt.total_seconds()/60), round(dt.total_seconds()%60)))


    def calc_prices(self, model):
        """
        Calculate costs of reference fuels with regards to
            a) sum of produced fuel [ref_prod_disc]
            b) sum of fuel satisfied demand [ref_demand_disc]
            c) sum of fuel exported [ref_export_disc]
            d) sum of fuel crossed the system boundaries (demand + export) [ref_out_disc]
        """
        start = datetime.now()
        logger_main.info('Calculating fuel costs')

        # Calc reference values (discounted)
        model.ref_prod_disc = pyo.value(sum(self.calc_disc_opex_postprocessing(model, model.f_prod_y, index1=N, index2=T, index3=F1) for N in model.Node for T in model.Tech-model.StorageTech for F1 in model.Fuel1 for F in model.RefFuel if model.F_subst[F,F1] and sum(model.F_subst[F1,F2] for F2 in model.Fuel1) == 1))
        model.ref_demand_disc = pyo.value(sum(self.calc_disc_opex_postprocessing(model, model.F_demand_y, index1=F) for F in model.RefFuel))
        model.ref_export_disc = pyo.value(sum(self.calc_disc_opex_postprocessing(model, model.f_export_y, index1=F, index2=F1) for F1 in model.Fuel1 for F in model.RefFuel if model.F_subst[F,F1] and sum(model.F_subst[F1,F2] for F2 in model.Fuel1) == 1))
        model.ref_out_disc = model.ref_demand_disc + model.ref_export_disc


        # Total costs
        model.tc = pyo.value(model.tc_obj - sum(model.high_storage_level_incentive[StoreT] for StoreT in model.StorageTech))

        # Costs of reference fuels [EUR/ unit of fuel]
        if model.ref_prod_disc > self.zero_threshold:
            model.ref_prod_price = model.tc / model.ref_prod_disc
        else:
            model.ref_prod_price = 0

        if model.ref_demand_disc > self.zero_threshold:
            model.ref_demand_price = model.tc / model.ref_demand_disc
        else:
            model.ref_demand_price = 0

        if model.ref_export_disc > self.zero_threshold:
            model.ref_export_price = model.tc / model.ref_export_disc
        else:
            model.ref_export_price = 0

        if model.ref_out_disc > self.zero_threshold:
            model.ref_out_price = model.tc / model.ref_out_disc
        else:
            model.ref_out_price = 0


        # Discouting of costs
        model.total_Capex_disc = pyo.value(sum(model.Capex_disc[T] for T in model.Tech | model.ExternalTech))
        model.total_Capex_system_disc = pyo.value(model.Capex_system_disc)

        model.total_Opex_disc = pyo.value(sum(model.Opex_disc[T] for T in model.Tech | model.ExternalTech))
        model.total_Opex_system_disc = pyo.value(model.Opex_system_disc)
        model.total_Opex_fuel_disc = pyo.value(sum(model.Opex_fuel_disc[F1] for F1 in model.Fuel1))
        model.total_Opex_timeseries_disc = pyo.value(sum(model.Opex_timeseries_disc[F1] for F1 in model.Fuel1))
        model.total_Opex_auxmedium_disc = pyo.value(sum(model.Opex_auxmedium_disc[A] for A in model.AuxMedium))
        model.total_Opex_network_capacity_disc = pyo.value(sum(model.Opex_network_capacity_disc[F] for F in model.Fuel))

        model.total_Revenue_disc = pyo.value(sum(model.Revenue_disc[F] for F in model.Fuel))
        model.total_Revenue_timeseries_disc = pyo.value(sum(model.Revenue_timeseries_disc[F] for F in model.Fuel))

        dt = datetime.now() - start
        logger_main.info('\tDuration: {}min {}sec'.format(int(dt.total_seconds()/60), round(dt.total_seconds()%60)))


    def write_ldc(self, model, file_name='model_load_duration_curve'):
        """Load Duration Curve"""
        logger_main.info('Writing load duration curves to file')
        start = datetime.now()

        # Define file name
        file_name = self.pre_output + file_name + ('_fixed' if self.fixed else '') + '.csv'

        # Collect technology names
        searchTech = 'Electrolysis'
        addTech = [T for T in model.Tech - model.StorageTech if searchTech.lower() in T.lower()]

        searchTech = 'Wind'
        windTech = [T for T in model.Tech - model.StorageTech if searchTech.lower() in T.lower()]

        searchTech = 'Solar'
        solarTech = [T for T in model.Tech - model.StorageTech if searchTech.lower() in T.lower()]

        Tech = [windTech + solarTech] + windTech + solarTech + addTech

        value = []
        header = ['Index']

        for T in Tech:
            for N in model.Node:
                for Y in model.Year:

                    (head, ldc) = self.calc_ldc(model, N, T, Y)

                    value.append(ldc)
                    header.append(head)

        values = np.column_stack(value)
        index = [i+1 for i in range(values.shape[0])]

        table = np.column_stack((index, values))

        file = open(self.repository + file_name, "w")
        np.savetxt(file, table, fmt='%s', header=';'.join(header), delimiter=';', comments='')
        file.close()

        dt = datetime.now() - start
        logger_main.info('\tDuration: {}min {}sec'.format(int(dt.total_seconds()/60), round(dt.total_seconds()%60)))


    def write_energy_share(self, model, file_name='energy_share'):

        start = datetime.now()
        logger_main.info('Writing energy share matrix:')
        F = 'Electricity'
        unit = self.unitConv[model.F_unit[F]]['newUnit']
        conv = self.unitConv[model.F_unit[F]]['conv']

        # Header
        header = '{} share over project time [{}] (rows: producer; cols: consumer)'.format(F, unit)

        # Scale values
        df = model.energy_share * conv

        # Set file_name
        file_name = self.repository + self.pre_output + file_name + ('_fixed' if self.fixed else '') + '.dat'

        f = open(file_name, 'w')
        f.write('# ' + header + '\n')
        df.to_csv(f, sep='\t', index=True, header=True, line_terminator='\n')
        f.close()

        dt = datetime.now() - start
        logger_main.info('\tDuration: {}min {}sec'.format(int(dt.total_seconds()/60), round(dt.total_seconds()%60)))


    def write_timeseries(self, model, file_name='model_timeseries'):
        """ """
        start = datetime.now()
        logger_main.info('Writing timeseries to file')

        # Set file_name concerning run
        file_name = self.pre_output + file_name + ('_fixed' if self.fixed else '') + '.csv'

        # Variables type 1: [N,T,F1, Time] for Tech - StorageTech
        var1 = ['f_prod']

        # Variable type 2: [N,T,F,F1, Time] for Tech - StorageTech
        var2 = ['f_cons']

        # Variables type 3: [N,F1, Time]
        var3 = ['f_import', 'f_import_timeseries', 'f_fix_quant_import', 'f_slack_pos', 'f_slack_neg']

        # Variables type 4: [N,F,F1 Time]
        var4 = ['f_export', 'f_export_timeseries', 'f_delivery']

        # Storage variables
        # Variables type 5: [N,T,F1, Time] for StorageTech
        var5 = ['storage_energy_level', 'f_prod']

        # Variables type 6: [N,T,F,F1, Time] for StorageTech
        var6 = ['f_cons']


        # Parameters type 1: Node Tech - Time -
        param1 = ['Availability']

        # Parameters type 2: Node Fuel - Time -
        param2 = ['F_demand']

        # Parameters type 3: Fuel - Time -
        param3 = ['F_export_timeseries_price', 'F_import_timeseries_price']

        # Parameters type 4: Node StorageTech Fuel - Time -
        param4 = ['Rolling_energy_reserve']

       # Note: model resolution is transformed into hourly resolution (with: for _ in range(model.Delta_H))

        # Define time stamps
        timestamp = [datetime.strptime('{} {} {} {}'.format(Y+2018, D, H+dH, int(sH*model.Delta_T*60)), '%Y %j %H %M') for Y in model.Year for D in model.Day for H in model.Hour for dH in range(pyo.value(model.Delta_H)) for sH in model.SubHour]
        timestamp_str = [t.strftime('%Y-%m-%d %H:%M:%S') for t in timestamp]

        values = []
        header = ['timestamp']

        # Variables type 1
        for v in var1:
            for T in model.Tech-model.StorageTech:
                for N in model.Node:
                    # Determine output fuel of tech
                    F1 = [F1 for F1 in model.Fuel1 if model.E_output[T,F1]][0]
                    # Get values (order of for loops is IMPORTANT)
                    temp_values = [pyo.value(model.component(v)[N,T,F1,Y,D,H,sH]) for Y in model.Year for D in model.Day for H in model.Hour for _ in range(pyo.value(model.Delta_H)) for sH in model.SubHour]
                    if sum([abs(v) for v in temp_values]) == 0:
                        continue
                    else:
                        values.append(temp_values)
                        header.extend(['{} {} {} {}'.format(v, T, N, F1)])

        # Variable type 2
        for v in var2:
            for T in model.Tech-model.StorageTech:
                for N in model.Node:
                    # Determine input fuel of tech
                    F = [F for F in model.Fuel if model.E_input[T,F]][0]
                    # Total: sum over F1 if F_subst[F,F1]
                    temp_values = [pyo.value(sum(model.component(v)[N,T,F,F1,Y,D,H,sH] for F1 in model.Fuel1 if model.F_subst[F,F1])) for Y in model.Year for D in model.Day for H in model.Hour for _ in range(pyo.value(model.Delta_H)) for sH in model.SubHour]
                    if sum([abs(v) for v in temp_values]) == 0:
                        continue
                    else:
                        values.append(temp_values)
                        header.extend(['{}_total {} {} {}'.format(v, T, N, F)])

                    # Wrt. F1
                    for F1 in (F1 for F1 in model.Fuel1 if model.F_subst[F,F1]):
                        temp_values = [pyo.value(model.component(v)[N,T,F,F1,Y,D,H,sH]) for Y in model.Year for D in model.Day for H in model.Hour for _ in range(pyo.value(model.Delta_H)) for sH in model.SubHour]
                        if sum([abs(v) for v in temp_values]) == 0:
                            continue
                        else:
                            values.append(temp_values)
                            header.extend(['{} {} {} {} {}'.format(v, T, N, F, F1)])

        # Work around for auxiliary electricity demand 8eg. Compressor_LP)
        v='f_cons'; F='Electricity'
        for T in (T for T in model.Tech if model.Aux_ed[T] > 0):
            for N in model.Node:
                temp_values = [pyo.value(sum(model.component(v)[N,T,F,F1,Y,D,H,sH] for F1 in model.Fuel1 if model.F_subst[F,F1])) for Y in model.Year for D in model.Day for H in model.Hour for _ in range(pyo.value(model.Delta_H)) for sH in model.SubHour]
                if sum([abs(v) for v in temp_values]) == 0:
                    continue
                else:
                    values.append(temp_values)
                    header.extend(['{}_total {} {} {}'.format(v, T, N, F)])

        # Variables type 3
        for v in var3:
            for N in model.Node:
                for F1 in model.Fuel1:
                    # Get values (order of for loops is IMPORTANT)
                    temp_values = [pyo.value(model.component(v)[N,F1,Y,D,H,sH]) for Y in model.Year for D in model.Day for H in model.Hour for _ in range(pyo.value(model.Delta_H)) for sH in model.SubHour]
                    if sum([abs(v) for v in temp_values]) == 0:
                        continue
                    else:
                        values.append(temp_values)
                        header.extend(['{} {} {}'.format(v, N, F1)])

        # Variables type 4
        for v in var4:
            for N in model.Node:
                for F in model.Fuel:
                    # Total: sum over F1 if F_subst[F,F1]
                    temp_values = [pyo.value(sum(model.component(v)[N,F,F1,Y,D,H,sH] for F1 in model.Fuel1 if model.F_subst[F,F1])) for Y in model.Year for D in model.Day for H in model.Hour for _ in range(pyo.value(model.Delta_H)) for sH in model.SubHour]
                    if sum([abs(v) for v in temp_values]) == 0:
                        continue
                    else:
                        values.append(temp_values)
                        header.extend(['{}_total {} {}'.format(v, N, F)])

                    # Wrt. F1
                    for F1 in (F1 for F1 in model.Fuel1 if model.F_subst[F,F1]):
                        temp_values = [pyo.value(model.component(v)[N,F,F1,Y,D,H,sH]) for Y in model.Year for D in model.Day for H in model.Hour for _ in range(pyo.value(model.Delta_H)) for sH in model.SubHour]
                        if sum([abs(v) for v in temp_values]) == 0:
                            continue
                        else:
                            values.append(temp_values)
                            header.extend(['{} {} {} {}'.format(v, N, F, F1)])

        # Variable type 5
        for v in var5:
            for T in model.StorageTech:
                for N in model.Node:

                    # Determine input fuel of tech
                    F = [F for F in model.Fuel if model.E_output[T,F]][0]
                    # Total: sum over F1 if F_subst[F,F1]
                    temp_values = [pyo.value(sum(model.component(v)[N,T,F1,Y,D,H,sH] for F1 in model.Fuel1 if model.F_subst[F,F1])) for Y in model.Year for D in model.Day for H in model.Hour for _ in range(pyo.value(model.Delta_H)) for sH in model.SubHour]
                    if sum([abs(v) for v in temp_values]) == 0:
                        continue
                    else:
                        values.append(temp_values)
                        header.extend(['{}_total {} {} {}'.format(v, T, N, F)])

                    # Wrt. F1
                    for F1 in (F1 for F1 in model.Fuel1 if model.F_subst[F,F1]):
                        temp_values = [pyo.value(model.component(v)[N,T,F1,Y,D,H,sH]) for Y in model.Year for D in model.Day for H in model.Hour for _ in range(pyo.value(model.Delta_H)) for sH in model.SubHour]
                        if sum([abs(v) for v in temp_values]) == 0:
                            continue
                        else:
                            values.append(temp_values)
                            header.extend(['{} {} {} {} {}'.format(v, T, N, F, F1)])

        # Variable type 6
        for v in var6:
            for T in model.StorageTech:
                for N in model.Node:

                    # Determine input fuel of tech
                    F = [F for F in model.Fuel if model.E_input[T,F]][0]
                    # Total: sum over F1 if F_subst[F,F1]
                    temp_values = [pyo.value(sum(model.component(v)[N,T,F,F1,Y,D,H,sH] for F1 in model.Fuel1 if model.F_subst[F,F1])) for Y in model.Year for D in model.Day for H in model.Hour for _ in range(pyo.value(model.Delta_H)) for sH in model.SubHour]
                    if sum([abs(v) for v in temp_values]) == 0:
                        continue
                    else:
                        values.append(temp_values)
                        header.extend(['{}_total {} {} {}'.format(v, T, N, F)])

                    # Wrt. F1
                    for F1 in (F1 for F1 in model.Fuel1 if model.F_subst[F,F1]):
                        temp_values = [pyo.value(model.component(v)[N,T,F,F1,Y,D,H,sH]) for Y in model.Year for D in model.Day for H in model.Hour for _ in range(pyo.value(model.Delta_H)) for sH in model.SubHour]
                        if sum([abs(v) for v in temp_values]) == 0:
                            continue
                        else:
                            values.append(temp_values)
                            header.extend(['{} {} {} {} {}'.format(v, T, N, F, F1)])

        # Parameters type 1
        for v in param1:
            for T in model.Tech:
                for N in model.Node:
                    if pyo.value(sum(model.Max_inst_cap[N,T,Y] for Y in model.Year)) > 0:
                        temp_values = [model.component(v)[N,T,Y,D,H,sH] for Y in model.Year for D in model.Day for H in model.Hour for _ in range(pyo.value(model.Delta_H)) for sH in model.SubHour]
                        values.append(temp_values)
                        header.extend(['{} {} {}'.format(v, T, N)])

        # Parameters type 2
        for v in param2:
            for F in model.Fuel:
                for N in model.Node:
                    temp_values = [model.component(v)[N,F,Y,D,H,sH] for Y in model.Year for D in model.Day for H in model.Hour for _ in range(pyo.value(model.Delta_H)) for sH in model.SubHour]
                    if sum([abs(v) for v in temp_values]) == 0:
                        continue
                    else:
                        values.append(temp_values)
                        header.extend(['{} {} {}'.format(v, F, N)])

        # Parameters type 3
        for v in param3:
            for F in model.Fuel:
                temp_values = [model.component(v)[F,Y,D,H,sH] for Y in model.Year for D in model.Day for H in model.Hour for _ in range(pyo.value(model.Delta_H)) for sH in model.SubHour]
                if sum([abs(v) for v in temp_values]) == 0:
                    continue
                else:
                    values.append(temp_values)
                    header.extend(['{} {}'.format(v, F)])

        # Parameters type 4
        for v in param4:
            for StoreT in model.StorageTech:
                F = [F for F in model.Fuel if model.E_output[StoreT,F]][0]
                for N in model.Node:

                    temp_values = [pyo.value(model.component(v)[N,StoreT,F,Y,D,H,sH]) for Y in model.Year for D in model.Day for H in model.Hour for _ in range(pyo.value(model.Delta_H)) for sH in model.SubHour]
                    if sum([abs(v) for v in temp_values]) == 0:
                        continue
                    else:
                        values.append(temp_values)
                        header.extend(['{} {} {} {}'.format(v, StoreT, N, F)])



        value_array = np.array(values).transpose()

        table = np.column_stack((timestamp_str, value_array))

        file = open(self.repository + file_name, "w")
        np.savetxt(file, table, fmt='%s', header=';'.join(header), delimiter=';', comments='')
        file.close()

        dt = datetime.now() - start
        logger_main.info('\tDuration: {}min {}sec'.format(int(dt.total_seconds()/60), round(dt.total_seconds()%60)))


    def write_parameters_to_file(self, model, file_name='model_parameter', round_digit=2, exclude=[]):
        start = datetime.now()
        logger_main.info('Writing parameters to file')

        # Define file name
        file_name = self.pre_output + file_name + ('_fixed' if self.fixed else '') + '.txt'
        scenario = self.scenario + ('_fixed' if self.fixed else '')

        header = 'OptSys Scenario: ' + str(scenario) + '\n\n' + 'Parameters \n' + '======================================'
        index = []
        value = []

        for param in model.component_objects(pyo.Param, active=True):

            if param.name in exclude:
                continue

            param_dict = param.extract_values()
            index.extend([param.name + '\t' + '\t'.join([str(i) for i in self._make_interable(k)]) for k in param_dict.keys()])
            #value.extend([round(v, round_digit) for v in param_dict.values()])
            value.extend([v for v in param_dict.values()])

        table = np.column_stack((index, value))

        file = open(self.repository + file_name, "w")
        np.savetxt(file, table, fmt='%s', header=header, delimiter='\t', comments='#')
        file.close()

        dt = datetime.now() - start
        logger_main.info('\tDuration: {}min {}sec'.format(int(dt.total_seconds()/60), round(dt.total_seconds()%60)))


    def write_second_run_config(self, model, file_name='optimizationModel_fixed.dat'):
        """
        Write configuration for second run:
        - fixed plant dimensions: installed capacities and storage volumes
        """
        start = datetime.now()
        logger_main.info('Writing configuration file with optim results.')

        file = open(self.repository + self.pre_output + file_name, 'w')

        # Defining Min and Max installed capacity
        param = ['Max_inst_cap', 'Min_inst_cap']
        for p in param:
            file.write('# Optim result: {}\n'.format(p))
            file.write('param: {} :=\n'.format(p))
            for Y in model.Year:
                for N in model.Node:
                    for T in model.Tech - model.StorageTech:
                        value = pyo.value(model.inst_cap[N, T, Y])
                        file.write('{}\t{}\t{}\t{}\n'.format(N, T, Y, value))

            file.write(';\n\n')

        # Defining Min and Max installed storage volume
        param = ['Max_inst_storage_vol', 'Min_inst_storage_vol']
        for p in param:
            file.write('# Optim result: {}\n'.format(p))
            file.write('param: {} :=\n'.format(p))
            for Y in model.Year:
                for N in model.Node:
                    for T in model.StorageTech:
                        value = pyo.value(model.inst_storage_vol[N, T, Y])
                        file.write('{}\t{}\t{}\t{}\n'.format(N, T, Y, value))

            file.write(';\n\n')

        file.close()

        dt = datetime.now() - start
        logger_main.info('\tDuration: {}min {}sec'.format(int(dt.total_seconds()/60), round(dt.total_seconds()%60)))


    def write_output(self, model, file_name='output'):
        start = datetime.now()
        logger_main.info('Writing output file:')

        # Set file_name concerning run
        file_name = self.pre_output + file_name + ('_fixed' if self.fixed else '') + '.txt'

        def create_keys(T='',F='', F1='', N='',Y=''):
            return [T, F, F1, N, Y]

        values = []

        # Prices of reference fuel
        keys = create_keys(F=' + '.join(model.RefFuel))
        unit = self.unitConv[model.F_unit[list(model.RefFuel)[0]]]['newUnit']
        conv = self.unitConv[model.F_unit[list(model.RefFuel)[0]]]['conv']

        values.append(['Price (ref: production)'] + keys + ['EUR/{}'.format(unit), model.ref_prod_price / conv])
        values.append(['Price (ref: demand)'] + keys + ['EUR/{}'.format(unit), model.ref_demand_price / conv])
        values.append(['Price (ref: export)'] + keys + ['EUR/{}'.format(unit), model.ref_export_price / conv])
        values.append(['Price (ref: output)'] + keys + ['EUR/{}'.format(unit), model.ref_out_price / conv])

        # Demand of reference fuel
        cat = 'F_demand'
        for F in model.Fuel:
            for Y in model.Year:
                value = pyo.value(model.F_demand_y[F, Y])
                if value != 0 or F in model.RefFuel:
                    unit = self.unitConv[model.F_unit[F]]['newUnit']
                    conv = self.unitConv[model.F_unit[F]]['conv']
                    values.append([cat] + create_keys(F=F,Y=Y) + ['{}/a'.format(unit), value * conv])


        # Objective (not objective value)
        values.append(['Objective function value'] + create_keys() + ['', round(pyo.value(model.tc_obj - sum(model.high_storage_level_incentive[StoreT] for StoreT in model.StorageTech)))])

        # Slack balance
        for F in model.Fuel:
            values.append(['Slack pos balance'] + create_keys(F=F) + [model.F_unit[F], round(pyo.value(model.slack_pos_balance[F]),2)])
            values.append(['Slack neg balance'] + create_keys(F=F) + [model.F_unit[F], round(pyo.value(model.slack_neg_balance[F]),2)])

        # Energy balance
        for F1 in model.Fuel:
            values.append(['Energy balance [F1]'] + create_keys(F1=F1) + [model.F_unit[F1], round(pyo.value(model.energy_balance[F1]),2)])

        # Optim dimensions
        mapping = {'Inst Capacity':     'inst_cap',
                   'Capacity add':      'cap_add',
                   'Capacity sub':      'cap_sub'}

        for k,i in mapping.items():
            component = model.component(i)

            for N in model.Node:
                for T in model.Tech:
                    for Y in model.Year:
                        values.append([k] + create_keys(N=N,T=T,Y=Y) + [model.T_unit[T], pyo.value(component[N,T,Y])])

        # Optim storage dimensions
        mapping = {'Inst Storage Capacity':     'inst_storage_vol',
                   'Storage Capacity add':      'storage_vol_add',
                   'Storage Capacity sub':      'storage_vol_sub'}

        for k,i in mapping.items():
            component = model.component(i)

            for N in model.Node:
                for T in model.StorageTech:
                    for Y in model.Year:
                        values.append([k] + create_keys(N=N,T=T,Y=Y) + [model.St_unit[T], pyo.value(component[N,T,Y])])

        # Costs
        mapping = {'Total_disc':         'tc',
                   'CAPEX_total_disc':   'total_Capex_disc',
                   'CAPEX_system_disc':  'total_Capex_system_disc',
                   'OPEX_total_disc':    'total_Opex_disc',
                   'OPEX_system_total_disc':    'total_Opex_system_disc',
                   'OPEX_fuel_total_disc': 'total_Opex_fuel_disc',
                   'OPEX_timeseries_disc': 'total_Opex_timeseries_disc',
                   'OPEX_auxmedium_total_disc': 'total_Opex_auxmedium_disc',
                   'OPEX_network_capacity_total_disc': 'total_Opex_network_capacity_disc',
                   'TAXES_disc':         'Opex_taxes_disc',
                   'MARGIN_disc':        'Project_Margin_disc',
                   'REVENUE_total_disc': 'total_Revenue_disc',
                   'REVENUE_timeseries_total_disc': 'total_Revenue_timeseries_disc',
                   'SUBSIDY_disc':        'Subsidy_CAPEX',
                   'SUBSIDY_share_disc':  'Capex_subsidy_disc'}

        for k,i in mapping.items():
            # Use 'getattr()' instead of 'component()', because some value are calculated during postprocessing
            values.append([k] + create_keys() + ['EUR', pyo.value(getattr(model, i))])

        # Discounted demand
        unit = self.unitConv[model.F_unit[list(model.RefFuel)[0]]]['newUnit']
        conv = self.unitConv[model.F_unit[list(model.RefFuel)[0]]]['conv']

        values.append(['Total_disc Production'] + create_keys(F=' + '.join(model.RefFuel)) + [unit, model.ref_prod_disc * conv])
        values.append(['Total_disc Demand'] + create_keys(F=' + '.join(model.RefFuel)) + [unit, model.ref_demand_disc * conv])
        values.append(['Total_disc Export'] + create_keys(F=' + '.join(model.RefFuel)) + [unit, model.ref_export_disc * conv])
        values.append(['Total_disc Output'] + create_keys(F=' + '.join(model.RefFuel)) + [unit, model.ref_out_disc * conv])

        # Partial costs
        mapping = {'Partial_CAPEX':     'Capex_disc',
                   'Partial_CAPEX_system':  'Capex_system_disc',
                   'Partial_SUBSIDY':   'Subsidy_CAPEX',
                   'Partial_SUBSIDY_share': 'Capex_subsidy_disc',
                   'Partial_MARGIN':    'Project_Margin_disc',
                   'Partial_OPEX':      'Opex_disc',
                   'Partial_OPEX_system': 'Opex_system_disc',
                   'Partial_OPEX_fuel': 'Opex_fuel_disc',
                   'Partial_OPEX_timeseries': 'Opex_timeseries_disc',
                   'Partial_OPEX_auxmedium': 'Opex_auxmedium_disc',
                   'Partial_OPEX_network_capacity': 'Opex_network_capacity_disc',
                   'Partial_REVENUE':   'Revenue_disc',
                   'Partial_REVENUE_timeseries':   'Revenue_timeseries_disc',
                   'Partial_TAXES': 'Opex_taxes_disc'}

        unit = self.unitConv[model.F_unit[list(model.RefFuel)[0]]]['newUnit']
        conv = self.unitConv[model.F_unit[list(model.RefFuel)[0]]]['conv']

        if not model.ref_prod_disc == 0:

            for k,i in mapping.items():
                component = model.component(i)

                for T in component.keys():
                    value = pyo.value(component[T]) / model.ref_prod_disc / conv
                    if value == 0:
                        continue
                    else:
                        values.append([k] + create_keys(T=T) + ['EUR/{}'.format(unit), value])


        # Annual costs
        mapping = {'CAPEX':     'capex',
                   'OPEX':      'opex'}

        for k,i in mapping.items():
            component = model.component(i)

            for N in model.Node:
                for T in model.Tech | model.ExternalTech:
                    for Y in model.Year:
                        value = pyo.value(component[N,T,Y])
                        if value == 0: continue
                        values.append([k] + create_keys(N=N,T=T,Y=Y) + ['EUR', value])

        values.append(['SUBSIDY'] + create_keys() + ['EUR', pyo.value(model.Subsidy_CAPEX)])
        values.append(['SUBSIDY_share'] + create_keys(N=N,Y=Y) + ['EUR', pyo.value(model.capex_subsidy[N,Y])])

        for N in model.Node:
            for Y in model.Year:
                values.append(['CAPEX system'] + create_keys(N=N,Y=Y) + ['EUR', pyo.value(model.capex_system[N,Y])])
                values.append(['OPEX system'] + create_keys(N=N,Y=Y) + ['EUR/a', pyo.value(model.opex_system[N,Y])])

        for F1 in model.Fuel1:
            for Y in model.Year:
                value = pyo.value(model.opex_timeseries[F1,Y])
                if value == 0: continue
                values.append(['OPEX_timeseries'] + create_keys(F1=F1,Y=Y) + ['EUR/a', value])

        for N in model.Node:
            for F1 in model.Fuel1:
                for Y in model.Year:
                    value = pyo.value(model.opex_fuel[N,F1,Y])
                    if value == 0: continue
                    values.append(['OPEX fuel'] + create_keys(N=N,F1=F1,Y=Y) + ['EUR/a', value])

        for N in model.Node:
            for F in model.Fuel:
                for Y in model.Year:
                    if model.F_network_capacity_charge[N,F,Y] == 0: continue
                    values.append(['OPEX network capacity'] + create_keys(N=N,F=F,Y=Y) + ['EUR/a', pyo.value(model.opex_network_capacity[N,F,Y])])

        for N in model.Node:
            for A in model.AuxMedium:
                for Y in model.Year:
                    values.append(['OPEX aux medium'] + create_keys(N=N,F=A,Y=Y) + ['EUR/a', pyo.value(sum(model.opex_auxmedium[N,T,A,Y] for T in model.Tech-model.StorageTech))])

        for Y in model.Year:
            values.append(['OPEX taxes'] + create_keys(Y=Y) + ['EUR/a', pyo.value(model.Opex_taxes[Y])])

        for F in model.Fuel:
            for Y in model.Year:
                value = pyo.value(model.revenue[F,Y])
                if value == 0: continue
                values.append(['REVENUE'] + create_keys(F=F,Y=Y) + ['EUR/a', pyo.value(model.revenue[F,Y])])

        for F in model.Fuel:
            for Y in model.Year:
                value = pyo.value(model.revenue_timeseries[F,Y])
                if value == 0: continue
                values.append(['REVENUE_timeseries'] + create_keys(F=F,Y=Y) + ['EUR/a', value])

        # Annual values
        # Annual fuel production
        for N in model.Node:
            for T in model.Tech - model.StorageTech:
                for F1 in (F1 for F1 in model.Fuel1 if model.E_output[T,F1]):
                    for Y in model.Year:
                        unit = self.unitConv[model.F_unit[F1]]['newUnit']
                        conv = self.unitConv[model.F_unit[F1]]['conv']
                        value = pyo.value(model.f_prod_y[N,T,F1,Y]) * conv
                        values.append(['Annual PRODUCTION'] + create_keys(N=N,T=T,F1=F1,Y=Y) + [unit, value])

        for N in model.Node:
            for T in model.StorageTech:
                for F in (F for F in model.Fuel if model.E_output[T,F]):
                    for F1 in (F1 for F1 in model.Fuel1 if model.F_subst[F,F1]):
                        for Y in model.Year:
                            unit = self.unitConv[model.F_unit[F1]]['newUnit']
                            conv = self.unitConv[model.F_unit[F1]]['conv']
                            value = pyo.value(model.f_prod_y[N,T,F1,Y]) * conv
                            values.append(['Annual PRODUCTION'] + create_keys(N=N,T=T,F1=F1,Y=Y) + [unit, value])

        # Annual fuel consumption
        for N in model.Node:
            for T in model.Tech:
                for F in (F for F in model.Fuel if model.E_input[T,F] or F == 'Electricity'):
                    for F1 in (F1 for F1 in model.Fuel1 if model.F_subst[F,F1]):
                        for Y in model.Year:
                            unit = self.unitConv[model.F_unit[F1]]['newUnit']
                            conv = self.unitConv[model.F_unit[F1]]['conv']
                            value = pyo.value(model.f_cons_y[N,T,F,F1,Y]) * conv
                            values.append(['Annual CONSUMPTION'] + create_keys(N=N,T=T,F=F,F1=F1,Y=Y) + [unit, value])

        # Annual aux medium volume
        for N in model.Node:
            for A in model.AuxMedium:
                for Y in model.Year:
                    values.append(['Annual VOLUME'] + create_keys(N=N,F=A,Y=Y) + ['Nm3', pyo.value(model.Aux_medium_flow_y[N,A,Y])])

        # Annual fuel import
        mapping = {'Annual IMPORT':             'f_import_y',
                   'Annual IMPORT_timeseries':  'f_import_timeseries_y',
                   'Annual IMPORT_fix_quantity':'f_fix_quant_import_y'}

        for k,i in mapping.items():
            component = model.component(i)
            for F1 in model.Fuel1:
                for Y in model.Year:
                    unit = self.unitConv[model.F_unit[F1]]['newUnit']
                    conv = self.unitConv[model.F_unit[F1]]['conv']
                    value = pyo.value(component[F1,Y]) * conv
                    if value == 0: continue
                    values.append([k] + create_keys(F1=F1,Y=Y) + [unit, value])

        # Annual fuel export
        mapping = {'Annual EXPORT':             'f_export_y',
                   'Annual EXPORT_timeseries':  'f_export_timeseries_y',
                   'Annual DELIVERY':           'f_delivery_y',
                   'Annual CONS_system':        'f_supply_cons_system_y'}

        for k,i in mapping.items():
            component = model.component(i)
            for F in model.Fuel:
                for F1 in (F1 for F1 in model.Fuel1 if model.F_subst[F,F1]):
                    for Y in model.Year:
                        unit = self.unitConv[model.F_unit[F1]]['newUnit']
                        conv = self.unitConv[model.F_unit[F1]]['conv']
                        value = pyo.value(component[F,F1,Y]) * conv
                        if value == 0: continue
                        values.append([k] + create_keys(F=F,F1=F1,Y=Y) + [unit, value])

        # Full Load Hours
        mapping = {'FLH':           'flh',
                   'FLH possible':  'flh_possible'}

        for k,i in mapping.items():
            component = getattr(model, i)
            for N in model.Node:
                for T in model.Tech - model.StorageTech:
                    for Y in model.Year:
                        values.append([k] + create_keys(N=N,T=T,Y=Y) + ['h', pyo.value(component[N,T,Y])])

        # Storage ratio
        for N in model.Node:
            for T in model.StorageTech:
                F = [F for F in model.Fuel if model.E_output[T,F]][0]
                for Y in model.Year:
                    values.append(['STORAGE ratio'] + create_keys(N=N,T=T,F=F,Y=Y) + ['d', pyo.value(model.storage_ratio[N,T,F,Y])])

        # Peak demand
        for N in model.Node:
            for Y in model.Year:
                values.append(['peak el demand'] + create_keys(N=N,Y=Y) + ['kW', pyo.value(model.peak_el_demand[N,Y])])

        # Network capacity
        for N in model.Node:
            for F in model.Fuel:
                for Y in model.Year:
                    if model.F_network_capacity_charge[N,F,Y] == 0: continue

                    unit = model.F_unit[F] + '/h'
                    value = pyo.value(model.f_network_capacity[N,F,Y])
                    values.append(['Network capacity required'] + create_keys(N=N,F=F,Y=Y) + [unit, value])

        # Land use
        for N in model.Node:
            for Y in model.Year:
                values.append(['Land use'] + create_keys(N=N,Y=Y) + ['m2', pyo.value(model.land_use[N,Y])])

        # Min of storage levels
        for N in model.Node:
            for T in model.StorageTech:
                for Y in model.Year:
                    unit = self.unitConv[model.St_unit[T]]['newUnit']
                    conv = self.unitConv[model.St_unit[T]]['conv']
                    value = conv * min([pyo.value(sum(model.storage_energy_level[N,T,F1,Y,D,H,sH] for F1 in model.Fuel1)) for D in model.Day for H in model.Hour for sH in model.SubHour])
                    values.append(['Min storage level'] + create_keys(N=N,T=T,Y=Y) + [unit, value])


        # Energy share electricity
        if 'Electrolysis' in model.Tech:
            # Look for technologies
            cons = 'Electrolysis'
            consumer = [T for T in model.Tech if cons.lower() in T.lower()]
            producer = [T for T in model.Tech] + ['f_import', 'f_import_timeseries']

            # Considered fuel
            F='Electricity'

            for C in consumer:
                for P in producer:
                    unit = self.unitConv[model.F_unit[F]]['newUnit']
                    conv = self.unitConv[model.F_unit[F]]['conv']
                    value = conv * model.energy_share.loc[P,C]

                    values.append(['Consumption share'] + create_keys(T=C,F=F,F1=P) + [unit, value])

                    #values.append([C + ' cons. share'] + create_keys(F=F,T=P) + [F, value])

        # Levelized Costs of Energy

        # TODO
        # calculations should take place at unit bottom level, meaning units defined by: model.F_unit and model.T_unit
        # costs and prices are often in EUR/MWh -> bottom level is kWh
        # use self.unitConv for unit conversion (unit dictionary is defined at __main__)

        mapping = {'LCOEnergy ideal':   'tech_LCOEnergy_ideal',
                   'LCOEnergy':         'tech_LCOEnergy'}
        for k,i in mapping.items():
            component = model.component(i)
            for N in model.Node:
                for T in model.Tech:

                    if T in model.StorageTech:
                        F_list = [F1 for F in model.Fuel if model.E_output[T,F] for F1 in model.Fuel1 if model.F_subst[F,F1]]
                    else:
                        F_list = [F for F in model.Fuel if model.E_output[T,F]]

                    for F in F_list:
                        unit = self.unitConv[model.F_unit[F]]['newUnit']
                        conv = self.unitConv[model.F_unit[F]]['conv']

                        for Y in model.Year:
                            value = pyo.value(component[N,T,F,Y]) / conv
                            values.append([k] + create_keys(N=N,T=T,F=F,Y=Y) + ['EUR/{}'.format(unit), value])

        for N in model.Node:
            for T in model.Tech:
                for F in (F for F in model.Fuel if model.E_input[T,F] or F == 'Electricity'):
                    unit = self.unitConv[model.F_unit[F]]['newUnit']
                    conv = self.unitConv[model.F_unit[F]]['conv']

                    for Y in model.Year:
                        value = pyo.value(model.tech_LCOEnergy_cons[N,T,F,Y]) / conv
                        values.append(['LCOEnergy consumed'] + create_keys(N=N,T=T,F=F,Y=Y) + ['EUR/{}'.format(unit), value])

        # LCOE of exported energy
        mapping = {'LCOEnergy demand':      'demand_LCOEnergy',
                   'LCOEnergy exported':    'export_LCOEnergy'}

        for k,i in mapping.items():
            component = model.component(i)

            for N in model.Node:
                for F in model.Fuel:
                    unit = self.unitConv[model.F_unit[F]]['newUnit']
                    conv = self.unitConv[model.F_unit[F]]['conv']

                    for Y in model.Year:
                        value = pyo.value(component[N,F,Y]) / conv
                        values.append([k] + create_keys(N=N,F=F,Y=Y) + ['EUR/{}'.format(unit), value])

        # Debugging (set True for debugging)
        if False:
            values.append(['--- DEBUGGING ---'] + create_keys(T='---',F='---', F1='---', N='---',Y='---') + ['----', 0])

            for N in model.Node:
                for T in model.StorageTech:
                    for F1 in model.Fuel1:
                        values.append(['start_storage_energy_level'] + create_keys(N=N,T=T,F=F1) + ['kWh', pyo.value(model.start_storage_energy_level[N,T,F1])])

            for N in model.Node:
                for T in model.StorageTech:
                    for F1 in model.Fuel1:
                        values.append(['end_storage_energy_level'] + create_keys(N=N,T=T,F=F1) + ['kWh', pyo.value(model.storage_energy_level[N,T,F1,max(model.Year),max(model.Day),max(model.Hour),max(model.SubHour)])])


            for N in model.Node:
                for T in model.StorageTech:
                    for F1 in model.Fuel1:
                        for Y in model.Year:
                            values.append(['Annual PRODUCTION'] + create_keys(N=N,T=T,F1=F1,Y=Y) + ['kWh', pyo.value(model.f_prod_y[N,T,F1,Y])])

            # Annual fuel consumption
            for N in model.Node:
                for T in model.StorageTech:
                    for F in model.Fuel:
                        for F1 in model.Fuel1:
                            for Y in model.Year:
                                values.append(['Annual CONSUMPTION'] + create_keys(N=N,T=T,F=F,F1=F1,Y=Y) + ['kWh', pyo.value(model.f_cons_y[N,T,F,F1,Y])])

        # Writing of results
        # Define column keys
        sets = ['Tech', 'Fuel', 'Fuel1', 'Node', 'Year']

        # Base data frame
        header = ['Category'] + sets + ['Unit', 'Value']
        df = pd.DataFrame(values, columns=header)

        files = [self.repository + file_name,
                 'scenario_collection' + os.path.sep + '{}_{}'.format(self.scenario, file_name)]

        for file in files:
            try:
                f = open(file, 'w')
                f.write(str(self.scenario) + '\n')
                df.to_csv(f, sep='\t', index=False, line_terminator='\n')
                f.close()
            except Exception as e:
                print(e)

        dt = datetime.now() - start
        logger_main.info('\tDuration: {}min {}sec'.format(int(dt.total_seconds()/60), round(dt.total_seconds()%60)))


    def load_data(self):
        start = datetime.now()
        logger_main.info('Loading data')

        # Input files
        data_config = self.repository + self.pre_input + 'optimizationModel.dat'
        data_wind = self.repository + self.pre_input + 'Availability_N1_Wind.dat'
        data_solar = self.repository + self.pre_input + 'Availability_N1_Solar.dat'
        data_electrolysis = self.repository + self.pre_input + 'Availability_N1_Electrolysis.dat'
        data_V_edp = self.repository + self.pre_input + 'V_edp.dat'
        data_F_edp = self.repository + self.pre_input + 'F_edp.dat'
        data_F_demand_exo_H2 = self.repository + self.pre_input + 'F_demand_exo_H2.dat'
        data_F_demand_exo_El = self.repository + self.pre_input + 'F_demand_exo_El.dat'
        data_F_demand_exo = self.repository + self.pre_input + 'F_demand_exo.dat'
        data_F_export_timeseries_price = self.repository + self.pre_input + 'F_export_timeseries_price.dat'
        data_F_import_timeseries_price = self.repository + self.pre_input + 'F_import_timeseries_price.dat'
        data_F_network_flow = self.repository + self.pre_input + 'F_network_flow.dat'

        # 2. run input files
        data_config_fixed = self.repository + self.pre_output + 'optimizationModel_fixed.dat'
        data_F_demand_exo_H2_fixed = self.repository + self.pre_input + 'F_demand_exo_H2_fixed.dat'
        data_F_demand_exo_El_fixed = self.repository + self.pre_input + 'F_demand_exo_El_fixed.dat'
        data_wind_fixed = self.repository + self.pre_input + 'Availability_N1_Wind_fixed.dat'
        data_solar_fixed = self.repository + self.pre_input + 'Availability_N1_Solar_fixed.dat'

        # Initialise data portal
        data = pyo.DataPortal()

        # Optional inputs: if input.dat is in repository than load data else raise warning
        # List content of repository
        ls = os.listdir(self.repository)

        # Import data sources for F_demand
        if data_F_demand_exo_H2.split(os.path.sep)[-1] in ls:
            data.load(filename=data_F_demand_exo_H2, index=(self.m.Node, self.m.Fuel, self.m.Year_All, self.m.Day_All, self.m.Hour, self.m.SubHour), param=self.m.F_demand)
        else: msg = 'Missing input: ' + str(data_F_demand_exo_H2); logger_main.warning(msg); print(msg)

        if data_F_demand_exo_El.split(os.path.sep)[-1] in ls:
            data.load(filename=data_F_demand_exo_El, index=(self.m.Node, self.m.Fuel, self.m.Year_All, self.m.Day_All, self.m.Hour, self.m.SubHour), param=self.m.F_demand)
        else: msg = 'Missing input: ' + str(data_F_demand_exo_El); logger_main.warning(msg); print(msg)

        if data_F_demand_exo.split(os.path.sep)[-1] in ls:
            data.load(filename=data_F_demand_exo, index=(self.m.Node, self.m.Fuel, self.m.Year_All, self.m.Day_All, self.m.Hour, self.m.SubHour), param=self.m.F_demand)
        else: msg = 'Missing input: ' + str(data_F_demand_exo); logger_main.warning(msg); print(msg)

        # Import data for F_export_timeseries_price
        if data_F_export_timeseries_price.split(os.path.sep)[-1] in ls:
            data.load(filename=data_F_export_timeseries_price, index=(self.m.Fuel, self.m.Year_All, self.m.Day_All, self.m.Hour, self.m.SubHour), param=self.m.F_export_timeseries_price)
        else: msg = 'Missing input: ' + str(data_F_export_timeseries_price); logger_main.warning(msg); print(msg)

        # Import data for F_import_timeseries_price
        if data_F_import_timeseries_price.split(os.path.sep)[-1] in ls:
            data.load(filename=data_F_import_timeseries_price, index=(self.m.Fuel, self.m.Year_All, self.m.Day_All, self.m.Hour, self.m.SubHour), param=self.m.F_import_timeseries_price)
        else: msg = 'Missing input: ' + str(data_F_import_timeseries_price); logger_main.warning(msg); print(msg)

        # Import data sportmarket prices
        if data_F_network_flow.split(os.path.sep)[-1] in ls:
            data.load(filename=data_F_network_flow, index=(self.m.Node, self.m.Fuel, self.m.Year_All, self.m.Day_All, self.m.Hour, self.m.SubHour), param=self.m.F_network_flow)
        else: msg = 'Missing input: ' + str(data_F_network_flow); logger_main.warning(msg); print(msg)

        # Import data sources for Availability
        if data_wind.split(os.path.sep)[-1] in ls:
            data.load(filename=data_wind, index=(self.m.Node, self.m.Tech, self.m.Year_All, self.m.Day_All, self.m.Hour, self.m.SubHour), param=self.m.Availability)
        else: msg = 'Missing input: ' + str(data_wind); logger_main.warning(msg); print(msg)

        if data_solar.split(os.path.sep)[-1] in ls:
            data.load(filename=data_solar, index=(self.m.Node, self.m.Tech, self.m.Year_All, self.m.Day_All, self.m.Hour, self.m.SubHour), param=self.m.Availability)
        else: msg = 'Missing input: ' + str(data_solar); logger_main.warning(msg); print(msg)

        if data_electrolysis.split(os.path.sep)[-1] in ls:
            data.load(filename=data_electrolysis, index=(self.m.Node, self.m.Tech, self.m.Year_All, self.m.Day_All, self.m.Hour, self.m.SubHour), param=self.m.Availability)
        else: msg = 'Missing input: ' + str(data_electrolysis); logger_main.warning(msg); print(msg)

        # Import data sources for Fixed and Variable Electricity Demand
        if data_F_edp.split(os.path.sep)[-1] in ls:
            data.load(filename=data_F_edp, index=(self.m.Node, self.m.Tech, self.m.Year_All, self.m.Day_All, self.m.Hour, self.m.SubHour), param=self.m.F_edp)
        else: msg = 'Missing input: ' + str(data_F_edp); logger_main.warning(msg); print(msg)

        if data_V_edp.split(os.path.sep)[-1] in ls:
            data.load(filename=data_V_edp, index=(self.m.Node, self.m.Tech, self.m.Year_All, self.m.Day_All, self.m.Hour, self.m.SubHour), param=self.m.V_edp)
        else: msg = 'Missing input: ' + str(data_V_edp); logger_main.warning(msg); print(msg)

        # Obligatory input:
        # Import general configuration
        data.load(filename=data_config)

        if self.fixed:
            logger_main.info('Loading fixed data')
            data.load(filename=data_config_fixed)

            if data_F_demand_exo_H2_fixed.split(os.path.sep)[-1] in ls:
                data.load(filename=data_F_demand_exo_H2_fixed)
            else: msg = 'Missing input: ' + str(data_F_demand_exo_H2_fixed); logger_main.warning(msg); print(msg)

            if data_F_demand_exo_El_fixed.split(os.path.sep)[-1] in ls:
                data.load(filename=data_F_demand_exo_El_fixed)
            else: msg = 'Missing input: ' + str(data_F_demand_exo_El_fixed); logger_main.warning(msg); print(msg)

            if data_wind_fixed.split(os.path.sep)[-1] in ls:
                data.load(filename=data_wind_fixed)
            else: msg = 'Missing input: ' + str(data_wind_fixed); logger_main.warning(msg); print(msg)

            if data_solar_fixed.split(os.path.sep)[-1] in ls:
                data.load(filename=data_solar_fixed)
            else: msg = 'Missing input: ' + str(data_solar_fixed); logger_main.warning(msg); print(msg)

        dt = datetime.now() - start
        logger_main.info('\tDuration: {}min {}sec'.format(int(dt.total_seconds()/60), round(dt.total_seconds()%60)))

        return data


    def create_instance(self):
        start = datetime.now()
        logger_main.info('Create instance of abstract model')

        instance = self.m.create_instance(data=self.data)

        dt = datetime.now() - start
        logger_main.info('\tDuration: {}min {}sec'.format(int(dt.total_seconds()/60), round(dt.total_seconds()%60)))

        return instance


    def solve(self):

        logger_main.info('Start solving model')
        start = datetime.now()

        # Define solver
        opt = SolverFactory('glpk', executable='C:\ProgramData\Miniconda3\pkgs\glpk-4.65\Library\bin\glpsol.exe', verbose=True)

        # Solver path for mac
        #opt = SolverFactory('glpk', verbose=True)

        # Testing other solvers
        #opt = SolverFactory(executable='C:\ProgramData\Anaconda3\Clp-master-win64-msvc15-md\bin\clp.exe')

        # Solve LP
        opt.solve(self.instance, tee=True)

        # Runtime
        dt = datetime.now() - start
        dt_min = int(dt.total_seconds()/60)
        dt_sec = round(dt.total_seconds()%60)

        # Write runtime report
        try:
            f = open(self.repository + self.pre_output + 'runtime_report.txt', 'w')
            f.write('Last optimization: ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n')
            f.write('Duration: ' + str(dt_min) + 'min ' + str(dt_sec) + 'sec \n')
            f.close()
        except Exception as e:
                print(e)

        logger_main.info('\tSolutions found: {}'.format(len(self.instance.solutions)))
        logger_main.info('\tDuration: ' + str(dt_min) + 'min ' + str(dt_sec) + 'sec')


    @staticmethod
    def _make_interable(x):
        """
        Create iterable object from integers and strings.
        """
        if isinstance(x, int) or isinstance(x, str):
            return [x]
        elif isinstance(x, type(None)):
            return ['']
        else:
            return x


    def store_model(self):
        start = datetime.now()

        file_name = self.repository + 'model_file' + ('_fixed' if self.fixed else '') + '.pkl'

        logger_main.info('Store model as pickle-file: {}'.format(file_name))

        with open(file_name, mode='wb') as f:
            cloudpickle.dump(self.instance, f)

        dt = datetime.now() - start
        logger_main.info('\tDuration: {}min {}sec'.format(int(dt.total_seconds()/60), round(dt.total_seconds()%60)))


    def load_model(self):
        start = datetime.now()
        file_name = self.repository + 'model_file' + ('_fixed' if self.fixed else '') + '.pkl'
        logger_main.info('Read model from pickle-file: {}'.format(file_name))

        with open(file_name, mode='rb') as f:
            model = cloudpickle.load(f)

        dt = datetime.now() - start
        logger_main.info('\tDuration: {}min {}sec'.format(int(dt.total_seconds()/60), round(dt.total_seconds()%60)))

        return model




    def run(self):

        self.m = self.build_model()
        self.data = self.load_data()

        self.instance = self.create_instance()
        self.instance.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT) # let solver apply dual variables (marginals)

        if self.write_lp:
            file_name = self.repository + 'optimizationModel.lp'
            self.instance.write(file_name, io_options={'symbolic_solver_labels': True})

        # Solve problem and store results to instance (self.results is unused)
        #self.results = self.solve()
        self.solve()

        # Store model with cloudpickle
        if self.pickle_model:
            self.store_model()



    def postprocessing(self):
        start = datetime.now()
        logger_main.info('Start postprocessing of {}'.format(self.scenario))

        # Define zero threshold
        self.zero_threshold = 1e-6

        if self.pickle_model:
            # Load model with cloudpickle
            model = self.load_model()
        else:
            model = self.instance


        # Postprocessing: order is important
        self.calc_slack_balance(model)
        self.calc_annual_values(model)
        self.calc_energy_balance(model)

        # NEW
        self.calc_LCOEnergy(model)
        #self.calc_LCOE(model)
        #self.calc_LCOH(model)

        self.calc_prices(model)
        self.calc_kpi(model)

        self.calc_energy_share(model)
        self.calc_land_use(model)

        # Write results
        self.write_energy_share(model)

        self.write_output(model)

        self.write_timeseries(model)
        self.write_ldc(model)
        self.write_parameters_to_file(model,exclude=['F_demand', 'F_edp', 'V_edp', 'Availability', 'F_export_timeseries_price'])

        # Write config of optim results
        self.write_second_run_config(model)

        dt = datetime.now() - start
        logger_main.info('Postprocessing duration: {}min {}sec'.format(int(dt.total_seconds()/60), round(dt.total_seconds()%60)))


    def optimize(scenario):

        # Settings

        # Enable/disable second run
        second_run = False

        # Enable/disable debugging mode (model saved to pickle file for separated postprocessing)
        pickle_model = False

        # Write LP file
        write_lp = False

        # Skip optimization and do only postProcessing
        skip_optim = False

        # Unit conversion
        unitConv = {'kW':       {'newUnit': 'MW',       'conv': 1/1000},
                    'kW_th':    {'newUnit': 'MW_th',    'conv': 1/1000},
                    'kWh':      {'newUnit': 'MWh',      'conv': 1/1000},
                    'kWh_th':   {'newUnit': 'MWh_th',   'conv': 1/1000},
                    'kg':       {'newUnit': 't',        'conv': 1/1000},
                    'h':        {'newUnit': 'd',        'conv': 1/24},
                    'd':        {'newUnit': 'a',        'conv': 1/365},
                    'm3':       {'newUnit': 'Ml',       'conv': 1000}}

        # Discount settings, define first year of discount
        start_disc = {'capex':  0,
                      'opex':   1}

        start = datetime.now()
        logger_run.info('Optimization of scenario {}'.format(scenario))

        # Use fixed dimensions
        fixed = False

        # Create class
        optSys = OptSys(scenario=scenario, fixed=fixed, pickle_model=pickle_model, write_lp=write_lp, unitConv=unitConv, start_disc=start_disc)

        # Run and build model
        if not skip_optim:
            try:
                optSys.run()
            except Exception as e:
                print(e)
                logger_run.error('Optimization error: ' + str(e))
                return 0

        # Evaluate results
        try:
            optSys.postprocessing()
        except Exception as e:
                print(e)
                logger_run.error('Postprocessing error: ' + str(e))
                return 0

        # clean up memory
        del(optSys)

        dt = datetime.now() - start
        logger_run.info('Finished: {} | duration: {}min {}sec'.format(scenario, int(dt.total_seconds()/60), round(dt.total_seconds()%60)))


        if second_run:
            start = datetime.now()
            logger_run.info('Second run started: {}'.format(scenario))

            # Use fixed dimensions
            fixed = True

            # Create class
            optSys2 = OptSys(scenario=scenario, fixed=fixed, pickle_model=pickle_model, write_lp=write_lp, unitConv=unitConv, start_disc=start_disc)

            # Run and build model
            try:
                optSys2.run()
            except Exception as e:
                print(e)
                return 0

            # Evaluate results
            try:
                optSys2.postprocessing()
            except Exception as e:
                print(e)
                return 0

            # clean up memory
            del(optSys2)

            dt = datetime.now() - start
            logger_run.info('Finished second run: {} | duration: {}min {}sec'.format(scenario, int(dt.total_seconds()/60), round(dt.total_seconds()%60)))

        # Return 1 for successful optimization
        return 1


if __name__ == '__main__':

    scenario = '20210214_template_grid'
    OptSys.optimize(scenario=scenario)




"""
# NOTES

## Electrolysis Efficiency

## Pyomo Sets: Operations
https://pyomo.readthedocs.io/en/stable/pyomo_modeling_components/Sets.html
>>> model.I = model.A | model.D # union
>>> model.J = model.A & model.D # intersection
>>> model.K = model.A - model.D # difference
>>> model.L = model.A ^ model.D # exclusive-or

The efficiency of electrolysis eff = 68% consideres LHV of hydrogen.


## Input Parameters
->eg.  mail

## Scalling
scale = 7
years = 21

m.Year = 1     2      3
         0     7      14

OPEX     11111 22222 33333


## Pyomo Coding

https://pyomo.readthedocs.io/en/latest/library_reference/aml/index.html

Binary variable
model.x = pyo.Var(... , within=pyo.Binary)

# Storage Efficiency
m.Eff -> efficiency for each charge process and discharge process

round trip eff. = m.Eff^2


"""
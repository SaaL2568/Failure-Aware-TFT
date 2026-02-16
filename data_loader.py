import pandas as pd
import numpy as np
from datetime import timedelta
from config import Config

class MIMICDataLoader:
    def __init__(self, hosp_path, icu_path):
        self.hosp_path = hosp_path
        self.icu_path = icu_path
        
    def load_patients(self):
        patients = pd.read_csv(f"{self.hosp_path}patients.csv")
        return patients
    
    def load_admissions(self):
        admissions = pd.read_csv(f"{self.hosp_path}admissions.csv")
        admissions['admittime'] = pd.to_datetime(admissions['admittime'])
        admissions['dischtime'] = pd.to_datetime(admissions['dischtime'])
        if 'deathtime' in admissions.columns:
            admissions['deathtime'] = pd.to_datetime(admissions['deathtime'])
        return admissions
    
    def load_icustays(self):
        icustays = pd.read_csv(f"{self.icu_path}icustays.csv")
        icustays['intime'] = pd.to_datetime(icustays['intime'])
        icustays['outtime'] = pd.to_datetime(icustays['outtime'])
        return icustays
    
    def load_chartevents(self, itemids=None):
        chartevents = pd.read_csv(f"{self.icu_path}chartevents.csv")
        chartevents['charttime'] = pd.to_datetime(chartevents['charttime'])
        if itemids:
            chartevents = chartevents[chartevents['itemid'].isin(itemids)]
        return chartevents
    
    def load_labevents(self, itemids=None):
        labevents = pd.read_csv(f"{self.hosp_path}labevents.csv")
        labevents['charttime'] = pd.to_datetime(labevents['charttime'])
        if itemids:
            labevents = labevents[labevents['itemid'].isin(itemids)]
        return labevents
    
    def load_inputevents(self):
        inputevents = pd.read_csv(f"{self.icu_path}inputevents.csv")
        inputevents['starttime'] = pd.to_datetime(inputevents['starttime'])
        inputevents['endtime'] = pd.to_datetime(inputevents['endtime'])
        return inputevents
    
    def load_outputevents(self):
        outputevents = pd.read_csv(f"{self.icu_path}outputevents.csv")
        outputevents['charttime'] = pd.to_datetime(outputevents['charttime'])
        return outputevents
    
    def load_d_items(self):
        d_items = pd.read_csv(f"{self.icu_path}d_items.csv")
        return d_items
    
    def load_d_labitems(self):
        d_labitems = pd.read_csv(f"{self.hosp_path}d_labitems.csv")
        return d_labitems
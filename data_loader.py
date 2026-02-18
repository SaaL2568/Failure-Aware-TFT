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
    
    def load_chartevents(self, itemids=None, stay_ids=None, chunksize=100000):
        chunks = []
        total_rows = 0
        
        print("Loading chartevents in chunks (this may take a few minutes)...")
        
        for i, chunk in enumerate(pd.read_csv(f"{self.icu_path}chartevents.csv", chunksize=chunksize)):
            if itemids is not None:
                chunk = chunk[chunk['itemid'].isin(itemids)]
            
            if stay_ids is not None:
                chunk = chunk[chunk['stay_id'].isin(stay_ids)]
            
            if len(chunk) > 0:
                chunks.append(chunk)
                total_rows += len(chunk)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {(i+1)*chunksize:,} rows, kept {total_rows:,} relevant rows")
        
        if len(chunks) == 0:
            return pd.DataFrame()
        
        chartevents = pd.concat(chunks, ignore_index=True)
        chartevents['charttime'] = pd.to_datetime(chartevents['charttime'])
        
        print(f"  Final chartevents size: {len(chartevents):,} rows")
        return chartevents
    
    def load_labevents(self, itemids=None, subject_ids=None, chunksize=100000):
        chunks = []
        total_rows = 0
        
        print("Loading labevents in chunks (this may take a few minutes)...")
        
        for i, chunk in enumerate(pd.read_csv(f"{self.hosp_path}labevents.csv", chunksize=chunksize)):
            if itemids is not None:
                chunk = chunk[chunk['itemid'].isin(itemids)]
            
            if subject_ids is not None:
                chunk = chunk[chunk['subject_id'].isin(subject_ids)]
            
            if len(chunk) > 0:
                chunks.append(chunk)
                total_rows += len(chunk)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {(i+1)*chunksize:,} rows, kept {total_rows:,} relevant rows")
        
        if len(chunks) == 0:
            return pd.DataFrame()
        
        labevents = pd.concat(chunks, ignore_index=True)
        labevents['charttime'] = pd.to_datetime(labevents['charttime'])
        
        print(f"  Final labevents size: {len(labevents):,} rows")
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

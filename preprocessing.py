import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import timedelta
from config import Config

class MIMICPreprocessor:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.scalers = {}
        self.label_encoders = {}
        
    def create_cohort(self):
        patients = self.data_loader.load_patients()
        admissions = self.data_loader.load_admissions()
        icustays = self.data_loader.load_icustays()
        
        cohort = icustays.merge(admissions, on=['subject_id', 'hadm_id'], how='inner')
        cohort = cohort.merge(patients, on='subject_id', how='inner')
        
        cohort = cohort[cohort['los'] >= Config.LOOKBACK_WINDOW / 24]
        
        return cohort
    
    def create_labels(self, cohort):
        labels = []
        
        for idx, row in cohort.iterrows():
            subject_id = row['subject_id']
            hadm_id = row['hadm_id']
            stay_id = row['stay_id']
            
            hospital_expire_flag = row.get('hospital_expire_flag', 0)
            
            if pd.notna(row.get('deathtime')):
                death_time = row['deathtime']
                icu_intime = row['intime']
                time_to_death = (death_time - icu_intime).total_seconds() / 3600
                
                if time_to_death > Config.LOOKBACK_WINDOW and time_to_death <= Config.LOOKBACK_WINDOW + Config.PREDICTION_WINDOW:
                    label = 1
                else:
                    label = 0
            else:
                label = 0
            
            labels.append({
                'subject_id': subject_id,
                'hadm_id': hadm_id,
                'stay_id': stay_id,
                'label': label
            })
        
        return pd.DataFrame(labels)
    
    def extract_static_features(self, cohort):
        static_features = cohort[['subject_id', 'hadm_id', 'stay_id']].copy()
        
        if 'gender' in cohort.columns:
            if 'gender' not in self.label_encoders:
                self.label_encoders['gender'] = LabelEncoder()
                static_features['gender'] = self.label_encoders['gender'].fit_transform(cohort['gender'].fillna('Unknown'))
            else:
                static_features['gender'] = self.label_encoders['gender'].transform(cohort['gender'].fillna('Unknown'))
        
        if 'anchor_age' in cohort.columns:
            static_features['anchor_age'] = cohort['anchor_age'].fillna(cohort['anchor_age'].median())
        
        if 'admission_type' in cohort.columns:
            if 'admission_type' not in self.label_encoders:
                self.label_encoders['admission_type'] = LabelEncoder()
                static_features['admission_type'] = self.label_encoders['admission_type'].fit_transform(cohort['admission_type'].fillna('Unknown'))
            else:
                static_features['admission_type'] = self.label_encoders['admission_type'].transform(cohort['admission_type'].fillna('Unknown'))
        
        if 'insurance' in cohort.columns:
            if 'insurance' not in self.label_encoders:
                self.label_encoders['insurance'] = LabelEncoder()
                static_features['insurance'] = self.label_encoders['insurance'].fit_transform(cohort['insurance'].fillna('Unknown'))
            else:
                static_features['insurance'] = self.label_encoders['insurance'].transform(cohort['insurance'].fillna('Unknown'))
        
        if 'marital_status' in cohort.columns:
            if 'marital_status' not in self.label_encoders:
                self.label_encoders['marital_status'] = LabelEncoder()
                static_features['marital_status'] = self.label_encoders['marital_status'].fit_transform(cohort['marital_status'].fillna('Unknown'))
            else:
                static_features['marital_status'] = self.label_encoders['marital_status'].transform(cohort['marital_status'].fillna('Unknown'))
        
        if 'race' in cohort.columns:
            if 'race' not in self.label_encoders:
                self.label_encoders['race'] = LabelEncoder()
                static_features['race'] = self.label_encoders['race'].fit_transform(cohort['race'].fillna('Unknown'))
            else:
                static_features['race'] = self.label_encoders['race'].transform(cohort['race'].fillna('Unknown'))
        
        return static_features
    
    def extract_time_series(self, cohort):
        chartevents = self.data_loader.load_chartevents(Config.VITAL_ITEMIDS)
        labevents = self.data_loader.load_labevents(Config.LAB_ITEMIDS)
        
        time_series_data = []
        
        for idx, row in cohort.iterrows():
            stay_id = row['stay_id']
            intime = row['intime']
            window_end = intime + timedelta(hours=Config.LOOKBACK_WINDOW)
            
            vitals = chartevents[chartevents['stay_id'] == stay_id].copy()
            vitals = vitals[(vitals['charttime'] >= intime) & (vitals['charttime'] < window_end)]
            
            labs = labevents[
                (labevents['subject_id'] == row['subject_id']) & 
                (labevents['hadm_id'] == row['hadm_id'])
            ].copy()
            labs = labs[(labs['charttime'] >= intime) & (labs['charttime'] < window_end)]
            
            vitals['hours_from_intime'] = (vitals['charttime'] - intime).dt.total_seconds() / 3600
            labs['hours_from_intime'] = (labs['charttime'] - intime).dt.total_seconds() / 3600
            
            time_bins = np.arange(0, Config.LOOKBACK_WINDOW + Config.TIME_STEP_HOURS, Config.TIME_STEP_HOURS)
            
            ts_record = {
                'subject_id': row['subject_id'],
                'hadm_id': row['hadm_id'],
                'stay_id': stay_id,
                'time_steps': []
            }
            
            for t in range(len(time_bins) - 1):
                t_start = time_bins[t]
                t_end = time_bins[t + 1]
                
                vitals_bin = vitals[(vitals['hours_from_intime'] >= t_start) & (vitals['hours_from_intime'] < t_end)]
                labs_bin = labs[(labs['hours_from_intime'] >= t_start) & (labs['hours_from_intime'] < t_end)]
                
                features = {}
                
                for itemid in Config.VITAL_ITEMIDS:
                    item_data = vitals_bin[vitals_bin['itemid'] == itemid]['valuenum']
                    if len(item_data) > 0:
                        features[f'vital_{itemid}'] = item_data.mean()
                    else:
                        features[f'vital_{itemid}'] = np.nan
                
                for itemid in Config.LAB_ITEMIDS:
                    item_data = labs_bin[labs_bin['itemid'] == itemid]['valuenum']
                    if len(item_data) > 0:
                        features[f'lab_{itemid}'] = item_data.mean()
                    else:
                        features[f'lab_{itemid}'] = np.nan
                
                features['time_idx'] = t
                ts_record['time_steps'].append(features)
            
            time_series_data.append(ts_record)
        
        return time_series_data
    
    def normalize_features(self, time_series_data, fit=True):
        all_features = []
        feature_names = None
        
        for record in time_series_data:
            for ts in record['time_steps']:
                if feature_names is None:
                    feature_names = [k for k in ts.keys() if k != 'time_idx']
                all_features.append([ts.get(fn, np.nan) for fn in feature_names])
        
        all_features = np.array(all_features)
        
        if fit:
            self.scalers['time_series'] = StandardScaler()
            all_features_scaled = self.scalers['time_series'].fit_transform(all_features)
        else:
            all_features_scaled = self.scalers['time_series'].transform(all_features)
        
        all_features_scaled = np.nan_to_num(all_features_scaled, nan=0.0)
        
        idx = 0
        for record in time_series_data:
            for ts in record['time_steps']:
                for j, fn in enumerate(feature_names):
                    ts[fn] = all_features_scaled[idx, j]
                idx += 1
        
        return time_series_data, feature_names
    
    def create_sequences(self, time_series_data, static_features, labels):
        sequences = []
        
        for record in time_series_data:
            stay_id = record['stay_id']
            
            static = static_features[static_features['stay_id'] == stay_id]
            label = labels[labels['stay_id'] == stay_id]
            
            if len(static) == 0 or len(label) == 0:
                continue
            
            static_values = static.drop(['subject_id', 'hadm_id', 'stay_id'], axis=1).values[0]
            label_value = label['label'].values[0]
            
            ts_values = []
            for ts in record['time_steps']:
                ts_vec = [ts[k] for k in sorted(ts.keys()) if k != 'time_idx']
                ts_values.append(ts_vec)
            
            ts_values = np.array(ts_values)
            
            sequences.append({
                'static': static_values,
                'time_series': ts_values,
                'label': label_value,
                'stay_id': stay_id
            })
        
        return sequences
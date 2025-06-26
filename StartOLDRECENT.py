#!/usr/bin/env python3
"""
Flask Blower App
============================

This is a Flask-based web application for recommending blower models
based on required airflow and pressure, with pricing and weight information.
It replaces the simple built-in HTTP server with a more robust Flask framework.

Usage:
    1. Make sure you have Flask installed: pip install Flask pandas numpy scipy
    2. Place this script and your CSV data files (Heliflow, Sutorbilt, Cycloblower)
       in the same directory.
    3. Run the script: python your_script_name.py (e.g., python flask_blower_app.py)
    4. Then visit: http://localhost:8000 in your web browser.
"""

from flask import Flask, request, jsonify, render_template_string
import json
import urllib.parse
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import os
import glob
import re
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# --- Blower Pricing Data (from SSI Main 2025 Price List) ---

# Sutorbilt Legend Series Pricing
SUTORBILT_PRICING_DATA = {
    "2": {"packagePrice": 6122.00, "enclosurePrice": 2795.00, "packageWeight": 450, "enclosureWeight": 255},
    "3": {"packagePrice": 6303.00, "enclosurePrice": 2795.00, "packageWeight": 470, "enclosureWeight": 255},
    "5": {"packagePrice": 7657.00, "enclosurePrice": 2900.00, "packageWeight": 505, "enclosureWeight": 261},
    "7": {"packagePrice": 8180.00, "enclosurePrice": 4164.00, "packageWeight": 650, "enclosureWeight": 310},
    "10": {"packagePrice": 11496.00, "enclosurePrice": 4164.00, "packageWeight": 700, "enclosureWeight": 310},
    "15": {"packagePrice": 12083.00, "enclosurePrice": 5739.00, "packageWeight": 1100, "enclosureWeight": 400},
    "20": {"packagePrice": 13157.00, "enclosurePrice": 6609.00, "packageWeight": 1250, "enclosureWeight": 450},
    "25": {"packagePrice": 19870.00, "enclosurePrice": 7054.00, "packageWeight": 2300, "enclosureWeight": 500},
    "30": {"packagePrice": 21607.00, "enclosurePrice": 7406.00, "packageWeight": 2450, "enclosureWeight": 550},
    "40": {"packagePrice": 24963.00, "enclosurePrice": 8566.00, "packageWeight": 2650, "enclosureWeight": 650},
    "50": {"packagePrice": 34561.00, "enclosurePrice": 12410.00, "packageWeight": 4490, "enclosureWeight": 1071},
    "60": {"packagePrice": 36238.00, "enclosurePrice": 12410.00, "packageWeight": 4600, "enclosureWeight": 1071},
    "75": {"packagePrice": 51893.00, "enclosurePrice": 12410.00, "packageWeight": 2440, "enclosureWeight": 1071},
    "100": {"packagePrice": 74833.00, "enclosurePrice": 11349.00, "packageWeight": 4200, "enclosureWeight": 1022},
    "125": {"packagePrice": 107068.00, "enclosurePrice": 14280.00, "packageWeight": 5255, "enclosureWeight": 1267},
    "150": {"packagePrice": 149471.00, "enclosurePrice": 14280.00, "packageWeight": 6500, "enclosureWeight": 1267},
    "200": {"packagePrice": 154835.00, "enclosurePrice": 14280.00, "packageWeight": 5450, "enclosureWeight": 1267},
    "250": {"packagePrice": 73584.00, "enclosurePrice": 13038.00, "packageWeight": 5450, "enclosureWeight": 1200}
}

# Cycloblower HE Series Pricing  
CYCLOBLOWER_PRICING_DATA = {
    "75": {"packagePrice": 51348.00, "enclosurePrice": 9596.00, "packageWeight": 3562, "enclosureWeight": 876},
    "100": {"packagePrice": 74833.00, "enclosurePrice": 10237.00, "packageWeight": 4681, "enclosureWeight": 929},
    "125": {"packagePrice": 107068.00, "enclosurePrice": 14280.00, "packageWeight": 7355, "enclosureWeight": 1267},
    "150": {"packagePrice": 149471.00, "enclosurePrice": 14280.00, "packageWeight": 9766, "enclosureWeight": 1267},
    "200": {"packagePrice": 154835.00, "enclosurePrice": 14280.00, "packageWeight": 10550, "enclosureWeight": 1267}
}

# Heliflow Series Pricing (using representative values from ranges)
HELIFLOW_PRICING_DATA = {
    "10": {"packagePrice": 11300.00, "enclosurePrice": 3200.00, "packageWeight": 800, "enclosureWeight": 280},
    "15": {"packagePrice": 12000.00, "enclosurePrice": 3557.00, "packageWeight": 940, "enclosureWeight": 303},
    "20": {"packagePrice": 12900.00, "enclosurePrice": 3557.00, "packageWeight": 1040, "enclosureWeight": 303},
    "25": {"packagePrice": 13600.00, "enclosurePrice": 3557.00, "packageWeight": 1100, "enclosureWeight": 303},
    "30": {"packagePrice": 13900.00, "enclosurePrice": 3557.00, "packageWeight": 1180, "enclosureWeight": 303},
    "40": {"packagePrice": 23033.00, "enclosurePrice": 6750.00, "packageWeight": 2070, "enclosureWeight": 610},
    "50": {"packagePrice": 23800.00, "enclosurePrice": 6750.00, "packageWeight": 2200, "enclosureWeight": 610},
    "60": {"packagePrice": 30800.00, "enclosurePrice": 7400.00, "packageWeight": 2900, "enclosureWeight": 650},
    "75": {"packagePrice": 42000.00, "enclosurePrice": 8500.00, "packageWeight": 3500, "enclosureWeight": 750},
    "100": {"packagePrice": 50000.00, "enclosurePrice": 8600.00, "packageWeight": 4200, "enclosureWeight": 800},
    "125": {"packagePrice": 55000.00, "enclosurePrice": 10200.00, "packageWeight": 4900, "enclosureWeight": 950},
    "150": {"packagePrice": 62000.00, "enclosurePrice": 11500.00, "packageWeight": 6200, "enclosureWeight": 1100},
    "200": {"packagePrice": 75000.00, "enclosurePrice": 13600.00, "packageWeight": 7200, "enclosureWeight": 1250}
}

# --- Helper Functions ---

def get_motor_hp_from_bhp(bhp):
    """Calculate required motor HP from brake HP with tiered safety margin rules."""
    available_hps = [2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150, 200, 250]
    
    # Tiered motor sizing rules:
    # 1. Up to 60 HP: If within 10% of motor size, go to next size up
    # 2. 75 HP: Use if BHP < 70, otherwise go to next size
    # 3. 100+ HP: Allow 10 HP range (90% rule)
    
    for i, motor_hp in enumerate(available_hps):
        if bhp <= motor_hp:
            # Rule 1: Up to 60 HP - strict 10% rule (go up one size if within 10%)
            if motor_hp <= 60:
                if bhp > motor_hp * 0.9:
                    # Within 10% of this motor size, go to next size up
                    if i + 1 < len(available_hps):
                        return available_hps[i + 1]
                    else:
                        return motor_hp  # Use current if it's the largest
                else:
                    return motor_hp
            
            # Rule 2: 75 HP special case
            elif motor_hp == 75:
                if bhp < 70:
                    return 75
                else:
                    # BHP >= 70, go to next size (100 HP)
                    if i + 1 < len(available_hps):
                        return available_hps[i + 1]
                    else:
                        return motor_hp
            
            # Rule 3: 100+ HP - allow 10 HP range (90% rule)
            else:  # motor_hp >= 100
                if bhp > motor_hp * 0.9:
                    # Within 10% (10 HP range for 100+), go to next size up
                    if i + 1 < len(available_hps):
                        return available_hps[i + 1]
                    else:
                        return motor_hp  # Use current if it's the largest
                else:
                    return motor_hp
    
    return None  # No motor large enough

def get_blower_pricing(bhp, series):
    """Get package and enclosure pricing for given brake HP and blower series."""
    motor_hp = get_motor_hp_from_bhp(bhp)
    
    # Select the appropriate pricing table based on series
    if series == 'Sutorbilt Legend':
        pricing_data = SUTORBILT_PRICING_DATA
    elif series == 'Cycloblower HE':
        pricing_data = CYCLOBLOWER_PRICING_DATA
    elif series == 'Heliflow':
        pricing_data = HELIFLOW_PRICING_DATA
    else:
        # Default to Sutorbilt if unknown series
        pricing_data = SUTORBILT_PRICING_DATA
    
    if motor_hp and str(motor_hp) in pricing_data:
        pricing = pricing_data[str(motor_hp)]
        return {
            'motor_hp': motor_hp,
            'package_price': pricing['packagePrice'],
            'enclosure_price': pricing['enclosurePrice'],
            'package_weight': pricing['packageWeight'],
            'enclosure_weight': pricing['enclosureWeight']
        }
    
    # If motor_hp is not found or is None, return a dictionary with None for prices and weights
    return {
        'motor_hp': motor_hp, # Keep motor_hp if it was determined
        'package_price': None,
        'enclosure_price': None,
        'package_weight': None,
        'enclosure_weight': None
    }

def clean_model_name(original_name):
    """Clean up model names to show just the essential model identifier."""
    # Remove common prefixes and clean up the name
    cleaned = original_name
    
    # Handle Heliflow models
    if 'Heliflow' in cleaned and 'HF' in cleaned:
        # Extract just the HF part (e.g., "HF 514")
        hf_match = re.search(r'HF\s*\d+', cleaned)
        if hf_match:
            return hf_match.group(0).replace(' ', ' ')  # Ensure single space
    
    # Handle Sutorbilt models
    if 'Sutorbilt' in cleaned:
        # Remove the long prefix and extract model (e.g., "8MP_8MVP" -> "8MP")
        if '_' in cleaned:
            # Get the part before the underscore
            parts = cleaned.split('_')
            if len(parts) > 0:
                model_part = parts[0]
                # Extract just the model identifier (e.g., "8MP")
                model_match = re.search(r'\d+[A-Z]+', model_part)
                if model_match:
                    return model_match.group(0)
        else:
            # Try to extract model pattern like "2LP", "3MP", etc.
            model_match = re.search(r'\d+[A-Z]+', cleaned)
            if model_match:
                return model_match.group(0)
    
    # Handle Cycloblower models
    if 'Cycloblower' in cleaned or 'CDL' in cleaned:
        # Extract CDL model (e.g., "125CDL375B" -> "125CDL")
        cdl_match = re.search(r'\d+CDL', cleaned)
        if cdl_match:
            return cdl_match.group(0)
    
    # If no specific pattern matched, return a cleaned version
    # Remove common long prefixes
    prefixes_to_remove = [
        'SUTORBILT LEGEND PRESSURE PERFORMANCE DATA LOGGER',
        'HELIFLOW PERFORMANCE DATA LOGGER',
        'CYCLOBLOWER HE PERFORMANCE DATA LOGGER',
        'CYCLOBLOWER HE PRESSURE PERFORMANCE DATA LOGGER',
        'Heliflow',
        'Sutorbilt',
        'Cycloblower'
    ]
    
    for prefix in prefixes_to_remove:
        if prefix in cleaned:
            cleaned = cleaned.replace(prefix, '').strip()
            # Remove leading dashes and extra spaces
            cleaned = cleaned.lstrip('- ').strip()
            break
    
    return cleaned if cleaned else original_name

# --- BlowerModel Class ---
class BlowerModel:
    def __init__(self, csv_file):
        self.name = Path(csv_file).stem.upper()
        if 'Heliflow Performance Data Logger' in self.name:
            model_part = self.name.split('HELIFLOW PERFORMANCE DATA LOGGER')[-1].strip().replace('  ', ' ')
            self.name = f"Heliflow {model_part}"
        elif 'Sutorbilt Legend Pressure Performance Data Logger' in self.name:
            model_part = self.name.split('SUTORBILT LEGEND PRESSURE DATA LOGGER')[-1].strip().replace('  ', ' ')
            self.name = f"Sutorbilt {model_part}"
        elif 'CycloBlower HE Performance Data Logger' in self.name or 'Cycloblower HE Performance Data Logger' in self.name:
            # Handle both possible spellings and clean up the name
            if 'CYCLOBLOWER HE PERFORMANCE DATA LOGGER' in self.name:
                model_part = self.name.split('CYCLOBLOWER HE PERFORMANCE DATA LOGGER')[-1].strip().replace('  ', ' ')
            elif 'CYCLOBLOWER HE PRESSURE PERFORMANCE DATA LOGGER' in self.name:
                model_part = self.name.split('CYCLOBLOWER HE PRESSURE PERFORMANCE DATA LOGGER')[-1].strip().replace('  ', ' ')
            else:
                model_part = self.name
            self.name = f"Cycloblower {model_part}"
        
        self.csv_file = csv_file
        self.data = None
        self.rpm_model = None
        self.power_model = None
        self.max_rpm = None
        self.max_pressure = None
        self.model_accuracy = {}
        self.airflow_range = None
        self.pressure_range = None
        self.load_success = False
        self.error_message = None
        self.series = self._determine_series()
        self.category = self._determine_category()
        
        try:
            self._load_data_from_csv(csv_file)
            self._train_models()
            self._validate_model_automatically()
            self.load_success = True
        except Exception as e:
            self.error_message = str(e)
            self.load_success = False
    
    def _determine_series(self):
        if 'HF' in self.name:
            return 'Heliflow'
        elif any(x in self.name for x in ['LP', 'LR', 'LVR', 'LVP', 'MP', 'MR', 'MVR', 'MVP', 'HP', 'HR', 'HVR', 'HVP']):
            return 'Sutorbilt Legend'
        elif 'CDL' in self.name or 'cycloblower' in self.csv_file.lower():
            return 'Cycloblower HE'
        else:
            return 'Unknown'
    
    def _determine_category(self):
        if 'HF' in self.name:
            if any(x in self.name for x in ['406', '408', '412']):
                return 'HF 400 Compact'
            elif any(x in self.name for x in ['514', '516', '524']):
                return 'HF 500 Industrial'
            elif any(x in self.name for x in ['817', '825']):
                return 'HF 800 Heavy-Duty'
            else:
                return 'Heliflow'
        elif 'LP' in self.name or 'LR' in self.name or 'LVR' in self.name or 'LVP' in self.name:
            return 'Low Pressure'
        elif 'MP' in self.name or 'MR' in self.name or 'MVR' in self.name or 'MVP' in self.name:
            return 'Medium Pressure'
        elif 'HP' in self.name or 'HR' in self.name or 'HVR' in self.name or 'HVP' in self.name:
            return 'High Pressure'
        elif 'CDL' in self.name:
            # Categorize Cycloblower HE models by size
            if any(x in self.name for x in ['125CDL', '160CDL']):
                return 'CDL Compact'
            elif any(x in self.name for x in ['200CDL', '250CDL']):
                return 'CDL Industrial'
            elif any(x in self.name for x in ['300CDL', '350CDL']):
                return 'CDL Heavy-Duty'
            else:
                return 'Cycloblower HE'
        else:
            return 'Unknown'
    
    def _load_data_from_csv(self, csv_file):
        """Load using the EXACT working method from bulk processor."""
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        
        with open(csv_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        header_line_idx = None
        for i, line in enumerate(lines):
            if 'Test Point' in line and 'Airflow' in line and 'Pressure' in line:
                header_line_idx = i
                break
        
        if header_line_idx is None:
            raise ValueError("Could not find header line with required format")
        
        header_line = lines[header_line_idx].strip().replace('\r', '')
        data_lines = lines[header_line_idx + 1:]
        
        clean_data_lines = []
        for line in data_lines:
            clean_line = line.strip().replace('\r', '')
            if clean_line and not clean_line.startswith(',,,'):
                clean_data_lines.append(clean_line)
        
        csv_content = header_line + '\n' + '\n'.join(clean_data_lines)
        df = pd.read_csv(StringIO(csv_content))
        
        column_mapping = {
            'Test Point': 'Test_Point',
            'Airflow (SCFM)': 'Airflow_SCFM',
            'Pressure (PSIG)': 'Pressure_PSIG',
            'BHP': 'Power_BHP',
            'RPM': 'RPM',
            '% Max Speed': 'Pct_Max_Speed',
            '% Max Pressure': 'Pct_Max_Pressure',
            'Notes': 'Notes'
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        numeric_cols = ['Test_Point', 'Airflow_SCFM', 'Pressure_PSIG', 'Power_BHP', 
                       'RPM', 'Pct_Max_Speed', 'Pct_Max_Pressure']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].replace('', np.nan)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['Airflow_SCFM', 'Pressure_PSIG', 'RPM']).reset_index(drop=True)
        
        if len(df) < 3:
            raise ValueError(f"Insufficient data points: {len(df)} (need at least 3)")
        
        self.data = df
        self.airflow_range = (df['Airflow_SCFM'].min(), df['Airflow_SCFM'].max())
        self.pressure_range = (df['Pressure_PSIG'].min(), df['Pressure_PSIG'].max())
        
        self._calculate_max_values()
    
    def _calculate_max_values(self):
        """Calculate maximum RPM and pressure."""
        if 'Pct_Max_Speed' in self.data.columns:
            speed_data = self.data.dropna(subset=['Pct_Max_Speed', 'RPM'])
            if len(speed_data) > 0:
                max_speed_pct = speed_data['Pct_Max_Speed'].max()
                max_speed_row = speed_data[speed_data['Pct_Max_Speed'] == max_speed_pct].iloc[0]
                self.max_rpm = max_speed_row['RPM'] / (max_speed_row['Pct_Max_Speed'] / 100)
        
        # Initialize max_pressure to None, it will be updated if conditions met
        self.max_pressure = None 

        if self.series == 'Cycloblower HE':
            # All Cycloblower HE models use 17 PSIG max pressure (RC1 configuration)
            self.max_pressure = 17.0
        
        # If max_pressure is still None (i.e., not a Cycloblower HE or not explicitly set above)
        # or if it's not set for other series, calculate it from data
        if self.max_pressure is None and 'Pct_Max_Pressure' in self.data.columns:
            pressure_data = self.data.dropna(subset=['Pct_Max_Pressure', 'Pressure_PSIG'])
            if len(pressure_data) > 0:
                max_pressure_pct = pressure_data['Pct_Max_Pressure'].max()
                max_pressure_row = pressure_data[pressure_data['Pct_Max_Pressure'] == max_pressure_pct].iloc[0]
                self.max_pressure = max_pressure_row['Pressure_PSIG'] / (max_pressure_row['Pct_Max_Pressure'] / 100)
    
    def _train_models(self):
        """Train prediction models."""
        airflow = self.data['Airflow_SCFM'].values
        pressure = self.data['Pressure_PSIG'].values
        rpm = self.data['RPM'].values
        points = np.column_stack((airflow, pressure))
        
        self.rpm_model = {
            'points': points,
            'values': rpm
        }
        
        if 'Power_BHP' in self.data.columns:
            power_data = self.data.dropna(subset=['Power_BHP'])
            if len(power_data) > 0:
                power_points = np.column_stack((
                    power_data['Airflow_SCFM'].values,
                    power_data['Pressure_PSIG'].values
                ))
                power_values = power_data['Power_BHP'].values
                
                self.power_model = {
                    'points': power_points,
                    'values': power_values
                }
    
    def _validate_model_automatically(self):
        """Calculate R²."""
        actual_rpm = []
        predicted_rpm = []
        
        for _, row in self.data.iterrows():
            try:
                pred_rpm = self.predict_rpm(row['Airflow_SCFM'], row['Pressure_PSIG'])
                if pred_rpm is not None:
                    actual_rpm.append(row['RPM'])
                    predicted_rpm.append(pred_rpm)
            except:
                continue
        
        if len(actual_rpm) > 1:
            actual_rpm = np.array(actual_rpm)
            predicted_rpm = np.array(predicted_rpm)
            
            ss_res = np.sum((actual_rpm - predicted_rpm) ** 2)
            ss_tot = np.sum((actual_rpm - np.mean(actual_rpm)) ** 2)
            r2_rpm = 1 - (ss_res / ss_tot)
            
            self.model_accuracy['rpm_r2'] = r2_rpm
    
    def predict_rpm(self, airflow, pressure):
        """Predict RPM."""
        if not self.rpm_model:
            return None
        
        points = self.rpm_model['points']
        values = self.rpm_model['values']
        
        result = griddata(points, values, (airflow, pressure), method='linear', fill_value=np.nan)
        
        if np.isnan(result):
            result = griddata(points, values, (airflow, pressure), method='nearest')
        
        return float(result) if not np.isnan(result) else None
    
    def predict_power(self, airflow, pressure):
        """Predict power."""
        if not self.power_model:
            return None
        
        points = self.power_model['points']
        values = self.power_model['values']
        
        result = griddata(points, values, (airflow, pressure), method='linear', fill_value=np.nan)
        
        if np.isnan(result):
            result = griddata(points, values, (airflow, pressure), method='nearest')
        
        return float(result) if not np.isnan(result) else None
    
    def predict(self, airflow, pressure):
        """Predict performance for given airflow and pressure with reasonable range checking."""
        
        # Check if the request is within reasonable bounds of the training data
        min_airflow, max_airflow = self.airflow_range
        min_pressure, max_pressure = self.pressure_range
        
        # Allow reasonable extrapolation (15% beyond training data for established blower curves)
        airflow_margin = (max_airflow - min_airflow) * 0.15
        pressure_margin = (max_pressure - min_pressure) * 0.15
        
        airflow_valid = (airflow >= min_airflow - airflow_margin) and (airflow <= max_airflow + airflow_margin)
        pressure_valid = (pressure >= min_pressure - pressure_margin) and (pressure <= max_pressure + pressure_margin)
        
        if not airflow_valid or not pressure_valid:
            return {'error': f'Request outside valid range. Airflow: {min_airflow:.0f}-{max_airflow:.0f} SCFM (±{airflow_margin:.0f}), Pressure: {min_pressure:.1f}-{max_pressure:.1f} PSI (±{pressure_margin:.1f})'}
        
        rpm = self.predict_rpm(airflow, pressure)
        power = self.predict_power(airflow, pressure)
        
        if rpm is None:
            return {'error': 'Prediction failed - unable to interpolate'}
        
        pct_max_rpm = (rpm / self.max_rpm * 100) if rpm and self.max_rpm else None
        pct_max_pressure = (pressure / self.max_pressure * 100) if self.max_pressure else None
        
        return {
            'hp': power,
            'rpm': rpm,
            'pct_max_rpm': pct_max_rpm,
            'pct_max_pressure': pct_max_pressure,
            'r2': self.model_accuracy.get('rpm_r2')
        }

# --- Load Blower Models ---
def load_blower_models():
    models = {}
    successful_models = []
    failed_models = []
    
    csv_files = glob.glob("*.csv")
    
    # Updated to include Cycloblower HE files
    blower_csv_files = [f for f in csv_files if 
                       'Heliflow Performance Data Logger' in f or 
                       'Sutorbilt Legend Pressure Performance Data Logger' in f or
                       'CycloBlower HE Performance Data Logger' in f or
                       'Cycloblower HE Performance Data Logger' in f or
                       f in ['2LP.csv']]
    
    print(f"Found {len(blower_csv_files)} blower CSV files:")
    for f in blower_csv_files:
        print(f"  • {f}")
    
    for csv_file in blower_csv_files:
        try:
            model = BlowerModel(csv_file)
            if model.load_success:
                models[model.name] = model
                successful_models.append(model.name)
                print(f"✅ Loaded: {model.name} ({len(model.data)} points, R²: {model.model_accuracy.get('rpm_r2', 'N/A'):.3f})")
            else:
                failed_models.append((csv_file, model.error_message))
                print(f"❌ Failed: {csv_file} - {model.error_message}")
        except Exception as e:
            failed_models.append((csv_file, str(e))) # Convert exception to string
            print(f"❌ Failed: {csv_file} - {e}")
    
    # Show summary by series
    series_counts = {}
    for model in models.values():
        series_counts[model.series] = series_counts.get(model.series, 0) + 1
    
    print(f"\n📊 Summary: {len(successful_models)} successful, {len(failed_models)} failed")
    print("📈 By Series:")
    for series, count in series_counts.items():
        print(f"   • {series}: {count} models")
    
    return models

def find_best_blowers(models, airflow, pressure, electricity_cost=0.10):
    """Find best blower recommendations using REAL CSV data with reasonable range checking."""
    results = []
    valid_results = []
    filtered_out = []
    
    print(f"🔍 Checking {len(models)} models for {airflow} SCFM @ {pressure} PSI:")
    
    for model_name, model in models.items():
        # Check if model can handle the request with reasonable extrapolation
        min_airflow, max_airflow = model.airflow_range
        min_pressure, max_pressure = model.pressure_range
        
        # Allow 15% extrapolation beyond training data ranges
        airflow_margin = (max_airflow - min_airflow) * 0.15
        pressure_margin = (max_pressure - min_pressure) * 0.15
        
        airflow_min_ext = min_airflow - airflow_margin
        airflow_max_ext = max_airflow + airflow_margin
        pressure_min_ext = min_pressure - pressure_margin  
        pressure_max_ext = max_pressure + pressure_margin
        
        airflow_capable = airflow_min_ext <= airflow <= airflow_max_ext
        pressure_capable = pressure_min_ext <= pressure <= pressure_max_ext
        
        if not airflow_capable or not pressure_capable:
            reason = []
            if not airflow_capable:
                reason.append(f"airflow {airflow} outside {airflow_min_ext:.0f}-{airflow_max_ext:.0f} (base: {min_airflow:.0f}-{max_airflow:.0f})")
            if not pressure_capable:
                reason.append(f"pressure {pressure} outside {pressure_min_ext:.1f}-{pressure_max_ext:.1f} (base: {min_pressure:.1f}-{max_pressure:.1f})")
            
            filtered_out.append(f"{model_name}: {', '.join(reason)}")
            continue
        
        prediction = model.predict(airflow, pressure)
        
        if 'error' in prediction:
            filtered_out.append(f"{model_name}: {prediction['error']}")
            continue
        
        if not prediction['hp'] or not prediction['rpm']:
            filtered_out.append(f"{model_name}: prediction returned None values")
            continue
        
        # Calculate annual cost
        motor_efficiency = 0.95
        belt_drive_efficiency = 0.97
        power_factor = 0.9
        
        motor_shaft_bhp = prediction['hp'] / belt_drive_efficiency
        kw = (motor_shaft_bhp * 0.746) / (motor_efficiency * power_factor)
        annual_cost = kw * 8760 * electricity_cost
        
        # Get pricing information using series-specific pricing
        pricing_info = get_blower_pricing(prediction['hp'], model.series)
        
        result = {
            'model': str(model.name),
            'display_name': clean_model_name(str(model.name)),  # Add cleaned display name
            'series': str(model.series),
            'category': str(model.category),
            'bhp': float(prediction['hp']),
            'rpm': int(prediction['rpm']),
            'motor_hp': pricing_info['motor_hp'],
            'package_price': pricing_info['package_price'],
            'enclosure_price': pricing_info['enclosure_price'],
            'package_weight': pricing_info.get('package_weight'), # Safely get package_weight
            'enclosure_weight': pricing_info.get('enclosure_weight'), # Safely get enclosure_weight
            'pct_max_rpm': float(prediction['pct_max_rpm']) if prediction['pct_max_rpm'] else None,
            'pct_max_pressure': float(prediction['pct_max_pressure']) if prediction['pct_max_pressure'] else None,
            'r2': float(prediction['r2']) if prediction['r2'] else None,
            'annual_cost': int(round(annual_cost)),
            'airflow_range': [float(model.airflow_range[0]), float(model.airflow_range[1])],
            'pressure_range': [float(model.pressure_range[0]), float(model.pressure_range[1])],
            'data_points': int(len(model.data))
        }
        
        # Determine validity and optimality
        result['is_valid'] = (result['pct_max_rpm'] <= 100 and result['pct_max_pressure'] <= 100)
        result['is_optimal'] = (result['pct_max_rpm'] >= 80 and result['pct_max_rpm'] <= 95 and result['pct_max_pressure'] < 95)
        result['is_best'] = False  # Will be set later
        
        results.append(result)
        
        if result['is_valid']:
            valid_results.append(result)
        
        # Show if extrapolating
        extrap_note = ""
        if airflow < min_airflow or airflow > max_airflow or pressure < min_pressure or pressure > max_pressure:
            extrap_note = " (extrapolated)"
        
        print(f"✅ {model_name}: {prediction['hp']:.1f} BHP, {prediction['rpm']:.0f} RPM ({result['pct_max_rpm']:.1f}% RPM){extrap_note}")
    
    print(f"\n📊 Results: {len(results)} capable models, {len(valid_results)} valid")
    if filtered_out:
        print(f"❌ Filtered out {len(filtered_out)} models:")
        for reason in filtered_out[:8]:  # Show first 8
            print(f"   • {reason}")
        if len(filtered_out) > 8:
            print(f"   • ... and {len(filtered_out) - 8} more")
    
    # Sort results
    results.sort(key=lambda x: (
        not x['is_valid'],  # Valid results first
        not x['is_optimal'],  # Then optimal results
        abs(x['pct_max_rpm'] - 87.5) if x['is_valid'] and x['pct_max_rpm'] else float('inf')  # Then closest to 87.5% RPM
    ))
    
    # Find best fit
    best_fit = None
    optimal_results = [r for r in valid_results if r['is_optimal']]
    
    if optimal_results:
        best_fit = min(optimal_results, key=lambda x: abs(x['pct_max_rpm'] - 87.5))
    elif valid_results:
        best_fit = min(valid_results, key=lambda x: x['pct_max_pressure'] + abs(x['pct_max_rpm'] - 85) * 0.5)
    
    # Mark best fit
    if best_fit:
        for result in results:
            if result['model'] == best_fit['model']:
                result['is_best'] = True
                break
    
    return {
        'results': results,
        'valid_results': valid_results,
        'best_fit': best_fit,
        'total_models': len(models),
        'filtered_count': len(filtered_out)
    }

# HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Blower Recommendation Tool</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
        .container { background: linear-gradient(135deg, #f0f9ff 0%, #e0e7ff 100%); min-height: 100vh; }
        .card { background: white; border-radius: 15px; padding: 30px; margin: 20px 0; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }
        .input-row { display: flex; gap: 20px; margin: 20px 0; }
        .input-group { flex: 1; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input { width: 100%; padding: 12px; border: 2px solid #ddd; border-radius: 8px; font-size: 16px; }
        button { background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); color: white; padding: 12px 30px; border: none; border-radius: 8px; font-size: 16px; cursor: pointer; }
        button:hover { background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%); }
        .results { margin-top: 20px; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f8fafc; font-weight: bold; }
        .optimal { background-color: #f0fdf4; border-left: 4px solid #22c55e; }
        .best { background-color: #fffde7; border-left: 4px solid #facc15; }
        .loading { opacity: 0.6; pointer-events: none; }
        .series-heliflow { border-left: 4px solid #3b82f6; }
        .series-sutorbilt { border-left: 44px solid #ef4444; }
        .series-cycloblower { border-left: 4px solid #8b5cf6; }
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <h1 style="text-align: center; color: #1e40af; margin-bottom: 10px;">🔧 Blower Recommendation Tool</h1>
            <p style="text-align: center; color: #64748b;">Heliflow • Sutorbilt Legend • Cycloblower HE</p>
            
            <div class="input-row">
                <div class="input-group">
                    <label for="airflow">Required Airflow (SCFM)</label>
                    <input type="number" id="airflow" placeholder="Enter SCFM">
                </div>
                <div class="input-group">
                    <label for="pressure">Required Pressure (PSI)</label>
                    <input type="number" id="pressure" placeholder="Enter PSI">
                </div>
                <div class="input-group">
                    <label for="cost">Electricity Cost ($/kWh)</label>
                    <input type="number" id="cost" value="0.10" step="0.01">
                </div>
                <div class="input-group">
                    <button onclick="findBlowers()" id="searchBtn">Find Blowers</button>
                </div>
            </div>
        </div>
        
        <div class="card" id="results" style="display: none;">
            <h2>📊 Results</h2>
            <div id="resultsContent"></div>
        </div>
    </div>

    <script>
        async function findBlowers() {
            const airflow = document.getElementById('airflow').value;
            const pressure = document.getElementById('pressure').value;
            const cost = document.getElementById('cost').value;
            
            if (!airflow || !pressure) {
                alert('Please enter airflow and pressure values');
                return;
            }
            
            const btn = document.getElementById('searchBtn');
            btn.textContent = 'Searching...';
            btn.classList.add('loading');
            
            try {
                const response = await fetch(`/api/recommend?airflow=${airflow}&pressure=${pressure}&cost=${cost}`);
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                const contentType = response.headers.get('content-type');
                if (!contentType || !contentType.includes('application/json')) {
                    const text = await response.text();
                    throw new Error(`Expected JSON but got: ${text.substring(0, 100)}...`);
                }
                
                const data = await response.json();
                
                if (data.error) {
                    throw new Error(data.error);
                }
                
                document.getElementById('results').style.display = 'block';
                document.getElementById('resultsContent').innerHTML = generateResultsHTML(data);
                
            } catch (error) {
                console.error('Full error:', error);
                alert('Error: ' + error.message);
            } finally {
                btn.textContent = 'Find Blowers';
                btn.classList.remove('loading');
            }
        }
        
function generateResultsHTML(data) {
    if (!data.results || data.results.length === 0) {
        return '<p>No suitable blowers found.</p>';
    }
    
    let html = '<table><thead><tr>';
    // Corrected headers to include all relevant data points as per result object
    html += '<th>Model</th><th>Series</th><th>BHP</th><th>Motor HP</th><th>RPM</th>';
    html += '<th>% Max RPM</th><th>% Max Press</th><th>Package Price</th><th>Enclosure Price</th><th>Package Weight</th><th>Enclosure Weight</th><th>Annual Cost</th><th>Status</th>';
    html += '</tr></thead><tbody>';
    
    data.results.forEach(result => {
        let rowClass = '';
        let status = '✅ Valid';
        
        // Add series-specific styling
        if (result.series === 'Heliflow') {
            rowClass += ' series-heliflow';
        } else if (result.series === 'Sutorbilt Legend') {
            rowClass += ' series-sutorbilt';
        } else if (result.series === 'Cycloblower HE') {
            rowClass += ' series-cycloblower';
        }
        
        if (result.is_optimal) {
            rowClass += ' optimal';
            status = '⭐ Optimal';
        }
        if (result.is_best) {
            rowClass += ' best';
            status = '🏆 Best Fit';
        }
        if (!result.is_valid) {
            status = '❌ Over Limit';
        }
        
        html += `<tr class="${rowClass}">`;
        html += `<td><strong>${result.display_name}</strong></td>`;
        html += `<td>${result.series}</td>`;
        html += `<td>${result.bhp.toFixed(2)}</td>`;
        html += `<td>${result.motor_hp || 'N/A'}</td>`;
        html += `<td>${result.rpm.toLocaleString()}</td>`;
        html += `<td>${result.pct_max_rpm.toFixed(1)}%</td>`;
        html += `<td>${result.pct_max_pressure.toFixed(1)}%</td>`;
        // Corrected price display syntax: removed leading ' + '
        html += `<td>${result.package_price ? '$' + Math.round(result.package_price).toLocaleString() : 'N/A'}</td>`;
        html += `<td>${result.enclosure_price ? '$' + Math.round(result.enclosure_price).toLocaleString() : 'N/A'}</td>`;
        // Added weight columns
        html += `<td>${result.package_weight ? Math.round(result.package_weight).toLocaleString() + ' lbs' : 'N/A'}</td>`;
        html += `<td>${result.enclosure_weight ? Math.round(result.enclosure_weight).toLocaleString() + ' lbs' : 'N/A'}</td>`;
        html += `<td>$${result.annual_cost.toLocaleString()}</td>`;
        html += `<td>${status}</td>`;
        html += '</tr>';
    });
    
    html += '</tbody></table>';
    
    // Add summary statistics
    html += '<div style="margin-top: 20px; padding: 15px; background-color: #f8fafc; border-radius: 8px;">';
    html += `<h3>📈 Summary</h3>`;
    html += `<p><strong>Total Models Evaluated:</strong> ${data.total_models}</p>`;
    html += `<p><strong>Capable Models:</strong> ${data.results.length}</p>`;
    html += `<p><strong>Valid Results:</strong> ${data.valid_results.length}</p>`;
    
    if (data.best_fit) {
        html += `<p><strong>🏆 Best Recommendation:</strong> ${data.best_fit.display_name} (${data.best_fit.series})</p>`;
        html += `<p style="margin-left: 20px;">• ${data.best_fit.bhp.toFixed(2)} BHP with ${data.best_fit.motor_hp || 'N/A'} HP motor at ${data.best_fit.rpm.toLocaleString()} RPM</p>`;
        html += `<p style="margin-left: 20px;">• ${data.best_fit.pct_max_rpm.toFixed(1)}% of max RPM, ${data.best_fit.pct_max_pressure.toFixed(1)}% of max pressure</p>`;
        // Corrected price and weight display syntax in summary
        html += `<p style="margin-left: 20px;">• Package Price: ${data.best_fit.package_price ? '$' + Math.round(data.best_fit.package_price).toLocaleString() : 'N/A'}, Enclosure: ${data.best_fit.enclosure_price ? '$' + Math.round(data.best_fit.enclosure_price).toLocaleString() : 'N/A'}</p>`;
        html += `<p style="margin-left: 20px;">• Package Weight: ${data.best_fit.package_weight ? Math.round(data.best_fit.package_weight).toLocaleString() + ' lbs' : 'N/A'}, Enclosure Weight: ${data.best_fit.enclosure_weight ? Math.round(data.best_fit.enclosure_weight).toLocaleString() + ' lbs' : 'N/A'}</p>`;
        html += `<p style="margin-left: 20px;">• Annual operating cost: $${data.best_fit.annual_cost.toLocaleString()}</p>`;
    }
    
    html += '</div>';
    return html;
}

    </script>
</body>
</html>
"""

# --- Flask Application Setup ---
app = Flask(__name__)
PORT = 8000 # Changed port back to 8000 as per usage instructions

# Global variable to store loaded models
# This will be loaded once when the Flask app starts
MODELS = {}

@app.before_request
def load_models_once():
    """
    Load blower models once when the first request comes in.
    This ensures models are loaded only when the server is actually used.
    """
    global MODELS
    if not MODELS: # Only load if not already loaded
        print("🔄 Loading blower models for Flask app...")
        try:
            MODELS = load_blower_models()
            print(f"✅ Ready with {len(MODELS)} models\n")
        except Exception as e:
            print(f"❌ Error loading models during Flask app startup: {e}")
            print("Please check that your CSV files are in the same directory as this script.")
            # In a real app, you might want to handle this more gracefully,
            # perhaps by serving an error page or stopping the app.

@app.route('/')
def index():
    """
    Serves the main HTML page for the blower recommendation tool.
    """
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/recommend', methods=['GET'])
def recommend_blower():
    """
    API endpoint to get blower recommendations based on airflow, pressure, and electricity cost.
    """
    try:
        airflow = float(request.args.get('airflow'))
        pressure = float(request.args.get('pressure'))
        cost = float(request.args.get('cost', 0.10)) # Default to 0.10 if not provided
        
        if not airflow or not pressure:
            return jsonify({'error': 'Airflow and pressure are required parameters.'}), 400
        
        if airflow <= 0 or pressure <= 0:
            return jsonify({'error': 'Airflow and pressure must be positive values.'}), 400

        print(f"🔍 Searching: {airflow} SCFM @ {pressure} PSI")
        
        # Ensure models are loaded before processing requests
        if not MODELS:
            return jsonify({'error': 'Blower models not loaded. Server may have encountered an issue during startup.'}), 500

        recommendations = find_best_blowers(MODELS, airflow, pressure, cost)
        
        print(f"📊 Found {len(recommendations['results'])} results, {len(recommendations['valid_results'])} valid")
        
        return jsonify(recommendations), 200
        
    except ValueError as ve:
        print(f"❌ Invalid input: {ve}")
        return jsonify({'error': f'Invalid input: {str(ve)}'}), 400
    except Exception as e:
        print(f"❌ Server error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/models', methods=['GET'])
def get_models_info():
    """
    API endpoint to return information about all loaded blower models.
    """
    if not MODELS:
        return jsonify({'error': 'Blower models not loaded.'}), 500

    model_info = []
    for model_name, model in MODELS.items():
        model_info.append({
            'name': model.name,
            'series': model.series,
            'category': model.category,
            'airflow_range': [float(model.airflow_range[0]), float(model.airflow_range[1])] if model.airflow_range else None,
            'pressure_range': [float(model.pressure_range[0]), float(model.pressure_range[1])] if model.pressure_range else None,
            'data_points': len(model.data),
            'r2': float(model.model_accuracy.get('rpm_r2')) if model.model_accuracy.get('rpm_r2') else 'N/A',
            'max_rpm': float(model.max_rpm) if model.max_rpm else None,
            'max_pressure': float(model.max_pressure) if model.max_pressure else None
        })
    
    # Group by series for better overview
    series_summary = {}
    for model in MODELS.values():
        if model.series not in series_summary:
            series_summary[model.series] = {
                'count': 0,
                'categories': set(),
                'airflow_range': [float('inf'), 0],
                'pressure_range': [float('inf'), 0]
            }
        
        series_summary[model.series]['count'] += 1
        series_summary[model.series]['categories'].add(model.category)
        
        # Update overall ranges
        min_air, max_air = model.airflow_range if model.airflow_range else (float('inf'), 0)
        min_press, max_press = model.pressure_range if model.pressure_range else (float('inf'), 0)
        
        if min_air < series_summary[model.series]['airflow_range'][0]:
            series_summary[model.series]['airflow_range'][0] = min_air
        if max_air > series_summary[model.series]['airflow_range'][1]:
            series_summary[model.series]['airflow_range'][1] = max_air
        if min_press < series_summary[model.series]['pressure_range'][0]:
            series_summary[model.series]['pressure_range'][0] = min_press
        if max_press > series_summary[model.series]['pressure_range'][1]:
            series_summary[model.series]['pressure_range'][1] = max_press
    
    # Convert sets to lists for JSON serialization
    for series in series_summary:
        series_summary[series]['categories'] = list(series_summary[series]['categories'])
    
    return jsonify({
        'models': model_info,
        'total_models': len(MODELS),
        'series_summary': series_summary
    }), 200

if __name__ == '__main__':
    # Get port from environment variable, default to 8000 for local development
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting Flask Blower Tool with Series-Specific Pricing & Weights")
    print(f"📡 Visit: http://localhost:{port}")
    print(f"📡 Or try: http://127.0.0.1:{port}")
    print("Press Ctrl+C to stop\n")
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

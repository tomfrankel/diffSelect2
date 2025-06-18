#!/usr/bin/env python3
"""
Simple HTTP Server Blower App
============================

Uses Python's built-in HTTP server instead of Flask.
This often works when Flask has connection issues.

Usage:
    python simple_server.py
    Then visit: http://localhost:8000
"""

import http.server
import socketserver
import json
import urllib.parse
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import os
import glob
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# Blower pricing data from SSI Main 2025 Price List
HP_PRICING_DATA = {
    "2": {"packagePrice": 6122.38, "enclosurePrice": 2794.93},
    "3": {"packagePrice": 6303.35, "enclosurePrice": 2794.93},
    "5": {"packagePrice": 7657.06, "enclosurePrice": 2899.61},
    "7": {"packagePrice": 8180.11, "enclosurePrice": 4164.03},
    "10": {"packagePrice": 11496.46, "enclosurePrice": 4164.03},
    "15": {"packagePrice": 12083.62, "enclosurePrice": 5739.37},
    "20": {"packagePrice": 13156.94, "enclosurePrice": 6608.97},
    "25": {"packagePrice": 19870.50, "enclosurePrice": 7053.77},
    "30": {"packagePrice": 21607.11, "enclosurePrice": 7406.46},
    "40": {"packagePrice": 24963.15, "enclosurePrice": 8565.65},
    "50": {"packagePrice": 34561.36, "enclosurePrice": 12410.49},
    "60": {"packagePrice": 36237.67, "enclosurePrice": 12410.49},
    "75": {"packagePrice": 51892.74, "enclosurePrice": 12410.49},
    "100": {"packagePrice": 74833.22, "enclosurePrice": 11349.22},
    "125": {"packagePrice": 107067.97, "enclosurePrice": 14280.47},
    "150": {"packagePrice": 149471.16, "enclosurePrice": 14280.47},
    "200": {"packagePrice": 154835.01, "enclosurePrice": 14280.47},
    "250": {"packagePrice": 73583.68, "enclosurePrice": 13037.56}
}

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

def get_blower_pricing(bhp):
    """Get package and enclosure pricing for given brake HP."""
    motor_hp = get_motor_hp_from_bhp(bhp)
    
    if motor_hp and str(motor_hp) in HP_PRICING_DATA:
        pricing = HP_PRICING_DATA[str(motor_hp)]
        return {
            'motor_hp': motor_hp,
            'package_price': pricing['packagePrice'],
            'enclosure_price': pricing['enclosurePrice']
        }
    
    return {
        'motor_hp': None,
        'package_price': None,
        'enclosure_price': None
    }

def clean_model_name(original_name):
    """Clean up model names to show just the essential model identifier."""
    # Remove common prefixes and clean up the name
    cleaned = original_name
    
    # Handle Heliflow models
    if 'Heliflow' in cleaned and 'HF' in cleaned:
        # Extract just the HF part (e.g., "HF 514")
        import re
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
                import re
                model_match = re.search(r'\d+[A-Z]+', model_part)
                if model_match:
                    return model_match.group(0)
        else:
            # Try to extract model pattern like "2LP", "3MP", etc.
            import re
            model_match = re.search(r'\d+[A-Z]+', cleaned)
            if model_match:
                return model_match.group(0)
    
    # Handle Cycloblower models
    if 'Cycloblower' in cleaned or 'CDL' in cleaned:
        # Extract CDL model (e.g., "125CDL375B" -> "125CDL")
        import re
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

# Same BlowerModel class as before - COMPLETE VERSION
class BlowerModel:
    def __init__(self, csv_file):
        self.name = Path(csv_file).stem.upper()
        if 'Heliflow Performance Data Logger' in self.name:
            model_part = self.name.split('HELIFLOW PERFORMANCE DATA LOGGER')[-1].strip().replace('  ', ' ')
            self.name = f"Heliflow {model_part}"
        elif 'Sutorbilt Legend Pressure Performance Data Logger' in self.name:
            model_part = self.name.split('SUTORBILT LEGEND PRESSURE PERFORMANCE DATA LOGGER')[-1].strip().replace('  ', ' ')
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
        
        if 'Pct_Max_Pressure' in self.data.columns:
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

# Load blower models
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
            failed_models.append((csv_file, str(e)))
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
        
        # Get pricing information
        pricing_info = get_blower_pricing(prediction['hp'])
        
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
        .series-sutorbilt { border-left: 4px solid #ef4444; }
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
    // Corrected headers for the table
    html += '<th>Model</th><th>Series</th><th>BHP</th><th>Motor HP</th><th>RPM</th>';
    html += '<th>% Max RPM</th><th>% Max Press</th><th>Package Price</th><th>Enclosure Price</th><th>Package Weight</th><th>Enclosure Weight</th><th>Annual Cost</th><th>Status</th>'; // Added 'Status' header
    html += '</tr></thead><tbody>';
    
    data.results.forEach(result => {
        let rowClass = '';
        let status = '✅ Valid'; // Initialize status
        
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
            status = '⭐ Optimal'; // Update status for optimal
        }
        if (result.is_best) {
            rowClass += ' best';
            status = '🏆 Best Fit'; // Update status for best fit
        }
        if (!result.is_valid) {
            status = '❌ Over Limit'; // Update status for invalid
        }
        
        html += `<tr class="${rowClass}">`;
        html += `<td><strong>${result.display_name}</strong></td>`;
        html += `<td>${result.series}</td>`;
        html += `<td>${result.bhp.toFixed(2)}</td>`;
        html += `<td>${result.motor_hp || 'N/A'}</td>`;
        html += `<td>${result.rpm.toLocaleString()}</td>`;
        html += `<td>${result.pct_max_rpm.toFixed(1)}%</td>`;
        html += `<td>${result.pct_max_pressure.toFixed(1)}%</td>`;
        // Corrected price display syntax
        html += `<td>${result.package_price ? '$' + Math.round(result.package_price).toLocaleString() : 'N/A'}</td>`;
        html += `<td>${result.enclosure_price ? '$' + Math.round(result.enclosure_price).toLocaleString() : 'N/A'}</td>`;
        // Added weight columns
        html += `<td>${result.package_weight ? Math.round(result.package_weight).toLocaleString() + ' lbs' : 'N/A'}</td>`;
        html += `<td>${result.enclosure_weight ? Math.round(result.enclosure_weight).toLocaleString() + ' lbs' : 'N/A'}</td>`;
        html += `<td>${result.annual_cost.toLocaleString()}</td>`;
        html += `<td>${status}</td>`; // Added status column data
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
        html += `<p style="margin-left: 20px;">• Annual operating cost: ${data.best_fit.annual_cost.toLocaleString()}</p>`;
    }
    
    html += '</div>';
    return html;
}

    </script>
</body>
</html>
"""
   

class BlowerRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Load models once when server starts
        print("🔄 Loading blower models...")
        self.models = load_blower_models()
        print(f"✅ Ready with {len(self.models)} models\n")
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode())
        
        elif self.path.startswith('/api/recommend'):
            # Parse query parameters
            query = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(query)
            
            try:
                if 'airflow' not in params or 'pressure' not in params:
                    raise ValueError("Missing airflow or pressure parameters")
                
                airflow = float(params['airflow'][0])
                pressure = float(params['pressure'][0])
                cost = float(params.get('cost', [0.10])[0])
                
                if airflow <= 0 or pressure <= 0:
                    raise ValueError("Airflow and pressure must be positive values")
                
                print(f"🔍 Searching: {airflow} SCFM @ {pressure} PSI")
                
                # Use REAL recommendation logic with CSV data
                recommendations = find_best_blowers(self.models, airflow, pressure, cost)
                
                print(f"📊 Found {len(recommendations['results'])} results, {len(recommendations['valid_results'])} valid")
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(recommendations, indent=2).encode())
                
            except ValueError as ve:
                print(f"❌ Invalid input: {ve}")
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                error_response = json.dumps({'error': f'Invalid input: {str(ve)}'})
                self.wfile.write(error_response.encode())
                
            except Exception as e:
                print(f"❌ Server error: {e}")
                import traceback
                traceback.print_exc()
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                error_response = json.dumps({'error': f'Server error: {str(e)}'})
                self.wfile.write(error_response.encode())
        
        elif self.path == '/api/models':
            # Return information about loaded models
            model_info = []
            for model_name, model in self.models.items():
                model_info.append({
                    'name': model.name,
                    'series': model.series,
                    'category': model.category,
                    'airflow_range': model.airflow_range,
                    'pressure_range': model.pressure_range,
                    'data_points': len(model.data),
                    'r2': model.model_accuracy.get('rpm_r2'),
                    'max_rpm': model.max_rpm,
                    'max_pressure': model.max_pressure
                })
            
            # Group by series for better overview
            series_summary = {}
            for model in self.models.values():
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
                min_air, max_air = model.airflow_range
                min_press, max_press = model.pressure_range
                
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
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'models': model_info,
                'total_models': len(self.models),
                'series_summary': series_summary
            }, indent=2).encode())
        
        else:
            self.send_response(404)
            self.end_headers()

def main():
    PORT = 8000
    
    print("🚀 Starting Enhanced Blower Tool with Cycloblower HE Support & Pricing")
    print(f"📡 Visit: http://localhost:{PORT}")
    print(f"📡 Or try: http://127.0.0.1:{PORT}")
    print("Press Ctrl+C to stop\n")
    
    # Test if we can load models before starting server
    try:
        print("🧪 Testing model loading...")
        test_models = load_blower_models()
        if len(test_models) == 0:
            print("⚠️  WARNING: No models loaded! Make sure CSV files are in the same directory.")
            print("Expected files like:")
            print("  • Heliflow Performance Data Logger  HF 406.csv")
            print("  • Sutorbilt Legend Pressure Performance Data Logger  2MP_2MVP.csv")
            print("  • CycloBlower HE Performance Data Logger  125CDL375B.csv")
            print("  • etc.")
        else:
            print(f"✅ Model loading test successful: {len(test_models)} models")
            
            # Show breakdown by series
            series_counts = {}
            for model in test_models.values():
                series_counts[model.series] = series_counts.get(model.series, 0) + 1
            
            print(f"\n🔍 Models by Series:")
            for series, count in series_counts.items():
                print(f"   • {series}: {count} models")
                # Show a few examples
                examples = [name for name, model in test_models.items() if model.series == series][:3]
                for example in examples:
                    model = test_models[example]
                    print(f"     - {example}: {model.airflow_range[0]:.0f}-{model.airflow_range[1]:.0f} CFM, {model.pressure_range[0]:.1f}-{model.pressure_range[1]:.1f} PSI")
                if len(examples) < len([m for m in test_models.values() if m.series == series]):
                    remaining = len([m for m in test_models.values() if m.series == series]) - len(examples)
                    print(f"     - ... and {remaining} more")
                    
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("Please check that your CSV files are in the same directory as this script.")
        return
    
    try:
        with socketserver.TCPServer(("", PORT), BlowerRequestHandler) as httpd:
            print(f"\n🌐 Server running on port {PORT}")
            print(f"🎯 Ready to analyze Heliflow, Sutorbilt Legend, and Cycloblower HE models!")
            print(f"💰 With complete pricing from SSI 2025 Price List!")
            httpd.serve_forever()
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"❌ Port {PORT} is already in use. Try a different port or close other applications.")
        else:
            print(f"❌ Network error: {e}")
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Multi-Point Blower Optimization Web App
=======================================

A user-friendly web interface for multi-point blower optimization.
Allows easy input of multiple operating points with individual input boxes.
Modified to use whole numbers only.

Usage:
    1. Make sure you have Flask, pandas, numpy, scipy, and psychrolib installed.
       (e.g., pip install Flask pandas numpy scipy psychrolib)
    2. Place this script and your blower performance CSV data files
       (e.g., Heliflow, Sutorbilt, Cycloblower) in the same directory.
       (Note: package_pricing.csv is temporarily not used, pricing is hardcoded)
    3. Run this script: python multi_point_web_app.py
    4. Visit: http://localhost:8001 in your browser
    5. Select number of operating points and fill in the forms, including ambient conditions.
"""

from flask import Flask, request, jsonify, render_template_string
import json
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import os
import glob
import re
from io import StringIO
from itertools import combinations
from pathlib import Path
import warnings
import psychrolib as psylib

warnings.filterwarnings('ignore') # Suppress warnings, e.g., pandas FutureWarnings

# --- HTML Template with better UI (DEFINED AT THE TOP) ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Point Blower Optimization</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }
        h1 {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        .subtitle {
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 40px;
            font-size: 1.2em;
        }
        .setup-section {
            background: #f8f9fa;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
        }
        .form-group {
            margin-bottom: 25px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #2c3e50;
            font-size: 1.1em;
        }
        select, input {
            width: 100%;
            padding: 15px;
            border: 2px solid #e1e8ed;
            border-radius: 10px;
            font-size: 16px;
            transition: border-color 0.3s ease;
        }
        select:focus, input:focus {
            outline: none;
            border-color: #3498db;
            box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
        }
        .operating-points {
            display: none;
            background: #fff;
            padding: 30px;
            border-radius: 15px;
            margin-top: 20px;
            border: 2px solid #e1e8ed;
        }
        .point-container {
            display: flex;
            align-items: center;
            gap: 20px;
            margin-bottom: 20px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            border-left: 4px solid #3498db;
        }
        .point-number {
            background: #3498db;
            color: white;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 1.2em;
            flex-shrink: 0;
        }
        .input-pair {
            display: flex;
            gap: 15px;
            flex: 1;
        }
        .input-field {
            flex: 1;
        }
        .input-field label {
            margin-bottom: 5px;
            font-size: 0.9em;
            color: #666;
        }
        .input-field input {
            margin-bottom: 0;
        }
        button {
            background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
            color: white;
            padding: 15px 40px;
            border: none;
            border-radius: 10px;
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            width: 100%;
            margin-top: 20px;
        }
        button:hover {
            background: linear-gradient(135deg, #2980b9 0%, #1f4e79 100%);
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        button:disabled {
            background: #bdc3c7;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        .results {
            margin-top: 30px;
            display: none;
        }
        .results-header {
            background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
            color: white;
            padding: 20px;
            border-radius: 15px 15px 0 0;
            text-align: center;
        }
        .results-content {
            background: white;
            border: 2px solid #27ae60;
            border-top: none;
            border-radius: 0 0 15px 15px;
            padding: 30px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        th, td {
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #e1e8ed;
        }
        th {
            background: #34495e;
            color: white;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.9em;
            letter-spacing: 0.5px;
        }
        tr:hover {
            background: #f8f9fa;
        }
        .best-solution {
            background: linear-gradient(135deg, #f1c40f 0%, #f39c12 100%);
            color: #2c3e50;
            font-weight: bold;
        }
        .optimal {
            background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
            color: white;
        }
        .loading {
            text-align: center;
            padding: 40px;
            color: #7f8c8d;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .error {
            background: #e74c3c;
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin: 20px 0;
        }
        .summary-card {
            background: linear-gradient(135deg, #8e44ad 0%, #9b59b6 100%);
            color: white;
            padding: 25px;
            border-radius: 15px;
            margin: 20px 0;
        }
        .point-section {
            margin-bottom: 40px;
        }
        .point-title {
            background: #34495e;
            color: white;
            padding: 15px 20px;
            border-radius: 10px 10px 0 0;
            margin: 0;
            font-size: 1.2em;
        }
        .quick-fill {
            background: #ecf0f1;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .quick-fill h3 {
            margin-top: 0;
            color: #2c3e50;
        }
        .quick-fill-buttons {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        .quick-fill-btn {
            background: #95a5a6;
            color: white;
            padding: 8px 15px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            width: auto;
            margin: 0;
        }
        .quick-fill-btn:hover {
            background: #7f8c8d;
            transform: none;
            box-shadow: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔧 Multi-Point Blower Optimizer</h1>
        <p class="subtitle">Find the most efficient blower configurations for multiple operating points</p>

        <div class="setup-section">
            <div class="form-group">
                <label for="numPoints">How many operating points do you want to analyze?</label>
                <select id="numPoints" onchange="generatePointInputs()">
                    <option value="">Select number of points...</option>
                    <option value="2">2 Operating Points</option>
                    <option value="3">3 Operating Points</option>
                    <option value="4">4 Operating Points</option>
                    <option value="5">5 Operating Points</option>
                </select>
            </div>

            <div class="form-group">
                <label for="electricityCost">Electricity Cost (cents per kWh)</label>
                <input type="number" id="electricityCost" value="10" step="1" min="0">
            </div>

            <div class="form-group">
                <label for="ambientTempF">Ambient Temperature (&deg;F)</label>
                <input type="number" id="ambientTempF" value="68" step="1">
            </div>

            <div class="form-group">
                <label for="elevationFt">Elevation (feet above sea level)</label>
                <input type="number" id="elevationFt" value="0" step="1">
            </div>

            <div class="form-group">
                <label for="relativeHumidity">Relative Humidity (%)</label>
                <input type="number" id="relativeHumidity" value="36" step="1" min="0" max="100">
            </div>

            <button type="button" onclick="generatePointInputs()" style="background: #95a5a6; margin-top: 10px; padding: 10px 20px; width: auto;">
                Generate Input Forms
            </button>
        </div>

        <div id="operatingPoints" class="operating-points" style="display: none;">
            <div class="quick-fill">
                <h3>Quick Fill Examples</h3>
                <div class="quick-fill-buttons">
                    <button type="button" class="quick-fill-btn" onclick="quickFillExample1()">Low Flow System</button>
                    <button type="button" class="quick-fill-btn" onclick="quickFillExample2()">Medium Flow System</button>
                    <button type="button" onclick="quickFillExample3()">High Flow System</button>
                    <button type="button" onclick="clearAll()">Clear All</button>
                </div>
            </div>
            <div id="pointInputs"></div>
            <button onclick="optimizeBlowers()" id="optimizeBtn">🚀 Optimize Blower Selection</button>
        </div>

        <div class="results" id="results">
            <div class="results-header">
                <h2>📊 Optimization Results</h2>
            </div>
            <div class="results-content" id="resultsContent"></div>
        </div>
    </div>

<script>
        function generatePointInputs() {
            const numPoints = document.getElementById('numPoints').value;
            const container = document.getElementById('operatingPoints');
            const inputsContainer = document.getElementById('pointInputs');

            console.log('generatePointInputs called, numPoints:', numPoints);

            if (!numPoints || numPoints === '') {
                container.style.display = 'none';
                console.log('No points selected, hiding container');
                return;
            }

            console.log('Showing container and generating inputs for', numPoints, 'points');
            container.style.display = 'block';
            inputsContainer.innerHTML = '';

            const pointCount = parseInt(numPoints);
            for (let i = 1; i <= pointCount; i++) {
                const pointDiv = document.createElement('div');
                pointDiv.className = 'point-container';
                pointDiv.innerHTML = `
                    <div class="point-number">${i}</div>
                    <div class="input-pair">
                        <div class="input-field">
                            <label for="airflow${i}">Airflow (SCFM)</label>
                            <input type="number" id="airflow${i}" placeholder="Enter SCFM" min="0" step="1">
                        </div>
                        <div class="input-field">
                            <label for="pressure${i}">Pressure (PSI)</label>
                            <input type="number" id="pressure${i}" placeholder="Enter PSI" min="0" step="1">
                        </div>
                    </div>
                `;
                inputsContainer.appendChild(pointDiv);
                console.log('Added input container for point', i);
            }

            // Scroll to the newly created inputs
            container.scrollIntoView({ behavior: 'smooth' });
        }

        function quickFillExample1() {
            // Low Flow System: 500, 1200, 2000 SCFM at 5 PSI
            const examples = [
                {airflow: 500, pressure: 5},
                {airflow: 1200, pressure: 5},
                {airflow: 2000, pressure: 5}
            ];
            fillExample(examples);
        }

        function quickFillExample2() {
            // Medium Flow System: 1000, 2500, 5000 SCFM at 6 PSI
            const examples = [
                {airflow: 1000, pressure: 6},
                {airflow: 2500, pressure: 6},
                {airflow: 5000, pressure: 6}
            ];
            fillExample(examples);
        }

        function quickFillExample3() {
            // High Flow System: 2000, 6000, 10000 SCFM at 7 PSI
            const examples = [
                {airflow: 2000, pressure: 7},
                {airflow: 6000, pressure: 7},
                {airflow: 10000, pressure: 7}
            ];
            fillExample(examples);
        }

        function fillExample(examples) {
            const numPoints = document.getElementById('numPoints').value;
            if (!numPoints) {
                alert('Please select the number of operating points first');
                return;
            }

            for (let i = 0; i < Math.min(examples.length, parseInt(numPoints)); i++) {
                const airflowInput = document.getElementById(`airflow${i+1}`);
                const pressureInput = document.getElementById(`pressure${i+1}`);
                if (airflowInput && pressureInput) {
                    airflowInput.value = examples[i].airflow;
                    pressureInput.value = examples[i].pressure;
                }
            }
        }

        function clearAll() {
            const numPoints = document.getElementById('numPoints').value;
            if (!numPoints) return;

            for (let i = 1; i <= parseInt(numPoints); i++) {
                const airflowInput = document.getElementById(`airflow${i}`);
                const pressureInput = document.getElementById(`pressure${i}`);
                if (airflowInput) airflowInput.value = '';
                if (pressureInput) pressureInput.value = '';
            }
        }

        async function optimizeBlowers() {
            const numPoints = document.getElementById('numPoints').value;
            const electricityCost = document.getElementById('electricityCost').value;
            // Get ambient conditions from inputs
            const ambientTempF = document.getElementById('ambientTempF').value;
            const elevationFt = document.getElementById('elevationFt').value;
            const relativeHumidity = document.getElementById('relativeHumidity').value;

            if (!numPoints) {
                alert('Please select the number of operating points');
                return;
            }

            // Collect operating points
            const operatingPoints = [];
            for (let i = 1; i <= parseInt(numPoints); i++) {
                const airflow = document.getElementById(`airflow${i}`).value;
                const pressure = document.getElementById(`pressure${i}`).value;

                if (!airflow || !pressure) {
                    alert(`Please fill in all values for Operating Point ${i}`);
                    return;
                }

                if (parseFloat(airflow) <= 0 || parseFloat(pressure) <= 0) {
                    alert(`Operating Point ${i}: Airflow and pressure must be positive values`);
                    return;
                }

                operatingPoints.push([parseInt(airflow), parseInt(pressure)]);
            }

            // Show loading
            const btn = document.getElementById('optimizeBtn');
            const resultsDiv = document.getElementById('results');
            const resultsContent = document.getElementById('resultsContent');

            btn.disabled = true;
            btn.textContent = 'Optimizing...';
            resultsDiv.style.display = 'block';
            resultsContent.innerHTML = `
                <div class="loading">
                    <div class="spinner"></div>
                    <p>Analyzing blower configurations for ${operatingPoints.length} operating points...</p>
                    <p>This may take a moment while we evaluate hundreds of combinations.</p>
                </div>
            `;

            try {
                const response = await fetch('/api/optimize', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        operating_points: operatingPoints,
                        electricity_cost: parseInt(electricityCost) / 100, // Convert cents to dollars
                        // Pass ambient conditions to the backend
                        ambient_conditions: {
                            ambient_temp_f: parseInt(ambientTempF),
                            elevation_ft: parseInt(elevationFt),
                            relative_humidity: parseInt(relativeHumidity) / 100 // Convert percentage to decimal
                        }
                    })
                });

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }

                const data = await response.json();

                if (data.error) {
                    throw new Error(data.error);
                }

                displayResults(data);

            } catch (error) {
                console.error('Error:', error);
                resultsContent.innerHTML = `
                    <div class="error">
                        <h3>❌ Error</h3>
                        <p>${error.message}</p>
                    </div>
                `;
            } finally {
                btn.disabled = false;
                btn.textContent = '🚀 Optimize Blower Selection';
            }
        }

        function displayResults(data) {
            const resultsContent = document.getElementById('resultsContent');

            let html = '';

            // Summary
            html += `
                <div class="summary-card">
                    <h3>📋 Analysis Summary</h3>
                    <p><strong>Operating Points:</strong> ${data.operating_points.length}</p>
                    <p><strong>Models Evaluated:</strong> ${data.total_models_attempted}</p>
                    <p><strong>Successfully Loaded Models:</strong> ${data.total_models_loaded}</p>
                    <p><strong>Electricity Cost:</strong> ${Math.round(data.electricity_cost * 100)} cents/kWh</p>
                    <p><strong>Ambient Conditions:</strong> ${data.ambient_conditions.ambient_temp_f}&deg;F, ${data.ambient_conditions.elevation_ft} ft, ${Math.round(data.ambient_conditions.relative_humidity * 100)}% RH</p>
                </div>
            `;

            // Best solutions for each point
            html += '<h3>🎯 Best Single Blower for Each Operating Point</h3>';

            for (const [pointKey, pointData] of Object.entries(data.single_solutions)) {
                const [airflow, pressure] = pointData.operating_point;
                html += `
                    <div class="point-section">
                        <h4 class="point-title">${pointKey}: ${Math.round(airflow).toLocaleString()} SCFM @ ${Math.round(pressure)} PSI</h4>
                `;

                if (pointData.solutions.length > 0) {
                    html += '<table><thead><tr>';
                    html += '<th>Rank</th><th>Model</th><th>Series</th><th>BHP</th><th>Motor HP</th>';
                    html += '<th>RPM</th><th>% Max RPM</th><th>Isentropic Eff.</th><th>Package Price</th><th>Annual Cost</th>';
                    html += '</tr></thead><tbody>';

                    pointData.solutions.slice(0, 5).forEach((solution, index) => {
                        const rowClass = index === 0 ? 'best-solution' : '';
                        const rank = index === 0 ? '🏆' : `${index + 1}.`;

                        html += `<tr class="${rowClass}">`;
                        html += `<td>${rank}</td>`;
                        html += `<td><strong>${solution.display_name}</strong></td>`;
                        html += `<td>${solution.series}</td>`;
                        html += `<td>${Math.round(solution.bhp)}</td>`;
                        html += `<td>${solution.motor_hp || 'N/A'}</td>`;
                        html += `<td>${Math.round(solution.rpm).toLocaleString()}</td>`;
                        html += `<td>${Math.round(solution.pct_max_rpm)}%</td>`;
                        html += `<td>${Math.round(solution.isentropic_efficiency)}%</td>`;
                        html += `<td>${solution.package_price ? '$' + Math.round(solution.package_price).toLocaleString() : 'N/A'}</td>`;
                        html += `<td>$${Math.round(solution.annual_cost).toLocaleString()}</td>`;
                        html += '</tr>';
                    });

                    html += '</tbody></table>';
                } else {
                    html += '<p class="error">❌ No valid solutions found for this operating point or models not loaded.</p>';
                }

                html += '</div>';
            }

            // Overall recommendations
            html += `
                <div class="summary-card">
                    <h3>💡 Recommendations</h3>
                    <p><strong>For variable-demand applications:</strong></p>
                    <ul>
                        <li>Consider using multiple smaller blowers that can be staged on/off.</li>
                        <li>Optimal efficiency often occurs at 80-90% of maximum RPM.</li>
                        <li>Avoid operating below 60% of maximum RPM for positive displacement blowers.</li>
                        <li>VFD (Variable Frequency Drive) control may be beneficial for continuous modulation and energy savings.</li>
                    </ul>
                    <p><strong>Note:</strong> Only blowers with exact motor HP pricing matches are shown. Blowers requiring motor sizes without pricing data are automatically excluded.</p>
                </div>
            `;

            resultsContent.innerHTML = html;
        }
</script>
</body>
</html>
"""

app = Flask(__name__)
PORT = 8000

# --- Constants and Data Loading ---
CURRENT_DIR = Path(__file__).parent
DATA_DIR = CURRENT_DIR

# Psychrolib setup
psylib.SetUnitSystem(psylib.SI) # Set SI units

# Path to the package pricing CSV (No longer used for actual pricing data, but kept for consistency)
PACKAGE_PRICING_FILE = DATA_DIR / 'package_pricing.csv'

# --- HARDCODED PRICING DATA (for testing/debugging purposes) ---
SUTORBILT_PRICING_DATA = {
    2: {"packagePrice": 6122, "enclosurePrice": 2795, "packageWeight": 450, "enclosureWeight": 255},
    3: {"packagePrice": 6303, "enclosurePrice": 2795, "packageWeight": 470, "enclosureWeight": 255},
    5: {"packagePrice": 7657, "enclosurePrice": 2900, "packageWeight": 505, "enclosureWeight": 261},
    7: {"packagePrice": 8180, "enclosurePrice": 4164, "packageWeight": 650, "enclosureWeight": 310},
    10: {"packagePrice": 11496, "enclosurePrice": 4164, "packageWeight": 700, "enclosureWeight": 310},
    15: {"packagePrice": 12083, "enclosurePrice": 5739, "packageWeight": 1100, "enclosureWeight": 400},
    20: {"packagePrice": 13157, "enclosurePrice": 6609, "packageWeight": 1250, "enclosureWeight": 450},
    25: {"packagePrice": 19870, "enclosurePrice": 7054, "packageWeight": 2300, "enclosureWeight": 500},
    30: {"packagePrice": 21607, "enclosurePrice": 7406, "packageWeight": 2450, "enclosureWeight": 550},
    40: {"packagePrice": 24963, "enclosurePrice": 8566, "packageWeight": 2650, "enclosureWeight": 650},
    50: {"packagePrice": 34561, "enclosurePrice": 12410, "packageWeight": 4490, "enclosureWeight": 1071},
    60: {"packagePrice": 36238, "enclosurePrice": 12410, "packageWeight": 4600, "enclosureWeight": 1071},
    75: {"packagePrice": 51893, "enclosurePrice": 12410, "packageWeight": 2440, "enclosureWeight": 1071},
    100: {"packagePrice": 74833, "enclosurePrice": 11349, "packageWeight": 4200, "enclosureWeight": 1022},
    125: {"packagePrice": 107068, "enclosurePrice": 14280, "packageWeight": 5255, "enclosureWeight": 1267},
    150: {"packagePrice": 149471, "enclosurePrice": 14280, "packageWeight": 6500, "enclosureWeight": 1267},
    200: {"packagePrice": 154835, "enclosurePrice": 14280, "packageWeight": 5450, "enclosureWeight": 1267},
    250: {"packagePrice": 73584, "enclosurePrice": 13038, "packageWeight": 5450, "enclosureWeight": 1200}
}

CYCLOBLOWER_PRICING_DATA = {
    75: {"packagePrice": 51348, "enclosurePrice": 9596, "packageWeight": 3562, "enclosureWeight": 876},
    100: {"packagePrice": 74833, "enclosurePrice": 10237, "packageWeight": 4681, "enclosureWeight": 929},
    125: {"packagePrice": 107068, "enclosurePrice": 14280, "packageWeight": 7355, "enclosureWeight": 1267},
    150: {"packagePrice": 149471, "enclosurePrice": 14280, "packageWeight": 9766, "enclosureWeight": 1267},
    200: {"packagePrice": 154835, "enclosurePrice": 14280, "packageWeight": 10550, "enclosureWeight": 1267}
}

HELIFLOW_PRICING_DATA = {
    10: {"packagePrice": 11300, "enclosurePrice": 3200, "packageWeight": 800, "enclosureWeight": 280},
    15: {"packagePrice": 12000, "enclosurePrice": 3557, "packageWeight": 940, "enclosureWeight": 303},
    20: {"packagePrice": 12900, "enclosurePrice": 3557, "packageWeight": 1040, "enclosureWeight": 303},
    25: {"packagePrice": 13600, "enclosurePrice": 3557, "packageWeight": 1100, "enclosureWeight": 303},
    30: {"packagePrice": 13900, "enclosurePrice": 3557, "packageWeight": 1180, "enclosureWeight": 303},
    40: {"packagePrice": 23033, "enclosurePrice": 6750, "packageWeight": 2070, "enclosureWeight": 610},
    50: {"packagePrice": 23800, "enclosurePrice": 6750, "packageWeight": 2200, "enclosureWeight": 610},
    60: {"packagePrice": 30800, "enclosurePrice": 7400, "packageWeight": 2900, "enclosureWeight": 650},
    75: {"packagePrice": 42000, "enclosurePrice": 8500, "packageWeight": 3500, "enclosureWeight": 750},
    100: {"packagePrice": 50000, "enclosurePrice": 8600, "packageWeight": 4200, "enclosureWeight": 800},
    125: {"packagePrice": 55000, "enclosurePrice": 10200, "packageWeight": 4900, "enclosureWeight": 950},
    150: {"packagePrice": 62000, "enclosurePrice": 11500, "packageWeight": 6200, "enclosureWeight": 1100},
    200: {"packagePrice": 75000, "enclosurePrice": 13600, "packageWeight": 7200, "enclosureWeight": 1250}
}
# --- END HARDCODED PRICING DATA ---

# Global variables to store loaded models (PACKAGE_PRICING_DF is no longer used for data)
MODELS = {}
PACKAGE_PRICING_DF = pd.DataFrame() # Keep as empty for now, or remove if not passed to get_blower_pricing.

# --- Helper Functions (From Start.py, adapted for this context) ---

def get_motor_hp_from_bhp(bhp, series=None): # Removed package_pricing_df parameter
    """Calculate required motor HP from brake HP with tiered safety margin rules."""
    available_hps = [2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150, 200, 250]

    for i, motor_hp in enumerate(available_hps):
        if bhp <= motor_hp:
            # Tiered motor sizing rules
            if motor_hp <= 60:
                if bhp > motor_hp * 0.9: # If within 10% of motor size, go to next size up
                    if i + 1 < len(available_hps):
                        return available_hps[i + 1]
                    else:
                        return motor_hp  # Use current if it's the largest
                else:
                    return motor_hp
            elif motor_hp == 75: # 75 HP special case
                if bhp < 70:
                    return 75
                else:
                    if i + 1 < len(available_hps):
                        return available_hps[i + 1]
                    else:
                        return motor_hp
            else: # motor_hp >= 100 - allow 10 HP range (90% rule)
                if bhp > motor_hp * 0.9:
                    if i + 1 < len(available_hps):
                        return available_hps[i + 1]
                    else:
                        return motor_hp
                else:
                    return motor_hp
    return None

def get_blower_pricing(motor_hp, series): # Removed package_pricing_df parameter
    """Get package and enclosure pricing for given motor HP and blower series from hardcoded data.
    Returns None for package_price if exact HP match not found."""

    pricing_data = {}
    if series == 'Sutorbilt Legend':
        pricing_data = SUTORBILT_PRICING_DATA
    elif series == 'Cycloblower HE':
        pricing_data = CYCLOBLOWER_PRICING_DATA
    elif series == 'Heliflow':
        pricing_data = HELIFLOW_PRICING_DATA
    else:
        # Default or unknown series, no pricing available
        return {
            'motor_hp': motor_hp,
            'package_price': None,
            'enclosure_price': None,
            'package_weight': None,
            'enclosure_weight': None
        }

    # Check if motor_hp exists in the selected pricing_data dictionary
    if motor_hp in pricing_data: # Direct lookup for int key
        pricing = pricing_data[motor_hp]
        return {
            'motor_hp': motor_hp,
            'package_price': pricing.get('packagePrice'), # Use .get for robustness with dict
            'enclosure_price': pricing.get('enclosurePrice'),
            'package_weight': pricing.get('packageWeight'),
            'enclosure_weight': pricing.get('enclosureWeight')
        }
    
    # If no exact match found, return None values (this will exclude the blower)
    return {
        'motor_hp': motor_hp,
        'package_price': None,
        'enclosure_price': None,
        'package_weight': None,
        'enclosure_weight': None
    }

def clean_model_name(original_name):
    """Clean up model names to show just the essential model identifier."""
    cleaned = original_name

    if 'Heliflow' in cleaned and 'HF' in cleaned:
        hf_match = re.search(r'HF\s*\d+', cleaned)
        if hf_match:
            return hf_match.group(0).replace(' ', ' ')

    if 'Sutorbilt' in cleaned:
        if '_' in cleaned:
            parts = cleaned.split('_')
            if len(parts) > 0:
                model_part = parts[0]
                model_match = re.search(r'\d+[A-Z]+', model_part)
                if model_match:
                    return model_match.group(0)
        else:
            model_match = re.search(r'\d+[A-Z]+', cleaned)
            if model_match:
                return model_match.group(0)

    if 'Cycloblower' in cleaned or 'CDL' in cleaned:
        cdl_match = re.search(r'\d+CDL', cleaned)
        if cdl_match:
            return cdl_match.group(0)

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
            cleaned = cleaned.lstrip('- ').strip()
            break

    return cleaned if cleaned else original_name

# --- BlowerModel Class (from Start.py, adapted) ---
class BlowerModel:
    def __init__(self, csv_file):
        self.name = Path(csv_file).stem.upper()
        # The display_name should be set AFTER determining series to handle specific cleanups
        self.series_determined = self._determine_series_initial() # Temporary series for initial naming
        self.display_name = clean_model_name(self.name) # Use unified cleaner

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
        self.series = self._determine_series() # Final series determination
        self.category = self._determine_category()

        try:
            self._load_data_from_csv(csv_file)
            self._train_models()
            self._validate_model_automatically()
            self.load_success = True
        except Exception as e:
            self.error_message = str(e)
            self.load_success = False

    def _determine_series_initial(self):
        # Helper for initial naming to match the old naming logic if needed
        if 'HF' in self.name or 'HELIFLOW' in self.name:
            return 'Heliflow'
        elif any(x in self.name for x in ['LP', 'LR', 'LVP', 'MP', 'MR', 'MVP', 'HP', 'HR', 'HVP']) or 'SUTORBILT' in self.name:
            return 'Sutorbilt Legend'
        elif 'CDL' in self.name or 'CYCLOBLOWER' in self.name:
            return 'Cycloblower HE'
        else:
            return 'Unknown'

    def _determine_series(self):
        # More robust series determination after cleaning name
        if 'HELIFLOW' in self.name or 'HF' in self.name:
            return 'Heliflow'
        elif 'SUTORBILT' in self.name or any(s in self.name for s in ['LP', 'LR', 'LVP', 'MP', 'MR', 'MVP', 'HP', 'HR', 'HVP']):
            return 'Sutorbilt Legend'
        elif 'CYCLOBLOWER' in self.name or 'CDL' in self.name:
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
        """Load using the robust method from Start.py."""
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

        # Critical: Drop NaNs from essential columns for prediction
        df = df.dropna(subset=['Airflow_SCFM', 'Pressure_PSIG', 'RPM', 'Power_BHP']).reset_index(drop=True)

        if len(df) < 3:
            raise ValueError(f"Insufficient data points: {len(df)} (need at least 3)")

        self.data = df
        self.airflow_range = (df['Airflow_SCFM'].min(), df['Airflow_SCFM'].max())
        self.pressure_range = (df['Pressure_PSIG'].min(), df['Pressure_PSIG'].max())

        self._calculate_max_values()

    def _calculate_max_values(self):
        """Calculate maximum RPM and pressure based on % Max Speed/Pressure logic."""
        self.max_rpm = None
        if 'Pct_Max_Speed' in self.data.columns and not self.data.empty:
            speed_data = self.data.dropna(subset=['Pct_Max_Speed', 'RPM'])
            if len(speed_data) > 0 and speed_data['Pct_Max_Speed'].max() > 0:
                max_speed_pct = speed_data['Pct_Max_Speed'].max()
                max_speed_row = speed_data[speed_data['Pct_Max_Speed'] == max_speed_pct].iloc[0]
                self.max_rpm = max_speed_row['RPM'] / (max_speed_row['Pct_Max_Speed'] / 100)
        # Fallback if max_rpm still None
        if self.max_rpm is None and not self.data.empty:
            self.max_rpm = self.data['RPM'].max() * 1.05 # Assume max is 5% above highest data point for safety

        self.max_pressure = None
        if self.series == 'Cycloblower HE':
            self.max_pressure = 17.0 # RC1 configuration limit
        elif 'Pct_Max_Pressure' in self.data.columns and not self.data.empty:
            pressure_data = self.data.dropna(subset=['Pct_Max_Pressure', 'Pressure_PSIG'])
            if len(pressure_data) > 0 and pressure_data['Pct_Max_Pressure'].max() > 0:
                max_pressure_pct = pressure_data['Pct_Max_Pressure'].max()
                max_pressure_row = pressure_data[pressure_data['Pct_Max_Pressure'] == max_pressure_pct].iloc[0]
                self.max_pressure = max_pressure_row['Pressure_PSIG'] / (max_pressure_row['Pct_Max_Pressure'] / 100)
        # Fallback if max_pressure still None
        if self.max_pressure is None and not self.data.empty:
             self.max_pressure = self.data['Pressure_PSIG'].max() * 1.05 # Assume max is 5% above highest data point for safety


    def _train_models(self):
        """Train prediction models using griddata."""
        airflow = self.data['Airflow_SCFM'].values
        pressure = self.data['Pressure_PSIG'].values
        rpm = self.data['RPM'].values
        points = np.column_stack((airflow, pressure))

        self.rpm_model = {
            'points': points,
            'values': rpm
        }

        # Ensure Power_BHP exists and is not all NaN
        if 'Power_BHP' in self.data.columns and not self.data['Power_BHP'].isnull().all():
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
            else:
                self.power_model = None # No valid power data after dropping NaNs
        else:
            self.power_model = None # No Power_BHP column or all NaN

    def _validate_model_automatically(self):
        """Calculate R² for RPM prediction."""
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

        if len(actual_rpm) > 1 and np.sum((actual_rpm - np.mean(actual_rpm)) ** 2) != 0: # Avoid division by zero
            actual_rpm = np.array(actual_rpm)
            predicted_rpm = np.array(predicted_rpm)

            ss_res = np.sum((actual_rpm - predicted_rpm) ** 2)
            ss_tot = np.sum((actual_rpm - np.mean(actual_rpm)) ** 2)
            r2_rpm = 1 - (ss_res / ss_tot)

            self.model_accuracy['rpm_r2'] = r2_rpm
        else:
            self.model_accuracy['rpm_r2'] = None # Cannot calculate R2

    def predict_rpm(self, airflow, pressure):
        """Predict RPM using griddata."""
        if not self.rpm_model or len(self.rpm_model['points']) < 3:
            return None

        points = self.rpm_model['points']
        values = self.rpm_model['values']

        result = griddata(points, values, (airflow, pressure), method='linear', fill_value=np.nan)

        if np.isnan(result):
            result = griddata(points, values, (airflow, pressure), method='nearest')

        return float(result) if not np.isnan(result) else None

    def predict_power(self, airflow, pressure):
        """Predict power (BHP) using griddata."""
        if not self.power_model or len(self.power_model['points']) < 3:
            return None

        points = self.power_model['points']
        values = self.power_model['values']

        result = griddata(points, values, (airflow, pressure), method='linear', fill_value=np.nan)

        if np.isnan(result):
            result = griddata(points, values, (airflow, pressure), method='nearest')

        return float(result) if not np.isnan(result) else None

    def predict(self, airflow, pressure):
        """Predict performance for given airflow and pressure with reasonable range checking."""
        min_airflow, max_airflow = self.airflow_range
        min_pressure, max_pressure = self.pressure_range

        # Allow reasonable extrapolation (15% beyond training data for established blower curves)
        airflow_margin = (max_airflow - min_airflow) * 0.15
        pressure_margin = (max_pressure - min_pressure) * 0.15

        airflow_valid = (airflow >= min_airflow - airflow_margin) and (airflow <= max_airflow + airflow_margin)
        pressure_valid = (pressure >= min_pressure - pressure_margin) and (pressure <= max_pressure + pressure_margin)

        if not airflow_valid or not pressure_valid:
            return {'error': f'Request outside valid range. Airflow: {airflow} (base: {min_airflow:.0f}-{max_airflow:.0f}, margin: {airflow_margin:.0f}), Pressure: {pressure} (base: {min_pressure:.1f}-{max_pressure:.1f}, margin: {pressure_margin:.1f})'}

        rpm = self.predict_rpm(airflow, pressure)
        power = self.predict_power(airflow, pressure)

        if rpm is None:
            return {'error': 'Prediction failed - unable to interpolate RPM.'}
        if power is None:
            return {'error': 'Prediction failed - unable to interpolate BHP.'}
        if self.max_rpm is None:
             return {'error': 'Prediction failed - Max RPM for model is not determined.'}
        if self.max_pressure is None:
             return {'error': 'Prediction failed - Max Pressure for model is not determined.'}

        pct_max_rpm = (rpm / self.max_rpm * 100)
        pct_max_pressure = (pressure / self.max_pressure * 100)

        return {
            'bhp': power,
            'rpm': rpm,
            'pct_max_rpm': pct_max_rpm,
            'pct_max_pressure': pct_max_pressure,
            'r2': self.model_accuracy.get('rpm_r2')
        }

# --- Ambient Condition & Thermodynamic Calculations (Using psychrolib) ---

def convert_f_to_c(temp_f):
    """Converts Fahrenheit to Celsius."""
    return (temp_f - 32) * 5 / 9

def convert_ft_to_m(elevation_ft):
    """Converts feet to meters."""
    return elevation_ft * 0.3048

def calculate_air_properties(ambient_temp_f, elevation_ft, relative_humidity):
    """
    Calculates various air properties using psychrolib.
    Returns dry bulb temperature (C), atmospheric pressure (Pa), humidity ratio (kg_w/kg_da), and air density (kg/m^3).
    """
    psylib.SetUnitSystem(psylib.SI) # Ensure SI units

    Tdb_C = convert_f_to_c(ambient_temp_f)
    elevation_m = convert_ft_to_m(elevation_ft)

    P_atm_Pa = psylib.GetStandardAtmPressure(elevation_m)
    W = psylib.GetHumRatioFromRelHum(Tdb_C, relative_humidity, P_atm_Pa)
    rho_kg_m3 = psylib.GetMoistAirDensity(Tdb_C, W, P_atm_Pa)

    return Tdb_C, P_atm_Pa, W, rho_kg_m3

def calculate_isentropic_power_and_efficiency(
    airflow_scfm, pressure_psig, bhp_measured,
    ambient_temp_f, elevation_ft, relative_humidity
):
    """
    Calculates the ideal isentropic power and blower isentropic efficiency using psychrolib.
    Args:
        airflow_scfm (float): Airflow in Standard Cubic Feet Per Minute.
        pressure_psig (float): Gauge pressure rise in PSI.
        bhp_measured (float): Actual Brake Horsepower of the blower.
        ambient_temp_f (float): Ambient dry bulb temperature in Fahrenheit.
        elevation_ft (float): Elevation above sea level in feet.
        relative_humidity (0.0 to 1.0): Relative humidity.

    Returns:
        tuple: (P_isentropic_kW, isentropic_efficiency_pct)
               P_isentropic_kW is the ideal isentropic power in kW.
               isentropic_efficiency_pct is the calculated isentropic efficiency in %.
    """
    # Gas constant for dry air (J/(kg·K))
    R_dry_air = 287.058
    # Specific heat ratio for air (gamma or k)
    k_air = 1.4

    # Convert ambient conditions to SI units and get properties using psychrolib
    Tdb_C, P_atm_Pa, W, rho_actual_kg_m3 = calculate_air_properties(ambient_temp_f, elevation_ft, relative_humidity)
    Tdb_K = Tdb_C + 273.15 # Convert to Kelvin

    # Calculate gas constant for moist air
    R_moist_air = R_dry_air * (1 + 1.6078 * W) / (1 + W)

    # Convert SCFM (Standard Cubic Feet Per Minute) to Actual Mass Flow Rate (kg/s)
    # Standard conditions for SCFM (ASHRAE): 20C (68F), 101.325 kPa (14.696 psiA), 0% RH (dry air for consistency)
    T_std_C = 20
    P_std_Pa = 101325
    W_std = psylib.GetHumRatioFromRelHum(T_std_C, 0, P_std_Pa) # Direct call
    rho_std_kg_m3 = psylib.GetMoistAirDensity(T_std_C, W_std, P_std_Pa) # Direct call

    # Convert SCFM to Standard Volumetric Flow Rate (m^3/s)
    Q_std_m3_s = airflow_scfm * (0.0283168 / 60) # SCFM to m^3/s

    # Calculate mass flow rate (kg/s) based on standard conditions
    m_dot_kg_s = Q_std_m3_s * rho_std_kg_m3

    # Convert pressure_psig (gauge) to absolute pressure in Pascals
    P_in_abs_Pa = P_atm_Pa # Inlet pressure is atmospheric
    P_out_abs_Pa = P_atm_Pa + (pressure_psig * 6894.757) # P_out = P_in + delta_P

    if P_out_abs_Pa <= P_in_abs_Pa or m_dot_kg_s <= 0: # Safety check
        return 0.0, 0.0

    # Isentropic power calculation for compressible flow (Watts)
    # P_isen = m_dot * (k/(k-1)) * R_moist_air * T_in * [(P_out/P_in)^((k-1)/k) - 1]
    P_isentropic_W = m_dot_kg_s * (k_air / (k_air - 1)) * R_moist_air * Tdb_K * \
                     ((P_out_abs_Pa / P_in_abs_Pa)**((k_air - 1) / k_air) - 1)

    P_isentropic_kW = P_isentropic_W / 1000 # Convert to kW

    # Convert measured BHP to kW
    bhp_kw = bhp_measured * 0.7457 # 1 HP = 0.7457 kW

    # Isentropic Efficiency calculation (%)
    # Efficiency = (Ideal Isentropic Power / Actual Power Input) * 100
    isentropic_efficiency_pct = (P_isentropic_kW / bhp_kw) * 100.0 if bhp_kw > 0 else 0.0

    # Cap efficiency at 100% and return both values
    isentropic_efficiency_pct = max(0.0, min(100.0, isentropic_efficiency_pct))
    return P_isentropic_kW, isentropic_efficiency_pct

# --- Optimization Logic ---

class MultiPointOptimizer:
    def __init__(self, models, electricity_cost=0.10):
        self.models = models
        self.electricity_cost = electricity_cost
        self.min_rpm_pct = 60 # Minimum recommended operating RPM %
        self.max_rpm_pct = 100 # Maximum operating RPM % (blower limit)

    def find_single_blower_solutions(self, operating_points, ambient_conditions):
        """
        Finds the best single blower solutions for each given operating point,
        including isentropic efficiency and cost calculations.
        """
        single_point_solutions = {}

        for i, (airflow, pressure) in enumerate(operating_points):
            point_key = f"Operating Point {i + 1}"
            solutions_for_this_point = []

            for model_name, blower_model in self.models.items():
                if not blower_model.load_success:
                    continue # Skip models that failed to load

                prediction = blower_model.predict(airflow, pressure)

                if 'error' in prediction:
                    continue # Skip if prediction failed or outside range

                bhp = prediction['bhp']
                rpm = prediction['rpm']
                pct_max_rpm = prediction['pct_max_rpm']
                pct_max_pressure = prediction['pct_max_pressure']

                # Filter based on valid operating range and reasonable output
                if bhp is None or bhp <= 0 or rpm is None or rpm <=0 or pct_max_rpm is None or \
                   pct_max_rpm > self.max_rpm_pct or pct_max_rpm < self.min_rpm_pct:
                    continue

                # Calculate motor HP based on BHP and tiered rules
                motor_hp = get_motor_hp_from_bhp(bhp, blower_model.series)
                if motor_hp is None:
                    continue # Skip if no suitable motor HP found

                # Get package pricing from the hardcoded data
                pricing_info = get_blower_pricing(motor_hp, blower_model.series)
                package_price = pricing_info['package_price']
                
                # CRITICAL: If package_price is None, skip this option
                if package_price is None:
                    continue  # Don't include blowers without valid pricing

                # Calculate isentropic efficiency
                P_isentropic_kW, isentropic_efficiency_pct = calculate_isentropic_power_and_efficiency(
                    airflow, pressure, bhp,
                    ambient_conditions['ambient_temp_f'],
                    ambient_conditions['elevation_ft'],
                    ambient_conditions['relative_humidity']
                )

                # Calculate annual cost (using common efficiencies for motor/belt)
                motor_efficiency = 0.95 # Assumed average motor efficiency
                belt_drive_efficiency = 0.97 # Assumed average belt drive efficiency
                power_factor = 0.9 # Assumed average power factor

                # Electrical input power (kW)
                electrical_input_kw = (bhp * 0.7457) / (motor_efficiency * belt_drive_efficiency * power_factor) if bhp > 0 else 0
                annual_cost = electrical_input_kw * 24 * 365 * self.electricity_cost

                solutions_for_this_point.append({
                    'model': str(blower_model.name),
                    'display_name': str(blower_model.display_name),
                    'series': str(blower_model.series),
                    'bhp': round(bhp),
                    'rpm': round(rpm),
                    'motor_hp': int(motor_hp),
                    'pct_max_rpm': round(pct_max_rpm),
                    'pct_max_pressure': round(pct_max_pressure),
                    'isentropic_efficiency': round(isentropic_efficiency_pct),
                    'package_price': round(package_price) if package_price is not None else None,
                    'annual_cost': round(annual_cost),
                    'airflow_scfm_interpolated': round(airflow),
                    'pressure_psi_interpolated': round(pressure),
                    'is_valid': bool(pct_max_rpm <= 100 and pct_max_pressure <= 100),
                    'is_optimal': bool(pct_max_rpm >= 80 and pct_max_rpm <= 95 and pct_max_pressure < 95)
                })

            # Sort solutions for the current point
            solutions_for_this_point.sort(key=lambda x: (
                not x['is_valid'],  # Invalid last
                not x['is_optimal'],  # Non-optimal last among valid
                x['annual_cost'],  # Lowest annual cost first
                -x['isentropic_efficiency']  # Highest efficiency first
            ))

            single_point_solutions[point_key] = {
                'operating_point': [round(airflow), round(pressure)],
                'solutions': solutions_for_this_point[:10] # Return top 10 solutions per point
            }

        return single_point_solutions

# --- Load Blower Models & Pricing Data ---
def load_all_blower_data():
    """
    Loads all blower models from CSV files.
    Note: package pricing is now hardcoded, so PACKAGE_PRICING_DF is not used for loading pricing data from CSV.
    """
    global MODELS

    models = {}
    successful_models_count = 0
    total_csv_files_attempted = 0

    # Get all CSV files, then filter based on expected blower file names
    csv_files_in_dir = glob.glob(str(DATA_DIR / "*.csv"))

    # Filter for known blower performance data patterns, excluding package_pricing.csv
    blower_csv_files = [f for f in csv_files_in_dir if
                        ('Heliflow Performance Data Logger' in f or
                         'Sutorbilt Legend Pressure Performance Data Logger' in f or
                         'CycloBlower HE Performance Data Logger' in f or
                         'Cycloblower HE Performance Data Logger' in f or
                         # Specific files that might not match above patterns but are blowers:
                         'Heliflow 2_ Belt Drive.csv' in f or
                         re.search(r'\d+[A-Z]+\.csv', os.path.basename(f)) # e.g., '2LP.csv'
                        ) and 'package_pricing' not in f.lower() # Exclude pricing file
                       ]
    total_csv_files_attempted = len(blower_csv_files)

    if not blower_csv_files:
        print("No blower performance CSV files found. Please ensure they are in the same directory.")
        return 0, 0 # Return 0, 0 for attempted and loaded if no files found

    for csv_file in blower_csv_files:
        try:
            model = BlowerModel(csv_file)
            if model.load_success:
                models[model.name] = model
                successful_models_count += 1
            else:
                pass # Suppress verbose error messages for skipped files
        except Exception as e:
            pass # Suppress verbose error messages for skipped files

    MODELS = models
    return total_csv_files_attempted, successful_models_count

# --- Flask Routes ---

@app.before_request
def load_data_once():
    """
    Load blower models once when the Flask app starts.
    Note: package pricing is now hardcoded, so the CSV pricing file is not loaded here.
    """
    global MODELS
    if not MODELS: # Only load if not already loaded
        print("🔄 Loading blower models for Flask app (using hardcoded pricing)...")
        try:
            total_attempted, total_loaded = load_all_blower_data()
            print(f"✅ Successfully loaded {total_loaded} of {total_attempted} blower models.")
            print("✅ All necessary data loading complete.\n")
        except Exception as e:
            print(f"❌ Error during initial data load: {e}")
            import traceback
            traceback.print_exc()
            print("Please ensure your CSV files are in the same directory as this script.")

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/optimize', methods=['POST'])
def optimize_blowers():
    try:
        data = request.get_json()
        operating_points = data.get('operating_points', [])
        electricity_cost = data.get('electricity_cost', 0.10)
        ambient_conditions = data.get('ambient_conditions', {}) # Get ambient conditions

        if not operating_points:
            return jsonify({'error': 'No operating points provided'}), 400

        if not MODELS:
            return jsonify({'error': 'Blower models not loaded. Server may have encountered an issue during startup.'}), 500

        print(f"🔍 Optimizing for {len(operating_points)} operating points")

        # Create optimizer instance
        optimizer = MultiPointOptimizer(MODELS, electricity_cost)

        # Find single blower solutions, passing ambient conditions
        single_solutions = optimizer.find_single_blower_solutions(operating_points, ambient_conditions)

        # Prepare response - all data is already in basic Python types
        results = {
            'operating_points': operating_points,
            'electricity_cost': round(electricity_cost, 2),
            'ambient_conditions': ambient_conditions,
            'total_models_attempted': int(len(glob.glob(str(DATA_DIR / "*.csv"))) - 1 if PACKAGE_PRICING_FILE.exists() else len(glob.glob(str(DATA_DIR / "*.csv")))),
            'total_models_loaded': int(len(MODELS)),
            'single_solutions': single_solutions,
        }

        return jsonify(results)

    except Exception as e:
        print(f"❌ Server error during optimization: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    print(f"🚀 Starting Multi-Point Blower Optimization Web App")
    print(f"📡 Visit: http://localhost:{PORT}")
    print("Press Ctrl+C to stop\n")
    app.run(host='0.0.0.0', port=PORT, debug=False, use_reloader=False)

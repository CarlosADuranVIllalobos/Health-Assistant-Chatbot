# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 00:42:03 2024

new_entries.py
This script is designed to update existing datasets with new entries for illnesses and their associated health advice and symptoms.
It primarily modifies two key datasets: Health_Advice_for_Illnesses.csv and symbipredict_2022.csv.

@author: Carlos A. Duran Villalobos
"""

import pandas as pd

# Load the health advice CSV
health_advice_df = pd.read_csv('../data/Health_Advice_for_Illnesses.csv')

# Add entries for Flu and Coronavirus
new_entries = pd.DataFrame({
    'Illness': ['Flu', 'Coronavirus'],
    'Advice': [
        'Stay hydrated, rest, and take over-the-counter fever reducers.',
        'Isolate, monitor symptoms, seek medical attention if symptoms worsen.'
    ],
    'See Doctor': [
        'if symptoms persist more than 5 days or worsen.',
        'Yes, especially if experiencing difficulty breathing or high fever.'
    ],
    'Explanation': [
        'Influenza (Flu) is a respiratory illness caused by influenza viruses.',
        'COVID-19 is caused by the SARS-CoV-2 virus and can lead to severe respiratory illness.'
    ]
})

# Append new entries to the health advice dataframe
health_advice_df = health_advice_df.append(new_entries, ignore_index=True)

# Save the updated CSV
health_advice_df.to_csv('../data/Health_Advice_for_Illnesses.csv', index=False)

# Load the symptom prediction CSV
symptom_predict_df = pd.read_csv('../data/symbipredict_2022.csv')

# Add simplified entries for Flu and Coronavirus
# Assuming a simplified symptom set; adjust according to your model's features
new_symptom_entries = pd.DataFrame({
    'prognosis': ['Flu', 'Coronavirus'],
    'fever': [1, 1],
    'cough': [1, 1],
    'fatigue': [1, 1],
    'body aches': [1, 0],
    'difficulty breathing': [0, 1],
    # Add other symptom columns with default value of 0
    # This should include all other symptom columns present in your dataframe
}).reindex(columns=symptom_predict_df.columns, fill_value=0)

# Append new entries to the symptom prediction dataframe
symptom_predict_df = symptom_predict_df.append(new_symptom_entries, ignore_index=True)

# Save the updated CSV
symptom_predict_df.to_csv('../data/symbipredict_2022.csv', index=False)
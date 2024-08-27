# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 00:13:33 2024

fine_tuning_model.py
Script for fine-tuning Rasa pipelines.

@author: Carlos A. Duran-Villalobos
"""

import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

# Define configurations to test
epochs = [100, 200, 500,  1000]
thresholds = [0.1, 0.3, 0.7]
learning_rates = [0.01, 0.001, 0.0001]

# Results DataFrame
results_df = pd.DataFrame(columns=['Epoch', 'Threshold', 'Learning Rate', 'F1 Score'])

def parse_results(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    f1_scores = []
    for key, value in data.items():
        if isinstance(value, dict) and 'f1-score' in value:
            f1_score = value['f1-score']
            if isinstance(f1_score, (float, int)):
                f1_scores.append(f1_score)
    if f1_scores:
        return sum(f1_scores) / len(f1_scores)
    return None  # Return None if no F1 scores were found

def run_rasa_test(epoch, threshold, learning_rate):
    config_filename = f'../scripts/config_{epoch}_{threshold}_{learning_rate}.yml'
    with open(config_filename, 'w') as file:
        file.write(f"""
language: en
pipeline:
  - name: WhitespaceTokenizer
  - name: RegexFeaturizer
  - name: LexicalSyntacticFeaturizer
  - name: CountVectorsFeaturizer
  - name: CountVectorsFeaturizer
    analyzer: char_wb
    min_ngram: 1
    max_ngram: 4
  - name: DIETClassifier
    epochs: {epoch}
    learning_rate: {learning_rate}
    constrain_similarities: true
  - name: EntitySynonymMapper
  - name: ResponseSelector
    epochs: 100
    constrain_similarities: true
  - name: FallbackClassifier
    threshold: {threshold}

policies:
  - name: MemoizationPolicy
  - name: RulePolicy
  - name: UnexpecTEDIntentPolicy
    max_history: 5
    epochs: 100
  - name: TEDPolicy
    max_history: 5
    epochs: 100
    constrain_similarities: true
""")

    # Ensure the results directory exists
    if not os.path.exists('../results'):
        os.makedirs('../results')

    # Run Rasa test and capture output
    result = subprocess.run(
        ['rasa', 'test', 'nlu', '--nlu', '../data/nlu.yml', '--config', config_filename, '--cross-validation', '--folds', '5', '--out', '../results'],
        capture_output=True, text=True
    )
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    return parse_results('../results/intent_report.json')

# Test configurations and log results
for epoch in epochs:
    for threshold in thresholds:
        for learning_rate in learning_rates:
            f1_score = run_rasa_test(epoch, threshold, learning_rate)
            results_df = results_df.append({
                'Epoch': epoch,
                'Threshold': threshold,
                'Learning Rate': learning_rate,
                'F1 Score': f1_score
            }, ignore_index=True)

# Save results to CSV
results_df.to_csv('../results/nlu_test_results.csv', index=False)

# Plotting the results
# Setting up the figure and axis
fig, ax = plt.subplots(figsize=(12, 8))

# Plotting results with default color palette
for (threshold, learning_rate), grp in results_df.groupby(['Threshold', 'Learning Rate']):
    ax.plot(grp['Epoch'], grp['F1 Score'], label=f'Thresh: {threshold}, LR: {learning_rate}', linewidth=2)

# Adding legend, grid, title, and labels with larger fonts
ax.legend(loc='best', fontsize='large')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.set_title('DIET Classifier Performance with Varying Configurations', fontsize=16, fontweight='bold')
ax.set_xlabel('Epochs', fontsize=14, fontweight='bold')
ax.set_ylabel('F1 Score', fontsize=14, fontweight='bold')

# Set larger tick labels
ax.tick_params(axis='both', which='major', labelsize=12)

# Save plot to file
fig_path = os.path.join('../results', 'performance_plot.png')
plt.savefig(fig_path, format='png', dpi=300)

# Show plot
plt.tight_layout()
plt.show()
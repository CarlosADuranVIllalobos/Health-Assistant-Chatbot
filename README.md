# Travel Spanish Chatbot

This repository demonstrates how to build a customized chatbot AI for learning and practicing useful Spanish for traveling using Python and the ChatterBot Corpus. It covers data preprocessing, chatbot training, and evaluation.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Chatbot Training](#chatbot-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project focuses on building a customized travel Spanish chatbot using Python. It includes scripts for data preprocessing, training a chatbot model, and evaluating the chatbot. The chatbot can be fine-tuned to respond to specific travel-related queries and provide relevant information based on the trained dataset.

## Dataset

The dataset used is the ChatterBot Corpus, which includes a variety of conversational data. For this project, we have customized the corpus to include travel-specific dialogues in Spanish.

### Example Dataset Structure

The dataset is organized in .yml files with the following structure:

- `categories`: Categories of the conversation
- `conversations`: List of dialogues

## Project Structure

Travel_Spanish_Chatbot/

```plaintext
Travel_Spanish_Chatbot/
├── data/
│   ├── travel_corpus.yml             # Custom travel-specific dialogues
├── notebooks/
│   ├── data_preprocessing.ipynb      # Data preprocessing and EDA
│   ├── chatbot_training.ipynb        # Chatbot training
│   ├── chatbot_evaluation.ipynb      # Chatbot evaluation
├── scripts/
│   ├── preprocess.py                 # Data preprocessing script
│   ├── train_chatbot.py              # Chatbot training script
│   ├── evaluate_chatbot.py           # Chatbot evaluation script
├── models/
│   ├── chatbot_model.pkl             # Trained chatbot model
├── results/
│   ├── evaluation_metrics.csv        # Evaluation metrics
│   ├── confusion_matrix.png          # Confusion matrix
├── README.md                         # Project README

# Health Assistant Chatbot


This repository demonstrates how to build a customized AI chatbot designed to provide basic medical advice. The chatbot is intended for demonstration purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.

## Table of Contents

- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Results](#results)
- [Disclaimer](#disclaimer)
- [Contributing](#contributing)
- [License](#license)

## Dataset

The chatbot is trained using a custom dataset that includes conversations related to common medical inquiries. The focus is on providing informative responses to general health-related questions.

### Citation of the Dataset

The `medical_corpus.yml` file contains hand-crafted conversations specifically designed for this project. If you use this repository, please cite it appropriately.

## Project Structure

```plaintext
Health_Assistant_Chatbot/
├── actions/
│   ├── actions.py                      # Contains custom action code for Rasa chatbot
├── data/
│   ├── Healt_advice_for_Illnesses.csv  # Data with health advice based on symptoms
│   ├── nlu.yml                         # Contains NLU training data
│   ├── rules.yml                       # Rules for the chatbot's behavior
│   ├── stories.yml                     # Stories to train the dialogue model
│   ├── symbipredict_2022.csv           # Additional dataset used for model training
├── scripts/
│   ├── fine_tuning_model.py            # Script to fine-tune the chatbot model
│   ├── new_entries.py                  # Script to add new entries to the dataset
├── models/
│   ├── chatbot_model.tar.gz            # Trained chatbot model
├── results/  
│   ├── DIETCLassifier_confusion_matrix.png  # Confusion matrix for DIET Classifier
│   ├── DIETCLassifier_histogram.png         # Histogram for DIET Classifier performance
│   ├── intent_confusion_matrix.png          # Confusion matrix for intent recognition
│   ├── intent_histogram.png                 # Histogram for intent recognition performance
│   ├── performance_plot.png                 # Performance plot of the model
│   ├── nlu_test_results.csv                 # Test results for the NLU component
├── test/
│   ├── test_stories.yml                # Test stories for validating the chatbot
├── background.png                      # Background image for the chatbot interface
├── config.yml                          # Configuration for Rasa model
├── credentials.yml                     # Credentials for connecting the chatbot to messaging platforms
├── domain.yml                          # Domain file defining intents, entities, and slots
├── endpoints.yml                       # Endpoints configuration for action server
├── index.html                          # Web interface for the chatbot
├── README.md                           # Project README
├── requirements.txt                    # Python dependencies
```

## Usage

1. **Fine-Tuning the Model**:
   - Use the `fine_tuning_model.py` script to fine-tune the chatbot model with the latest data.
   - [Fine-Tuning Script](scripts/fine_tuning_model.py)

2. **Adding New Entries**:
   - Add new health-related entries to the dataset using the `new_entries.py` script.
   - [New Entries Script](scripts/new_entries.py)

3. **Running the Chatbot**:
   - Run the chatbot locally by executing the following command:
     ```bash
     rasa run actions --debug
     rasa run --enable-api --cors "*" --debug
     ```

4. **Testing the Chatbot**:
   - Validate the chatbot's responses by using the test stories provided in the `test_stories.yml` file.

## Modeling

The project uses the Rasa framework to build and train the chatbot. The chatbot is trained with a custom medical corpus (`nlu.yml`) that includes a variety of common medical inquiries and responses.

### Example Medical Corpus

The `nlu.yml` includes conversations like:

```yaml
categories:
- health

conversations:
- - What are the symptoms of the flu?
  - Common symptoms include fever, cough, sore throat, and body aches.
- - How can I treat a cold at home?
  - Rest, drink plenty of fluids, and take over-the-counter medications to relieve symptoms.
- - Should I go to the doctor if I have a fever?
  - If your fever is high or persistent, it's a good idea to consult a healthcare professional.

## Evaluation

Model performance is assessed using metrics such as:

- Accuracy
- Precision
- Recall
- F1-Score

Visualizations include confusion matrices.

## Results

### Confusion Matrices

![Confusion Matrices](results/confusion_matrices.png)

### Evaluation Metrics

| Metric    | Score |
|-----------|-------|
| Accuracy  | 0.95  |
| Precision | 0.96  |
| Recall    | 0.94  |
| F1-Score  | 0.95  |

### Discussion and Conclusion

The chatbot demonstrates high accuracy and performs well in understanding and responding to both English and Spanish queries. It is particularly useful for travelers looking to practice and learn Spanish travel phrases.

## Contributing

Contributions are welcome! Please create an issue or submit a pull request for any feature requests or improvements.

## License

This project is licensed under the MIT License.

If you use this repository in your research, please cite it as shown in the right sidebar.

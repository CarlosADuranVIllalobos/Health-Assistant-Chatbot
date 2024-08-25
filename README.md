# Travel Spanish Chatbot

This repository demonstrates how to build a customized AI chatbot to help users practice and learn useful Spanish for traveling. The chatbot is designed to handle conversations in both English and Spanish, making it an ideal tool for English speakers looking to improve their Spanish language skills.

## Table of Contents

- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Dataset

The chatbot is trained using a custom bilingual corpus that includes conversations in both English and Spanish. This corpus helps the chatbot understand and respond in both languages, facilitating language learning.

### Citation of the Dataset

The `bilingual_corpus.yml` file contains hand-crafted conversations specifically designed for this project. If you use this repository, please cite it appropriately.

## Project Structure
Travel_Chatbot/

```plaintext
Travel_Chatbot/
├── data/
│   ├── bilingual_corpus.yml          # Bilingual conversation corpus
├── notebooks/
│   ├── data_preprocessing.ipynb      # Data preprocessing and EDA
│   ├── model_training.ipynb          # Model training and tuning
│   ├── model_evaluation.ipynb        # Model evaluation
├── scripts/
│   ├── preprocess.py                 # Data preprocessing script
│   ├── train.py                      # Model training script
│   ├── evaluate.py                   # Model evaluation script
├── models/
│   ├── chatbot_model.pkl             # Trained chatbot model
├── results/
│   ├── evaluation_metrics.csv        # Evaluation metrics
│   ├── confusion_matrix.png          # Confusion matrix
├── README.md                         # Project README
```

## Usage

1. **Data Preprocessing**:
   - Execute the `data_preprocessing.ipynb` notebook to clean and preprocess the data.
   - [Data Preprocessing Notebook](notebooks/data_preprocessing.ipynb)

2. **Model Training**:
   - Use the `model_training.ipynb` notebook to train the chatbot model using the bilingual corpus.
   - [Model Training Notebook](notebooks/model_training.ipynb)

3. **Model Evaluation**:
   - Evaluate the performance of the trained model using the `model_evaluation.ipynb` notebook.
   - [Model Evaluation Notebook](notebooks/model_evaluation.ipynb)

## Modeling

The project uses the ChatterBot library to build and train the chatbot. The chatbot is trained with the custom bilingual corpus (`bilingual_corpus.yml`) that includes a wide range of travel-related conversations in both English and Spanish.

### Example Bilingual Corpus

The `bilingual_corpus.yml` includes conversations like:

```yaml
categories:
- travel

conversations:
- - Where is the bathroom?
  - El baño está al fondo a la derecha.
- - Thank you very much.
  - No hay de qué, fue un placer ayudarte.
- - Excuse me, where is the restroom?
  - El baño está al fondo a la derecha.
- - Please, can you help me?
  - Claro, ¿en qué puedo ayudarte?
...
```

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

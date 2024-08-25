from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer, ListTrainer
import spacy

# Load SpaCy models
nlp_en = spacy.load("en_core_web_sm")
nlp_es = spacy.load("es_core_news_sm")

def create_and_train_chatbot(corpus_path):
    chatbot = ChatBot(
        'TravelBot',
        storage_adapter='chatterbot.storage.SQLStorageAdapter',
        database_uri='sqlite:///database.sqlite3',
        logic_adapters=[
            'chatterbot.logic.BestMatch'
        ],
        preprocessors=[
            'chatterbot.preprocessors.clean_whitespace'
        ]
    )

    trainer = ChatterBotCorpusTrainer(chatbot)
    trainer.train(corpus_path)

    # Optionally, train with built-in English and Spanish datasets
    trainer.train("chatterbot.corpus.english")
    trainer.train("chatterbot.corpus.spanish")

    print("Chatbot trained successfully")

if __name__ == "__main__":
    path = "../data/bilingual_corpus.yml"
    create_and_train_chatbot(path)

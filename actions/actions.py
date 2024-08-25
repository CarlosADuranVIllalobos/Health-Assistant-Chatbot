# -*- coding: utf-8 -*-
"""
Created on Sat Aug 3 00:42:03 2024

actions.py
This script integrates GPT-Neo and NLTK libraries for natural language processing and implements several actions for Rasa SDK to manage user interactions within a chatbot context. The actions include generating responses using GPT-Neo, analyzing symptoms, and providing health advice based on CSV data sources.

@author: Carlos A. Duran Villalobos
"""
# Importing required libraries
#from transformers import GPTNeoForCausalLM, GPT2Tokenizer #if you are running your own server
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import UserUtteranceReverted, SlotSet, FollowupAction
from typing import Any, Text, Dict, List
import pandas as pd
import numpy as np
import re
from nltk.stem import WordNetLemmatizer
import nltk
import requests
# Load environment variables and initialize NLP tools
from dotenv import load_dotenv
load_dotenv()
lemmatizer = WordNetLemmatizer()

## If you re running your own server, Initialize the GPT-Neo tokenizer and model 
#tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
#model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
def ensure_nltk_data():
    """Ensure necessary NLTK data packages are downloaded."""
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("WordNet not found; downloading...")
        nltk.download('wordnet')
    else:
        print("WordNet is already installed.")

ensure_nltk_data()
## If you are running your own server
# def get_text_from_gpt(user_text):
#     """Generate text using GPT-Neo model."""
#     input_ids = tokenizer(user_text, return_tensors="pt").input_ids
#     max_length = len(input_ids[0]) + 50
#     attention_mask = input_ids.ne(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id).long()
#     response_ids = model.generate(input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id, attention_mask=attention_mask, do_sample=True, temperature=0.7)
#     return tokenizer.decode(response_ids[0], skip_special_tokens=True)


def get_text_from_gpt(user_text):
    """Generate text using GPT-Neo 2.7B model via Hugging Face API."""
    API_URL = "https://api-inference.huggingface.co/models/EleutherAI/gpt-neo-2.7B"
    headers = {"Authorization": "Bearer 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"}
    
    # Prepare the request data with the parameters you previously used for the local model
    prompt_parameters = {
        "inputs": user_text,
        "parameters": {
            "max_length": len(user_text.split()) + 100,
            "temperature": 0.7
        },
        "options": {
            "use_cache": False
        }
    }

    response = requests.post(API_URL, headers=headers, json=prompt_parameters)

    # Check if the request was successful
    if response.status_code == 200:
        result = response.json()
        if result and 'generated_text' in result[0]:  # API returns a list with the generated text inside a dictionary
            return result[0]['generated_text']
        else:
            return "Generated text not found in the response."
    else:
        return f"Failed to generate text, status code: {response.status_code}, response: {response.text}"

def normalize_text(text):
    """Normalize text by removing extra spaces, punctuation, and converting to lower case."""
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lower case
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])  # Lemmatize words

def construct_health_question(intent, user_text):
    if intent == "ask_symptoms":
        return f"What illness might I have if {user_text}?"
    elif intent == "ask_advice":
        return f"What should I do if {user_text}?"
    elif intent == "ask_doctor":
        return f"Do I need to see a doctor if {user_text}?"
    else:
        return user_text

class ActionNotifyResponseGeneration(Action):
    """Action to notify users that the response is being generated."""
    def name(self) -> Text:
        return "action_notify_response_generation"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message("Generating response... Please wait.")
        return [FollowupAction("action_print_done")]

class ActionPrintDone(Action):
    """Action to signify that processing is complete."""
    def name(self) -> Text:
        return "action_print_done"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        return [FollowupAction("symptom_analysis_action")]



class Simple_GPT_Action(Action):
    """Fallback action using GPT to handle unrecognizable input."""
    def name(self) -> Text:
        return "action_gpt_default_fallback"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        user_text = tracker.latest_message.get('text')
        response = get_text_from_gpt(user_text + "?")
        dispatcher.utter_message('GPT (custom_action): ' + response)
        return [UserUtteranceReverted()]

class SymptomAnalysisAction(Action):
    """
    This class defines an action to analyze user-reported symptoms and predict possible illnesses.
    It utilizes data from a CSV file to match symptoms to possible diagnoses.
    """

    def name(self) -> Text:
        """Defines the name of the action which Rasa will use to call this class in actions."""
        return "symptom_analysis_action"

    def run(self, dispatcher, tracker, domain):
        """
        Execute the action.
        
        Args:
        dispatcher: The dispatcher is used to send messages back to the user.
        tracker: The tracker stores the state of the conversation and can retrieve user inputs and other data.
        domain: The domain specifies the intents, entities, slots, and templates required for the application.

        Returns:
        A list of events or actions to be taken in response to the user's message.
        """
        # Retrieve the latest message, its intent, and entities
        user_text = tracker.latest_message.get('text', '')
        intent = tracker.latest_message.get('intent', {}).get('name', '')
        entities = tracker.latest_message.get('entities', [])

        # Normalize the user input to standardize it for processing
        user_text = normalize_text(user_text)
        intent = normalize_text(intent)

        # Convert entities to a dictionary, with symptoms normalized
        symptoms_input = {normalize_text(entity['value']): 1 for entity in entities if entity['entity'] == 'symptom'}
        if not symptoms_input:
            dispatcher.utter_message("Please specify your symptoms for a proper diagnosis.")
            return []

        print(f"Intent received: {intent}, Symptoms: {symptoms_input}")
        # Load and normalize the symptom prediction data
        df = pd.read_csv('data/symbipredict_2022.csv')
        df.columns = [normalize_text(col) for col in df.columns]

        # Process the symptoms to predict possible illnesses
      
        if intent == "ask_symptoms" and symptoms_input:
            num_diagnoses, predicted_illness = self.predict_illness(symptoms_input)
            if num_diagnoses == 1 and predicted_illness:
                # Respond with the predicted illness and offer further assistance or advice
                dispatcher.utter_message(f"Based on your symptoms, you might have: {predicted_illness}.")
                return [SlotSet("illness", predicted_illness), FollowupAction("action_provide_illness_advice")]
            elif num_diagnoses > 1:
                # Recommend seeing a doctor or provide an option to specify symptoms further
                dispatcher.utter_message(f"{predicted_illness} Please see a doctor for a precise diagnosis or provide more specific symptoms.")
                return [SlotSet("illness", predicted_illness)]
            else:
                # If no illness is confidently predicted, seek clarification or provide generic advice
                dispatcher.utter_message("This is a GPT answer. If you need to specify an illness, please provide more specific symptoms.")
                gpt_response = self.get_answers_from_gpt(intent, user_text)
                dispatcher.utter_message(gpt_response)
                return [UserUtteranceReverted()]
        else:
            # Use a generic GPT-based response if the intent does not match or if symptoms are not detected
            response = self.get_answers_from_gpt(intent, user_text)
            dispatcher.utter_message(response)
            return [UserUtteranceReverted()]

    def predict_illness(self, symptoms_input):
        """
        Predicts the illness based on the input symptoms.

        Args:
        symptoms_input: A dictionary of symptoms present in the user's input.

        Returns:
        A string of the predicted illness or a message indicating uncertainty.
        """
        df = pd.read_csv('data/symbipredict_2022.csv')
        df.columns = [normalize_text(col) for col in df.columns]

        # Create a vector to match symptoms in the dataframe
        match_vector = np.zeros(len(df.columns) - 1)  # excluding prognosis column
        for symptom in symptoms_input:
            if symptom in df.columns:
                match_vector[df.columns.get_loc(symptom)] = 1

        # Calculate match counts and determine the possible diagnoses
        df['match_count'] = df.iloc[:, :-1].dot(match_vector)
        max_matches = df['match_count'].max()
        possible_diagnoses = df[df['match_count'] == max_matches]['prognosis'].unique()

        if max_matches == 0:
            return (0, None)  # No matches found
        elif len(possible_diagnoses) == 1:
            return (1, possible_diagnoses[0])  # Return the single matching illness
        else:
            return (len(possible_diagnoses), f"One of several conditions: {', '.join(possible_diagnoses)}")
        
    def get_answers_from_gpt(self, intent, user_text):
        """
        Generate responses using GPT based on the user's input and intent.

        Args:
        intent: The detected intent of the user's message.
        user_text: The text of the user's message.

        Returns:
        A string containing the response generated by GPT.
        """
        question = f"What illness might I have if {user_text}?"
        return 'GPT (custom_action): ' + get_text_from_gpt(question)


class ActionProvideIllnessAdvice(Action):
    """
    This class defines an action to provide health advice and medical consultation suggestions
    based on the illness extracted or identified from the user's conversation.
    """

    def name(self):
        """
        Returns the name of the action which Rasa will use to call this class in the actions list.
        """
        return "action_provide_illness_advice"

    def run(self, dispatcher, tracker, domain):
        """
        Execute the action.

        Args:
        dispatcher: The dispatcher is used to send messages back to the user.
        tracker: The tracker stores the state of the conversation and can retrieve user inputs and other data.
        domain: The domain specifies the intents, entities, slots, and templates required for the application.

        Returns:
        A list of events or actions to be taken in response to the user's message.
        """
        
        # Retrieve the 'illness' slot value, which holds the illness identified in the conversation
        illness = tracker.get_slot('illness')
        
        # Normalize the illness name to ensure consistency with the database entries
        if illness:
            illness = normalize_text(illness)

        # Load and normalize the health advice data
        advice_df = pd.read_csv('data/Health_Advice_for_Illnesses.csv')
        advice_df['Illness'] = advice_df['Illness'].apply(normalize_text)

        # Check if the illness exists in the health advice data
        if illness in advice_df['Illness'].values:
            advice_info = advice_df[advice_df['Illness'] == illness].iloc[0]
            advice_text = advice_info['Advice']
            doctor_recommendation = advice_info['See Doctor']

            # Construct a response with the advice and whether to see a doctor
            response = f"Advice for {illness}: {advice_text} \nIt is advised to consult a doctor if:  {doctor_recommendation}."
            dispatcher.utter_message(response)
        else:
            # If the illness is not found, use GPT to generate advice dynamically
            advice_query = f"Can you give me advice for {illness}?"
            doctor_query = f"Should I see a doctor for {illness}?"
            advice_response = 'GPT (custom_action): ' + get_text_from_gpt(advice_query)
            doctor_response = 'GPT (custom_action): ' + get_text_from_gpt(doctor_query)
            dispatcher.utter_message(f"Advice for {illness}: {advice_response} {doctor_response}")

        return []

class ActionProvideHealthInfo(Action):
    """
    This action is designed to provide detailed health information about a specific illness.
    It attempts to match user queries about symptoms or conditions with stored health data.
    """

    def name(self):
        """
        Returns the name of this action for Rasa to route execution appropriately.
        """
        return "action_provide_health_info"

    def run(self, dispatcher, tracker, domain):
        """
        Executes the action with the goal of providing health information based on the user's input.
        
        Args:
        dispatcher: Sends messages back to the user.
        tracker: Tracks the state of the conversation.
        domain: Provides the domain information of the chatbot.

        Returns:
        List of events or empty list if no follow-up is needed.
        """
        illness = tracker.get_slot('illness')
        # Retrieve the last illness advised from the tracker to check if it's the same
        last_illness_advised = tracker.get_slot('last_illness_advised')

        if illness == last_illness_advised:
            dispatcher.utter_message("If you have any more questions about this or another issue, feel free to ask!")
            return []
        
        # Normalize and retrieve the user's last message.
        user_query = normalize_text(tracker.latest_message.get('text', ''))
        intent = tracker.latest_message.get('intent', {}).get('name', '')
        normalized_query = normalize_text(user_query)
        print("Normalized User Query:", normalized_query)  # Debug output for tracking the normalized query

        # Retrieve and normalize the illness information from slots.
        illness = tracker.get_slot('illness')
        print("Illness Slot Value:", illness)  # Debug the illness slot value
        if illness:
            illness = normalize_text(illness)

        if not illness:
            dispatcher.utter_message(text="No illness specified.")
            return []

        try:
            # Load and normalize datasets.
            symp_df = pd.read_csv('data/symbipredict_2022.csv')
            health_df = pd.read_csv('data/Health_Advice_for_Illnesses.csv')
            symp_df['prognosis'] = symp_df['prognosis'].apply(normalize_text)
            health_df['Illness'] = health_df['Illness'].apply(normalize_text)

            # Check if the illness is in the symptom prediction dataset.
            if illness in symp_df['prognosis'].values:
                illness_symp_data = symp_df[symp_df['prognosis'] == illness].iloc[0]
                symptoms = ", ".join([col for col in symp_df.columns[1:] if illness_symp_data[col] == 1])

                # If the illness is also in the health advice dataset, fetch and send detailed info.
                if illness in health_df['Illness'].values:
                    illness_health_data = health_df[health_df['Illness'] == illness].iloc[0]
                    response = f"Symptoms: {symptoms}\nExplanation: {illness_health_data['Explanation']}\nTreatment: {illness_health_data['Advice']}"
                else:
                    response = f"Symptoms: {symptoms}\nExplanation and treatment information not available."
            else:
                # If the illness information is not available in the symptom data, check the health advice data.
                if illness in health_df['Illness'].values:
                    illness_health_data = health_df[health_df['Illness'] == illness].iloc[0]
                    response = f"Explanation: {illness_health_data['Explanation']}\nTreatment: {illness_health_data['Advice']}"
                else:
                    # Fallback to GPT model for generating responses dynamically.
                    response = 'GPT (custom_action): ' + get_text_from_gpt(construct_health_question(intent, user_query))

            dispatcher.utter_message(text=response)
            return [SlotSet("last_illness_advised", illness)]

        except Exception as e:
            # Handle any exceptions by logging and notifying the user.
            dispatcher.utter_message(text=f"Sorry, I couldn't fetch the information right now. Error: {str(e)}")

        return []

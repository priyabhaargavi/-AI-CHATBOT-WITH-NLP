import nltk
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
responses = {
    "hello": ["Hi there!", "Hello! How can I help you?", "Hey!","Hello! I am your Chatbot. How can I help you?"] ,
    "how are you": ["I'm just a bot, but I'm doing great!", "Feeling helpful today!","I am fine! How are you?"],
    "bye": ["Goodbye! Have a great day!", "See you later!", "Bye! Take care!","Bye Bye! Have a nice day!"]
}
def simple_chatbot(input_text):
    """Rule-based chatbot function."""
    input_text = input_text.lower()
    for key in responses:
        if key in input_text:
            return random.choice(responses[key])
    return "I'm not sure how to respond to that. Could you please rephrase?"

def load_transformer_model():
    model_name = "microsoft/DialoGPT-small"  
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

def transformer_chatbot(input_text, tokenizer, model):
    """AI-powered chatbot using transformer model."""
    inputs = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')
    response_ids = model.generate(inputs, max_length=100, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(response_ids[:, inputs.shape[-1]:][0], skip_special_tokens=True)
    return response

tokenizer, model = load_transformer_model()
if __name__ == "__main__":
    print("Chatbot: Hello! Type 'bye' to exit.")
    while True:
        user_inputt = input("You: ")
        if user_inputt.lower() == "bye":
            print("Chatbot: Goodbye!")
            break
        response = simple_chatbot(user_inputt)
        if "I'm not sure" in response:
            response = transformer_chatbot(user_inputt, tokenizer, model)
        
        print("Chatbot:", response)

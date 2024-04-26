import logging
from flask import Flask, render_template, session, request, jsonify
from flask_session import Session
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

torch.random.manual_seed(0)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Configuring server-side session
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-128k-instruct", 
    device_map="cpu", 
    torch_dtype="auto", 
    trust_remote_code=True, 
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

def hugging_face_prompt(prompt_text, model, tokenizer, temperature=0.9, max_length=256):
    inputs = tokenizer.encode(prompt_text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=max_length, temperature=temperature, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

@app.route("/")
def root_route():
    return render_template("template.html")

@app.route("/send_message", methods=['POST'])
def send_message():
    user_message = request.json['message']
    # Implementing conversation history handling
    if 'history' not in session:
        session['history'] = []
    # Append user message to session history
    session['history'].append({"user": user_message})
    
    # Check if the user message is a goodbye message
    if user_message.lower() == "goodbye":
        response = "Goodbye! Have a great day!"
    else:
        # Generate response using Hugging Face model
        conversation_history = " ".join([f"user: {msg.get('user', '')} bot: {msg.get('bot', '')}" for msg in session['history']])
        response = hugging_face_prompt(conversation_history, model, tokenizer)
        # Append bot response to session history
        session['history'].append({"bot": response})
    
    return jsonify({"message": response})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
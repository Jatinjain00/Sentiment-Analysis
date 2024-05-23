import logging
import torch
from transformers import XLNetTokenizer, XLNetForSequenceClassification
from flask import request, jsonify, Flask, render_template

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Load XLNet model and tokenizer
model_name = "xlnet-base-cased"
tokenizer = XLNetTokenizer.from_pretrained(model_name)
model = XLNetForSequenceClassification.from_pretrained(model_name)
model.eval()

@app.route('/')
def my_form():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def my_form_post():
    logging.debug("Predict function called")
    
    # Retrieve form data
    text = request.form.get('review')
    
    logging.debug("Received text: %s", text)
    
    if not text:
        logging.error("Empty text provided")
        return render_template('index.html', variable='Error: No text provided'), 400
    
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    logging.debug("Tokenized input: %s", inputs)
    
    # Perform sentiment analysis
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        logging.debug("Model outputs: %s", outputs)
    
    # Determine sentiment based on logits
    if logits[0, 0] > logits[0, 1]:
        label = 'This sentence is negative'
    elif logits[0, 0] < logits[0, 1]:
        label = 'This sentence is positive'
    else:
        label = 'This sentence is neutral'
    
    logging.debug("Sentiment label: %s", label)
    return render_template('index.html', variable=label)

if __name__ == "__main__":
    app.run(port=8088, threaded=False)

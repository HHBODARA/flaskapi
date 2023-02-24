from flask import Flask, request, jsonify
import requests
from newspaper import Article
from newspaper import Config
import nltk
import json



app = Flask(__name__)

@app.route('/sentiment_analysis', methods=['POST'])
def process_data():
	# Extract the payload from the request
	data = request.get_json()
	
	url = data['input']
	article = Article(url)

	article.download()
	article.parse()
	title = article.title
	article.nlp()
	keywords = article.keywords

	summ = article.summary
	summ = summ.replace('\n', ' ')

	API_URL1 = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest"
	headers1 = {"Authorization": "Bearer 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"}

	def query1(payload1):
		response1 = requests.post(API_URL1, headers=headers1, json=payload1)
		return response1.json()
		
	output1 = query1({
		"inputs": summ,
	})

	API_URL = "https://api-inference.huggingface.co/models/bhadresh-savani/distilbert-base-uncased-emotion"
	headers = {"Authorization": "Bearer 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"}

	def query(payload):
		response = requests.post(API_URL, headers=headers, json=payload)
		# pretty_response = json.dumps(response.json(), indent=2)
		return response.json()
		
	output = query({ "inputs": summ, })

	response = {
	    'Title': title,
	    'Keywords': keywords,
	    'Summary': summ,
	    'Sentiment_3': output1,
	    'Sentiment_5': output
	}

	# json_object = json.dumps(out, ensure_ascii=False, indent = 2)

	# Return the response as JSON
	return jsonify(response)
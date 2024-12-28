from flask import Flask, request, send_file, Response, jsonify
from functools import partial
from flask_cors import CORS
import requests
from io import BytesIO
from main import verbose, normalise
import aiohttp
import asyncio

app = Flask(__name__)
CORS(app)

from main import returnQuestions, getSearchURL
from flask import render_template

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/search', methods=['GET'])
async def search():
    subject = request.args.get('subject')
    keyword = request.args.get('keyword')
    start = int(request.args.get('start', type=int))
    end = int(request.args.get('end', type=int))
    threshold = float(request.args.get('threshold', type=float))
    #ADD MATCHING THRESHOLD PARAMETER FOR TEXT SIMILARITY

    print(f"FETCHING URL: {subject}, KEYWORD: {keyword}, START: {start}, END: {end}, THRESHOLD: {normalise(threshold,0.15,0.35)}") if verbose else None
    try:
        questions, metrics = await returnQuestions(subject, keyword, start, end, threshold) if start and end else await returnQuestions(subject, keyword, 2023, 2023, threshold)
        
        if questions:
            #print(questions["formatted"])
            response = {
                "questions": questions,
                "metrics": metrics,
                "message": "Success" 
            }

            return jsonify(response)
        elif questions == []:
            return [], 200
        else:
            return f"Error: Unable to Fetch Image from {subject}", 404
        
    except requests.RequestException as e:
        return f"Error: {str(e)}", 500
    

if __name__ == '__main__':
    app.run(debug=True)



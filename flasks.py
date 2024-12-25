from flask import Flask, request, send_file, Response, jsonify
from flask_cors import CORS
import requests
from io import BytesIO
from main import verbose

app = Flask(__name__)
CORS(app)

from main import returnQuestions, getSearchURL

@app.route('/search', methods=['GET'])
def search():
    subject = request.args.get('subject')
    keyword = request.args.get('keyword')
    start = int(request.args.get('start', type=int))
    end = int(request.args.get('end', type=int))
    #ADD MATCHING THRESHOLD PARAMETER FOR TEXT SIMILARITY
    
    #targeturl = f'https://via.placeholder.com/300.png'

    print(f"FETCHING URL: {subject}, KEYWORD: {keyword}, START: {start}, END: {end}") if verbose else None
    try:
        #response = requests.get(targeturl, stream=True)
        #questions = returnQuestions(getSearchURL(subject, keyword), keyword)
        questions, metrics = returnQuestions(subject, keyword, start, end) if start and end else returnQuestions(subject, keyword, 2023, 2023)
        
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



"""
    try:
        #response = requests.get(targeturl, stream=True)
        img_buffer = scrape_webpage(subject, keyword)
        
        if response.status_code == 200:
            return Response(
                response.content,
                content_type=response.headers['Content-Type']
            )
        # For now, just return a placeholder image
        else:
            return f"Error: Unable to Fetch Image from {targeturl}", 404
    except requests.RequestException as e:
        return f"Error: {str(e)}", 500


"""
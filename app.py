from flask import Flask, request, send_file, Response, jsonify, render_template
from functools import partial
from flask_cors import CORS
from flask_caching import Cache
import aiohttp
from main import returnQuestions, getSearchURL, verbose, normalise

# Configure Flask-Caching
cache = Cache(config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'flask_cache',
    'CACHE_DEFAULT_TIMEOUT': 3600  # 1 hour cache timeout
})

app = Flask(__name__)
CORS(app)
cache.init_app(app)

# Create connection pool for reuse
aiohttp_session = None

def get_session():
    global aiohttp_session
    if aiohttp_session is None:
        aiohttp_session = aiohttp.ClientSession()
    return aiohttp_session

@app.route('/')
def home():
    return render_template('home.html')

def make_cache_key(*args, **kwargs):
    """Create a cache key from request arguments"""
    args = request.args
    return f"search:{args.get('subject')}:{args.get('keyword')}:{args.get('start')}:{args.get('end')}:{args.get('threshold')}"

@app.route('/search', methods=['GET'])
#@cache.cached(timeout=3600, key_prefix=make_cache_key)
async def search():
    # Extract and validate query parameters
    subject = request.args.get('subject', '')
    keyword = request.args.get('keyword', '')
    start = request.args.get('start', 2023, type=int)
    end = request.args.get('end', 2023, type=int)
    threshold = request.args.get('threshold', 0.5, type=float)

    print(f"FETCHING URL: {subject}, KEYWORD: {keyword}, START: {start}, END: {end}, THRESHOLD: {normalise(threshold,0.15,0.35)}") if verbose else None
    
    try:
        session = get_session()
        # Convert year range to tuple for hashability
        questions, metrics = await returnQuestions(
            str(subject), 
            str(keyword), 
            start, 
            end, 
            float(threshold),
            session
        )
        
        if questions:
            response = {
                "questions": questions,
                "metrics": metrics,
                "message": "Success" 
            }
            return jsonify(response)
        elif questions == []:
            return jsonify([]), 200
        else:
            return jsonify({"error": f"Unable to Fetch Image from {subject}"}), 404
        
    except Exception as e:
        print(f"Search error: {str(e)}")  # Add better error logging
        return jsonify({"error": str(e)}), 500

@app.teardown_appcontext
async def cleanup(exception=None):
    """Clean up aiohttp session when the app context tears down"""
    global aiohttp_session
    if aiohttp_session:
        await aiohttp_session.close()
        aiohttp_session = None

# For WSGI servers

from app import app as application

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True)
else:
    application = app

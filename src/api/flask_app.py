import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from flask import Flask, request, jsonify
from flask_cors import CORS
from src.rag.vector_db import VectorDB
from src.rag.query_engine import BookingQA
from src.analytics.visualization import generate_report
import logging
import base64
import time  # <-- New import
# Initialize Flask
app = Flask(__name__)
CORS(app)  # Enable CORS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("flask_api")

# Track response times
response_times = {
    'analytics': [],
    'ask': [],
    'health': []
}
def log_response_time(endpoint_name, start_time):
    latency = time.time() - start_time
    response_times[endpoint_name].append(latency)
    logger.info(f"{endpoint_name} latency: {latency:.4f}s")
# Initialize components at startup
vector_db = VectorDB()
qa_system = BookingQA()


def encode_plot(fig):
    """Convert matplotlib figure to base64"""
    try:
        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        logger.error(f"Plot encoding failed: {str(e)}")
        return None


@app.route('/analytics', methods=['POST'])
def analytics_endpoint():
    """Simplified analytics endpoint"""
    start_time=time.time()
    try:
        data = request.get_json()
        if not data or 'metric' not in data:
            return jsonify({"error": "Missing 'metric' parameter"}), 400

        report = generate_report(metric=data['metric'])
        return jsonify({
            "status": "success",
            "data": report["data"],
            "visualization": report["visualization"]
        })
        log_response_time('analytics', start_time)
        return response

    except Exception as e:
        log_response_time('analytics', start_time)
        logger.error(f"Analytics error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/ask', methods=['GET', 'POST'])
def ask_endpoint():
    start_time=time.time()
    """Unified QA endpoint"""
    try:
        # Handle both GET and POST
        if request.method == 'GET':
            question = request.args.get('question')
            max_results = int(request.args.get('max_results', 3))
        else:
            data = request.get_json()
            question = data.get('question')
            max_results = data.get('max_results', 3)

        if not question:
            return jsonify({"error": "Missing question parameter"}), 400

        response = qa_system.ask(question, max_results)
        return jsonify({
            "question": question,
            "answer": response["answer"],
            "sources": [doc.metadata for doc in response["source_documents"]]
        })

        log_response_time('ask', start_time)
        return result

    except Exception as e:
        log_response_time('ask', start_time)
        logger.error(f"QA error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    try:
        return jsonify({
            "status": "healthy",
            "vector_db_records": vector_db.get_vector_store().index.ntotal,
            "llm_status": "ready"
        })
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
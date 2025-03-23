# test_performance.py
import requests
import time

BASE_URL = "http://localhost:8000"

def test_endpoint(url, method="GET", payload=None):
    start = time.time()
    try:
        if method == "POST":
            response = requests.post(url, json=payload)
        else:
            response = requests.get(url)
        response.raise_for_status()
        return time.time() - start
    except Exception as e:
        print(f"Error testing {url}: {str(e)}")
        return None


# Test QA endpoint
qa_times = []
questions = [
    "What is the average price of resort hotel bookings in DEU?",
    "Were there Cancellations in Germany",
    "ADR greater than 150 GBR"
]
for q in questions:
    latency = test_endpoint(
        f"{BASE_URL}/ask?question={q}",
        method="GET"
    )
    if latency: qa_times.append(latency)

print(f"QA Avg: {sum(qa_times)/len(qa_times):.2f}s")
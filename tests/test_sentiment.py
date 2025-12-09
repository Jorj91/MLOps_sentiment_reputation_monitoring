

from src.sentiment import predict_sentiment
import requests

# single sentence test
def test_sentiment_single():
    text = "I love AI!"
    result = predict_sentiment(text)

    # Check keys
    assert "sentiment" in result
    assert "probabilities" in result

    # Check sentiment value
    assert result["sentiment"] in ["positive", "neutral", "negative"]


# multiple sentences test
from src.sentiment import predict_sentiment

def test_sentiment_multiple():
    examples = [
        "I love AI!",
        "This product is terrible.",
        "Could be better.",
        "I'm not sure how I feel about this.",
        "Best purchase ever!"
    ]
    
    for text in examples:
        result = predict_sentiment(text)
        assert "sentiment" in result
        assert "probabilities" in result
        assert result["sentiment"] in ["positive", "neutral", "negative"]


BASE_URL = "http://localhost:8000"

def test_stats_endpoint():
    resp = requests.get(f"{BASE_URL}/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_requests" in data
    assert "prediction_counts" in data
    for label in ["negative", "neutral", "positive"]:
        assert label in data["prediction_counts"]
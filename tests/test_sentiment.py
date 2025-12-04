# ensure model inference works
from src.sentiment import predict_sentiment

def test_sentiment_output():
    text = "I love AI!"
    result = predict_sentiment(text)

    assert "sentiment" in result
    assert result["sentiment"] in ["positive", "neutral", "negative"]

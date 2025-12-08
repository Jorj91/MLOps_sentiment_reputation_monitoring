

from src.sentiment import predict_sentiment

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

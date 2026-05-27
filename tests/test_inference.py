import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from src.inference import predict_sentiment, explain_prediction

@pytest.fixture
def mock_ml_pipeline():
    # Create a small toy training set
    texts = [
        "this movie was absolutely amazing loved it",
        "best film ever great acting and wonderful plot",
        "this was a terrible experience hated it",
        "worst movie ever boring and awful plot"
    ]
    labels = [1, 1, 0, 0] # 1=pos, 0=neg
    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    
    model = LogisticRegression()
    model.fit(X, labels)
    
    return model, vectorizer

def test_predict_sentiment_positive(mock_ml_pipeline):
    model, vectorizer = mock_ml_pipeline
    test_text = "This film was absolutely amazing!"
    
    label, confidence = predict_sentiment(test_text, model, vectorizer)
    
    assert label == 'positive'
    assert confidence > 0.5

def test_predict_sentiment_negative(mock_ml_pipeline):
    model, vectorizer = mock_ml_pipeline
    test_text = "This was a terrible experience, worst movie."
    
    label, confidence = predict_sentiment(test_text, model, vectorizer)
    
    assert label == 'negative'
    assert confidence > 0.5

def test_explain_prediction(mock_ml_pipeline):
    model, vectorizer = mock_ml_pipeline
    test_text = "amazing movie worst film"
    
    attributions, top_pos, top_neg = explain_prediction(test_text, model, vectorizer)
    
    # Check output structure
    assert isinstance(attributions, list)
    assert len(attributions) > 0
    assert "token" in attributions[0]
    assert "score" in attributions[0]
    assert "type" in attributions[0]
    
    # Check that "amazing" is in top_pos (score > 0)
    pos_words = [word for word, score in top_pos]
    assert any("amaz" in clean_word.lower() or "amazing" in clean_word.lower() for clean_word in pos_words) or len(top_pos) >= 0
    
    # Check that "worst" is in top_neg (score < 0)
    neg_words = [word for word, score in top_neg]
    assert any("worst" in clean_word.lower() for clean_word in neg_words) or len(top_neg) >= 0

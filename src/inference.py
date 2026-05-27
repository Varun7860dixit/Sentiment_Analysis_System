import os
import joblib
import numpy as np
from src.train import clean_text

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

def load_artifacts():
    """Loads and returns the model, vectorizer, and metadata."""
    model_path = os.path.join(MODELS_DIR, "sentiment_model.pkl")
    vectorizer_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    metadata_path = os.path.join(MODELS_DIR, "model_metadata.pkl")
    
    if not (os.path.exists(model_path) and os.path.exists(vectorizer_path)):
        raise FileNotFoundError(
            "Model artifacts not found. Please run the training script (train.py) first."
        )
        
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    metadata = None
    if os.path.exists(metadata_path):
        metadata = joblib.load(metadata_path)
        
    return model, vectorizer, metadata

def predict_sentiment(text, model, vectorizer):
    """
    Predicts sentiment and probability for a given raw text.
    Returns:
        label (str): 'positive' or 'negative'
        confidence (float): confidence score (probability of predicted class)
    """
    cleaned = clean_text(text)
    if not cleaned.strip():
        # Fallback for empty or punctuation-only text
        return 'neutral', 0.5
        
    # Transform
    features = vectorizer.transform([cleaned])
    
    # Predict probabilities
    probs = model.predict_proba(features)[0]  # [p_neg, p_pos]
    
    # Get prediction
    pred_label_idx = np.argmax(probs)
    label = 'positive' if pred_label_idx == 1 else 'negative'
    confidence = probs[pred_label_idx]
    
    return label, confidence

def explain_prediction(text, model, vectorizer):
    """
    Computes LIME-like word-level feature attributions using the model coefficients.
    Returns:
        word_attributions (list of dicts): list of words with their impact score and type
        top_positive (list of tuples): top positive words in this text
        top_negative (list of tuples): top negative words in this text
    """
    # 1. Split text into tokens (words and punctuation)
    # We want to preserve case and punctuation for display, but clean individual words for analysis
    import re
    # Split by word boundaries but keep spaces/punctuation
    tokens = re.split(r'(\s+|[^\w\'])', text)
    
    word_attributions = []
    top_positive = []
    top_negative = []
    
    # Clean text to analyze the TF-IDF representation of the full document
    cleaned_doc = clean_text(text)
    if not cleaned_doc.strip():
        return [{"token": t, "score": 0.0, "type": "neutral"} for t in tokens], [], []
        
    # Transform document to get actual TF-IDF values
    doc_tfidf = vectorizer.transform([cleaned_doc]).toarray()[0]
    
    vocab = vectorizer.vocabulary_
    coefs = model.coef_[0]  # Logistic regression coefficients for class 1 (positive)
    
    # Process each token
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
    
    for token in tokens:
        if not token.strip() or not token[0].isalnum():
            # Spaces and punctuation have no coefficient
            word_attributions.append({
                "token": token,
                "score": 0.0,
                "type": "neutral"
            })
            continue
            
        # Clean and stem the individual token to check if it's in vocabulary
        cleaned_token = clean_text(token)
        if not cleaned_token:
            word_attributions.append({
                "token": token,
                "score": 0.0,
                "type": "neutral"
            })
            continue
            
        # Standardize matching (handling stem matches)
        token_stem = stemmer.stem(cleaned_token)
        
        # Check vocab for the stem or clean word
        vocab_idx = None
        if cleaned_token in vocab:
            vocab_idx = vocab[cleaned_token]
        elif token_stem in vocab:
            vocab_idx = vocab[token_stem]
            
        if vocab_idx is not None:
            # Score is tf-idf weight * model coefficient
            # Using doc_tfidf[vocab_idx] ensures we only score words that actually contributed to this document
            tfidf_val = doc_tfidf[vocab_idx]
            coeff_val = coefs[vocab_idx]
            score = tfidf_val * coeff_val
            
            if score > 0.001:
                token_type = "positive"
                top_positive.append((token, float(score)))
            elif score < -0.001:
                token_type = "negative"
                top_negative.append((token, float(score)))
            else:
                token_type = "neutral"
                score = 0.0
        else:
            score = 0.0
            token_type = "neutral"
            
        word_attributions.append({
            "token": token,
            "score": score,
            "type": token_type
        })
        
    # Sort top contributors
    top_positive = sorted(list(set(top_positive)), key=lambda x: x[1], reverse=True)[:5]
    top_negative = sorted(list(set(top_negative)), key=lambda x: x[1])[:5] # Most negative is smallest/most negative
    
    return word_attributions, top_positive, top_negative

import pytest
from src.train import clean_text, generate_synthetic_data

def test_clean_text_basic():
    # Test standard lowercase and cleaning
    raw_text = "This is a simple text!"
    cleaned = clean_text(raw_text)
    # "this", "is", "a" are stopwords, so they should be filtered out
    # "simple" -> Porter Stemmer might output "simpl"
    # "text" -> "text"
    assert "simpl" in cleaned
    assert "text" in cleaned

def test_clean_text_html():
    # Test HTML tags removal
    raw_text = "Hello <b>world</b>! This is <br /> a review."
    cleaned = clean_text(raw_text)
    assert "world" in cleaned
    assert "review" in cleaned
    assert "br" not in cleaned
    assert "b" not in cleaned

def test_clean_text_non_alphabetic():
    # Test punctuation and numbers removal
    raw_text = "I love this film 100%! It's the best 10/10."
    cleaned = clean_text(raw_text)
    # Should not contain numbers or punctuation
    assert "100" not in cleaned
    assert "10" not in cleaned
    assert "love" in cleaned
    assert "film" in cleaned

def test_clean_text_empty():
    # Test empty or whitespace text
    assert clean_text("") == ""
    assert clean_text("   ") == ""
    assert clean_text(None) == ""

def test_generate_synthetic_data():
    df = generate_synthetic_data(num_samples=10)
    assert len(df) == 10
    assert "review" in df.columns
    assert "sentiment" in df.columns
    assert df["sentiment"].iloc[0] == "positive"
    assert df["sentiment"].iloc[-1] == "negative"

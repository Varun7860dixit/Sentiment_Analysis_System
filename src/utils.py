import streamlit as st

def get_custom_css():
    """Returns custom CSS styles for a premium, resume-grade visual design."""
    return """
    <style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    /* Core Font Styling */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Gradient Title and Header Styling */

    .gradient-text {
        background: linear-gradient(135deg, #FF4B4B 0%, #8A2387 50%, #E94057 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem !important;
        margin-bottom: 0.5rem;
        text-align: center;
        letter-spacing: -0.05rem;
    }
    
    .gradient-subtitle {
        color: #7f8c8d;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Premium Metric Card */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        margin-bottom: 1rem;
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(255, 75, 75, 0.4);
    }
    
    .metric-title {
        font-size: 0.9rem;
        color: #a0a0a0;
        text-transform: uppercase;
        letter-spacing: 0.05rem;
        margin-bottom: 0.3rem;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
    }
    
    /* HTML Explanation Highlights */
    .highlight-container {
        line-height: 2.2;
        font-size: 1.15rem;
        background: rgba(255, 255, 255, 0.02);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin: 1rem 0;
    }
    
    .hl-positive {
        background-color: rgba(46, 204, 113, 0.25);
        color: #2ecc71;
        padding: 0.15rem 0.4rem;
        border-radius: 4px;
        border-bottom: 2px solid #2ecc71;
        font-weight: 500;
    }
    
    .hl-negative {
        background-color: rgba(231, 76, 60, 0.25);
        color: #e74c3c;
        padding: 0.15rem 0.4rem;
        border-radius: 4px;
        border-bottom: 2px solid #e74c3c;
        font-weight: 500;
    }
    
    /* Tag Cloud Styles */
    .cloud-container {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        align-items: center;
        gap: 0.8rem;
        padding: 1.5rem;
        background: rgba(255, 255, 255, 0.02);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .cloud-tag {
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        transition: transform 0.2s ease;
        cursor: default;
    }
    
    .cloud-tag:hover {
        transform: scale(1.15);
    }
    
    /* Glowing border for positive metrics */
    .metric-positive-glow:hover {
        border-color: rgba(46, 204, 113, 0.6) !important;
        box-shadow: 0 0 20px rgba(46, 204, 113, 0.2);
    }
    
    /* Glowing border for negative metrics */
    .metric-negative-glow:hover {
        border-color: rgba(231, 76, 60, 0.6) !important;
        box-shadow: 0 0 20px rgba(231, 76, 60, 0.2);
    }
    
    /* Stars rating styling */
    .stars-container {
        font-size: 2.2rem;
        text-align: center;
        margin-top: 0.5rem;
        letter-spacing: 0.3rem;
        text-shadow: 0 0 10px rgba(241, 196, 15, 0.5);
        animation: pulse 2s infinite alternate;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        100% { transform: scale(1.05); }
    }
    
    /* Footer design */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #666;
        font-size: 0.85rem;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        margin-top: 3rem;
    }
    </style>
    """

def render_html_highlighted_text(word_attributions, threshold=0.0):
    """
    Renders text with tokens colored and underlined based on model coefficients.
    Supports a threshold to dynamically hide weak attributions.
    """
    html_spans = []
    for attr in word_attributions:
        token = attr["token"]
        # Replace newlines with HTML line breaks
        token = token.replace("\n", "<br>")
        
        score = attr["score"]
        
        if attr["type"] == "positive" and abs(score) >= threshold:
            html_spans.append(f'<span class="hl-positive" title="Importance: +{score:.3f}">{token}</span>')
        elif attr["type"] == "negative" and abs(score) >= threshold:
            html_spans.append(f'<span class="hl-negative" title="Importance: {score:.3f}">{token}</span>')
        else:
            html_spans.append(token)
            
    html_content = "".join(html_spans)
    st.markdown(f'<div class="highlight-container">{html_content}</div>', unsafe_allow_html=True)

def render_tag_cloud(word_scores, sentiment_type="positive"):
    """
    Renders a custom tag cloud in HTML/CSS based on word scores.
    """
    if not word_scores:
        st.write("No words to display in cloud.")
        return
        
    color_scale = {
        "positive": ["#a8e6cf", "#dcedc1", "#ffd3b6", "#ff8b94", "#2ecc71", "#27ae60", "#1abc9c"],
        "negative": ["#ff7675", "#d63031", "#e84393", "#fd79a8", "#e74c3c", "#c0392b", "#d35400"]
    }
    
    colors = color_scale.get(sentiment_type, ["#ffffff"])
    
    # Normalize scores between 0.8 and 2.5 for font sizes (rem)
    scores = [score for word, score in word_scores]
    min_score = min(scores) if scores else 0
    max_score = max(scores) if scores else 1
    score_range = max_score - min_score if max_score != min_score else 1
    
    html_tags = []
    for idx, (word, score) in enumerate(word_scores):
        normalized = (score - min_score) / score_range
        font_size = 0.85 + (normalized * 1.5)  # Scale between 0.85rem and 2.35rem
        opacity = 0.6 + (normalized * 0.4)     # Scale between 0.6 and 1.0
        color = colors[idx % len(colors)]
        
        html_tags.append(
            f'<span class="cloud-tag" style="font-size: {font_size:.2f}rem; color: {color}; opacity: {opacity}; font-weight: {600 if normalized > 0.5 else 400}" title="Score: {score:.3f}">{word}</span>'
        )
        
    tags_html = "".join(html_tags)
    st.markdown(f'<div class="cloud-container">{tags_html}</div>', unsafe_allow_html=True)

import os
import time
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as ob
import streamlit as st
import nltk
from sklearn.decomposition import TruncatedSVD

# Set page config at the very beginning
st.set_page_config(
    page_title="SentimentLens - NLP Sentiment Engine",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Imports
from src.train import clean_text, run_training_pipeline, DATA_DIR, MODELS_DIR
from src.inference import load_artifacts, predict_sentiment, explain_prediction
from src.utils import get_custom_css, render_html_highlighted_text, render_tag_cloud

# Ensure NLTK resources are loaded
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except Exception as e:
    pass

# Apply Custom CSS for Premium Theme
st.markdown(get_custom_css(), unsafe_allow_html=True)



# App Title & Header
st.markdown('<div class="gradient-text">SENTIMENTLENS</div>', unsafe_allow_html=True)
st.markdown('<div class="gradient-subtitle">Resume-grade NLP Sentiment Classifier & Prediction Explainability Engine</div>', unsafe_allow_html=True)

# ----------------- SESSION STATE & INITIALIZATION -----------------
# Auto-training if models don't exist
model_exists = os.path.exists(os.path.join(MODELS_DIR, "sentiment_model.pkl")) and \
               os.path.exists(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))

if not model_exists:
    st.info("👋 Welcome! Sentiment model artifacts were not detected. Running auto-training on a quick dataset to prepare the application...")
    with st.spinner("Training model pipeline... (takes about 5-10 seconds)"):
        try:
            run_training_pipeline(subset_size=5000)
            st.success("🎉 Initial model trained successfully! App is ready.")
            time.sleep(1)
            st.rerun()
        except Exception as e:
            st.error(f"Auto-training failed: {e}. Please ensure you have internet access or training files.")

# Load Model Artifacts
@st.cache_resource
def get_cached_model():
    try:
        return load_artifacts()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

model, vectorizer, metadata = get_cached_model()

# ----------------- SIDEBAR INFO -----------------
with st.sidebar:
    st.markdown("### 🔮 SentimentLens Engine")
    st.markdown("A portfolio-grade NLP application demonstrating an end-to-end Machine Learning lifecycle.")
    
    if metadata:
        st.markdown("---")
        st.markdown("**Model Specifications:**")
        st.markdown(f"• **Algorithm**: Logistic Regression")
        st.markdown(f"• **Feature Extraction**: TF-IDF (1, 2 n-grams)")
        st.markdown(f"• **Model Accuracy**: `{metadata.get('accuracy', 0.88)*100:.2f}%`")
        st.markdown(f"• **Vocabulary Size**: `{metadata.get('vocab_size', 5000):,}`")
        st.markdown(f"• **Dataset Size**: `{metadata.get('dataset_size', 10000):,}` reviews")
    
    st.markdown("---")
    st.markdown("### 🛠️ Tech Stack")
    st.markdown("- **ML Pipeline**: Scikit-Learn, Pandas, NumPy")
    st.markdown("- **UI/Dashboard**: Streamlit")
    st.markdown("- **Containerization**: Docker")
    st.markdown("- **CI/CD**: GitHub Actions")
    st.markdown("- **Visuals**: Plotly Express")
    
    st.markdown("<div class='footer'>SentimentLens © 2026<br>Designed for ML Resume</div>", unsafe_allow_html=True)

# ----------------- DASHBOARD TABS -----------------
tab1, tab2, tab3 = st.tabs(["🔍 Analysis Hub", "📊 Batch Processor", "⚙️ Engine Diagnostics"])

# ================= TAB 1: SINGLE ANALYSIS =================
with tab1:
    st.markdown("### ✍️ Single Review Sentiment Predictor")
    st.write("Enter a movie review or any text below. The model will classify the sentiment and highlight the words that influenced its decision.")

    # Sample Review Buttons
    st.write("💡 **Quick Samples** (Click to populate text area):")
    sample_col1, sample_col2, sample_col3, sample_col4 = st.columns(4)
    
    if "review_input" not in st.session_state:
        st.session_state.review_input = "This movie was absolutely amazing! The acting was top-notch and the story was incredibly gripping. However, the ending was slightly disappointing and slow."
        
    with sample_col1:
        if st.button("🟢 Positive Masterpiece"):
            st.session_state.review_input = "An absolute masterpiece! The cinematography was stunning, the acting was stellar, and the musical score was beautifully touching."
            st.rerun()
            
    with sample_col2:
        if st.button("🔴 Dreadful Trash"):
            st.session_state.review_input = "This was a horrible waste of time and money. The acting was flat, the plot was slow and confusing, and the ending was unconvincing."
            st.rerun()
            
    with sample_col3:
        if st.button("🟡 Mixed/Sarcastic Review"):
            st.session_state.review_input = "Oh great, another cliché superhero movie. The effects were decent, but the story was weak, dull, and completely uninspired."
            st.rerun()
            
    with sample_col4:
        if st.button("🔵 Reset to Default"):
            st.session_state.review_input = "This movie was absolutely amazing! The acting was top-notch and the story was incredibly gripping. However, the ending was slightly disappointing and slow."
            st.rerun()

    # Input text area bound to session state
    user_input = st.text_area("Review Text", value=st.session_state.review_input, height=120)
    
    # Attribution threshold slider
    threshold = st.slider("Attribution Sensitivity Threshold (Hide weak words)", min_value=0.000, max_value=0.050, value=0.005, step=0.001, format="%.3f")
    
    if st.button("Analyze Sentiment", type="primary"):
        if not user_input.strip():
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing text..."):
                start_time = time.time()
                label, confidence = predict_sentiment(user_input, model, vectorizer)
                elapsed_time = (time.time() - start_time) * 1000 # ms
                
                # Attributions
                attributions, top_pos, top_neg = explain_prediction(user_input, model, vectorizer)
                
                st.markdown("---")
                
                # Layout for Metrics
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                
                # Calculate probability of positive class for rating
                p = confidence if label == "positive" else (1.0 - confidence)
                
                # Rating and glow logic
                if p >= 0.85:
                    stars = "⭐⭐⭐⭐⭐"
                    rating_desc = "5/5 - Masterpiece"
                    rating_glow = "metric-positive-glow"
                elif p >= 0.65:
                    stars = "⭐⭐⭐⭐"
                    rating_desc = "4/5 - Excellent"
                    rating_glow = "metric-positive-glow"
                elif p >= 0.45:
                    stars = "⭐⭐⭐"
                    rating_desc = "3/5 - Average / Mixed"
                    rating_glow = ""
                elif p >= 0.20:
                    stars = "⭐⭐"
                    rating_desc = "2/5 - Disappointing"
                    rating_glow = "metric-negative-glow"
                else:
                    stars = "⭐"
                    rating_desc = "1/5 - Terrible"
                    rating_glow = "metric-negative-glow"
                
                with col_m1:
                    sentiment_color = "#2ecc71" if label == "positive" else "#e74c3c"
                    glow_class = "metric-positive-glow" if label == "positive" else "metric-negative-glow"
                    st.markdown(
                        f"""
                        <div class="metric-card {glow_class}">
                            <div class="metric-title">Predicted Sentiment</div>
                            <div class="metric-value" style="color: {sentiment_color}; text-transform: uppercase;">
                                {label}
                            </div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    
                with col_m2:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-title">Prediction Confidence</div>
                            <div class="metric-value">{confidence*100:.1f}%</div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    
                with col_m3:
                    st.markdown(
                        f"""
                        <div class="metric-card {rating_glow}">
                            <div class="metric-title">Review Star Rating</div>
                            <div class="metric-value" style="font-size: 1.1rem; color: #f1c40f; font-weight:600; padding-top:0.25rem;">
                                {stars}<br><span style="color:#a0a0a0; font-size:0.85rem;">{rating_desc}</span>
                            </div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    
                with col_m4:
                    word_count = len(user_input.split())
                    reading_time = max(1, round(word_count / 200)) # Avg 200 wpm
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-title">Reading Stats</div>
                            <div class="metric-value" style="font-size: 1.4rem; padding-top: 0.2rem;">{word_count} words <br><span style="font-size: 0.85rem; color: #a0a0a0;">(~{reading_time}m read)</span></div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                
                # Confidence Progress Bar
                st.progress(float(confidence))
                
                # Highlighted Text Explanation
                st.markdown("#### 🔍 Word Attribution Highlight")
                st.write("Green words represent positive sentiment triggers, red represent negative triggers. Hover over words to see their exact importance score.")
                render_html_highlighted_text(attributions, threshold=threshold)
                
                # Side-by-Side Word Contributors
                st.markdown("#### 📊 Key Feature Contributors")
                expl_col1, expl_col2 = st.columns(2)
                
                with expl_col1:
                    st.markdown("<p style='color:#2ecc71; font-weight:600; font-size:1.1rem;'>🟢 Top Positive Triggers</p>", unsafe_allow_html=True)
                    if top_pos:
                        pos_df = pd.DataFrame(top_pos, columns=["Word", "Importance"])
                        st.dataframe(pos_df.style.background_gradient(cmap="Greens", subset=["Importance"]), use_container_width=True)
                    else:
                        st.info("No positive triggers found in this text.")
                        
                with expl_col2:
                    st.markdown("<p style='color:#e74c3c; font-weight:600; font-size:1.1rem;'>🔴 Top Negative Triggers</p>", unsafe_allow_html=True)
                    if top_neg:
                        neg_df = pd.DataFrame(top_neg, columns=["Word", "Importance"])
                        st.dataframe(neg_df.style.background_gradient(cmap="Reds_r", subset=["Importance"]), use_container_width=True)
                    else:
                        st.info("No negative triggers found in this text.")

# ================= TAB 2: BATCH PROCESSOR =================
with tab2:
    st.markdown("### 📂 Bulk Processing & Sentiment Analytics")
    st.write("Upload a CSV file containing reviews. SentimentLens will classify all reviews and display interactive dashboard charts.")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            
            # Column selector
            columns = df.columns.tolist()
            text_col = st.selectbox("Select the column containing review text:", columns)
            
            if st.button("Process Batch Data"):
                with st.spinner("Processing batch predictions..."):
                    # Progress Bar
                    progress_text = "Analyzing batch..."
                    my_bar = st.progress(0, text=progress_text)
                    
                    results = []
                    confidences = []
                    
                    # Row-by-row prediction to update progress bar (or batch vectorization which is faster)
                    # For performance, we'll vectorise everything, then predict, then update progress
                    texts = df[text_col].fillna("").astype(str).tolist()
                    
                    # Batch predict
                    batch_start = time.time()
                    cleaned_texts = [clean_text(t) for t in texts]
                    
                    # Handing empty texts
                    cleaned_texts = [c if c.strip() else "neutral" for c in cleaned_texts]
                    
                    features = vectorizer.transform(cleaned_texts)
                    probs = model.predict_proba(features) # matrix of size [num_samples, 2]
                    
                    for i, prob in enumerate(probs):
                        pred_label_idx = np.argmax(prob)
                        label = 'positive' if pred_label_idx == 1 else 'negative'
                        confidence = prob[pred_label_idx]
                        
                        results.append(label)
                        confidences.append(confidence)
                        
                        if i % max(1, len(texts)//10) == 0:
                            my_bar.progress(int((i/len(texts))*100), text=f"Analyzed {i}/{len(texts)} reviews")
                            
                    my_bar.progress(100, text="Analysis complete!")
                    time.sleep(0.5)
                    my_bar.empty()
                    
                    # Add results to dataframe
                    df['predicted_sentiment'] = results
                    df['confidence'] = confidences
                    
                    st.markdown("---")
                    st.markdown("### 📊 Batch Analytics Dashboard")
                    
                    # Layout
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Donut chart of sentiment distribution
                        sentiment_counts = df['predicted_sentiment'].value_counts().reset_index()
                        sentiment_counts.columns = ['Sentiment', 'Count']
                        
                        fig = px.pie(
                            sentiment_counts, 
                            values='Count', 
                            names='Sentiment', 
                            hole=0.45,
                            title='Sentiment Distribution',
                            color='Sentiment',
                            color_discrete_map={'positive': '#2ecc71', 'negative': '#e74c3c'}
                        )
                        fig.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font_color='#ffffff',
                            title_x=0.25
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                    with col2:
                        # Sentiment vs review length
                        df['review_length'] = df[text_col].apply(lambda x: len(str(x).split()))
                        
                        fig = px.histogram(
                            df, 
                            x='review_length', 
                            color='predicted_sentiment',
                            barmode='overlay',
                            title='Review Word Count Distribution by Sentiment',
                            labels={'review_length': 'Word Count', 'predicted_sentiment': 'Sentiment'},
                            color_discrete_map={'positive': '#2ecc71', 'negative': '#e74c3c'}
                        )
                        fig.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font_color='#ffffff',
                            title_x=0.2
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # HTML Custom Tag Clouds for Positive and Negative sentiments in Uploaded File
                    st.markdown("#### ☁️ Dataset Tag Clouds")
                    cloud_col1, cloud_col2 = st.columns(2)
                    
                    # Compile text counts
                    with cloud_col1:
                        st.markdown("<p style='text-align:center; color:#2ecc71; font-weight:600;'>🟢 Positive Reviews Tag Cloud</p>", unsafe_allow_html=True)
                        pos_corpus = " ".join(df[df['predicted_sentiment'] == 'positive'][text_col].fillna("").astype(str).tolist())
                        pos_clean = clean_text(pos_corpus).split()
                        
                        # Get frequencies
                        if pos_clean:
                            from collections import Counter
                            pos_freq = Counter(pos_clean).most_common(25)
                            render_tag_cloud(pos_freq, "positive")
                        else:
                            st.write("No words to display.")
                            
                    with cloud_col2:
                        st.markdown("<p style='text-align:center; color:#e74c3c; font-weight:600;'>🔴 Negative Reviews Tag Cloud</p>", unsafe_allow_html=True)
                        neg_corpus = " ".join(df[df['predicted_sentiment'] == 'negative'][text_col].fillna("").astype(str).tolist())
                        neg_clean = clean_text(neg_corpus).split()
                        
                        if neg_clean:
                            from collections import Counter
                            neg_freq = Counter(neg_clean).most_common(25)
                            render_tag_cloud(neg_freq, "negative")
                        else:
                            st.write("No words to display.")
                            
                    # SVD projection for semantic visualization
                    st.markdown("#### 🗺️ Review Semantic Map (2D Semantic Projection)")
                    st.write("This interactive scatter plot shows a 2D projection of the uploaded reviews using Latent Semantic Analysis (TruncatedSVD). Each point represents a review, colored by its predicted sentiment.")
                    
                    try:
                        # Vectorize all cleaned texts
                        X_tfidf = vectorizer.transform(cleaned_texts)
                        
                        # Project to 2D
                        svd = TruncatedSVD(n_components=2, random_state=42)
                        X_2d = svd.fit_transform(X_tfidf)
                        
                        # Build DataFrame for plotting
                        svd_df = pd.DataFrame(X_2d, columns=['Semantic Dimension 1', 'Semantic Dimension 2'])
                        svd_df['Sentiment'] = results
                        svd_df['Confidence'] = confidences
                        
                        # Truncate review text for hover data
                        svd_df['Review Preview'] = [t[:100] + "..." if len(t) > 100 else t for t in texts]
                        
                        fig_svd = px.scatter(
                            svd_df, 
                            x='Semantic Dimension 1', 
                            y='Semantic Dimension 2', 
                            color='Sentiment',
                            hover_data=['Review Preview', 'Confidence'],
                            color_discrete_map={'positive': '#2ecc71', 'negative': '#e74c3c'},
                            title='Reviews mapped in Latent Semantic Space'
                        )
                        
                        fig_svd.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font_color='#ffffff',
                            title_x=0.35
                        )
                        
                        st.plotly_chart(fig_svd, use_container_width=True)
                    except Exception as ex:
                        st.info(f"Visualizing Semantic Space is not supported for this small batch: {ex}")
                        
                    # Download section
                    st.markdown("#### 📥 Download Predictions")
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Processed CSV",
                        data=csv,
                        file_name="sentimentlens_predictions.csv",
                        mime="text/csv",
                        type="primary"
                    )
                    
                    st.markdown("#### 📋 Data Preview")
                    st.dataframe(df[[text_col, 'predicted_sentiment', 'confidence', 'review_length']], use_container_width=True)
                    
        except Exception as e:
            st.error(f"Error parsing file: {e}")
            
    else:
        # Template download helper
        st.info("💡 Don't have a dataset? Download a template CSV, add movie reviews, and upload it back!")
        template_df = pd.DataFrame({
            "review": [
                "This film was a beautiful masterclass in visual storytelling.",
                "Worst waste of money. The acting was atrocious and the plot made absolutely no sense.",
                "A decent movie with some slow moments, but the lead actor saves it.",
                "Loved it! Best soundtrack of the year and very emotional."
            ]
        })
        csv_template = template_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Template CSV",
            data=csv_template,
            file_name="sentimentlens_batch_template.csv",
            mime="text/csv"
        )

# ================= TAB 3: DIAGNOSTICS & RETRAINING =================
with tab3:
    st.markdown("### ⚙️ Engine Diagnostics & Local Model Training")
    st.write("Explore how the model performs, inspect feature weights, or retrain the model locally.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### ⚙️ Local Retraining Engine")
        st.write("Upload clean data or use default dataset configurations to retrain the underlying model.")
        
        subset_slider = st.slider("Select Dataset Size (Subsetting)", min_value=1000, max_value=30000, value=10000, step=1000)
        
        if st.button("Start Model Retraining", type="primary"):
            with st.spinner("Retraining sentiment classifier..."):
                train_placeholder = st.empty()
                train_placeholder.info("🔄 Preprocessing data & extracting features...")
                
                try:
                    run_training_pipeline(subset_size=subset_slider)
                    
                    # Clear cache to load new model
                    get_cached_model.clear()
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Training failed: {e}")
                    
        st.markdown("---")
        st.markdown("#### 📊 Evaluation Metrics")
        if metadata:
            st.markdown(f"• **Accuracy Score**: `{metadata['accuracy']*100:.2f}%`")
            st.markdown(f"• **Vocabulary Size**: `{metadata['vocab_size']:,} words`")
            st.markdown(f"• **Training Sample Size**: `{metadata['dataset_size']:,} reviews`")
            
    with col2:
        st.markdown("#### 🧩 Confusion Matrix")
        if metadata and 'confusion_matrix' in metadata:
            cm = np.array(metadata['confusion_matrix'])
            
            fig = px.imshow(
                cm,
                text_auto=True,
                aspect="auto",
                labels=dict(x="Predicted Sentiment", y="Actual Sentiment", color="Count"),
                x=['Negative', 'Positive'],
                y=['Negative', 'Positive'],
                color_continuous_scale="RdBu_r"
            )
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#ffffff',
                title_x=0.5
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Confusion matrix statistics not available.")
            
    st.markdown("---")
    st.markdown("#### 🔎 Interactive Model Vocabulary & Coefficient Explorer")
    st.write("Search the model's vocabulary to see the raw coefficients. A positive coefficient pushes the prediction towards **Positive Sentiment**, while a negative coefficient pulls it towards **Negative Sentiment**.")
    
    if model and vectorizer:
        vocab = vectorizer.vocabulary_
        coefs = model.coef_[0]
        
        # Build dictionary
        vocab_list = []
        for word, idx in vocab.items():
            vocab_list.append({
                "Word": word,
                "Coefficient": float(coefs[idx]),
                "Influence": "Positive 🟢" if coefs[idx] > 0 else "Negative 🔴"
            })
            
        vocab_df = pd.DataFrame(vocab_list).sort_values(by="Coefficient", key=abs, ascending=False).reset_index(drop=True)
        
        # Add Search Bar
        search_query = st.text_input("Type a word to search model coefficient (e.g., 'worst', 'amaz', 'bore', 'excel'):")
        
        if search_query:
            filtered_df = vocab_df[vocab_df['Word'].str.contains(search_query.lower(), case=False)]
            st.dataframe(filtered_df, use_container_width=True)
        else:
            st.dataframe(vocab_df.head(100), use_container_width=True)
            st.caption("Displaying top 100 most influential words by default.")

# 🔮 SentimentLens: Advanced Sentiment Analytics & Explanation Engine
SentimentLens NLP CI/CD Docker Streamlit App

SentimentLens is a production-ready, resume-grade Natural Language Processing (NLP) web application built to classify movie review sentiments (Positive/Negative) and deliver interactive, real-time predictions through a clean clinical UI.

Instead of treating the machine learning model as a "black box," SentimentLens implements a local mathematical feature attribution method. It maps TF-IDF coefficients directly back to the source text, highlighting exactly which words triggered a positive (emerald green) or negative (ruby red) classification — fully containerized and backed by an automated CI/CD pipeline.

## 🚀 Key Features

### 🔍 Interactive Analysis Hub:
* **Single-Review Prediction**: Predict sentiment in real-time with confidence scores.
* **LIME-Style Local Attribution**: Visually highlights positive/negative words in the review based on mathematical feature importances (TF-IDF weights × coefficients).
* **Text Statistics**: Displays reading time estimates and text length metrics.

### 📊 Batch Processor:
* **Bulk Upload**: Upload CSV files containing customer reviews for batch analysis.
* **Interactive Data Analytics**: Render Plotly donut charts of sentiment distribution and word count histograms by class.
* **Vocabulary Tag Clouds**: Dynamic HTML/CSS word clouds representing positive and negative vocabularies in the uploaded dataset.
* **Downloadable Results**: Export predictions and confidence scores as CSV.

### ⚙️ Engine Diagnostics & Retraining:
* **Confusion Matrix Heatmap**: Interactive heatmap evaluating model accuracy.
* **Vocabulary Coefficient Explorer**: Searchable dictionary containing all model terms and their exact positive/negative weights.
* **On-Demand Local Retraining**: Slide to select dataset size and trigger live retraining, reloading the model state dynamically.

### 🐳 Production Containerization:
* A fully optimized Docker image pre-trained with 10k+ IMDB samples for instant startup.
* Serves the Streamlit app on port 8501 on any machine without environment setup.

### 🛡️ CI/CD Integration:
* GitHub Actions workflow automatically installs dependencies, runs linter and unit tests, and verifies project health on every push or pull request.

---

## 🛠️ Technology Stack

| Layer | Tools |
| :--- | :--- |
| **Frontend & UI** | Streamlit, HTML5, Custom CSS |
| **Machine Learning** | Scikit-Learn (LogisticRegression), NumPy, Joblib |
| **Data Processing** | Pandas, NLTK (Tokenization, Stopwords, Porter Stemming) |
| **Visualizations** | Plotly Express |
| **Testing & Linter** | PyTest, Flake8 |
| **DevOps & Containerization** | Docker, GitHub Actions (YAML Workflows) |

---

## 📐 Architecture & Workflow

```
Dataset (IMDB/Synthetic Reviews)
       ↓
Data Preprocessing
(HTML Tag Removal → Non-alphabetic Cleaning → Porter Stemming)
       ↓
Feature Extraction
(TfidfVectorizer — vocab mapping)
       ↓
Logistic Regression
(train_test_split → model.fit → model.predict)
       ↓
Model Persistence
(joblib.dump → models/sentiment_model.pkl)
       ↓
Streamlit Web Application
(User Input/CSV → Inference → Feature Attribution)
       ↓
GitHub Repository
       ↓
GitHub Actions CI/CD
(Checkout → Install Deps → Run Lints → Run PyTest → Verify Docker)
       ↓
Docker Containerization
(Build Image → Run Container → Serve on :8501)
       ↓
Streamlit Cloud Deployment
```

---

## 📖 Model Details

SentimentLens uses a **TF-IDF + Logistic Regression** model. The prediction for a document $d$ is determined by the log-odds:

$$\text{log-odds} = \beta_0 + \sum_{i \in \text{vocab}} \beta_i \cdot \text{TF-IDF}(w_i, d)$$

Where:
* $\beta_0$ is the model intercept.
* $\beta_i$ is the coefficient weight of word $w_i$.
* $\text{TF-IDF}(w_i, d)$ is the TF-IDF weight of word $w_i$ in document $d$.

To explain the prediction for a single review, we compute the individual attribution score $S$ for each word in the input text:

$$S(w_i, d) = \beta_i \cdot \text{TF-IDF}(w_i, d)$$

Preprocessing pipeline applied before training and inference:

| Step | Method |
| :--- | :--- |
| **Text Cleaning** | Removal of HTML tags and non-alphabetic characters |
| **Text Tokenization** | Case normalization, stopword filtering, and Porter stemming |
| **Feature Extraction** | TfidfVectorizer (max_features=5000, n_gram range=(1, 2)) |
| **Classification Model** | LogisticRegression (class_weight='balanced', max_iter=1000) |
| **Train/Test Split** | train_test_split() (80/20 split) |

* **Target Variable**: `sentiment` (positive / negative)
* **Achieved Accuracy**: ~88% on the test split.

> [!WARNING]
> This application is built for educational and portfolio demonstration purposes only. It is not intended for clinical or critical operational decision-making.

---

## 🏃 Local Setup & Installation

### Prerequisites
* Python 3.11+
* Git

### Step-by-Step Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Varun7860dixit/Sentiment_Analysis_System.git
   cd Sentiment_Analysis_System
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Training Pipeline**:
   Train the classification model on movie reviews:
   ```bash
   python -m src.train
   ```
   * **Output**: Model saved successfully to the `models/` directory.

5. **Run Streamlit**:
   Launch the web application locally:
   ```bash
   streamlit run app.py
   ```
   Access the app in your browser at `http://localhost:8501`.

---

## 🐳 Docker Deployment

To build and run the containerized application:

1. **Build the Docker Image**:
   ```bash
   docker build -t sentiment-lens-app .
   ```

2. **Run the Container**:
   ```bash
   docker run -p 8501:8501 sentiment-lens-app
   ```

> [!WARNING]
> Ensure Docker Desktop is running before executing these commands. The `dockerDesktopLinuxEngine not found` error occurs when Docker Desktop is not active.

The application will download NLTK, train the initial model state, and serve on port `8501`.

---

## 🛡️ CI/CD Pipeline

Every git push or pull request triggers the following GitHub Actions workflow (`.github/workflows/ci-cd.yml`):

```yaml
name: Sentura NLP CI/CD Pipeline

on:
  push:
    branches: [ main, master ]
  pull_request:
    branches: [ main, master ]
  workflow_dispatch:
```

### What happens on every push:
1. **GitHub detects the code change**: Spins up a fresh Ubuntu virtual machine.
2. **Installs all project dependencies**: Upgrades pip and installs packages from `requirements.txt`.
3. **Executes Flake8 Linter**: Scans the codebase for syntax errors or undefined names.
4. **Executes PyTest Unit Tests**: Runs the test suite in the `tests/` directory.
5. **Verifies Docker Build**: Builds the Dockerfile to ensure containerization health.
6. **Reports pipeline success or failure**.

---

## ☁️ Streamlit Cloud Deployment

The application is deployed on Streamlit Community Cloud:
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub account.
3. Select the repository: `Varun7860dixit/Sentiment_Analysis_System`
4. Branch: `main` | Main file: `app.py`
5. Click **Deploy**.
6. 🔗 **Live App**: *(Add your Streamlit deployment URL here)*

---

## 📁 Project Structure

```text
Sentiment_Analysis_System/
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml           # GitHub Actions CI/CD pipeline
│
├── data/
│   └── movie_data.csv          # Movie reviews dataset
│
├── models/
│   ├── sentiment_model.pkl     # Trained Logistic Regression model (auto-generated)
│   ├── tfidf_vectorizer.pkl    # Serialized TF-IDF vectorizer (auto-generated)
│   └── model_metadata.pkl      # Training metadata (auto-generated)
│
├── src/
│   ├── utils.py                # Preprocessing and text cleaning utilities
│   ├── train.py                # Training pipeline script
│   └── inference.py            # Predictor and explanation engine
│
├── tests/
│   ├── test_train.py           # Unit tests for training pipeline
│   └── test_inference.py       # Unit tests for inference pipeline
│
├── app.py                      # Streamlit dashboard and UI app
├── Dockerfile                  # Multi-stage Docker configuration
├── requirements.txt            # Python dependencies
├── README.md
└── .gitignore
```

---

## 📦 Requirements

```text
streamlit>=1.30.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.2.0
plotly>=5.15.0
nltk>=3.8.0
pytest>=7.4.0
flake8>=6.1.0
joblib>=1.3.0
wordcloud>=1.9.0
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## ✅ Objectives Achieved

| Objective | Status |
| :--- | :--- |
| Develop NLP classification model (Logistic Regression) | ✅ Achieved |
| Mathematical explainability (Attribution Engine) | ✅ Achieved |
| Model training & serialization pipeline | ✅ Achieved |
| Interactive Streamlit UI | ✅ Achieved |
| Docker containerization | ✅ Achieved |
| GitHub Actions CI/CD pipeline | ✅ Achieved |
| Automated testing & code quality linter | ✅ Achieved |
| Streamlit Cloud deployment | ✅ Achieved |

---

## ⚠️ Disclaimer

This application is built for educational and portfolio demonstration purposes only. The sentiment classification and words highlighted by the model should not be used as a substitute for professional research, business intelligence, or customer sentiment metrics.

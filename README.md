# ✨ Semantic Product Review Analyzer

A powerful, next-generation NLP tool for analyzing product reviews with granular sentiment understanding, aspect extraction, and advanced linguistic nuance detection.

## 🚀 Key Features

### 🧠 Advanced Sentiment Engine
- **10-Tier Granular Analysis**: Goes beyond positive/negative. Detects nuance on a 0-100% scale:
    - *Very Positive (90-100%)* to *Very Negative (0-10%)*
    - nuances like *Slightly Positive*, *Moderately Negative*, *Leaning Positive*, etc.
- **Context-Aware Scoring**: Understands contrastive conjunctions ("but", "however") and shifts in tone within a single sentence.
- **Intensity Detection**: Calculates the intensity of the sentiment (e.g., "good" vs "phenomenal").

### 🔍 Deep Understanding
- **Aspect Extraction**: Automatically identifies product aspects (e.g., "battery life", "build quality", "customer service") using hybrid regex and POS tagging.
- **Humor & Sarcasm Detection**: Identifies irony, exaggeration, and humorous elements in reviews.
- **Contradiction Analysis**: flags conflicting statements (e.g., "The design is great but the build is cheap") and calculates a contradiction score.

### 📊 Interactive Visualization
- **Sentiment Spectrum**: Visual bar showing exactly where the sentiment falls on the 0-100% scale.
- **Dynamic Word Clouds**: Visual representation of key phrases.
- **Aspect Sentiment Charts**: Horizontal bar charts breaking down satisfaction by feature.
- **Interactive Gauges**: Real-time visualization of overall satisfaction and specific metrics.

## 🛠 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/semantic-review-analyzer.git
   cd semantic-review-analyzer
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: This project requires PyTorch, Transformers, SpaCy, and Streamlit.*

4. **Install NLP Models:**
   ```bash
   python -m spacy download en_core_web_sm
   python -m nltk.downloader punkt stopwords wordnet
   ```

## 🚦 Usage

1. **Launch the Application:**
   ```bash
   streamlit run app.py
   ```

2. **Analyze Reviews:**
   - Open your browser to `http://localhost:8501`.
   - Paste a review into the text area.
   - Click **Analyze Review** to see the magic! 🚀

## 🧪 Example Analysis

**Input:**
> "The headphones look amazing and the sound is decent, however, they are quite uncomfortable to wear for long periods. I want to love them but I can't."

**Output:**
- **Overall Score**: ~40% (Mixed/Leaning Negative)
- **Granular Sentiment**: Leaning Negative 😕
- **Aspects Detected**:
    - *Appearance*: Very Positive (100%) - "looks amazing"
    - *Audio*: Neutral/Slightly Positive (60%) - "sound is decent"
    - *Comfort*: Negative (20%) - "uncomfortable"
- **Contradictions**: Detected! ("look amazing" vs "uncomfortable")

## 📂 Project Structure

```
semantic-review-analyzer/
├── app.py               # Main Streamlit dashboard application
├── semantic_analyzer.py # Core NLP engine (Aspect extraction, Sentiment logic, Humor detection)
├── config.py            # Configuration (Model names, Regex patterns)
├── requirements.txt     # Python dependencies
└── README.md            # Documentation
```

## 🔧 Under the Hood

The analyzer uses a hybrid approach:
1. **Transformer Models**: Uses `distilbert-base-uncased-finetuned-sst-2-english` for raw sentiment scoring.
2. **Linguistic Rules**: Applies a complex layer of rules to handle negation, intensifiers ("very", "extremely"), and diminishers ("slightly").
3. **Lexicon Adjustment**: Fine-tunes scores based on specific words to ensure "okay" maps to ~50% and "excellent" maps to >90%.
4. **SpaCy Dependency Parsing**: Used to bind adjectives to their specific nouns for accurate aspect-based sentiment.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Sentence-Transformers](https://www.sbert.net/)
- [Streamlit](https://streamlit.io/)
- [spaCy](https://spacy.io/)
- [NLTK](https://www.nltk.org/)

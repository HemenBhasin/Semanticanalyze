# ✨ Semantic Product Review Analyzer

A powerful, next-generation NLP tool for analyzing product reviews with granular sentiment understanding, aspect extraction, and advanced linguistic nuance detection.

## 🚀 Key Features

### 🌐 Live Cross-Platform Web Scraper
- **Universal URL Support**: Paste a single Amazon link, and the engine dynamically fetches reviews.
- **Auto-Matcher**: Automatically searches Yahoo for the equivalent product on Flipkart to scrape both platforms concurrently.
- **Anti-Bot Defenses**: Headless Playwright integration with stealth scripts, ad-blocking, parallel tabs, and diverse sorting (Recent/Top) proxying to bypass rigorous e-commerce captchas.

### 🧠 Advanced Sentiment Engine
- **10-Tier Granular Analysis**: Goes beyond positive/negative. Detects nuance on a 0-100% scale.
- **Cross-Platform Discrepancy Alerts**: Detects and visually warns if a product is strangely rated highly on one site but hated on another.
- **Short-Review Indian Market Optimization**: Heuristically scores short generic reviews ("Good", "Awesome") accurately to reflect true product sentiment mathematically without UI clutter.

### 🔍 Deep Understanding & Auth Filter
- **Authenticity Filter**: Automatically cleans dataset by dropping exact duplicates, scam links, and keyboard smashes.
- **Aspect Extraction**: Automatically identifies product aspects (e.g., "battery life", "build quality") using hybrid regex and POS tagging.
- **Humor & Contradiction Detection**: Flags conflicting statements ("The design is great but the build is cheap").

### 📊 Interactive Dashboard
- **Segmented Final Ratings**: Unified Verdict displaying individual platform breakdown alongside the combined rating.
- **Ranked Feature Tags**: Visual horizontal bar charts breaking down satisfaction by feature.
- **Trend Alerts**: Flags sudden dips in recent review scores compared to historical averages.

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

- [Hugging Face Transformers](https://huggingface.co/transformers/) (Core Sentiment Engine)
- [Sentence-Transformers](https://www.sbert.net/) (Vector Embeddings)
- [Streamlit](https://streamlit.io/) (Interactive Dashboard)
- [Playwright](https://playwright.dev/) (Headless E-commerce Web Scraping)
- [BeautifulSoup 4](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) (Parsing HTML DOM elements)
- [spaCy](https://spacy.io/) & [NLTK](https://www.nltk.org/) (Linguistic Processing)

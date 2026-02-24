import re
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from spacy.matcher import Matcher
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import spacy
from transformers import pipeline
import torch
from collections import defaultdict
import logging

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class SemanticReviewAnalyzer:
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        """Initialize the semantic analyzer with a pre-trained sentiment model."""
        try:
            # Try to load the spaCy model
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # If model not found, download it
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize the sentiment analyzer pipeline with batching
        device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis", 
                model=model_name,
                device=device,
                truncation=True,
                batch_size=8  # Process multiple clauses in parallel
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize sentiment analyzer: {str(e)}")
            raise
        
        # Initialize the Matcher for aspect extraction
        self.matcher = Matcher(self.nlp.vocab)
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Add aspect patterns to matcher
        self._initialize_aspect_patterns()
    
    def _initialize_aspect_patterns(self):
        """Initialize patterns for aspect extraction."""
        # Adjective + Noun patterns
        adj_noun = [
            {"POS": "ADJ", "OP": "?"},
            {"POS": "NOUN"}
        ]
        
        # Noun + Preposition + Noun (e.g., "quality of the screen")
        noun_prep_noun = [
            {"POS": "NOUN"},
            {"POS": "ADP"},  # Preposition
            {"POS": "DET", "OP": "?"},
            {"POS": "NOUN"}
        ]
        
        # Add patterns to matcher
        self.matcher.add("ADJ_NOUN", [adj_noun])
        self.matcher.add("NOUN_PREP_NOUN", [noun_prep_noun])
        
        # Add common aspect patterns
        self.aspect_patterns = [
            (r'\b(screen|display|monitor)\b', 'display'),
            (r'\b(battery|battery life|battery time)\b', 'battery'),
            (r'\b(performance|speed|processing|processor|cpu|gpu|ram|memory|storage|ssd|hard drive)\b', 'performance'),
            (r'\b(keyboard|trackpad|touchpad|mouse|input device)\b', 'input devices'),
            (r'\b(speaker|audio|sound|microphone|mic)\b', 'audio'),
            (r'\b(design|build|looks?|appearance|aesthetics)\b', 'design'),
            (r'\b(port|usb|hdmi|thunderbolt|connector|jack|slot|sd card|headphone)\b', 'ports'),
            (r'\b(weight|lightweight|heavy|portability|size|dimension)\b', 'portability'),
            (r'\b(price|cost|value|worth|expensive|cheap|affordable)\b', 'price'),
            (r'\b(camera|webcam|video call|selfie)\b', 'camera'),
            (r'\b(software|os|operating system|windows|macos|linux|driver|firmware)\b', 'software'),
            (r'\b(heat|temperature|cooling|fan|noise|ventilation)\b', 'thermal performance'),
            (r'\b(keyboard|key|typing|backlit|backlight|illuminated)\b', 'keyboard'),
            (r'\b(trackpad|touchpad|gesture|pointing device)\b', 'trackpad'),
            (r'\b(bluetooth|wifi|wireless|connectivity|nfc|gps|ethernet|lan)\b', 'connectivity'),
            (r'\b(upgrade|ram upgrade|storage upgrade|expandability|user replaceable)\b', 'upgradability'),
            (r'\b(keyboard|touchpad|trackpad|mouse|stylus|pen|touch screen|touchscreen)\b', 'input devices'),
            (r'\b(hinge|build quality|durability|sturdiness|material|aluminum|plastic|metal|magnesium)\b', 'build quality'),
            (r'\b(service|support|warranty|return policy|customer service|repair|replacement)\b', 'customer support'),
            (r'\b(bloatware|pre-installed|trial|adware|unnecessary software|bloat)\b', 'bloatware')
        ]
        
    def preprocess_text(self, text: str) -> str:
        """Preprocess the input text."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs, special characters, and numbers
        text = re.sub(r'http\S+|www.\S+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenization and lemmatization
        tokens = word_tokenize(text)
        tokens = [self.nlp(token)[0].lemma_ for token in tokens]
        
        # Remove stopwords and short tokens
        tokens = [token for token in tokens if token not in stopwords.words('english') and len(token) > 2]
        
        return ' '.join(tokens)
    
    def get_granular_sentiment_label(self, score: float) -> str:
        """
        Convert sentiment score (0-1) to granular sentiment label.
        
        Args:
            score: Sentiment score between 0 and 1
            
        Returns:
            Detailed sentiment label string
        """
        if score >= 0.90:
            return "Very Positive"
        elif score >= 0.80:
            return "Highly Positive"
        elif score >= 0.70:
            return "Moderately Positive"
        elif score >= 0.60:
            return "Slightly Positive"
        elif score >= 0.50:
            return "Leaning Positive"
        elif score >= 0.40:
            return "Leaning Negative"
        elif score >= 0.30:
            return "Slightly Negative"
        elif score >= 0.20:
            return "Moderately Negative"
        elif score >= 0.10:
            return "Highly Negative"
        else:
            return "Very Negative"
    
    def detect_humor(self, text: str) -> Dict:
        """
        Detect humor, sarcasm, and irony in text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with humor analysis results:
            {
                'has_humor': bool,
                'humor_type': str (sarcasm/irony/exaggeration/wordplay/none),
                'confidence': float (0-1),
                'indicators': List[str],
                'humorous_segments': List[str]
            }
        """
        doc = self.nlp(text.lower())
        text_lower = text.lower()
        
        # Humor indicators
        sarcasm_patterns = [
            'yeah right', 'sure', 'as if', 'oh great', 'just what i needed',
            'wonderful', 'perfect', 'exactly what i wanted', 'fantastic',
            'brilliant idea', 'oh wow', 'real genius', 'nice job'
        ]
        
        exaggeration_words = [
            'literally', 'absolutely', 'completely', 'totally', 'utterly',
            'extremely', 'incredibly', 'unbelievably', 'ridiculously',
            'insanely', 'mind-blowing', 'earth-shattering', 'world-ending'
        ]
        
        irony_patterns = [
            ('love', ['hate', 'terrible', 'awful', 'bad', 'worst']),
            ('great', ['disappointing', 'failed', 'broken', 'useless']),
            ('perfect', ['defective', 'broken', 'malfunctioned', 'crashed']),
            ('excellent', ['poor', 'bad', 'terrible', 'awful'])
        ]
        
        # Track indicators
        indicators = []
        humorous_segments = []
        humor_score = 0.0
        humor_type = 'none'
        
        # Check for sarcasm patterns
        sarcasm_count = 0
        for pattern in sarcasm_patterns:
            if pattern in text_lower:
                sarcasm_count += 1
                indicators.append(f"Sarcasm: '{pattern}'")
                humorous_segments.append(pattern)
        
        if sarcasm_count > 0:
            humor_score += min(0.4, sarcasm_count * 0.2)
            humor_type = 'sarcasm'
        
        # Check for exaggeration
        exaggeration_count = sum(1 for word in exaggeration_words if word in text_lower)
        if exaggeration_count >= 2:
            humor_score += min(0.3, exaggeration_count * 0.1)
            indicators.append(f"Exaggeration detected ({exaggeration_count} markers)")
            if humor_type == 'none':
                humor_type = 'exaggeration'
        
        # Check for irony (positive words with negative context)
        for positive_word, negative_words in irony_patterns:
            if positive_word in text_lower:
                for neg_word in negative_words:
                    if neg_word in text_lower:
                        # Check if they're in proximity (within 10 words)
                        pos_idx = text_lower.find(positive_word)
                        neg_idx = text_lower.find(neg_word)
                        if abs(pos_idx - neg_idx) < 50:  # Rough word proximity
                            humor_score += 0.3
                            indicators.append(f"Irony: '{positive_word}' near '{neg_word}'")
                            humorous_segments.append(f"{positive_word}...{neg_word}")
                            if humor_type == 'none':
                                humor_type = 'irony'
        
        # Check for exclamation marks (often used sarcastically)
        exclamation_count = text.count('!')
        if exclamation_count >= 2:
            humor_score += min(0.2, exclamation_count * 0.05)
            indicators.append(f"Multiple exclamations ({exclamation_count})")
        
        # Check for ALL CAPS (sarcasm indicator)
        caps_words = [token.text for token in doc if token.text.isupper() and len(token.text) > 2]
        if caps_words:
            humor_score += min(0.2, len(caps_words) * 0.1)
            indicators.append(f"Emphasis caps: {', '.join(caps_words)}")
        
        # Normalize confidence score
        confidence = min(1.0, humor_score)
        has_humor = confidence > 0.2
        
        return {
            'has_humor': has_humor,
            'humor_type': humor_type if has_humor else 'none',
            'confidence': confidence,
            'indicators': indicators,
            'humorous_segments': humorous_segments
        }
    
    def detect_contradictions(self, text: str) -> Dict:
        """
        Detect contradictory statements in the text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with contradiction analysis:
            {
                'has_contradictions': bool,
                'contradiction_score': float (0-1),
                'contradictory_pairs': List[Dict],
                'sentiment_variance': float,
                'contrast_markers': List[str]
            }
        """
        doc = self.nlp(text)
        
        # Contrast markers
        contrast_markers = [
            'but', 'however', 'although', 'though', 'yet', 'except',
            'despite', 'whereas', 'while', 'nevertheless', 'nonetheless',
            'on the other hand', 'even though', 'in contrast', 'conversely',
            'still', 'instead', 'unfortunately', 'sadly'
        ]
        
        # Find all contrast markers in text
        found_markers = []
        for marker in contrast_markers:
            if marker in text.lower():
                found_markers.append(marker)
        
        # Split text at contrast markers
        contradictory_pairs = []
        sentiment_scores = []
        
        # Analyze each sentence
        sentences = list(doc.sents)
        for i, sent in enumerate(sentences):
            sent_text = sent.text.strip()
            
            # Check if sentence contains contrast marker
            has_contrast = any(marker in sent_text.lower() for marker in contrast_markers)
            
            if has_contrast:
                # Split at the contrast marker
                for marker in contrast_markers:
                    if marker in sent_text.lower():
                        parts = sent_text.lower().split(marker, 1)
                        if len(parts) == 2:
                            before_text = parts[0].strip()
                            after_text = parts[1].strip()
                            
                            # Analyze sentiment of both parts
                            if before_text and after_text:
                                try:
                                    before_sentiment = self._analyze_sentence_sentiment(before_text)
                                    after_sentiment = self._analyze_sentence_sentiment(after_text)
                                    
                                    # Check if sentiments are opposite
                                    score_diff = abs(before_sentiment['score'] - after_sentiment['score'])
                                    
                                    if score_diff > 0.3:  # Significant difference
                                        contradictory_pairs.append({
                                            'before': before_text,
                                            'after': after_text,
                                            'marker': marker,
                                            'before_sentiment': before_sentiment,
                                            'after_sentiment': after_sentiment,
                                            'contrast_strength': score_diff
                                        })
                                        
                                        sentiment_scores.extend([
                                            before_sentiment['score'],
                                            after_sentiment['score']
                                        ])
                                except Exception as e:
                                    self.logger.warning(f"Error analyzing contradiction: {str(e)}")
                        break
        
        # Calculate sentiment variance
        if len(sentiment_scores) > 1:
            sentiment_variance = float(np.var(sentiment_scores))
        else:
            sentiment_variance = 0.0
        
        # Calculate contradiction score
        contradiction_score = 0.0
        if contradictory_pairs:
            # Base score on number and strength of contradictions
            avg_strength = sum(pair['contrast_strength'] for pair in contradictory_pairs) / len(contradictory_pairs)
            contradiction_score = min(1.0, (len(contradictory_pairs) * 0.3) + (avg_strength * 0.5))
        
        has_contradictions = contradiction_score > 0.2
        
        return {
            'has_contradictions': has_contradictions,
            'contradiction_score': contradiction_score,
            'contradictory_pairs': contradictory_pairs,
            'sentiment_variance': sentiment_variance,
            'contrast_markers': found_markers
        }
    
    def extract_aspects(self, text: str) -> List[str]:
        """
        Enhanced aspect extraction using spaCy's POS tagging, dependency parsing, and pattern matching.
        
        Args:
            text: The input text to extract aspects from
            
        Returns:
            List of extracted and grouped aspects
        """
        if not text.strip():
            return []
            
        doc = self.nlp(text.lower().strip())
        
        # Define more comprehensive aspect patterns
        patterns = [
            # Adjective + Noun (e.g., "fast performance", "good quality")
            [{"POS": "ADJ"}, {"POS": "NOUN"}],
            # Adverb + Adjective + Noun (e.g., "very good performance")
            [{"POS": "ADV"}, {"POS": "ADJ"}, {"POS": "NOUN"}],
            # Noun + Preposition + Noun (e.g., "quality of service")
            [{"POS": "NOUN"}, {"POS": "ADP"}, {"POS": "NOUN"}],
            # Compound nouns (e.g., "battery life", "screen resolution")
            [{"POS": "NOUN"}, {"POS": "NOUN"}],
            # Adjective + Adjective + Noun (e.g., "long battery life")
            [{"POS": "ADJ"}, {"POS": "ADJ"}, {"POS": "NOUN"}]
        ]
        
        # Initialize matcher with patterns
        matcher = Matcher(self.nlp.vocab)
        matcher.add("ASPECT_PATTERNS", patterns)
        
        # Extract aspects using multiple strategies
        aspects = set()
        
        # 1. Extract using pattern matching
        matches = matcher(doc)
        for match_id, start, end in matches:
            span = doc[start:end]
            if 1 <= len(span) <= 4:  # Limit to 1-4 word spans
                aspect_text = span.text.lower()
                # Skip if it's just a determiner or very common word
                if len(aspect_text) > 2 and not any(t.is_stop for t in span):
                    aspects.add(aspect_text)
        
        # 2. Extract noun chunks with better filtering
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.lower()
            # Skip pronouns, single letters, and very common words
            if (chunk.root.pos_ in ["NOUN", "PROPN"] and 
                len(chunk_text) > 2 and
                not any(t.is_stop or t.is_punct for t in chunk) and
                chunk_text not in ["it", "this", "that", "they", "them"] and
                not chunk_text.replace("'", "").isnumeric()):
                aspects.add(chunk_text)
        
        # 3. Extract compound nouns and named entities
        for token in doc:
            # Handle compound nouns (e.g., "battery_life")
            if token.dep_ in ["compound", "amod", "nmod"] and token.head.pos_ == "NOUN":
                compound = f"{token.text} {token.head.text}".lower()
                if len(compound) > 4:  # Minimum length check
                    aspects.add(compound)
            
            # Add individual nouns that might have been missed
            elif (token.pos_ in ["NOUN", "PROPN"] and 
                  len(token.text) > 2 and 
                  not token.is_stop and 
                  not any(token.text in a for a in aspects)):
                aspects.add(token.text.lower())
                
        # 4. Explicitly allow aspects from our regex patterns (handling verbs like "look")
        text_lower = text.lower()
        for pattern, _ in self.aspect_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                # Add the matched text (e.g. "look", "looks", "design")
                # We use the regex match directly
                aspects.add(match.group(0))
        
        # Filter and clean aspects
        filtered_aspects = set()
        common_terms = {
            "product", "item", "thing", "something", "anything", "one", "way", 
            "time", "day", "people", "lot", "bit", "part", "kind", "sort",
            "use", "make", "take", "give", "get", "like", "see"
        }
        
        for aspect in aspects:
            # Skip if aspect is too short or in common terms
            if len(aspect) < 3:
                continue
                
            # Skip if any word is a stopword or common term
            words = aspect.split()
            if any(word in common_terms for word in words):
                continue
                
            # Skip if it's just a number
            if all(w.replace("'", "").replace("-", "").isnumeric() for w in words):
                continue
                
            filtered_aspects.add(aspect)
        
        # Group similar aspects using word embeddings for better clustering
        aspect_list = list(filtered_aspects)
        if not aspect_list:
            return []
            
        # Group by head noun
        aspect_groups = {}
        for aspect in aspect_list:
            doc = self.nlp(aspect)
            head = next((t for t in reversed(doc) if t.dep_ == "ROOT" or t.head == t), doc[-1] if doc else None)
            if head:
                head_lemma = head.lemma_
                if head_lemma not in aspect_groups:
                    aspect_groups[head_lemma] = []
                aspect_groups[head_lemma].append(aspect)
        
        # Select the most descriptive aspect from each group
        final_aspects = []
        for group in aspect_groups.values():
            if len(group) == 1:
                final_aspects.append(group[0])
            else:
                # Prefer longer, more specific phrases
                best_aspect = max(group, key=lambda x: (
                    len(x.split()),  # Prefer multi-word phrases
                    sum(1 for t in self.nlp(x) if t.pos_ in ["ADJ", "NOUN"]),  # Prefer more content words
                    -len(x) / 10  # Slight preference for shorter phrases among equals
                ))
                final_aspects.append(best_aspect)
        
        # Sort by length (longer phrases first) then alphabetically
        final_aspects.sort(key=lambda x: (-len(x.split()), x))
        
        return final_aspects
    
    def _split_at_contrast_markers(self, text: str) -> list[tuple[str, bool]]:
        """Split text at contrast markers and return chunks with their sentiment influence."""
        # Define contrast markers and their impact on following text
        contrast_markers = {
            'but': True, 'however': True, 'although': False, 'though': False,
            'yet': True, 'except': True, 'despite': False, 'whereas': True,
            'while': True, 'nevertheless': True, 'nonetheless': True,
            'on the other hand': True, 'in contrast': True, 'conversely': True,
            'that said': True, 'having said that': True, 'then again': True
        }
        
        # Convert to lowercase for case-insensitive matching
        lower_text = text.lower()
        chunks = []
        last_pos = 0
        
        # Find all contrast markers and their positions
        markers = []
        for marker in sorted(contrast_markers.keys(), key=len, reverse=True):
            pos = 0
            while True:
                pos = lower_text.find(marker, pos)
                if pos == -1:
                    break
                # Check if it's a whole word
                if (pos == 0 or not lower_text[pos-1].isalnum()) and \
                   (pos + len(marker) >= len(lower_text) or not lower_text[pos + len(marker)].isalnum()):
                    markers.append((pos, marker, contrast_markers[marker]))
                pos += len(marker)
        
        # Sort markers by position
        markers.sort()
        
        # Split text at markers
        for pos, marker, is_strong in markers:
            if pos > last_pos:
                chunk = text[last_pos:pos].strip()
                if chunk:
                    chunks.append((chunk, False))  # Text before marker
            
            # Add the marker itself
            chunks.append((text[pos:pos+len(marker)], is_strong))
            last_pos = pos + len(marker)
        
        # Add remaining text
        if last_pos < len(text):
            chunk = text[last_pos:].strip()
            if chunk:
                chunks.append((chunk, False))
        
        # If no chunks were created, return the whole text
        if not chunks:
            return [(text, False)]
            
        return chunks

    def _analyze_sentence_sentiment(self, text: str) -> dict:
        """Analyze sentiment of a single sentence and return score and label."""
        try:
            # Skip very short texts that might be just punctuation or stopwords
            if len(text.split()) <= 2 and not any(c.isalpha() for c in text):
                return {'score': 0.5, 'label': 'neutral'}
                
            result = self.sentiment_analyzer(text)[0]
            raw_score = result['score']
            label = result['label']
            
            # Convert to 0-1 scale where 0=very negative, 1=very positive
            if label == 'POSITIVE':
                base_score = raw_score
            else:
                base_score = 1 - raw_score
            
            # --- Lexicon-Based Score Adjustment ---
            # The model is often too confident (0.9 vs 0.1). We use keywords to
            # force the score into intermediate ranges (Granular Sentiment).
            
            text_lower = text.lower()
            doc = self.nlp(text_lower)
            lemmas = set(token.lemma_ for token in doc)
            
            # 1. Neutral/Weak Terms (Target: 0.45 - 0.55)
            # Phrases like "it is okay", "average", "fine", "decent" often get high positive scores
            neutral_terms = {
                'okay', 'ok', 'average', 'standard', 'fine', 'decent', 'acceptable', 
                'adequate', 'passable', 'so-so', 'fair', 'mediocre'
            }
            if any(term in text_lower.split() for term in neutral_terms) or \
               any(term in lemmas for term in neutral_terms):
               # Dampen heavily towards neutral
               base_score = 0.5 + (base_score - 0.5) * 0.3
            
            # 2. Moderate Negative (Target: 0.20 - 0.35)
            # Words that mean "bad" but not "terrible"
            mod_neg_terms = {
                'disappointed', 'disappointing', 'poor', 'bad', 'issue', 'issues',
                'problem', 'problems', 'weak', 'slow', 'confusing', 'hard', 'difficult',
                'weird', 'strange', 'annoying', 'below expectations', 'unimpressed',
                'lacking', 'flawed'
            }
            has_mod_neg = any(term in text_lower for term in mod_neg_terms) or \
                          any(term in lemmas for term in mod_neg_terms)
                          
            # 3. Strong Negative (Target: 0.0 - 0.15)
            strong_neg_terms = {
                'terrible', 'awful', 'horrible', 'worst', 'garbage', 'trash', 'useless',
                'hate', 'pathetic', 'disaster', 'nightmare', 'broken', 'dead'
            }
            has_strong_neg = any(term in text_lower for term in strong_neg_terms)
            
            # 4. Moderate Positive (Target: 0.65 - 0.80)
            mod_pos_terms = {
                'good', 'nice', 'cool', 'happy', 'satisfied', 'useful', 'helpful',
                'solid', 'smooth', 'clean', 'pretty good', 'fun', 'worth'
            }
            has_mod_pos = any(term in text_lower for term in mod_pos_terms) or \
                          any(term in lemmas for term in mod_pos_terms)

            # 5. Strong Positive (Target: 0.85 - 1.0)
            strong_pos_terms = {
                'excellent', 'amazing', 'perfect', 'awesome', 'fantastic', 'superb',
                'best', 'outstanding', 'brilliant', 'incredible', 'love', 'beautiful',
                'gorgeous', 'masterpiece'
            }
            has_strong_pos = any(term in text_lower for term in strong_pos_terms)

            # --- Logic to Override Model Confidence ---
            
            if has_strong_neg:
                # Force into 0.0 - 0.15 range
                base_score = min(0.15, base_score)
                base_score = min(base_score, 0.1) # Push lower
            
            elif has_mod_neg:
                # Force into 0.20 - 0.35 range
                # If model thinks it's positive (e.g. "Below expectations" -> 0.9), flip it
                if base_score > 0.5:
                    base_score = 0.3 # Flip to negative
                
                # Clamp to range
                base_score = max(0.2, min(0.35, base_score))
                
            elif has_strong_pos:
                # Force into 0.85 - 1.0 range
                base_score = max(0.85, base_score)
                
            elif has_mod_pos:
                # Force into 0.65 - 0.80 range
                # If model is super confident (0.99), pull it down
                base_score = min(0.80, max(0.65, base_score))
                
                # If model thought it was negative, flip it (unlikely for these words but possible)
                if base_score < 0.5:
                    base_score = 0.65

            # Intensity Modifiers
            intensifiers = {'very', 'extremely', 'really', 'absolutely', 'completely', 'totally', 'highly'}
            weakeners = {'slightly', 'somewhat', 'mostly', 'kind of', 'sort of', 'a bit', 'little'}
            
            has_intr = any(t in text_lower.split() for t in intensifiers)
            has_weak = any(t in text_lower.split() for t in weakeners)
            
            if has_weak:
                # Pull scores towards neutral (0.5)
                if base_score > 0.5:
                    base_score = max(0.55, base_score - 0.15) # e.g. 0.7 -> 0.55
                else:
                    base_score = min(0.45, base_score + 0.15) # e.g. 0.3 -> 0.45
            
            if has_intr:
                 # Push scores away from neutral
                if base_score > 0.5:
                    base_score = min(0.98, base_score + 0.1)
                else:
                    base_score = max(0.02, base_score - 0.1)

            return {'score': base_score, 'label': label.lower()}
            
        except Exception as e:
            self.logger.warning(f"Error analyzing sentence '{text}': {str(e)}")
            return {'score': 0.5, 'label': 'neutral'}

    def _detect_sentiment_shift(self, doc) -> tuple[list, float, float]:
        """
        Detect sentiment shift within a document.
        
        Returns:
            tuple: (sentiment_changes, avg_shift, shift_ratio)
                - sentiment_changes: List of (sentence, score, label, shift_from_previous)
                - avg_shift: Average magnitude of sentiment shifts
                - shift_ratio: Ratio of sentences with significant shifts
        """
        if len(doc) < 2:
            return [("", 0.5, 'neutral', 0.0)], 0.0, 0.0
            
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 0]
        if not sentences:
            return [("", 0.5, 'neutral', 0.0)], 0.0, 0.0
            
        # Analyze each sentence
        sentiment_data = []
        prev_score = None
        shift_magnitude = 0.0
        shift_count = 0
        
        for i, sent in enumerate(sentences):
            score, label = self._analyze_sentence_sentiment(sent)
            shift = abs(score - prev_score) if prev_score is not None else 0.0
            
            # Check for significant shift
            if prev_score is not None and shift > 0.3:  # Threshold for significant shift
                shift_magnitude += shift
                shift_count += 1
                
            sentiment_data.append((sent, score, label, shift))
            prev_score = score
        
        # Calculate metrics
        avg_shift = (shift_magnitude / shift_count) if shift_count > 0 else 0.0
        shift_ratio = shift_count / (len(sentences) - 1) if len(sentences) > 1 else 0.0
        
        return sentiment_data, avg_shift, shift_ratio

    def _summarize_sentiment_chunks(self, chunks: list[tuple[str, bool]]) -> tuple[float, list]:
        """
        Analyze and summarize sentiment across text chunks.
        
        Args:
            chunks: List of (text, is_after_contrast) tuples
            
        Returns:
            tuple: (final_score, analysis_results)
                - final_score: Combined sentiment score (0-1)
                - analysis_results: List of (text, score, label, weight)
        """
        if not chunks:
            return 0.5, []
            
        results = []
        total_weight = 0
        weighted_sum = 0
        
        # First pass: analyze all chunks
        for i, (chunk, is_contrast) in enumerate(chunks):
            # Skip very short chunks that are just contrast markers
            if len(chunk.split()) <= 2 and not any(c.isalpha() for c in chunk):
                continue
                
            # Analyze chunk sentiment
            sentiment_result = self._analyze_sentence_sentiment(chunk)
            score = sentiment_result['score']
            label = sentiment_result['label']
            
            # Weight chunks after contrast markers more heavily
            weight = 2.0 if is_contrast else 1.0
            
            # Slightly increase weight of negative sentiments (negativity bias)
            if score < 0.4:
                weight *= 1.2
                
            results.append({
                'text': chunk,
                'score': score,
                'label': label,
                'weight': weight,
                'is_contrast': is_contrast
            })
            
            weighted_sum += score * weight
            total_weight += weight
        
        if not results:
            return 0.5, []
            
        # Calculate weighted average score
        final_score = weighted_sum / total_weight if total_weight > 0 else 0.5
        
        return final_score, results

    def _analyze_sentiment_context(self, text: str) -> tuple[float, float, list]:
        """
        Analyze sentiment with context awareness and contrast handling.
        
        Returns:
            tuple: (base_score, shift_ratio, sentiment_changes)
                - base_score: Overall sentiment score (0-1)
                - shift_ratio: Ratio of sentences with significant shifts
                - sentiment_changes: List of (sentence, score, label, shift, weight)
        """
        # Split text into chunks at contrast markers
        chunks = self._split_at_contrast_markers(text)
        
        # Analyze each chunk with proper weighting
        base_score, chunk_analysis = self._summarize_sentiment_chunks(chunks)
        
        # Convert to sentiment changes format
        sentiment_changes = []
        shift_count = 0
        prev_score = None
        
        for i, chunk in enumerate(chunk_analysis):
            score = chunk['score']
            shift = abs(score - prev_score) if prev_score is not None else 0.0
            
            # Check for significant shift
            if prev_score is not None and shift > 0.3:
                shift_count += 1
                
            sentiment_changes.append((
                chunk['text'],
                score,
                chunk['label'],
                shift,
                chunk['weight']
            ))
            prev_score = score
        
        # Calculate shift ratio
        shift_ratio = shift_count / len(chunk_analysis) if chunk_analysis else 0.0
        
        # If there are significant shifts, adjust final score
        if shift_ratio > 0.3:
            # Give more weight to negative chunks
            negative_weight = sum(
                (1 - chunk['score']) * chunk['weight'] 
                for chunk in chunk_analysis 
                if chunk['score'] < 0.4
            )
            positive_weight = sum(
                chunk['score'] * chunk['weight'] 
                for chunk in chunk_analysis 
                if chunk['score'] > 0.6
            )
            
            if negative_weight > positive_weight * 1.5:
                base_score = min(base_score, 0.4)  # Cap at slightly negative
            
        return base_score, shift_ratio, sentiment_changes

    def _analyze_clause_sentiment(self, clause: str, aspect: str = None) -> Dict:
        """
        Analyze sentiment of a single clause with enhanced context awareness and contrast handling.
        
        Args:
            clause: The text clause to analyze
            aspect: Optional aspect to focus the analysis on
            
        Returns:
            Dictionary with sentiment analysis results including:
            - score: Sentiment score (0-1)
            - label: Sentiment label (positive/negative/neutral/mixed)
            - confidence: Confidence in the prediction (0-1)
            - aspects: If aspect is provided, aspect-specific sentiment
            - sentiment_changes: Detailed sentiment analysis of chunks
        """
        try:
            if not clause.strip():
                return {'score': 0.5, 'label': 'neutral', 'confidence': 0.0, 'text': ''}
            
            # Preprocess the clause
            clause = clause.strip()
            
            # If analyzing a specific aspect, add context
            if aspect:
                # Add aspect context to help the model focus
                clause = f"When considering {aspect}, {clause}"
            
            # Sentiment analysis with context and contrast handling
            base_score, shift_ratio, sentiment_changes = self._analyze_sentiment_context(clause)
            
            # Calculate confidence based on sentiment strength and consistency
            confidence = 0.8  # Base confidence
            
            # Adjust confidence based on sentiment shifts
            if shift_ratio > 0.3:  # High shift ratio indicates mixed sentiment
                confidence *= (1.0 - (shift_ratio * 0.7))  # Reduce confidence for mixed sentiment
                
            # Adjust confidence based on sentiment strength
            sentiment_strength = abs(base_score - 0.5) * 2  # 0-1, higher is stronger
            confidence = max(0.1, min(0.95, confidence * (0.7 + (sentiment_strength * 0.3))))
            
            # Determine final label
            if shift_ratio > 0.4 and len(sentiment_changes) > 1:
                # If significant shifts, label as mixed sentiment
                label = 'mixed'
                # For mixed sentiment, move score towards neutral based on shift ratio
                base_score = 0.5 + ((base_score - 0.5) * (1.0 - (shift_ratio * 0.7)))
            elif base_score > 0.6:
                label = 'positive'
            elif base_score < 0.4:
                label = 'negative'
            else:
                label = 'neutral'
            
            # Prepare result
            result = {
                'score': float(base_score),
                'label': label,
                'confidence': float(confidence),
                'text': clause,
                'sentiment_changes': [
                    {
                        'text': chunk[0],
                        'score': float(chunk[1]),
                        'label': chunk[2],
                        'shift': float(chunk[3]),
                        'weight': float(chunk[4]) if len(chunk) > 4 else 1.0
                    }
                    for chunk in sentiment_changes
                ]
            }
            
            # Add aspect-specific information if needed
            if aspect:
                result['aspect'] = aspect
                result['aspect_score'] = base_score
                
            return result
            
        except Exception as e:
            self.logger.warning(f"Error analyzing clause: {str(e)}")
            return {'score': 0.5, 'label': 'neutral', 'confidence': 0.0, 'text': clause.strip()}
    
    def _get_aspect_weight(self, aspect: str) -> float:
        """
        Get importance weight for an aspect.
        
        Args:
            aspect: The aspect to get weight for
            
        Returns:
            Weight value between 0.5 and 1.5
        """
        # Default weight
        weight = 1.0
        
        # Important aspects get higher weight
        important_aspects = {
            'quality', 'price', 'performance', 'battery', 'camera',
            'screen', 'display', 'sound', 'audio', 'value'
        }
        
        # Check if aspect contains any important keywords
        aspect_lower = aspect.lower()
        if any(keyword in aspect_lower for keyword in important_aspects):
            weight = 1.3
        
        return weight

    def analyze_sentiment(self, text: str, aspect: str = None) -> Dict:
        """
        Analyze sentiment of the given text with clause-based analysis.
        
        Args:
            text: The text to analyze
            aspect: Optional aspect to focus the analysis on
            
        Returns:
            Dictionary with sentiment analysis results
        """
        try:
            doc = self.nlp(text)
            
            # If aspect is provided, find all its occurrences
            if aspect:
                aspect_tokens = [t.lower() for t in aspect.split()]
                aspect_occurrences = []
                
                # Find all occurrences of the aspect in the text
                for i in range(len(doc) - len(aspect_tokens) + 1):
                    if all(doc[i + j].text.lower() == aspect_tokens[j] for j in range(len(aspect_tokens))):
                        aspect_occurrences.append(i)
                
                if not aspect_occurrences:
                    return {
                        'label': 'neutral',
                        'score': 0.5,
                        'confidence': 0.0,
                        'aspect': aspect,
                        'clauses_analyzed': 0
                    }
                
                # Analyze each occurrence with context
                aspect_scores = []
                
                for pos in aspect_occurrences:
                    # Get context window and analyze
                    context_span, contrast_data, _ = self._get_context_window(doc, pos)
                    context_text = ' '.join([token.text for token in context_span])
                    
                    try:
                        # Get base sentiment of the context
                        result = self.sentiment_analyzer(context_text)[0]
                        base_score = float(result['score'])
                        
                        # Normalize score (Handle model output which is 0.5-1.0 confidence)
                        if result['label'] == 'NEGATIVE':
                             base_score = 1.0 - base_score
                        
                        # Adjust for negation in context
                        negation_terms = {'not', 'no', 'never', "n't", 'hardly', 'barely'}
                        has_negation = any(t.text.lower() in negation_terms for t in context_span)
                        
                        if has_negation:
                            # Invert sentiment if negation is present
                            # But only if the model didn't catch it (simple heuristic)
                            # Actually, text-classification models usually catch negation.
                            # But sometimes with aspect slicing it misses.
                            pass # Rely on model for now, but ensure score is normalized
                        
                        adjusted_score = base_score
                        
                        # Check for contrast ("good but expensive")
                        # If contrast is present, reduce confidence slightly as it's mixed
                        if contrast_data: # contrast_data is is_contrastive boolean
                             # If we are in the "but ..." part, we might want to boost importance?
                             # For now, just pass the score through.
                             pass
                        
                        # Calculate confidence based on score distance from neutral and context length
                        confidence = abs(adjusted_score - 0.5) * 2
                        context_length = len(context_text.split())
                        
                        # Adjust confidence based on context length
                        if context_length < 5:
                            confidence *= 0.7
                        
                        aspect_scores.append({
                            'score': adjusted_score,
                            'confidence': confidence,
                            'context': context_text,
                            'position': pos,
                            'base_score': base_score
                        })
                        
                    except Exception as e:
                        self.logger.warning(f"Error analyzing aspect '{aspect}' at position {pos}: {str(e)}")
                        continue
                
                if not aspect_scores:
                    return {
                        'label': 'neutral',
                        'score': 0.5,
                        'confidence': 0.0,
                        'aspect': aspect,
                        'clauses_analyzed': 0
                    }
                
                # Calculate weighted average score
                total_weight = sum(s['confidence'] ** 2 for s in aspect_scores)
                if total_weight > 0:
                    avg_score = sum(s['score'] * (s['confidence'] ** 2) for s in aspect_scores) / total_weight
                else:
                    avg_score = sum(s['score'] for s in aspect_scores) / len(aspect_scores)
                
                # Calculate average confidence
                avg_confidence = sum(s['confidence'] for s in aspect_scores) / len(aspect_scores)
                
                # Determine label
                if avg_score > 0.6:
                    label = 'positive'
                elif avg_score < 0.4:
                    label = 'negative'
                else:
                    label = 'neutral'
                
                self.logger.info(
                    f"Aspect Analysis | '{aspect}' | "
                    f"Score: {avg_score:.3f} ({label}) | "
                    f"Confidence: {avg_confidence:.2f} | "
                    f"Occurrences: {len(aspect_scores)}"
                )
                
                return {
                    'label': label,
                    'score': avg_score,
                    'confidence': avg_confidence,
                    'aspect': aspect,
                    'clauses_analyzed': len(aspect_scores)
                }
            
            # For general sentiment (no specific aspect)
            # Enhanced clause splitting that preserves contrastive relationships
            clauses = []
            current_clause = []
            contrast_indicators = ['but', 'however', 'although', 'though', 'yet', 'except', 'despite', 'whereas', 'while', 'nevertheless', 'nonetheless']
            
            # First, split into sentences
            for sent in doc.sents:
                sent_text = sent.text
                
                # Check for contrastive conjunctions
                has_contrast = any(conj in sent_text.lower() for conj in contrast_indicators)
                
                if has_contrast:
                    # For contrastive sentences, split into segments
                    segments = []
                    current_segment = []
                    
                    for token in sent:
                        if token.text.lower() in contrast_indicators:
                            if current_segment:
                                segments.append(' '.join(current_segment).strip())
                                current_segment = []
                            segments.append(token.text)  # Keep the contrast word as a separate segment
                        else:
                            current_segment.append(token.text)
                    
                    if current_segment:  # Add the last segment
                        segments.append(' '.join(current_segment).strip())
                    
                    # Process segments, giving more weight to segments after contrast words
                    for i, segment in enumerate(segments):
                        if segment.lower() in contrast_indicators:
                            continue  # Skip the contrast word itself
                            
                        # Segments after contrast words get higher weight
                        weight = 1.5 if i > 0 and segments[i-1].lower() in contrast_indicators else 1.0
                        clauses.append({'text': segment, 'weight': weight, 'is_after_contrast': weight > 1.0})
                else:
                    # For non-contrastive sentences, add as is with normal weight
                    clauses.append({'text': sent_text, 'weight': 1.0, 'is_after_contrast': False})
            
            # Analyze each clause with context awareness
            clause_results = []
            for i, clause_info in enumerate(clauses):
                clause = clause_info['text']
                if not clause.strip():
                    continue
                
                # Analyze the clause
                result = self._analyze_clause_sentiment(clause)
                
                # Adjust confidence based on position (clauses after contrast get higher confidence)
                if clause_info['is_after_contrast']:
                    result['confidence'] = min(1.0, result['confidence'] * 1.3)  # 30% boost
                
                # Only include clauses with sufficient confidence
                if result['confidence'] > 0.1:
                    result['weight'] = clause_info['weight']
                    clause_results.append(result)
            
            if not clause_results:
                # Fallback to full text analysis if no clauses were confident
                result = self.sentiment_analyzer(text)[0]
                overall_score = float(result['score'])
                label = 'positive' if overall_score > 0.6 else 'negative' if overall_score < 0.4 else 'neutral'
                return {
                    'label': label,
                    'score': overall_score,
                    'confidence': abs(overall_score - 0.5) * 2,
                    'aspect': 'overall',
                    'clauses_analyzed': 1
                }
            
            # Calculate sentiment strengths for mixed sentiment detection
            positive_scores = []
            negative_scores = []
            
            for result in clause_results:
                score = result['score']
                weight = result['weight'] * result['confidence']
                
                if score > 0.5:
                    positive_scores.append((score, weight))
                elif score < 0.5:
                    negative_scores.append((1 - score, weight))
            
            pos_strength = sum(s * w for s, w in positive_scores) / sum(w for _, w in positive_scores) if positive_scores else 0
            neg_strength = sum(s * w for s, w in negative_scores) / sum(w for _, w in negative_scores) if negative_scores else 0
            
            # Calculate weighted average score directly for accuracy
            total_weighted_score = sum(r['score'] * r['weight'] * r['confidence'] for r in clause_results)
            total_weight = sum(r['weight'] * r['confidence'] for r in clause_results)
            
            if total_weight > 0:
                overall_score = total_weighted_score / total_weight
            else:
                overall_score = 0.5
                
            # Additional check: If meaningful clauses exist but total_weight is somehow 0
            if total_weight == 0 and clause_results:
                 overall_score = sum(r['score'] for r in clause_results) / len(clause_results)
            
            # Calculate average confidence
            avg_confidence = sum(r['confidence'] * r['weight'] for r in clause_results) / sum(r['weight'] for r in clause_results)
            
            # Determine overall label with more nuanced thresholds
            if overall_score > 0.7:
                label = 'positive'
            elif overall_score < 0.3:
                label = 'negative'
            else:
                if pos_strength > 0 and neg_strength > 0:
                    if abs(pos_strength - neg_strength) < 0.2:  # If strengths are close
                        label = 'mixed'
                    else:
                        label = 'slightly positive' if overall_score > 0.5 else 'slightly negative'
                else:
                    label = 'neutral'
                
            return {
                'label': label,
                'score': float(overall_score),
                'confidence': float(avg_confidence),
                'aspect': 'overall',
                'clauses_analyzed': len(clause_results)
            }
            
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {str(e)}")
            return {'label': 'neutral', 'score': 0.5}
    
    def map_aspects_to_categories(self, aspects: List[str]) -> Dict[str, List[str]]:
        """Map extracted aspects to predefined categories."""
        from config import ASPECT_KEYWORDS
        
        aspect_categories = defaultdict(list)
        
        for aspect in aspects:
            aspect_lower = aspect.lower()
            matched = False
            
            for category, keywords in ASPECT_KEYWORDS.items():
                if any(keyword in aspect_lower for keyword in keywords):
                    aspect_categories[category].append(aspect)
                    matched = True
                    break
            
            if not matched:
                # If no category matches, create a new one with the aspect name
                aspect_categories[aspect] = [aspect]
        
        return dict(aspect_categories)
    
    def _get_context_window(self, doc, index: int, window_size: int = 5) -> tuple:
        """Get the context window around a token index."""
        start = max(0, index - window_size)
        end = min(len(doc), index + window_size + 1)
        span = doc[start:end]
        
        # Check for contrastive terms in the window
        contrast_terms = {'but', 'however', 'although', 'though', 'yet', 'except', 'despite'}
        is_contrastive = any(token.text.lower() in contrast_terms for token in span)
        
        return span, is_contrastive, None

    def analyze_review(self, review_text: str) -> Dict:
        """
        Analyze a product review with enhanced context-aware sentiment analysis.
        
        Args:
            review_text: The review text to analyze
            
        Returns:
            Dictionary containing analysis results with sentiment, aspects, and key phrases
            {
                'review': str,
                'overall_sentiment': {'label': str, 'score': float, 'confidence': float},
                'aspects': [{
                    'aspect': str, 
                    'sentiment': str, 
                    'score': float,
                    'weight': float,
                    'confidence': float,
                    'occurrences': int
                }],
                'key_phrases': List[str],
                'summary': str,
                'contextual_sentences': List[dict]
            }
        """
        if not review_text.strip():
            return {
                'review': '',
                'overall_sentiment': {'label': 'neutral', 'score': 0.5, 'confidence': 0.0},
                'aspects': [],
                'key_phrases': [],
                'summary': 'No review text provided.',
                'contextual_sentences': []
            }
            
        try:
            # Preprocess the review text
            preprocessed_text = self.preprocess_text(review_text)
            doc = self.nlp(review_text)
            
            # Extract aspects with context
            # Use raw review_text to preserve sentence structure for SpaCy
            aspects = self.extract_aspects(review_text)
            
            # If no aspects found, use the whole review for sentiment
            if not aspects:
                sentiment = self.analyze_sentiment(review_text)
                granular_label = self.get_granular_sentiment_label(float(sentiment['score']))
                # Calculate intensity
                intensity = abs(float(sentiment['score']) - 0.5) * 2
                
                # Detect humor/contradictions even if no aspects
                humor_analysis = self.detect_humor(review_text)
                contradiction_analysis = self.detect_contradictions(review_text)
                
                return {
                    'review': review_text,
                    'overall_sentiment': {
                        'label': sentiment['label'],
                        'granular_label': granular_label,
                        'score': float(sentiment['score']),
                        'confidence': float(sentiment.get('confidence', 0.0)),
                        'intensity': intensity
                    },
                    'humor_analysis': humor_analysis,
                    'contradiction_analysis': contradiction_analysis,
                    'aspects': [],
                    'key_phrases': self.extract_key_phrases(review_text),
                    'summary': 'General review (no specific aspects found).',
                    'contextual_sentences': []
                }
            
            # Analyze sentiment for each aspect with context
            aspect_sentiments = {}
            contextual_sentences = []
            
            for aspect in aspects:
                # Get aspect weight based on importance
                aspect_weight = self._get_aspect_weight(aspect)
                
                # Analyze sentiment with the enhanced analyzer
                sentiment_result = self.analyze_sentiment(review_text, aspect)
                
                # Store the result with additional metadata
                aspect_sentiments[aspect] = {
                    'label': sentiment_result['label'],
                    'score': float(sentiment_result['score']),
                    'weight': aspect_weight,
                    'confidence': float(sentiment_result.get('confidence', 0.0)),
                    'occurrences': review_text.lower().count(aspect.lower())
                }
                
                # Store context for debugging/insights
                try:
                    aspect_tokens = aspect.lower().split()
                    for i in range(len(doc) - len(aspect_tokens) + 1):
                        if all(doc[i + j].text.lower() == aspect_tokens[j] for j in range(len(aspect_tokens))):
                            context_span, is_contrastive, _ = self._get_context_window(doc, i, window_size=7)
                            contextual_sentences.append({
                                'aspect': aspect,
                                'sentence': ' '.join([token.text for token in context_span]),
                                'sentiment': sentiment_result,
                                'position': i,
                                'is_contrastive': is_contrastive
                            })
                except Exception as e:
                    pass # Ignore context extraction errors to prevent failure
            
            # Generate key phrases and summary
            key_phrases = self.extract_key_phrases(review_text)
            summary = self.generate_summary(review_text, aspect_sentiments)
            
            # Calculate weighted overall sentiment
            weighted_scores = []
            total_weight = 0
            confidences = []
            
            for aspect, data in aspect_sentiments.items():
                current_weight = data['weight']
                current_score = data['score']
                
                # Negativity Bias: Negative aspects often weigh more heavily on user satisfaction
                if current_score < 0.45:
                    current_weight *= 1.2
                
                # Extreme Bias: Very strong opinions (positive or negative) matter more
                if abs(current_score - 0.5) > 0.4:
                    current_weight *= 1.1
                    
                # Normalize score to [-0.5, 0.5] range for weighted average
                normalized_score = (current_score - 0.5) * current_weight
                weighted_scores.append(normalized_score)
                total_weight += current_weight
                confidences.append(data['confidence'])
            
            # Calculate final scores
            if total_weight > 0:
                final_score = (sum(weighted_scores) / total_weight) + 0.5  # Back to [0, 1] range
                final_score = max(0.0, min(1.0, final_score))  # Clamp to [0, 1]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                
                # NLP models often struggle with extremely short reviews ("Good", "Nice product")
                # and assign them low confidence or neutral scores.
                # In Indian e-commerce, these are overwhelmingly positive (usually 5-star).
                # We apply a heuristic bump for short, positive-leaning inputs to fix the skewed rating mathematical averages.
                words_in_review = list(filter(None, re.split(r'\W+', review_text.lower())))
                if len(words_in_review) <= 5:
                    positive_keywords = {"good", "nice", "awesome", "excellent", "super", "superb", "best", "great", "loved", "amazing", "perfect"}
                    if any(word in positive_keywords for word in words_in_review):
                        # Force a high positive score
                        final_score = max(final_score, 0.85)
                        avg_confidence = max(avg_confidence, 0.8)
                
                if final_score > 0.6:
                    label = 'positive'
                elif final_score < 0.4:
                    label = 'negative'
                else:
                    label = 'neutral'
                    
                overall_sentiment = {
                    'label': label,
                    'score': float(final_score),
                    'confidence': float(avg_confidence)
                }
            else:
                # Fallback to simple analysis if weighting fails (e.g. no aspects found)
                overall_sentiment = self.analyze_sentiment(review_text)
                
                # Apply the same short-review heuristic to the fallback
                words_in_review = list(filter(None, re.split(r'\W+', review_text.lower())))
                if len(words_in_review) <= 5:
                    positive_keywords = {"good", "nice", "awesome", "excellent", "super", "superb", "best", "great", "loved", "amazing", "perfect"}
                    if any(word in positive_keywords for word in words_in_review):
                        overall_sentiment['score'] = max(overall_sentiment.get('score', 0), 0.85)
                        overall_sentiment['label'] = 'positive'
            
            # Format aspect results
            aspect_results = []
            for aspect, data in aspect_sentiments.items():
                aspect_results.append({
                    'aspect': aspect,
                    'sentiment': data['label'],
                    'granular_sentiment': self.get_granular_sentiment_label(data['score']),
                    'score': data['score'],
                    'weight': data['weight'],
                    'confidence': data['confidence'],
                    'occurrences': data['occurrences']
                })
            
            # Sort aspects by importance (weight * score magnitude)
            aspect_results.sort(
                key=lambda x: abs(x['score'] - 0.5) * x['weight'], 
                reverse=True
            )
            
            # Detect humor and sarcasm
            humor_analysis = self.detect_humor(review_text)
            
            # Detect contradictions
            contradiction_analysis = self.detect_contradictions(review_text)
            
            # Get granular sentiment label
            granular_label = self.get_granular_sentiment_label(overall_sentiment['score'])
            overall_sentiment['granular_label'] = granular_label
            
            # Calculate sentiment intensity (how far from neutral)
            sentiment_intensity = abs(overall_sentiment['score'] - 0.5) * 2  # 0-1 scale
            overall_sentiment['intensity'] = float(sentiment_intensity)
            
            return {
                'review': review_text,
                'overall_sentiment': overall_sentiment,
                'aspects': aspect_results,
                'key_phrases': key_phrases,
                'summary': summary,
                'contextual_sentences': contextual_sentences[:5],  # For debugging
                'humor_analysis': humor_analysis,
                'contradiction_analysis': contradiction_analysis
            }

            
        except Exception as e:
            self.logger.error(f"Error analyzing review: {str(e)}")
            # Fallback to simple analysis if something goes wrong
            sentiment = self.analyze_sentiment(review_text)
            return {
                'review': review_text,
                'overall_sentiment': {
                    'label': sentiment['label'],
                    'score': float(sentiment['score']),
                    'confidence': float(sentiment.get('confidence', 0.0))
                },
                'aspects': [],
                'key_phrases': [],
                'summary': 'Analysis incomplete due to an error.',
                'contextual_sentences': []
            }
    
    def extract_key_phrases(self, text: str) -> list[str]:
        """Extract key phrases from text."""
        try:
            doc = self.nlp(text)
            phrases = []
            for chunk in doc.noun_chunks:
                if len(chunk.text) > 3 and not chunk.root.is_stop:
                    phrases.append(chunk.text)
            return list(set(phrases))[:10]
        except Exception as e:
            self.logger.warning(f"Error extracting key phrases: {str(e)}")
            return []
    
    def generate_summary(self, text: str, aspect_sentiments: dict) -> str:
        """Generate a summary of the review."""
        try:
            if not aspect_sentiments:
                return "General review with mixed sentiments."
            
            positive = sum(1 for data in aspect_sentiments.values() if data['score'] > 0.6)
            negative = sum(1 for data in aspect_sentiments.values() if data['score'] < 0.4)
            
            if positive > negative:
                return f"Mostly positive review mentioning {positive} favorable aspects."
            elif negative > positive:
                return f"Mostly negative review with {negative} critical aspects."
            return "Mixed review with balanced feedback."
        except Exception as e:
            self.logger.warning(f"Error generating summary: {str(e)}")
            return "Review analyzed successfully."

    def analyze_batch(self, reviews: List[Dict]) -> List[Dict]:
        """
        Analyze a batch of reviews efficiently.
        Args:
            reviews: List of review dictionaries containing 'text' and other metadata
        Returns:
            List of dictionaries with original metadata plus analysis results
        """
        results = []
        for review in reviews:
            text = review.get('text', '')
            if text:
                try:
                    analysis = self.analyze_review(text)
                    combined = {**review, 'analysis': analysis}
                    results.append(combined)
                except Exception as e:
                    self.logger.error(f"Error analyzing batch item: {str(e)}")
        return results

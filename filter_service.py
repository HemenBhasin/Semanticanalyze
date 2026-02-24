import re
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class FilterService:
    def __init__(self):
        # Common spam/bot patterns (overly generic, repetitive characters, etc.)
        self.spam_patterns = [
            r"^(good|nice|awesome|bad|worst|excelent|excellent)$",  # Single word low-effort
            r"(.)\1{4,}",  # Repeated characters like "wooooow" or "awfulllll"
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", # Links in reviews
            r"buy this.*whatsapp", # Scam linking
            r"^[A-Za-z0-9]{15,}$" # Random keyboard smash
        ]
        
    def filter_reviews(self, reviews: List[Dict]) -> tuple[List[Dict], int]:
        """
        Filters out exact duplicates and severe bot spam.
        Crucially, it KEEPS short/one-word reviews (like "Good", "Nice") so their 
        sentiment and star rating still counts towards the mathematical unified score.
        However, it flags them with `is_short_or_generic=True` so they can be 
        hidden from the physical UI dashboard to avoid clutter.
        """
        filtered_reviews = []
        seen_texts = set()
        spam_regexes = [re.compile(pattern, re.IGNORECASE) for pattern in self.spam_patterns[2:]] # Only use actual spam patterns, not the "good/nice" pattern
        
        initial_count = len(reviews)
        
        for review in reviews:
            text = review.get('text', '').strip()
            review['is_short_or_generic'] = False
            
            # Check for exact duplicates
            text_lower = text.lower()
            if text_lower in seen_texts:
                continue
                
            # Check against severe spam regex patterns (links, keyboard smashes)
            is_spam = False
            for regex in spam_regexes:
                if regex.search(text):
                    is_spam = True
                    break
            
            if is_spam:
                continue
                
            # Flag short/generic reviews for UI omission (but KEEP them for scoring)
            words = text.split()
            generic_pattern = re.compile(r"^(good|nice|awesome|bad|worst|excelent|excellent|super|superb|best)$", re.IGNORECASE)
            
            if len(words) < 3 or generic_pattern.search(text) or re.search(r"(.)\1{4,}", text):
                review['is_short_or_generic'] = True
                
            # Add to the clean list
            seen_texts.add(text_lower)
            filtered_reviews.append(review)
            
        num_filtered_out = initial_count - len(filtered_reviews)
        logger.info(f"Dropped {num_filtered_out} severe spam/duplicates. Kept {len(filtered_reviews)} reviews (with {sum(1 for r in filtered_reviews if r['is_short_or_generic'])} flagged as short/generic).")
        
        return filtered_reviews, num_filtered_out

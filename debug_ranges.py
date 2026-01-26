from semantic_analyzer import analyzer
import json

# Specific cases user mentioned are failing
test_cases = [
    # Positive Ranges
    ("Excellent, impressive", "Highly Positive (80-90%)"),
    ("Great, satisfied", "Moderately Positive (70-80%)"),
    ("Good, minor issues", "Slightly Positive (60-70%)"),
    
    # Negative Ranges
    ("Mostly negative", "Leaning Negative (40-50%)"), # This one is tricky as "mostly negative" is a meta-comment
    ("Below expectations", "Slightly Negative (30-40%)"),
    ("Disappointed", "Moderately Negative (20-30%)"), 
    ("Very poor", "Highly Negative (10-20%)"),
    
    # Control cases
    ("It is okay", "Neutral"),
    ("Terrible", "Very Negative"),
    ("Perfect", "Very Positive")
]

print(f"{'Text':<30} | {'Expected':<30} | {'Actual Label':<20} | {'Score':<8} | {'Raw Model'}")
print("-" * 110)

analyzer._initialize_aspect_patterns() # Ensure initialized

for text, expected in test_cases:
    # We want to see the score coming out of analyze_review because that's what the UI uses
    result = analyzer.analyze_review(text)
    score = result['overall_sentiment']['score']
    label = result['overall_sentiment']['granular_label']
    
    # Also inspect the inner sentence analysis to see the raw model score if possible
    # We'll re-run the internal method to debug
    inner = analyzer._analyze_sentence_sentiment(text)
    
    print(f"{text:<30} | {expected:<30} | {label:<20} | {score*100:5.1f}%  | {inner['score']*100:5.1f}%")

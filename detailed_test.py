"""Test to see actual score values"""
from semantic_analyzer import analyzer
import json

test_reviews = [
    ("Absolutely terrible!", "Very Negative"),
    ("Bad product, disappointed", "Moderately Negative"),
    ("It's okay, nothing special", "Neutral"),
    ("Good product, happy with it", "Moderately Positive"),
    ("Excellent! Amazing quality!", "Very Positive"),
    ("This is just weird.", "Negative/Neutral"), # No obvious aspect
    ("Just no.", "Very Negative")
]

print("Score Distribution Test")
print("=" * 80)

for text, expected in test_reviews:
    result = analyzer.analyze_review(text)
    score = result['overall_sentiment']['score']
    label = result['overall_sentiment'].get('granular_label', 'NOT SET')
    
    print(f"\nText: {text}")
    print(f"  Expected: {expected}")
    print(f"  Score: {score:.4f} ({score*100:.1f}%)")
    print(f"  Granular Label: {label}")

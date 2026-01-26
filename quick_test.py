"""Quick test to check current score distribution"""
from semantic_analyzer import analyzer

tests = [
    "Terrible!",
    "Bad product", 
    "It's okay",
    "Good product",
    "Excellent!"
]

for text in tests:
    result = analyzer.analyze_review(text)
    score = result['overall_sentiment']['score']
    label = result['overall_sentiment'].get('granular_label', 'Unknown')
    print(f"{score:.3f} - {label:25s} - {text}")

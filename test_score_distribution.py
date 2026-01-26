"""
Test script to verify sentiment score distribution across all ranges.
"""

from semantic_analyzer import analyzer

# Test cases designed to hit different sentiment ranges
test_cases = [
    ("Absolutely terrible! Worst product ever! Complete garbage!", "Very Negative", 0.0, 0.1),
    ("Very bad quality. Highly disappointed. Poor performance.", "Highly Negative", 0.1, 0.2),
    ("Disappointed overall. Not worth the money. Below expectations.", "Moderately Negative", 0.2, 0.3),
    ("It's okay but not great. Some issues. Could be better.", "Slightly Negative", 0.3, 0.4),
    ("Average product. Nothing special. Some good, some bad.", "Leaning Negative", 0.4, 0.5),
    ("Decent product overall. Generally satisfied. Fair value.", "Leaning Positive", 0.5, 0.6),
    ("Pretty good. Happy with purchase. Minor issues.", "Slightly Positive", 0.6, 0.7),
    ("Very good product. Really satisfied. Great features.", "Moderately Positive", 0.7, 0.8),
    ("Excellent quality! Highly recommend. Very impressed!", "Highly Positive", 0.8, 0.9),
    ("Perfect! Outstanding! Best ever! Absolutely amazing!", "Very Positive", 0.9, 1.0),
]

print("Testing Sentiment Score Distribution")
print("=" * 80)

for text, expected_label, min_score, max_score in test_cases:
    result = analyzer.analyze_review(text)
    score = result['overall_sentiment']['score']
    granular_label = result['overall_sentiment'].get('granular_label', 'Unknown')
    
    in_range = min_score <= score <= max_score
    status = "✓" if in_range else "✗"
    
    print(f"\n{status} {expected_label}")
    print(f"   Text: {text[:60]}...")
    print(f"   Expected Range: {min_score:.1f} - {max_score:.1f}")
    print(f"   Actual Score: {score:.3f}")
    print(f"   Granular Label: {granular_label}")
    
    if not in_range:
        print(f"   ⚠️  SCORE OUT OF EXPECTED RANGE!")

print("\n" + "=" * 80)
print("Test Complete!")

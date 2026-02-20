import logging
import sys

# Configure logging to stdout
logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s: %(message)s',
    stream=sys.stdout
)

from semantic_analyzer import analyzer
import json

test_cases = [
    # The specific failure case
    ("The headphones have weak sound quality and feel uncomfortable. They look good, but I wouldn’t recommend them.", "Mixed (Negative Lean)"),
    ("The headphones look great.", "Positive"),
    ("It looks amazing", "Positive"),
]

print(f"{'Text':<80} | {'Expected':<20} | {'Label':<15} | {'Score':<8} | {'Range'}")
print("-" * 140)

analyzer._initialize_aspect_patterns()

for text, expected in test_cases:
    result = analyzer.analyze_review(text)
    score = result['overall_sentiment']['score']
    label = result['overall_sentiment']['label']
    granular = result['overall_sentiment'].get('granular_label', 'N/A')
    
    # Check if contradictions were detected
    has_contradictions = result.get('contradiction_analysis', {}).get('has_contradictions', False)
    
    print(f"{text[:80]:<80} | {expected:<20} | {label:<15} | {score*100:5.1f}%  | {granular}")
    if has_contradictions:
        print(f"   [!] Contradiction Detected")
        
    for a in result.get('aspects', []):
        print(f"      - {a['aspect']}: {a['score']:.2f} (Weight: {a['weight']:.2f})")

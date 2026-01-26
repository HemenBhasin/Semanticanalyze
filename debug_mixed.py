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
    # conflicting aspects
    ("The camera is great but the battery is terrible.", "Mixed (Negative Lean)"),
    ("The battery is terrible but the camera is great.", "Mixed (Positive Lean)"),
    
    # Contradictory statements
    ("I love the design but I hate the user interface.", "Mixed"),
    ("It's a beautiful app but it crashes constantly.", "Mixed (Negative Lean)"),
    ("The service was slow but the food was delicious.", "Mixed (Positive Lean)"),
    
    # Nuanced conflict
    ("Good features, poor execution.", "Mixed (Negative Lean)"),
    ("Expensive but worth it.", "Positive (Lean)"),
    
    # Simple aspectless
    ("I want to like it but I can't.", "Negative")
]

print(f"{'Text':<50} | {'Expected':<20} | {'Label':<15} | {'Score':<8} | {'Range'}")
print("-" * 110)

analyzer._initialize_aspect_patterns()

for text, expected in test_cases:
    result = analyzer.analyze_review(text)
    score = result['overall_sentiment']['score']
    label = result['overall_sentiment']['label']
    granular = result['overall_sentiment'].get('granular_label', 'N/A')
    
    # Check if contradictions were detected
    has_contradictions = result.get('contradiction_analysis', {}).get('has_contradictions', False)
    
    print(f"{text:<50} | {expected:<20} | {label:<15} | {score*100:5.1f}%  | {granular}")
    if has_contradictions:
        print(f"   [!] Contradiction Detected")
        
    for a in result.get('aspects', []):
        print(f"      - {a['aspect']}: {a['score']:.2f} (Weight: {a['weight']:.2f})")

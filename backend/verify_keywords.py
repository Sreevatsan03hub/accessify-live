"""Quick test to verify YAKE-based keyword detection works on any text."""
from services.keyword_detection_service import get_keyword_detector

detector = get_keyword_detector()

test_texts = [
    "The four of us, for all, tired, what a cry! The thing is that he's already gone, he is... Man, is this song so pop?",
    "The exam is tomorrow and I need to submit my assignment before the deadline.",
    "Climate change is affecting biodiversity in tropical rainforests across the globe.",
    "The machine learning algorithm achieved 95% accuracy on the validation dataset.",
]

for text in test_texts:
    print(f"\n{'='*80}")
    print(f"INPUT: {text}")
    keywords = detector.extract_keywords(text)
    print(f"KEYWORDS ({len(keywords)}):")
    for kw in keywords:
        print(f"  → '{kw['original']}' {kw['emoji']}  (score: {kw['score']})")
    print(f"ENRICHED: {detector.enrich_text(text)}")

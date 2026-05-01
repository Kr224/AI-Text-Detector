import pickle
import os
import sys
from ai_text_detector import SimpleAIDetector


class AITextPredictor:
    def __init__(self, model_path='models/ai_detector_model.pkl'):
        self.model_path = model_path
        self.detector = None

        if os.path.exists(model_path):
            self.load_model()
        else:
            print(f"Model not found at {model_path}")
            print("Please run train_model.py first to train a model.")
            self.detector = SimpleAIDetector()

    def load_model(self):
        """Load trained model"""
        try:
            with open(self.model_path, 'rb') as f:
                self.detector = pickle.load(f)
            print(f"Model loaded successfully from {self.model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            self.detector = SimpleAIDetector()
            return False

    def predict(self, text):
        """Make prediction on single text"""
        if self.detector.model is None:
            print("Model not trained. Using heuristic analysis only.")
            result = self.detector.analyze_text(text)
            ai_score = result['heuristic_score']
            is_ai = ai_score > 0.5

            return {
                'is_ai': is_ai,
                'ai_score': ai_score,
                'human_probability': 1 - ai_score,
                'ai_probability': ai_score,
                'analysis': result
            }
        else:
            return self.detector.predict(text)

    def analyze_text(self, text):
        """Detailed analysis of text"""
        prediction = self.predict(text)
        analysis = self.detector.analyze_text(text)

        # Combine results
        result = {
            'prediction': 'AI-Generated' if prediction['is_ai'] else 'Human-Written',
            'confidence': max(prediction['human_probability'], prediction['ai_probability']),
            'ai_probability': prediction['ai_probability'],
            'human_probability': prediction['human_probability'],
            'detailed_analysis': analysis
        }

        return result

    def print_results(self, text, result):
        """Print formatted results"""
        print("\n" + "=" * 60)
        print("AI TEXT DETECTOR RESULTS")
        print("=" * 60)

        print(f"\nText preview: \"{text[:100]}...\"")
        print(f"\nPrediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"AI Probability: {result['ai_probability']:.1%}")
        print(f"Human Probability: {result['human_probability']:.1%}")

        print("\nDetailed Analysis:")
        print("-" * 40)
        analysis = result['detailed_analysis']
        print(f"Text Length: {analysis['text_length']} characters")
        print(f"Word Count: {analysis['word_count']}")
        print(f"Unique Word Ratio: {analysis['unique_word_ratio']:.3f}")
        print(f"AI Word Ratio: {analysis['ai_word_ratio']:.3f}")
        print(f"Sentence Variation: {analysis['sentence_variation']:.3f}")
        print(f"Punctuation Density: {analysis['punctuation_density']:.3f}")

        # Interpretation
        print("\nInterpretation:")
        if analysis['ai_word_ratio'] > 0.01:
            print("  - High AI word usage detected")
        if analysis['sentence_variation'] < 0.3:
            print("  - Low sentence length variation (common in AI text)")
        if analysis['unique_word_ratio'] < 0.5:
            print("  - High word repetition detected")

        print("\n" + "=" * 60)


# Interactive mode
def interactive_mode():
    """Run interactive prediction mode"""
    predictor = AITextPredictor()

    print("AI Text Detector - Interactive Mode")
    print("Type 'quit' to exit")
    print("-" * 40)

    while True:
        text = input("\nEnter text to analyze (or 'quit'): ")

        if text.lower() in ['quit', 'exit', 'q']:
            break

        if len(text.strip()) < 10:
            print("Please enter at least 10 characters.")
            continue

        result = predictor.analyze_text(text)
        predictor.print_results(text, result)


# Batch prediction from file
def batch_predict(file_path):
    """Predict on multiple texts from a file"""
    predictor = AITextPredictor()

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]

        print(f"Analyzing {len(texts)} texts from {file_path}")

        results = []
        for i, text in enumerate(texts, 1):
            result = predictor.predict(text)
            results.append({
                'text': text[:50] + '...' if len(text) > 50 else text,
                'prediction': 'AI' if result['is_ai'] else 'Human',
                'ai_probability': result['ai_probability']
            })

            if i % 10 == 0:
                print(f"Processed {i}/{len(texts)} texts")

        # Summary
        ai_count = sum(1 for r in results if r['prediction'] == 'AI')
        human_count = len(results) - ai_count

        print(f"\nSummary:")
        print(f"Total texts: {len(results)}")
        print(f"Predicted AI: {ai_count} ({ai_count / len(results):.1%})")
        print(f"Predicted Human: {human_count} ({human_count / len(results):.1%})")

        return results

    except Exception as e:
        print(f"Error reading file: {e}")
        return []


# Main execution
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='AI Text Detector')
    parser.add_argument('--text', type=str, help='Text to analyze')
    parser.add_argument('--file', type=str, help='File containing texts to analyze')
    parser.add_argument('--interactive', action='store_true', help='Start interactive mode')

    args = parser.parse_args()

    predictor = AITextPredictor()

    if args.text:
        result = predictor.analyze_text(args.text)
        predictor.print_results(args.text, result)

    elif args.file:
        batch_predict(args.file)

    elif args.interactive:
        interactive_mode()

    else:
        # Default: test with sample texts
        print("Testing with sample texts...\n")

        sample_texts = [
            # Human-like
            "I'm not really sure what to write here. Just testing out this AI detector thing. Hope it works!",

            # AI-like
            "The multifaceted nature of contemporary computational paradigms necessitates a comprehensive reevaluation of traditional algorithmic frameworks in order to optimize operational efficiency and facilitate robust analytical methodologies.",

            # Mixed
            "Honestly, I think the main issue is that people don't communicate enough. However, one must consider the broader societal implications of digital communication platforms on interpersonal relationships and community cohesion."
        ]

        for i, text in enumerate(sample_texts, 1):
            print(f"\nSample {i}:")
            result = predictor.analyze_text(text)
            predictor.print_results(text, result)
#!/usr/bin/env python3
"""
Simple AI Text Detector Runner
"""

import os
import sys
import pandas as pd
import numpy as np
from utils_seed import set_global_seed  # For reproducible simple demo

# Set a fixed random seed so that the simple demo is reproducible
set_global_seed(42)

# Create directories
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Download NLTK data
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')
    nltk.download('punkt_tab')


def create_simple_dataset():
    """Create a simple 1000-sample dataset without external dependencies"""
    print("Creating simple dataset with 1000 samples...")

    human_texts = []
    ai_texts = []

    # Simple human-like texts
    human_templates = [
        "I can't believe {}. It's just {}.",
        "So {}. What do you think about {}?",
        "Remember when {}? That was {}.",
        "I've been thinking about {} lately. It's {}.",
        "Not sure if {}, but {}."
    ]

    human_fillers = [
        ['the weather today', 'traffic', 'that movie', 'the news', 'work'],
        ['crazy', 'awesome', 'terrible', 'weird', 'normal'],
        ['we went there', 'that happened', 'I tried that', 'we saw that'],
        ['good times', 'funny', 'embarrassing', 'memorable', 'awkward']
    ]

    # Simple AI-like texts
    ai_templates = [
        "The {} of {} requires a comprehensive understanding of {}.",
        "In order to optimize {}, it is necessary to implement effective {}.",
        "Modern {} approaches must adapt to accommodate {}.",
        "The {} of {} has significantly transformed {}.",
        "{} represents an important aspect of addressing challenges in {}."
    ]

    ai_fillers = [
        ['implementation', 'development', 'integration', 'deployment'],
        ['artificial intelligence', 'data analysis', 'cloud computing', 'machine learning'],
        ['key principles', 'fundamental concepts', 'core methodologies', 'essential frameworks'],
        ['efficiency', 'performance', 'security', 'reliability'],
        ['strategic planning', 'innovative solutions', 'technical approaches', 'systematic methods']
    ]

    # Generate 500 human texts
    for i in range(500):
        template = np.random.choice(human_templates)
        text = template
        for filler_set in human_fillers:
            if '{}' in text:
                text = text.replace('{}', np.random.choice(filler_set), 1)

        # Add human imperfections
        if np.random.random() > 0.7:
            text += " I think."
        if np.random.random() > 0.8:
            text = text.replace('.', '...')

        human_texts.append(text)

    # Generate 500 AI texts
    for i in range(500):
        template = np.random.choice(ai_templates)
        text = template
        filler_count = template.count('{}')

        for j in range(filler_count):
            filler_set = ai_fillers[j % len(ai_fillers)]
            text = text.replace('{}', np.random.choice(filler_set), 1)

        # Add AI characteristics
        if np.random.random() > 0.5:
            transitions = ['Furthermore,', 'Moreover,', 'Therefore,', 'Consequently,']
            text += f" {np.random.choice(transitions)} this demonstrates the need for continued development."

        # Make more formal
        replacements = {
            'use': 'utilize',
            'make': 'create',
            'help': 'facilitate',
            'show': 'demonstrate',
            'good': 'effective',
            'bad': 'ineffective'
        }

        for old, new in replacements.items():
            if old in text:
                text = text.replace(old, new)

        ai_texts.append(text)

    # Create DataFrame
    df = pd.DataFrame({
        'text': human_texts + ai_texts,
        'label': [0] * 500 + [1] * 500,
        'source': ['generated'] * 1000
    })

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save
    df.to_csv('data/processed/simple_dataset.csv', index=False)

    print(f"Created dataset with {len(df)} samples")
    print(f"Human: {len(human_texts)}, AI: {len(ai_texts)}")

    # Show samples
    print("\nSample Human Text:")
    print(df[df['label'] == 0].iloc[0]['text'])

    print("\nSample AI Text:")
    print(df[df['label'] == 1].iloc[0]['text'])

    return df


def train_simple_model():
    """Train the model with the simple demo dataset"""
    print("\n" + "=" * 60)
    print("TRAINING MODEL (SIMPLE DEMO DATASET)")
    print("=" * 60)

    # Import here to avoid circular imports
    from train_model import ModelTrainer

    trainer = ModelTrainer()
    # Train using the simple demo dataset instead of the default large dataset
    accuracy, results = trainer.train(
        use_large_dataset=True,
        dataset_size=1000,
        save_model=True,
        dataset_path='data/processed/simple_dataset.csv'
    )

    print(f"\nModel trained with accuracy: {results['accuracy']:.1%}")
    return trainer


def test_model_interactive():
    """Test the model interactively"""
    print("\n" + "=" * 60)
    print("INTERACTIVE TESTING")
    print("=" * 60)

    from predict import AITextPredictor

    predictor = AITextPredictor()

    if predictor.detector.model is None:
        print("Model not found. Training first...")
        train_simple_model()
        predictor = AITextPredictor()  # Reload

    print("\nType sentences to analyze. Type 'quit' to exit.")
    print("-" * 40)

    while True:
        text = input("\nEnter text: ").strip()

        if text.lower() in ['quit', 'exit', 'q']:
            break

        if len(text) < 5:
            print("Please enter longer text.")
            continue

        result = predictor.analyze_text(text)
        print(f"\nPrediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"AI Probability: {result['ai_probability']:.1%}")


def main_menu():
    """Display main menu"""
    while True:
        print("\n" + "=" * 60)
        print("SIMPLE AI TEXT DETECTOR")
        print("=" * 60)

        print("\n1. Create dataset (1000 samples)")
        print("2. Train model")
        print("3. Test interactively")
        print("4. Run full pipeline (1+2+3)")
        print("5. Quick test")
        print("6. Exit")

        choice = input("\nChoose option (1-6): ").strip()

        if choice == '1':
            create_simple_dataset()
        elif choice == '2':
            train_simple_model()
        elif choice == '3':
            test_model_interactive()
        elif choice == '4':
            create_simple_dataset()
            train_simple_model()
            test_model_interactive()
        elif choice == '5':
            quick_test()
        elif choice == '6':
            print("Goodbye!")
            break
        else:
            print("Invalid choice")


def quick_test():
    """Quick test with sample texts"""
    from predict import AITextPredictor

    predictor = AITextPredictor()

    if predictor.detector.model is None:
        print("Model not trained. Training first...")
        create_simple_dataset()
        train_simple_model()
        predictor = AITextPredictor()

    test_cases = [
        ("I'm just typing my thoughts here. Not sure what to write.", "Human"),
        ("The implementation of innovative methodologies necessitates comprehensive analytical frameworks.", "AI"),
        ("My cat is being weird again. She keeps chasing her tail.", "Human"),
        ("Therefore, it is imperative to leverage data-driven approaches for optimal outcomes.", "AI"),
        ("I think the weather is nice today. Maybe I'll go for a walk.", "Human"),
        ("The multifaceted nature of contemporary paradigms underscores the importance of robust solutions.", "AI")
    ]

    print("\n" + "=" * 60)
    print("QUICK TEST RESULTS")
    print("=" * 60)

    correct = 0
    total = 0

    for text, expected in test_cases:
        result = predictor.predict(text)
        prediction = "AI" if result['is_ai'] else "Human"

        print(f"\nExpected: {expected}")
        print(f"Text: {text[:60]}...")
        print(f"Predicted: {prediction}")
        print(f"Confidence: {max(result['human_probability'], result['ai_probability']):.1%}")

        if expected == prediction:
            correct += 1
            print("✓ Correct!")
        else:
            print("✗ Wrong")

        total += 1

    print(f"\nAccuracy: {correct}/{total} ({correct / total * 100:.1f}%)")


if __name__ == "__main__":
    print("Setting up AI Text Detector...")
    main_menu()
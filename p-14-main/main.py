"""
Main script to run the entire AI Text Detector pipeline
"""

import sys
import os
from utils_seed import set_global_seed  # For reproducible pipeline

# Set a fixed random seed once at program start
set_global_seed(42)


def main():
    print("=" * 60)
    print("AI TEXT DETECTOR - COMPLETE PIPELINE")
    print("=" * 60)

    print("\nOptions:")
    print("1. Create large dataset")
    print("2. Train model")
    print("3. Test model")
    print("4. Run interactive detection")
    print("5. Full pipeline (dataset + train + test)")
    print("6. Exit")

    choice = input("\nEnter your choice (1-6): ").strip()

    if choice == '1':
        print("\nCreating large dataset...")
        from data_collector import DataCollector
        collector = DataCollector()
        # Check what methods are available
        if hasattr(collector, 'create_large_dataset'):
            collector.create_large_dataset(1000, 'data/processed/large_dataset.csv')
        elif hasattr(collector, 'collect_all_data'):
            collector.collect_all_data(use_openai=False)
        else:
            print("Error: No dataset creation method found in DataCollector")
            print("Available methods:", [m for m in dir(collector) if not m.startswith('_')])

    elif choice == '2':
        print("\nTraining model...")
        from train_model import ModelTrainer
        trainer = ModelTrainer()
        trainer.train(use_large_dataset=True)

    elif choice == '3':
        print("\nTesting model...")
        from predict import AITextPredictor
        predictor = AITextPredictor()

        # Test with built-in samples
        test_texts = [
            "This is clearly written by a human. I'm just typing random thoughts.",
            "The intricate tapestry of computational methodologies necessitates a paradigm shift in analytical frameworks.",
            "I dunno, maybe it's just me but this seems kinda weird? Like, why would anyone do that?"
        ]

        for text in test_texts:
            result = predictor.analyze_text(text)
            print(f"\nText: {text[:50]}...")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.1%}")

    elif choice == '4':
        print("\nStarting interactive mode...")
        from predict import interactive_mode
        interactive_mode()

    elif choice == '5':
        print("\nRunning full pipeline...")

        # 1. Create dataset
        # print("\n[1/3] Creating dataset...")
        # try:
        #     from data_collector import DataCollector
        #     collector = DataCollector()
        #
        #     # Try different method names
        #
        #

            # if hasattr(collector, 'create_large_dataset'):
            #     print("Using create_large_dataset method...")
            #     collector.create_large_dataset(1000, 'data/processed/large_dataset.csv')
            # # elif hasattr(collector, 'collect_all_data'):
            # #     print("Using collect_all_data method...")
            # #     collector.collect_all_data(use_openai=False)
            # else:
            #     # Create dataset using dataset_loader directly
            #     print("Creating dataset via dataset_loader...")
            #     from dataset_loader import DatasetLoader
            #     loader = DatasetLoader()
            #     human, ai = loader.create_synthetic_dataset_large(1000)
            #
            #     # Save dataset
            #     import pandas as pd
            #     df = pd.DataFrame({
            #         'text': human + ai,
            #         'label': [0] * len(human) + [1] * len(ai)
            #     })
            #     df.to_csv('data/processed/large_dataset.csv', index=False)
            #     print(f"Created dataset with {len(df)} samples")

        # except Exception as e:
        #     print(f"Error creating dataset: {e}")
        #     print("Continuing with existing dataset if available...")

        # 2. Train model
        print("\n[2/3] Training model...")
        try:
            from train_model import ModelTrainer
            trainer = ModelTrainer()
            trainer.train(use_large_dataset=False)
        except Exception as e:
            print(f"Error training model: {e}")

        # 3. Test
        print("\n[3/3] Testing model...")
        try:
            from predict import interactive_mode
            interactive_mode()
        except Exception as e:
            print(f"Error in interactive mode: {e}")
            print("\nYou can still test with: python predict.py --interactive")

    elif choice == '6':
        print("Exiting...")
        sys.exit(0)

    else:
        print("Invalid choice. Please try again.")


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('data/human', exist_ok=True)
    os.makedirs('data/ai', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Download NLTK data if needed
    try:
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading NLTK data (punkt)...")
            nltk.download('punkt')
            # punkt_tab is optional and may not exist in all NLTK versions
            try:
                nltk.download('punkt_tab')
            except Exception:
                pass
    except ImportError:
        print("NLTK is not installed. Please run `pip install nltk` in your virtual environment.")

    main()
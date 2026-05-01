import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import pickle
import os
import json
from datetime import datetime

from sklearn.model_selection import cross_val_score

from dataset_loader import DatasetLoader
from utils_seed import set_global_seed  # For reproducible training
from ai_text_detector import SimpleAIDetector


class ModelTrainer:
    def __init__(self, model_dir='models'):
        # Set a fixed random seed for training pipeline
        set_global_seed(42)
        self.model_dir = model_dir
        self.detector = SimpleAIDetector()
        self.dataset_loader = DatasetLoader()
        os.makedirs(model_dir, exist_ok=True)

    def train(self, use_large_dataset=True, dataset_size=1000, save_model=True, dataset_path=None):
        """Train the model with optional large dataset"""
        print("=" * 60)
        print("MODEL TRAINING STARTED")
        print("=" * 60)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Load data
        if use_large_dataset:
            print("\nUsing large dataset (1000+ samples)...")
            # If a custom dataset path is provided (e.g. simple_dataset.csv),
            # use it instead of the default large_dataset path.
            if dataset_path is not None:
                human_texts, ai_texts = self.dataset_loader.load_large_dataset(
                    dataset_path=dataset_path,
                    min_samples=dataset_size
                )
            else:
                human_texts, ai_texts = self.dataset_loader.load_large_dataset(
                    min_samples=dataset_size
                )
        else:
            print("\nUsing preprocessed dataset...")
            human_texts, ai_texts = self.dataset_loader.load_standard_datasets(
                dataset_path="data/processed/real_dataset.csv",
                min_samples_per_class=dataset_size // 2
            )

        # Check dataset size
        total_samples = len(human_texts) + len(ai_texts)
        print(f"\nDataset Statistics:")
        print(f"Total samples: {total_samples}")
        print(f"Human samples: {len(human_texts)} ({len(human_texts) / total_samples * 100:.1f}%)")
        print(f"AI samples: {len(ai_texts)} ({len(ai_texts) / total_samples * 100:.1f}%)")

        # Prepare train/test split
        X_train, X_test, y_train, y_test = self.dataset_loader.prepare_train_test_split(
            human_texts, ai_texts
        )

        # Extract human and AI texts for training
        human_train = [text for text, label in zip(X_train, y_train) if label == 0]
        ai_train = [text for text, label in zip(X_train, y_train) if label == 1]

        # Train model
        print("\nTraining the detector...")
        print(f"Training on {len(X_train)} samples...")
        start_time = datetime.now()
        accuracy = self.detector.train(human_train, ai_train)
        training_time = (datetime.now() - start_time).total_seconds()

        print(f"\nTraining completed in {training_time:.2f} seconds")

        # # ===== NEW" CROSS-VALIDATION  =====
        # # Perform cross-validation for robust evaluation
        # if len(human_train) >= 100 and len(ai_train) >= 100:
        #     print("\n" + "=" * 60)
        #     print("PERFORMING CROSS-VALIDATION")
        #     print("=" * 60)
        #     cv_scores = self.evaluate_with_cross_validation(human_train, ai_train)
        #     # Store CV scores in results
        #     results = {'cv_scores': cv_scores.tolist(), 'cv_mean': float(cv_scores.mean())}
        # # ===== END OF ADDITION =====

        # Evaluate on test set
        print("\n" + "=" * 60)
        print("TEST SET EVALUATION")
        print("=" * 60)

        results = self.evaluate_on_test_set(X_test, y_test)
        results['training_time_seconds'] = training_time
        results['dataset_size'] = total_samples
        results['train_samples'] = len(X_train)
        results['test_samples'] = len(X_test)


        # Save model
        if save_model:
            self.save_model()

            # Save training metadata
            metadata = {
                'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'dataset_size': total_samples,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'human_samples': len(human_texts),
                'ai_samples': len(ai_texts),
                'model_type': 'RandomForest',
                'features_used': 8,
                'training_time_seconds': training_time
            }

            with open(os.path.join(self.model_dir, 'training_metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"\nTraining metadata saved to {os.path.join(self.model_dir, 'training_metadata.json')}")

        # Feature analysis
        self.analyze_features(human_train, ai_train)

        # Additional analysis
        self.perform_additional_analysis(X_test, y_test)

        return accuracy, results

    def evaluate_on_test_set(self, X_test, y_test):
        """Evaluate model on test set"""
        print(f"Testing on {len(X_test)} samples...")
        y_pred = []
        y_prob = []

        start_time = datetime.now()
        for text in X_test:
            result = self.detector.predict(text)
            y_pred.append(result['is_ai'])
            y_prob.append(result['ai_probability'])
        inference_time = (datetime.now() - start_time).total_seconds()

        print(f"Inference completed in {inference_time:.2f} seconds")
        print(f"Average inference time: {inference_time / len(X_test) * 1000:.2f} ms per sample")

        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Calculate additional metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

        print(f"\nTest Set Results:")
        print(f"Accuracy:           {accuracy:.2%}")
        print(f"Precision:          {precision:.2%}")
        print(f"Recall (Sensitivity): {recall:.2%}")
        print(f"Specificity:        {specificity:.2%}")
        print(f"F1-Score:           {f1:.2%}")
        print(f"False Positive Rate: {false_positive_rate:.2%}")
        print(f"\nConfusion Matrix:")
        print(f"True Negatives (Human correctly identified): {tn}")
        print(f"False Positives (Human misclassified as AI): {fp}")
        print(f"False Negatives (AI misclassified as Human): {fn}")
        print(f"True Positives (AI correctly identified):    {tp}")

        # Confusion matrix visualization
        self.plot_confusion_matrix(y_test, y_pred)

        # ROC Curve
        self.plot_roc_curve(y_test, y_prob)

        # Precision-Recall Curve
        self.plot_precision_recall_curve(y_test, y_prob)

        # Save results
        results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'specificity': float(specificity),
            'false_positive_rate': float(false_positive_rate),
            'confusion_matrix': {
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp)
            },
            'test_size': len(X_test),
            'inference_time_seconds': inference_time,
            'avg_inference_time_ms': inference_time / len(X_test) * 1000
        }

        with open(os.path.join(self.model_dir, 'test_results.json'), 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nDetailed results saved to {os.path.join(self.model_dir, 'test_results.json')}")

        return results

    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Human', 'AI'],
                    yticklabels=['Human', 'AI'],
                    annot_kws={"size": 16})
        plt.title('Confusion Matrix', fontsize=16, pad=20)
        plt.ylabel('True Label', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
        plt.show()

    def plot_roc_curve(self, y_true, y_prob):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, pad=20)
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'roc_curve.png'), dpi=150, bbox_inches='tight')
        plt.show()

        # Save AUC value
        with open(os.path.join(self.model_dir, 'model_performance.txt'), 'a') as f:
            f.write(f"ROC AUC Score: {roc_auc:.4f}\n")

    def plot_precision_recall_curve(self, y_true, y_prob):
        """Plot Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)

        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='green', lw=3, label=f'PR curve (AUC = {pr_auc:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.title('Precision-Recall Curve', fontsize=16, pad=20)
        plt.legend(loc="upper right", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'precision_recall_curve.png'), dpi=150, bbox_inches='tight')
        plt.show()

    def analyze_features(self, human_texts, ai_texts):
        """Analyze feature differences between human and AI text"""
        print("\n" + "=" * 60)
        print("FEATURE ANALYSIS")
        print("=" * 60)

        # Extract features for analysis
        human_features = []
        ai_features = []

        # Use a sample for analysis (max 200 each)
        sample_size = min(200, len(human_texts), len(ai_texts))
        human_sample = human_texts[:sample_size]
        ai_sample = ai_texts[:sample_size]

        print(f"Analyzing {sample_size} samples from each class...")

        for text in human_sample:
            features = self.detector.extract_features(text)
            human_features.append(features)

        for text in ai_sample:
            features = self.detector.extract_features(text)
            ai_features.append(features)

        human_features = np.array(human_features)
        ai_features = np.array(ai_features)

        # Feature names (from your extract_features method)
        feature_names = [
            'Unique Word Ratio',
            'AI Word Ratio',
            'Sentence Variation',
            'Avg Word Length',
            'Punctuation Density',
            'Conjunction Ratio',
            'Paragraph Variation',

            'Burstiness',  # NEW
            'Transition Density',  # NEW
            'Avg Clause Length',  # NEW
            'Lexical Diversity',  # NEW
            'Pronoun Ratio',  # NEW
            'Contraction Ratio',  # NEW
            'Passive Voice Ratio'  # NEW
        ]


        # Calculate mean differences
        print("\nFeature Comparison (Human vs AI):")
        print("-" * 60)
        print(f"{'Feature':25} | {'Human Mean':12} | {'AI Mean':12} | {'Difference':12} | {'% Diff':10}")
        print("-" * 60)

        feature_stats = []
        for i, name in enumerate(feature_names):
            human_mean = human_features[:, i].mean()
            ai_mean = ai_features[:, i].mean()
            diff = ai_mean - human_mean
            pct_diff = (diff / human_mean * 100) if human_mean != 0 else 0

            print(f"{name:25} | {human_mean:12.4f} | {ai_mean:12.4f} | {diff:12.4f} | {pct_diff:10.1f}%")

            feature_stats.append({
                'feature': name,
                'human_mean': float(human_mean),
                'ai_mean': float(ai_mean),
                'difference': float(diff),
                'pct_difference': float(pct_diff)
            })

        # Save feature statistics
        with open(os.path.join(self.model_dir, 'feature_statistics.json'), 'w') as f:
            json.dump(feature_stats, f, indent=2)

        # Feature importance from model
        if hasattr(self.detector.model, 'feature_importances_'):
            print("\n" + "=" * 60)
            print("FEATURE IMPORTANCE (from Random Forest)")
            print("=" * 60)

            importances = self.detector.model.feature_importances_
            indices = np.argsort(importances)[::-1]

            print(f"{'Feature':25} | {'Importance':12} | {'Cumulative %':12}")
            print("-" * 60)

            cumulative = 0
            feature_importance_stats = []
            for i, idx in enumerate(indices):
                cumulative += importances[idx]
                print(f"{feature_names[idx]:25} | {importances[idx]:12.4f} | {cumulative:12.1%}")

                feature_importance_stats.append({
                    'feature': feature_names[idx],
                    'importance': float(importances[idx]),
                    'rank': i + 1
                })

            # Save feature importance
            with open(os.path.join(self.model_dir, 'feature_importance.json'), 'w') as f:
                json.dump(feature_importance_stats, f, indent=2)

            # Plot feature importance
            self.plot_feature_importance(feature_names, importances)

    def plot_feature_importance(self, feature_names, importances):
        """Plot feature importance"""
        indices = np.argsort(importances)[::-1]
        sorted_names = [feature_names[i] for i in indices]
        sorted_importances = importances[indices]

        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(sorted_names)), sorted_importances, align='center', color='steelblue')
        plt.yticks(range(len(sorted_names)), sorted_names, fontsize=12)
        plt.xlabel('Feature Importance', fontsize=14)
        plt.title('Feature Importance in AI Text Detection', fontsize=16, pad=20)
        plt.gca().invert_yaxis()  # Highest importance at top

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, sorted_importances)):
            plt.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{val:.3f}', va='center', fontsize=11)

        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'feature_importance.png'), dpi=150, bbox_inches='tight')
        plt.show()

    def perform_additional_analysis(self, X_test, y_test):
        """Perform additional analysis on test set"""
        print("\n" + "=" * 60)
        print("ADDITIONAL ANALYSIS")
        print("=" * 60)

        # Analyze prediction confidence
        confidences = []
        predictions = []

        for text in X_test:
            result = self.detector.predict(text)
            confidences.append(max(result['human_probability'], result['ai_probability']))
            predictions.append(result['is_ai'])

        # Calculate average confidence
        avg_confidence = np.mean(confidences)
        print(f"Average prediction confidence: {avg_confidence:.2%}")

        # Confidence by class
        human_confidences = [c for c, p in zip(confidences, predictions) if not p]
        ai_confidences = [c for c, p in zip(confidences, predictions) if p]

        if human_confidences:
            print(f"Average confidence for human predictions: {np.mean(human_confidences):.2%}")
        if ai_confidences:
            print(f"Average confidence for AI predictions: {np.mean(ai_confidences):.2%}")

        # Save confidence analysis
        confidence_stats = {
            'avg_confidence': float(avg_confidence),
            'avg_human_confidence': float(np.mean(human_confidences)) if human_confidences else 0,
            'avg_ai_confidence': float(np.mean(ai_confidences)) if ai_confidences else 0,
            'min_confidence': float(np.min(confidences)),
            'max_confidence': float(np.max(confidences))
        }

        with open(os.path.join(self.model_dir, 'confidence_analysis.json'), 'w') as f:
            json.dump(confidence_stats, f, indent=2)

    def save_model(self):
        """Save trained model"""
        model_path = os.path.join(self.model_dir, 'ai_detector_model.pkl')

        with open(model_path, 'wb') as f:
            pickle.dump(self.detector, f)

        print(f"\nModel saved to: {model_path}")

    def load_model(self):
        """Load trained model"""
        model_path = os.path.join(self.model_dir, 'ai_detector_model.pkl')

        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.detector = pickle.load(f)
            print(f"Model loaded from: {model_path}")

            # Also load metadata if available
            metadata_path = os.path.join(self.model_dir, 'training_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                print(f"Model trained on: {metadata['training_date']}")
                print(f"Training dataset size: {metadata['dataset_size']} samples")

            return True
        else:
            print(f"No model found at: {model_path}")


    def evaluate_with_cross_validation(self, human_texts, ai_texts, cv_folds=5):
        """Evaluate model using k-fold cross-validation"""
        print(f"\n{'=' * 60}")
        print(f"CROSS-VALIDATION EVALUATION ({cv_folds} folds)")
        print('=' * 60)

        X, y = self.detector.prepare_dataset(human_texts, ai_texts)

        # Perform cross-validation
        cv_scores = cross_val_score(
            self.detector.model, X, y,
            cv=cv_folds,
            scoring='accuracy'
        )

        print(f"\nCross-validation scores: {cv_scores}")
        print(f"Mean accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std() * 2:.2%})")

        return cv_scores

# Main training execution with options
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train AI Text Detector Model')
    parser.add_argument('--use-large-dataset', action='store_true',
                        help='Use large dataset (1000+ samples)')
    parser.add_argument('--dataset-size', type=int, default=1000,
                        help='Minimum dataset size (default: 1000)')
    parser.add_argument('--skip-save', action='store_true',
                        help='Skip saving model (for testing)')

    args = parser.parse_args()

    trainer = ModelTrainer()

    # Train model with specified options
    trainer.train(
        use_large_dataset=args.use_large_dataset,
        dataset_size=args.dataset_size,
        save_model=not args.skip_save
    )



    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run 'python predict.py --interactive' to test the model interactively")
    print("2. Check the 'models/' directory for:")
    print("   - ai_detector_model.pkl (trained model)")
    print("   - test_results.json (performance metrics)")
    print("   - feature_importance.png (visualization)")
    print("   - confusion_matrix.png (visualization)")
    print("3. Review feature analysis to understand what distinguishes AI text")
    print("\nTo train with large dataset: python train_model.py --use-large-dataset")
    print("To specify dataset size: python train_model.py --use-large-dataset --dataset-size 2000")


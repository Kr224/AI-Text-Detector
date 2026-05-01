import numpy as np
import pandas as pd
from collections import Counter
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from utils_seed import set_global_seed  # For reproducible model behavior
import warnings


warnings.filterwarnings('ignore')

# Download NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')
    nltk.download('punkt_tab')


class SimpleAIDetector:
    def __init__(self):
        # Set a fixed random seed for any stochastic behavior in the detector
        set_global_seed(42)
        # Common AI indicator words (based on research)
        self.ai_words = [
            'tapestry', 'delve', 'testament', 'landscape', 'realm',
            'testament', 'moreover', 'furthermore', 'however', 'therefore',
            'consequently', 'noteworthy', 'intricate', 'comprehensive',
            'paradigm', 'leverage', 'robust', 'seamless', 'foster',
            'holistic', 'nuanced', 'myriad', 'crucial', 'essential',
            'paramount', 'utilize', 'endeavor', 'henceforth', 'thus'

            # Additional AI-typical words
            'facilitate', 'implement', 'optimize', 'streamline','synergy',
            'multifaceted', 'exponential', 'transformative', 'innovative',
            'cutting-edge', 'state-of-the-art', 'groundbreaking', 'revolutionary',
            'unprecedented', 'unparalleled', 'burgeoning', 'proliferation',
            'encompasses', 'necessitates', 'underscores', 'exemplifies',
            'substantiate', 'elucidate', 'ascertain', 'epitomize',
            'meticulous', 'pivotal', 'integral', 'salient', 'quintessential',
            'plethora', 'gamut', 'spectrum', 'array', 'nexus',
            'catalyst', 'impetus', 'propensity', 'proclivity', 'disposition'
        ]

        self.model = None
        self.feature_names = None

    def extract_features(self, text):
        """Extract linguistic features from text"""
        features = []

        # Clean text
        text = str(text).lower()

        # 1. Perplexity-like metric (simplified - using word repetition)
        words = word_tokenize(text)
        word_counts = Counter(words)
        unique_ratio = len(set(words)) / max(len(words), 1)
        features.append(unique_ratio)  # Lower = more repetitive

        # 2. AI word frequency
        ai_word_count = sum(1 for word in words if word in self.ai_words)
        ai_word_ratio = ai_word_count / max(len(words), 1)
        features.append(ai_word_ratio)

        # 3. Sentence length variation
        sentences = sent_tokenize(text)
        if len(sentences) > 1:
            sent_lengths = [len(word_tokenize(s)) for s in sentences]
            sent_length_std = np.std(sent_lengths)
            sent_length_mean = np.mean(sent_lengths)
            features.append(sent_length_std / max(sent_length_mean, 1))
        else:
            features.append(0)

        # 4. Average word length
        avg_word_length = np.mean([len(word) for word in words if word.isalpha()]) if words else 0
        features.append(avg_word_length)

        # 5. Punctuation ratio
        punctuation = sum(1 for char in text if char in '.,;:!?')
        punct_ratio = punctuation / max(len(text), 1)
        features.append(punct_ratio)

        # 6. Conjunction ratio
        conjunctions = ['and', 'but', 'or', 'so', 'because', 'although', 'however']
        conj_count = sum(1 for word in words if word in conjunctions)
        conj_ratio = conj_count / max(len(words), 1)
        features.append(conj_ratio)

        # 7. Paragraph structure (simplified)
        paragraphs = text.split('\n\n')
        para_length_variation = np.std([len(p.split()) for p in paragraphs if p.strip()]) if len(paragraphs) > 1 else 0
        features.append(min(para_length_variation, 10))  # Cap at 10

        # 8. Readability score (simplified)
        long_words = sum(1 for word in words if len(word) > 6)
        readability = long_words / max(len(words), 1) if words else 0
        features.append(readability)

        # NEW FEATURES - Add these after existing features:

        # 9. Burstiness - variation in sentence complexity
        if len(sentences) > 1:
            sent_complexities = [len(word_tokenize(s)) / max(len(s.split(',')), 1) for s in sentences]
            burstiness = np.std(sent_complexities) / max(np.mean(sent_complexities), 1)
            features.append(burstiness)
        else:
            features.append(0)

        # 10. Transition word density (AI tends to use more)
        transition_words = ['however', 'therefore', 'furthermore', 'moreover',
                            'consequently', 'nevertheless', 'nonetheless']
        transition_count = sum(1 for word in words if word in transition_words)
        transition_density = transition_count / max(len(words), 1)
        features.append(transition_density)

        # 11. Average clause length (AI tends to be more uniform)
        clauses = re.split(r'[,;:]', text)
        clause_lengths = [len(word_tokenize(c)) for c in clauses if c.strip()]
        avg_clause_length = np.mean(clause_lengths) if clause_lengths else 0
        features.append(min(avg_clause_length, 20))

        # 12. Lexical diversity (Type-Token Ratio)
        ttr = len(set(words)) / len(words) if words else 0
        features.append(ttr)

        # 13. Pronoun usage ratio (humans use more personal pronouns)
        pronouns = ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'you', 'your']
        pronoun_count = sum(1 for word in words if word in pronouns)
        pronoun_ratio = pronoun_count / max(len(words), 1)
        features.append(pronoun_ratio)

        # 14. Contraction usage (humans use more contractions)
        contractions = ["n't", "'re", "'ve", "'ll", "'d", "'m", "'s"]
        contraction_count = sum(1 for word in words if any(c in word for c in contractions))
        contraction_ratio = contraction_count / max(len(words), 1)
        features.append(contraction_ratio)

        # 15. Passive voice detection (AI uses more passive voice)
        passive_indicators = ['is', 'are', 'was', 'were', 'been', 'being']
        past_participles = [w for w in words if w.endswith('ed')]
        passive_count = 0
        for i in range(len(words) - 1):
            if words[i] in passive_indicators and i + 1 < len(words) and words[i + 1] in past_participles:
                passive_count += 1
        passive_ratio = passive_count / max(len(sentences), 1)
        features.append(passive_ratio)

        return np.array(features)

        return np.array(features)

    def prepare_dataset(self, human_texts, ai_texts):
        """Prepare dataset from human and AI texts"""
        X = []
        y = []

        # Extract features from human texts
        for text in human_texts:
            features = self.extract_features(text)
            X.append(features)
            y.append(0)  # 0 for human

        # Extract features from AI texts
        for text in ai_texts:
            features = self.extract_features(text)
            X.append(features)
            y.append(1)  # 1 for AI

        return np.array(X), np.array(y)

    # new
    def train(self, human_texts, ai_texts):
        """Train the classifier using ensemble approach"""
        X, y = self.prepare_dataset(human_texts, ai_texts)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Create ensemble of multiple models
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15)
        gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42, learning_rate=0.1)
        lr_model = LogisticRegression(random_state=42, max_iter=1000)

        # Voting classifier combines predictions
        self.model = VotingClassifier(
            estimators=[
                ('rf', rf_model),
                ('gb', gb_model),
                ('lr', lr_model)
            ],
            voting='soft'  # Use probability averaging
        )

        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Ensemble model trained successfully!")
        print(f"Accuracy: {accuracy:.2%}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))

        return accuracy

    def predict(self, text):
        """Predict if text is AI-generated"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        features = self.extract_features(text).reshape(1, -1)
        prob = self.model.predict_proba(features)[0]
        prediction = self.model.predict(features)[0]

        return {
            'is_ai': bool(prediction),
            'human_probability': prob[0],
            'ai_probability': prob[1],
            'ai_score': prob[1]  # "AI-ness" score
        }

    def analyze_text(self, text):
        """Detailed analysis of text features"""
        features = self.extract_features(text)

        analysis = {
            'text_length': len(text),
            'word_count': len(word_tokenize(text)),
            'unique_word_ratio': features[0],
            'ai_word_ratio': features[1],
            'sentence_variation': features[2],
            'avg_word_length': features[3],
            'punctuation_density': features[4],
            'conjunction_ratio': features[5]
        }

        # Simple heuristic scoring
        score = 0
        if analysis['ai_word_ratio'] > 0.01:
            score += 1
        if analysis['sentence_variation'] < 0.3:
            score += 1
        if analysis['unique_word_ratio'] < 0.5:
            score += 1
        if analysis['punctuation_density'] < 0.01:
            score += 1

        analysis['heuristic_score'] = min(score / 4, 1.0)

        return analysis
    # new
    def predict_with_calibrated_confidence(self, text):
        """Predict with calibrated confidence scores"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        features = self.extract_features(text).reshape(1, -1)
        prob = self.model.predict_proba(features)[0]
        prediction = self.model.predict(features)[0]

        # Calibrate confidence based on proximity to decision boundary
        raw_confidence = max(prob)
        decision_margin = abs(prob[1] - prob[0])

        # Adjust confidence: lower if close to boundary (0.5)
        if decision_margin < 0.2:
            calibrated_confidence = raw_confidence * 0.8
            uncertainty = "HIGH"
        elif decision_margin < 0.4:
            calibrated_confidence = raw_confidence * 0.9
            uncertainty = "MEDIUM"
        else:
            calibrated_confidence = raw_confidence
            uncertainty = "LOW"

        return {
            'is_ai': bool(prediction),
            'human_probability': prob[0],
            'ai_probability': prob[1],
            'calibrated_confidence': calibrated_confidence,
            'uncertainty': uncertainty,
            'decision_margin': decision_margin
        }
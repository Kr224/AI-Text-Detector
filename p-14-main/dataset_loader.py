import pandas as pd
import numpy as np
import json
import os
from utils_seed import set_global_seed  # For reproducible dataset handling
from sklearn.model_selection import train_test_split
from typing import List, Dict  # Type hints for list/dict structures

class DatasetLoader:
    def __init__(self, data_dir='data'):
        # Set a fixed random seed for data loading and splitting
        set_global_seed(42)
        self.data_dir = data_dir

    def load_large_dataset(self, dataset_path='data/processed/large_dataset.csv', min_samples=1000):
        """Load the large dataset (1000+ samples)"""
        if os.path.exists(dataset_path):
            print(f"Loading large dataset from {dataset_path}")
            df = pd.read_csv(dataset_path)

            # Filter out any problematic texts
            df = df.dropna(subset=['text'])
            df = df[df['text'].str.len() > 30]

            human_texts = df[df['label'] == 0]['text'].tolist()
            ai_texts = df[df['label'] == 1]['text'].tolist()

            print(f"Loaded {len(human_texts)} human texts and {len(ai_texts)} AI texts")

            # # ===== ADD NORMALIZATION =====
            # # Normalize text lengths to reduce bias
            # human_texts = self.normalize_text_lengths(human_texts)
            # ai_texts = self.normalize_text_lengths(ai_texts)
            # print(f"After normalization: {len(human_texts)} human, {len(ai_texts)} AI")


            # Check if we have enough samples
            if len(human_texts) < min_samples // 2 or len(ai_texts) < min_samples // 2:
                print(f"Warning: Dataset has less than {min_samples // 2} samples per class.")
                print("Consider generating more samples using create_large_dataset.py")

            # Balance if needed (but keep as many as possible)
            min_class_samples = min(len(human_texts), len(ai_texts))
            if min_class_samples < len(human_texts) or min_class_samples < len(ai_texts):
                print(f"Balancing dataset to {min_class_samples} samples each...")
                human_texts = human_texts[:min_class_samples]
                ai_texts = ai_texts[:min_class_samples]

            print(f"Final dataset: {len(human_texts)} human, {len(ai_texts)} AI")
            return human_texts, ai_texts
        else:
            print(f"Large dataset not found at {dataset_path}")
            print("Creating new large dataset...")
            # Import here to avoid circular imports
            try:
                from data_collector import DataCollector
                collector = DataCollector()
                collector.create_large_dataset(min_samples, dataset_path)
                return self.load_large_dataset(dataset_path, min_samples)
            except ImportError:
                print("Could not import DataCollector. Using synthetic dataset instead.")
                return self.create_synthetic_dataset_large(min_samples)

    def load_standard_datasets(
            self,
            dataset_path: str = "data/processed/real_dataset.csv",
            min_samples_per_class: int = 50,
            shuffle: bool = True,
            random_state: int = 42,
    ):
        """
        Load a unified real-world dataset and return human and AI texts.

        Expected CSV format:
            text,label
            <string>,0 or 1

        Where:
            label = 0  -> human-written text
            label = 1  -> AI-generated text

        This method can be used as the real dataset entry point.
        """
        import os
        import pandas as pd

        print("Loading datasets...")

        if not os.path.exists(dataset_path):
            print(f"[WARN] Real dataset not found at: {dataset_path}")
            print("Falling back to large synthetic dataset...")
            return self.load_large_dataset()

        print(f"[INFO] Loading real dataset from: {dataset_path}")
        df = pd.read_csv(dataset_path)

        # Basic sanity checks
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError(
                "Real dataset must contain 'text' and 'label' columns.\n"
                f"Current columns: {', '.join(df.columns.astype(str))}"
            )

        # Drop rows with missing text or label
        df = df.dropna(subset=["text", "label"]).copy()
        df["text"] = df["text"].astype(str).str.strip()
        df["label"] = df["label"].astype(int)

        # Optional shuffle
        if shuffle:
            df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

        # Split into human (0) and AI (1)
        human_df = df[df["label"] == 0]
        ai_df = df[df["label"] == 1]

        print(f"[INFO] Human samples: {len(human_df)}")
        print(f"[INFO] AI samples:    {len(ai_df)}")

        if len(human_df) < min_samples_per_class or len(ai_df) < min_samples_per_class:
            print(
                "[WARN] Real dataset has fewer samples per class than "
                f"min_samples_per_class={min_samples_per_class}."
            )

        human_texts = human_df["text"].tolist()
        ai_texts = ai_df["text"].tolist()

        return human_texts, ai_texts

    def load_collected_data(self):
        """Load data collected by our data collector"""
        human_texts = []
        ai_texts = []

        # Load human data
        human_files = ['reddit_posts.json', 'wikipedia.json', 'news.json']
        for file in human_files:
            path = os.path.join(self.data_dir, 'human', file)
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data:
                        if isinstance(item, dict) and 'text' in item:
                            human_texts.append(item['text'])

        # Load AI data
        ai_files = ['openai_generated.json', 'local_generated.json']
        for file in ai_files:
            path = os.path.join(self.data_dir, 'ai', file)
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data:
                        if isinstance(item, dict) and 'text' in item:
                            ai_texts.append(item['text'])

        # If no data found, create synthetic data
        if len(human_texts) < 10 or len(ai_texts) < 10:
            print("Insufficient data found. Creating synthetic dataset...")
            human_texts, ai_texts = self.create_synthetic_dataset()

        # Clean and filter
        human_texts = self.clean_texts(human_texts)
        ai_texts = self.clean_texts(ai_texts)

        print(f"Loaded {len(human_texts)} human texts and {len(ai_texts)} AI texts")
        return human_texts, ai_texts

    def create_synthetic_dataset(self):
        """Create a synthetic dataset if no real data is available"""
        print("Creating synthetic dataset for initial testing...")

        # Human-like texts (more varied, emotional, imperfect)
        human_texts = [
            "I can't believe how bad the traffic was today. Seriously, it took me an hour to go 5 miles!",
            "My cat just knocked over my coffee mug. Again. This is like the third time this week.",
            "Tried that new pizza place downtown. The crust was way too thick for my liking, but the toppings were decent.",
            "Finished that book everyone's talking about. Honestly, didn't see what the big deal was. Ending felt rushed.",
            "Why do I always forget my umbrella when it's raining? Left it at home again today, got completely soaked.",
            "The meeting today could have been an email. Wasted two hours discussing things we already decided last week.",
            "My garden is finally starting to grow! The tomatoes are coming in, but something's eating the lettuce leaves.",
            "Can't decide what to watch tonight. Scrolled through Netflix for 30 minutes and still haven't picked anything.",
            "Lost my keys again. Found them in the fridge next to the milk. No idea how they got there.",
            "That movie was so much better than I expected. The special effects were actually pretty good for a low budget film."
        ]

        # AI-like texts (more structured, formal, "perfect")
        ai_texts = [
            "The implementation of sustainable energy solutions represents a critical component in addressing contemporary environmental challenges and fostering ecological resilience.",
            "In order to optimize organizational efficiency, it is imperative to leverage data-driven methodologies and implement robust analytical frameworks.",
            "The intricate interplay between cognitive processes and neural mechanisms underscores the complexity of human consciousness and perceptual experiences.",
            "Contemporary pedagogical approaches must evolve to accommodate diverse learning modalities and cultivate critical thinking competencies among students.",
            "The exponential growth of digital information necessitates the development of sophisticated cybersecurity protocols to ensure data integrity and privacy protection.",
            "Advancements in quantum computing hold the potential to revolutionize computational paradigms and facilitate unprecedented problem-solving capabilities.",
            "The integration of artificial intelligence with traditional manufacturing processes can significantly enhance productivity and operational efficiency.",
            "Climate change mitigation strategies require coordinated international efforts and the implementation of comprehensive policy frameworks across multiple sectors.",
            "The proliferation of social media platforms has fundamentally transformed interpersonal communication dynamics and information dissemination patterns.",
            "Biotechnological innovations in genetic engineering present both unprecedented opportunities and complex ethical considerations for scientific communities."
        ]

        # Expand dataset with variations
        topics = ['technology', 'science', 'education', 'health', 'environment', 'business']

        for topic in topics:
            human_texts.append(
                f"I read an article about {topic} yesterday. Some of it was interesting, but parts were hard to follow.")
            human_texts.append(
                f"Not sure what I think about the latest {topic} news. Need to read more about it before forming an opinion.")

            ai_texts.append(
                f"The domain of {topic} encompasses a multitude of interdisciplinary considerations that warrant comprehensive examination and analytical scrutiny.")
            ai_texts.append(
                f"Contemporary developments in the field of {topic} necessitate a paradigm shift in traditional methodological approaches and theoretical frameworks.")

        return human_texts * 20, ai_texts * 20  # Multiply to get more samples

    def create_synthetic_dataset_large(self, total_samples=1000):
        """Create a large synthetic dataset (500 human, 500 AI)"""
        print(f"Creating large synthetic dataset with {total_samples} samples...")

        human_texts = []
        ai_texts = []

        # Generate human texts
        human_base = [
            "I'm not sure about this, but it seems like {}.",
            "Honestly, {} is way more complicated than it needs to be.",
            "Can't believe how {} that was!",
            "So {} just happened and I don't know what to think.",
            "Why does {} always have to be so difficult?",
            "Tried {} for the first time today. It was {}.",
            "My experience with {} was {} to say the least.",
            "Not impressed with {}. The {} was really {}.",
            "{} gets a {}/10 from me.",
            "Would I recommend {}? Probably not. The {} was {}."
        ]

        human_fillers = [
            ['the new update', 'that policy', 'this app', 'the service', 'the system'],
            ['confusing', 'frustrating', 'annoying', 'surprising', 'expected'],
            ['good', 'bad', 'okay', 'great', 'terrible'],
            ['that', 'this thing', 'it', 'everything', 'the whole situation']
        ]

        # Generate 500 human texts
        for i in range(total_samples // 2):
            template = np.random.choice(human_base)
            # Replace placeholders
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

        # Generate AI texts
        ai_base = [
            "The {} of {} necessitates a comprehensive reevaluation of {}.",
            "In order to optimize {}, it is imperative to implement robust {}.",
            "Contemporary {} paradigms must evolve to accommodate {}.",
            "The {} of {} has fundamentally transformed {}.",
            "{} represents a critical component in addressing challenges related to {}.",
            "The multifaceted nature of {} underscores the importance of {}.",
            "{} serves as a testament to the transformative potential of {}.",
            "A comprehensive understanding of {} is essential for navigating {}.",
            "The convergence of {} and {} has given rise to {}.",
            "{} embodies a paradigm shift in {}, challenging conventional {}."
        ]

        ai_fillers = [
            ['implementation', 'integration', 'optimization', 'development', 'deployment'],
            ['artificial intelligence', 'machine learning', 'data analytics', 'cloud computing', 'blockchain'],
            ['traditional frameworks', 'existing methodologies', 'current approaches', 'conventional systems',
             'established protocols'],
            ['organizational efficiency', 'data security', 'user experience', 'system performance',
             'operational excellence'],
            ['strategic initiatives', 'innovative solutions', 'technological advancements', 'digital transformation',
             'business processes']
        ]

        # Generate 500 AI texts
        for i in range(total_samples // 2):
            template = np.random.choice(ai_base)
            text = template
            filler_count = template.count('{}')
            for j in range(filler_count):
                filler_set = ai_fillers[j % len(ai_fillers)]
                text = text.replace('{}', np.random.choice(filler_set), 1)

            # Add AI characteristics
            if np.random.random() > 0.5:
                transitions = ['Furthermore,', 'Moreover,', 'Consequently,', 'Therefore,']
                text += f" {np.random.choice(transitions)} this development underscores the need for continued innovation."

            # Make more formal
            replacements = {
                'use': 'utilize',
                'make': 'fabricate',
                'help': 'facilitate',
                'show': 'demonstrate'
            }
            for old, new in replacements.items():
                if old in text:
                    text = text.replace(old, new)

            ai_texts.append(text)

        print(f"Created {len(human_texts)} human and {len(ai_texts)} AI texts")
        return human_texts, ai_texts

    def clean_texts(self, texts):
        """Clean and normalize texts"""
        cleaned = []
        for text in texts:
            if isinstance(text, str):
                # Remove excessive whitespace
                text = ' '.join(text.split())
                # Remove very short texts
                if len(text) > 30:
                    cleaned.append(text)
        return cleaned

    def prepare_train_test_split(self, human_texts, ai_texts, test_size=0.2, random_state=42):
        """Prepare train/test split"""
        # Combine and label
        X = human_texts + ai_texts
        y = [0] * len(human_texts) + [1] * len(ai_texts)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Human in train: {sum(1 for y in y_train if y == 0)}")
        print(f"AI in train: {sum(1 for y in y_train if y == 1)}")

        return X_train, X_test, y_train, y_test

    def load_preprocessed(self, filepath='data/processed/labeled_dataset.csv'):
        """Load preprocessed dataset"""
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            human_texts = df[df['label'] == 0]['text'].tolist()
            ai_texts = df[df['label'] == 1]['text'].tolist()
            return human_texts, ai_texts
        else:
            print(f"Preprocessed file not found: {filepath}")
            print("Using large dataset instead...")
            return self.load_large_dataset()

    # New
    def normalize_text_lengths(self, texts, target_length=200):
        """Normalize text lengths to reduce bias"""
        normalized = []
        for text in texts:
            words = text.split()
            if len(words) > target_length:
                # Truncate long texts
                normalized.append(' '.join(words[:target_length]))
            elif len(words) < target_length // 2:
                # Skip very short texts
                continue
            else:
                normalized.append(text)
        return normalized
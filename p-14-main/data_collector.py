
import os
import random

import pandas as pd

from utils_seed import set_global_seed
from typing import List, Dict


class DataCollector(object):
    """
    DataCollector is responsible for generating synthetic human and AI texts
    and saving them as a labeled CSV dataset.

    This version focuses on more realistic and less trivially separable
    writing styles for human vs AI text.
    """

    def __init__(self, openai_api_key=None):
        """
        openai_api_key is kept for future extension, but not used in this project.
        """
        # Make all random operations reproducible
        set_global_seed(42)

        self.openai_api_key = openai_api_key

        # Basic topic pool used for both humans and AI
        self.topics = [
            "work", "food", "travel", "music", "books",
            "technology", "health", "education", "sports",
            "movies", "family", "finance", "social media",
            "mental health", "online courses", "remote work"
        ]

        # High-level categories for human texts
        self.human_categories = [
            "casual_chat",
            "personal_story",
            "opinion",
            "question",
            "review",
            "rant"
        ]

        # Soft "AI-like" keywords used for AI_word_ratio feature.
        # IMPORTANT: we will also occasionally inject a few of these
        # into human texts so that AI_word_ratio is no longer 0 vs non-0.
        self.ai_keywords = [
            "framework",
            "underlying",
            "in this context",
            "robust",
            "comprehensive",
            "significant",
            "insight",
            "perspective",
            "dimension",
            "dynamic",
        ]

        # Informal expressions, typos and emojis for human texts
        self.human_slang = [
            "lol", "tbh", "idk", "kinda", "sort of",
            "btw", "ngl", "omg", "haha", "lmao"
        ]
        self.human_emojis = [
            "😂", "😅", "🙂", "🙃", "🥲", "🤦‍♂️", "🤷‍♀️"
        ]

        # Transition phrases often used by modern LLMs
        self.ai_transitions = [
            "From a practical point of view, ",
            "In many everyday situations, ",
            "On the other hand, ",
            "That said, ",
            "However, ",
            "In addition, ",
            "More broadly speaking, ",
        ]

        # Soft openings and closings for AI-style paragraphs
        self.ai_openings = [
            "When we talk about {topic}, it helps to separate a few different aspects.",
            "The way people experience {topic} can vary a lot depending on their background.",
            "In recent years, {topic} has started to influence daily life more than most people expect.",
            "At a basic level, {topic} is about how we organize our time, energy and priorities.",
        ]

        self.ai_closings = [
            "Overall, there is no single right answer, but paying attention to context usually helps.",
            "In practice, small consistent habits matter more than any one big decision.",
            "So the real challenge is balancing convenience, long-term impact and personal values.",
            "Ultimately, the best approach depends on individual goals and constraints.",
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_large_dataset(self, num_samples=1000, output_file="data/processed/large_dataset.csv"):
        """
        Create a synthetic dataset with more realistic writing styles and
        save it as a CSV file.

        The dataset will contain both human and AI generated texts.
        """
        if num_samples < 2:
            raise ValueError("num_samples must be at least 2")

        set_global_seed(42)

        half = num_samples // 2
        human_samples = self.create_human_texts(half)
        ai_samples = self.create_ai_texts(num_samples - half)

        df = self.create_labeled_dataset(human_samples, ai_samples)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False, encoding="utf-8")
        print("[INFO] Large dataset saved to {} with {} samples".format(output_file, len(df)))

    # ------------------------------------------------------------------
    # Human / AI wrapper methods (keep original interface)
    # ------------------------------------------------------------------

    def create_human_texts(self, num_samples):
        """
        Wrapper kept for backward compatibility.

        Internally uses the more realistic generator.
        """
        return self.create_realistic_human_texts(num_samples)

    def create_ai_texts(self, num_samples):
        """
        Wrapper kept for backward compatibility.

        Internally uses the more realistic generator.
        """
        return self.create_realistic_ai_texts(num_samples)

    # ------------------------------------------------------------------
    # Realistic human text generation
    # ------------------------------------------------------------------

    def create_realistic_human_texts(self, num_samples):
        """
        Generate more realistic human texts with:
        1) Informal expressions and slang,
        2) Minor typos and imperfect grammar,
        3) Natural sentence structures,
        4) Emotions and personal opinions,
        5) Occasional use of "AI-like" words to avoid trivial separation.
        """
        set_global_seed(42)

        samples = []
        for _ in range(num_samples):
            topic = random.choice(self.topics)
            category = random.choice(self.human_categories)

            if category == "casual_chat":
                template = (
                    "Tbh I've been thinking about {topic} a lot lately and it kinda "
                    "messes with my schedule sometimes."
                )
            elif category == "personal_story":
                template = (
                    "Last week I had this random experience with {topic} and it "
                    "was way more intense than I expected."
                )
            elif category == "opinion":
                template = (
                    "I honestly feel like {topic} is getting a bit overhyped, "
                    "but at the same time I sort of understand why people care."
                )
            elif category == "question":
                template = (
                    "Does anyone else feel weird about how {topic} keeps changing "
                    "every few months, or is it just me?"
                )
            elif category == "review":
                template = (
                    "I tried something new related to {topic} yesterday and overall "
                    "it was decent, not amazing but not terrible either."
                )
            else:  # rant
                template = (
                    "I'm kinda tired of hearing about {topic} all the time, "
                    "it just pops up everywhere and it's low-key exhausting."
                )

            text = template.format(topic=topic)

            # Inject emotional emphasis or extra clauses
            if random.random() < 0.5:
                extra = random.choice([
                    " Honestly, it got stuck in my head the whole day.",
                    " It sounds small but it really threw off my mood.",
                    " I know it shouldn't bother me this much but it does.",
                    " Maybe I'm overthinking it, but still.",
                ])
                text += extra

            # Occasionally insert one AI-like keyword so AI_word_ratio is not 0
            if random.random() < 0.25:
                keyword = random.choice(self.ai_keywords)
                insert_phrase = f" From my perspective the {keyword} of the whole thing is how people adapt over time."
                text += insert_phrase

            # Add slang, emojis, typos, and casual punctuation
            text = self._human_add_slang_and_emojis(text)
            text = self._human_add_typos(text)

            samples.append({
                "text": text,
                "source": "synthetic_human",
                "category": category,
                "topic": topic,
            })

        return samples

    def _human_add_slang_and_emojis(self, text: str) -> str:
        """
        Add informal tokens and emojis at a moderate rate.
        This increases natural variation without making the text unreadable.
        """
        # Chance to append a slang token
        if random.random() < 0.5:
            slang = random.choice(self.human_slang)
            text += " " + slang

        # Chance to append an emoji
        if random.random() < 0.35:
            emoji = random.choice(self.human_emojis)
            text += " " + emoji

        # Chance to add ellipsis or repeated punctuation
        if random.random() < 0.4:
            ending = random.choice(["...", "?!", "!!!"])
            text += ending

        return text

    def _human_add_typos(self, text: str) -> str:
        """
        Introduce small, controlled typos to simulate human imperfections.
        """
        words = text.split()
        if len(words) < 4:
            return text

        typo_indices = random.sample(range(len(words)), k=max(1, len(words) // 15))
        for idx in typo_indices:
            w = words[idx]
            if len(w) > 4 and w.isalpha() and random.random() < 0.5:
                # Simple typo: duplicate a random character
                pos = random.randint(1, len(w) - 2)
                w = w[:pos] + w[pos] + w[pos:]
                words[idx] = w

        return " ".join(words)

    # ------------------------------------------------------------------
    # Realistic AI text generation
    # ------------------------------------------------------------------

    def create_realistic_ai_texts(self, num_samples):
        """
        Generate more realistic AI texts that:
        1) Do not rely on a single set of obvious keywords,
        2) Approximate modern LLM explanatory style,
        3) Use natural transitions and mixed sentence lengths,
        4) Are not perfectly formal (occasional contractions / mild informality).
        """
        set_global_seed(42)

        samples = []
        for _ in range(num_samples):
            topic = random.choice(self.topics)

            opening = random.choice(self.ai_openings).format(topic=topic)
            transition1 = random.choice(self.ai_transitions)
            transition2 = random.choice(self.ai_transitions)

            # Body sentences with mild structure and occasional contractions
            body1 = (
                f"{transition1}one way to look at {topic} is to check how it "
                f"affects people's daily routines, stress levels and long-term plans."
            )
            body2 = (
                f"{transition2}it's usually not just about {topic} itself, "
                f"but also about expectations, social pressure and available resources."
            )

            closing = random.choice(self.ai_closings)

            text = " ".join([opening, body1, body2, closing])

            # Occasionally insert a few AI-style keywords, but not too many
            if random.random() < 0.6:
                keyword = random.choice(self.ai_keywords)
                insert = (
                    f" In this context, a simple {keyword} can help people reason "
                    f"about trade-offs without getting lost in details."
                )
                text += insert

            # Add very light informal touches so the style is not unrealistically perfect
            if random.random() < 0.3:
                text += " In practice, it doesn't always work out as neatly as it sounds."

            samples.append({
                "text": text,
                "source": "synthetic_ai",
                "category": "formal_explanation",
                "topic": topic,
            })

        return samples

    # ------------------------------------------------------------------
    # Dataset creation
    # ------------------------------------------------------------------

    def create_labeled_dataset(
        self,
        human_samples: List[Dict[str, str]],
        ai_samples: List[Dict[str, str]]
    ) -> pd.DataFrame:
        """
        Combine human and AI samples into a single labeled DataFrame.

        Columns: text, label, source, category, topic
        label: 0 -> human, 1 -> ai
        """
        records = []

        for s in human_samples:
            records.append({
                "text": s["text"],
                "label": 0,
                "source": s.get("source", "synthetic_human"),
                "category": s.get("category", "unknown"),
                "topic": s.get("topic", "unknown"),
            })

        for s in ai_samples:
            records.append({
                "text": s["text"],
                "label": 1,
                "source": s.get("source", "synthetic_ai"),
                "category": s.get("category", "formal_explanation"),
                "topic": s.get("topic", "unknown"),
            })

        df = pd.DataFrame(records)
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        return df

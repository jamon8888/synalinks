"""
Explore Anthropic Interviewer Dataset

Analyzes the dataset structure and extracts insights to inform training.
"""

import pandas as pd
import re
from collections import Counter, defaultdict


def explore_dataset(split: str = "workforce"):
    """Explore and analyze the Anthropic Interviewer dataset."""

    splits = {
        "workforce": "interview_transcripts/workforce_transcripts.csv",
        "creatives": "interview_transcripts/creatives_transcripts.csv",
        "scientists": "interview_transcripts/scientists_transcripts.csv",
    }

    print("=" * 70)
    print(f"EXPLORING ANTHROPIC INTERVIEWER DATASET - {split.upper()}")
    print("=" * 70)

    # Load dataset
    print(f"\nLoading dataset...")
    df = pd.read_csv("hf://datasets/Anthropic/AnthropicInterviewer/" + splits[split])

    print(f"\n1. DATASET OVERVIEW")
    print("-" * 70)
    print(f"Number of interviews: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Shape: {df.shape}")

    # Show first few rows
    print(f"\n2. SAMPLE DATA")
    print("-" * 70)
    print(df.head())

    # Column info
    print(f"\n3. COLUMN INFORMATION")
    print("-" * 70)
    print(df.info())

    # Analyze transcript structure
    print(f"\n4. TRANSCRIPT ANALYSIS")
    print("-" * 70)

    # Find the transcript column
    transcript_col = None
    for col in ["transcript", "text", "interview", "content", "conversation"]:
        if col in df.columns:
            transcript_col = col
            break

    if transcript_col is None:
        transcript_col = df.columns[0]

    print(f"Transcript column: {transcript_col}")

    # Analyze first transcript
    if len(df) > 0:
        sample_transcript = df[transcript_col].iloc[0]
        print(f"\nSample transcript (first 500 chars):")
        print("-" * 70)
        print(str(sample_transcript)[:500])
        print("...")

        # Count turns
        turns = extract_turns(sample_transcript)
        print(f"\nNumber of turns in first interview: {len(turns)}")
        print(f"Interviewer turns: {sum(1 for t in turns if t['role'] == 'interviewer')}")
        print(f"Participant turns: {sum(1 for t in turns if t['role'] == 'participant')}")

    # Statistics across all transcripts
    print(f"\n5. DATASET STATISTICS")
    print("-" * 70)

    all_question_types = []
    all_lengths = []
    all_turn_counts = []

    for idx, row in df.iterrows():
        transcript = str(row[transcript_col])

        if pd.isna(transcript) or len(transcript) < 50:
            continue

        # Length
        all_lengths.append(len(transcript))

        # Turns
        turns = extract_turns(transcript)
        all_turn_counts.append(len(turns))

        # Question types
        for turn in turns:
            if turn["role"] == "interviewer":
                q_type = classify_question(turn["text"])
                all_question_types.append(q_type)

    print(f"\nTranscript lengths:")
    print(f"  Average: {sum(all_lengths)/len(all_lengths):.0f} characters")
    print(f"  Min: {min(all_lengths)}")
    print(f"  Max: {max(all_lengths)}")

    print(f"\nTurns per interview:")
    print(f"  Average: {sum(all_turn_counts)/len(all_turn_counts):.1f}")
    print(f"  Min: {min(all_turn_counts)}")
    print(f"  Max: {max(all_turn_counts)}")

    print(f"\nQuestion type distribution:")
    type_counts = Counter(all_question_types)
    for q_type, count in type_counts.most_common():
        pct = (count / len(all_question_types)) * 100
        print(f"  {q_type:15s}: {count:4d} ({pct:5.1f}%)")

    # Analyze content patterns
    print(f"\n6. CONTENT PATTERNS")
    print("-" * 70)

    all_text = " ".join(df[transcript_col].astype(str).values)

    # Common pain point indicators
    pain_words = ["struggle", "difficult", "hard", "frustrating", "annoying", "problem", "challenge"]
    need_words = ["need", "want", "wish", "would like", "looking for"]
    task_words = ["spend time", "have to", "need to", "work on"]

    print(f"\nPain point indicators:")
    for word in pain_words:
        count = len(re.findall(r'\b' + word + r'\b', all_text, re.IGNORECASE))
        print(f"  '{word}': {count} occurrences")

    print(f"\nNeed indicators:")
    for word in need_words:
        count = len(re.findall(r'\b' + word + r'\b', all_text, re.IGNORECASE))
        print(f"  '{word}': {count} occurrences")

    # Extract common topics
    print(f"\n7. COMMON TOPICS (based on keywords)")
    print("-" * 70)

    topics = {
        "coding": ["code", "programming", "development", "software", "debug"],
        "testing": ["test", "testing", "qa", "quality assurance"],
        "documentation": ["document", "documentation", "readme", "wiki"],
        "automation": ["automate", "automation", "script", "workflow"],
        "collaboration": ["team", "collaborate", "meeting", "communication"],
        "data": ["data", "analysis", "dataset", "database"],
    }

    topic_counts = {}
    for topic, keywords in topics.items():
        count = sum(
            len(re.findall(r'\b' + kw + r'\b', all_text, re.IGNORECASE))
            for kw in keywords
        )
        topic_counts[topic] = count

    for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {topic:15s}: {count:4d} mentions")

    # Training recommendations
    print(f"\n8. TRAINING RECOMMENDATIONS")
    print("-" * 70)
    print(f"✓ Dataset has {len(df)} interviews - good size for training")
    print(f"✓ Average {sum(all_turn_counts)/len(all_turn_counts):.0f} turns per interview")
    print(f"✓ Diverse question types detected")
    print(f"\nRecommended training configuration:")
    print(f"  - Epochs: 3-5")
    print(f"  - Batch size: 1 (for best results)")
    print(f"  - Validation split: 0.2")
    print(f"  - Focus on {type_counts.most_common(1)[0][0]} questions (most common)")

    return df


def extract_turns(transcript: str):
    """Extract conversation turns from transcript."""
    turns = []
    pattern = r"(Interviewer|Participant|User|Agent|AI|Human):\s*(.*?)(?=(?:Interviewer|Participant|User|Agent|AI|Human):|$)"
    matches = re.findall(pattern, transcript, re.DOTALL | re.IGNORECASE)

    for role, text in matches:
        role_normalized = "interviewer" if role.lower() in ["interviewer", "agent", "ai"] else "participant"
        turns.append({"role": role_normalized, "text": text.strip()})

    return turns


def classify_question(text: str) -> str:
    """Classify question type."""
    text_lower = text.lower()

    if any(p in text_lower for p in ["anything else", "final", "wrap up"]):
        return "concluding"
    elif any(p in text_lower for p in ["you mean", "clarify", "elaborate"]):
        return "clarifying"
    elif any(p in text_lower for p in ["you said", "earlier", "mentioned"]):
        return "follow_up"
    elif any(p in text_lower for p in ["who", "what specific", "when", "where", "how many", "how much"]):
        return "specific"
    else:
        return "open_ended"


if __name__ == "__main__":
    try:
        # Explore all splits
        for split in ["workforce", "creatives", "scientists"]:
            df = explore_dataset(split)
            print("\n" + "=" * 70)
            print("\n")

    except FileNotFoundError:
        print("\n❌ Dataset not found. Make sure you have huggingface_hub installed:")
        print("   pip install huggingface_hub")
        print("\nOr check the dataset at:")
        print("   https://huggingface.co/datasets/Anthropic/AnthropicInterviewer")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

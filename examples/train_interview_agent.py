"""
Train Interview Agent on Anthropic Interviewer Dataset

This script trains the Interview Agent using real interview transcripts from
the Anthropic Interviewer dataset to learn effective questioning strategies.

Dataset: https://huggingface.co/datasets/Anthropic/AnthropicInterviewer
Contains interview transcripts from workforce, creatives, and scientists.
"""

import asyncio
import pandas as pd
from typing import List, Dict, Tuple
import re
from datetime import datetime

import synalinks
from synalinks.src.modules.interview_agent import (
    InterviewAgent,
    PRDGenerator,
    EvolutionTracker,
    User,
    InterviewSession,
    Requirement,
    UseCase,
    Technology,
    Goal,
    InterviewQuestion,
    AnalyzedResponse,
    InterviewResult,
)


# ============================================================================
# DATASET LOADING
# ============================================================================


def load_anthropic_dataset(split: str = "workforce") -> pd.DataFrame:
    """
    Load the Anthropic Interviewer dataset.

    Args:
        split: One of 'workforce', 'creatives', 'scientists'

    Returns:
        DataFrame with interview transcripts
    """
    splits = {
        "workforce": "interview_transcripts/workforce_transcripts.csv",
        "creatives": "interview_transcripts/creatives_transcripts.csv",
        "scientists": "interview_transcripts/scientists_transcripts.csv",
    }

    if split not in splits:
        raise ValueError(f"Split must be one of {list(splits.keys())}")

    print(f"Loading {split} dataset from HuggingFace...")
    df = pd.read_csv("hf://datasets/Anthropic/AnthropicInterviewer/" + splits[split])

    print(f"✓ Loaded {len(df)} interview transcripts")
    print(f"  Columns: {list(df.columns)}")

    return df


# ============================================================================
# DATA PROCESSING
# ============================================================================


def parse_interview_transcript(transcript: str) -> List[Dict[str, str]]:
    """
    Parse interview transcript into question-answer pairs.

    Expected format:
    Interviewer: [question]
    Participant: [response]
    ...

    Returns:
        List of {'role': 'interviewer'|'participant', 'text': '...'} dicts
    """
    turns = []

    # Split by "Interviewer:" or "Participant:" markers
    pattern = r"(Interviewer|Participant):\s*(.*?)(?=(?:Interviewer|Participant):|$)"
    matches = re.findall(pattern, transcript, re.DOTALL | re.IGNORECASE)

    for role, text in matches:
        turns.append(
            {"role": role.lower(), "text": text.strip()}
        )

    return turns


def extract_requirements_from_response(
    response_text: str, question_text: str
) -> List[Requirement]:
    """
    Extract potential requirements from participant responses.

    Uses heuristics to identify:
    - Pain points ("I struggle with...", "It's hard to...")
    - Needs ("I need...", "I want...", "I wish...")
    - Tasks ("I spend time on...", "I have to...")
    """
    requirements = []

    # Patterns that indicate requirements
    pain_patterns = [
        r"(?:struggle|difficult|hard|challenging|frustrating|annoying).{0,50}(?:with|to)\s+(.{10,100})",
        r"(?:takes?|spend)\s+(?:too much |a lot of )?time\s+(?:on |to |doing )?(.{10,100})",
        r"(?:wish|want|need)\s+(?:to |I could |I had )?(.{10,100})",
    ]

    for i, pattern in enumerate(pain_patterns):
        matches = re.findall(pattern, response_text, re.IGNORECASE)
        for match in matches:
            # Clean up the matched text
            text = match.strip().rstrip(".,;:")

            if len(text) > 15:  # Meaningful requirement
                requirements.append(
                    Requirement(
                        label="Requirement",
                        name=f"Requirement {len(requirements) + 1}",
                        description=text,
                        priority=3,  # Default medium priority
                        category="general",
                        confidence_score=0.5,  # Heuristic extraction = lower confidence
                        discovered_at=datetime.now().isoformat(),
                    )
                )

    return requirements


def process_transcript_to_session(
    transcript: str,
    metadata: Dict = None,
    split: str = "workforce",
) -> Tuple[InterviewSession, List[Tuple[InterviewQuestion, str, AnalyzedResponse]]]:
    """
    Convert a raw transcript into an InterviewSession with training data.

    Args:
        transcript: Raw transcript text
        metadata: Additional metadata from dataset
        split: Dataset split (for categorization)

    Returns:
        (InterviewSession, training_examples)
        where training_examples = [(question, response, analysis), ...]
    """
    # Parse transcript
    turns = parse_interview_transcript(transcript)

    # Create user from metadata
    user = User(
        label="User",
        name=metadata.get("participant_name", "Anonymous") if metadata else "Anonymous",
        role=metadata.get("role", "Professional") if metadata else "Professional",
        experience_level=metadata.get("experience", "intermediate") if metadata else "intermediate",
        industry=split,  # workforce, creatives, scientists
    )

    # Initialize session
    session = InterviewSession(
        user=user,
        messages=[],
        requirements=[],
        use_cases=[],
        technologies=[],
        goals=[],
    )

    # Process turns into training examples
    training_examples = []
    current_question = None

    for turn in turns:
        if turn["role"] == "interviewer":
            # Create question object
            current_question = InterviewQuestion(
                text=turn["text"],
                question_type=classify_question_type(turn["text"]),
                purpose="Extract information from participant",
                is_concluding=False,
            )

        elif turn["role"] == "participant" and current_question:
            # Analyze response
            response_text = turn["text"]

            # Extract requirements using heuristics
            extracted_reqs = extract_requirements_from_response(
                response_text, current_question.text
            )

            # Create analysis
            analysis = AnalyzedResponse(
                extracted_requirements=extracted_reqs,
                extracted_use_cases=[],
                extracted_technologies=[],
                extracted_goals=[],
                sentiment=classify_sentiment(response_text),
                information_gain=calculate_information_gain(response_text),
                confidence=0.6,  # Heuristic-based
                follow_up_topics=[],
            )

            # Add to session
            session.requirements.extend(extracted_reqs)
            session.questions_asked += 1

            # Store training example
            training_examples.append((current_question, response_text, analysis))

            current_question = None

    return session, training_examples


def classify_question_type(question_text: str) -> str:
    """
    Classify question type based on patterns.

    Returns: 'open_ended', 'specific', 'follow_up', 'clarifying', or 'concluding'
    """
    question_lower = question_text.lower()

    # Concluding patterns
    if any(p in question_lower for p in ["anything else", "final", "last question", "to wrap up"]):
        return "concluding"

    # Clarifying patterns
    if any(p in question_lower for p in ["you mean", "clarify", "you mentioned", "elaborate"]):
        return "clarifying"

    # Follow-up patterns
    if any(p in question_lower for p in ["you said", "earlier you", "going back to", "regarding"]):
        return "follow_up"

    # Specific patterns (who, what, when, where, how many, how much)
    if any(p in question_lower for p in ["who", "what specific", "when", "where", "how many", "how much", "which"]):
        return "specific"

    # Default: open-ended
    return "open_ended"


def classify_sentiment(text: str) -> str:
    """Simple sentiment classification."""
    positive_words = ["love", "great", "excellent", "helpful", "enjoy", "like", "good"]
    negative_words = ["hate", "difficult", "struggle", "frustrating", "annoying", "hard", "problem"]

    text_lower = text.lower()
    pos_count = sum(1 for w in positive_words if w in text_lower)
    neg_count = sum(1 for w in negative_words if w in text_lower)

    if pos_count > neg_count:
        return "positive"
    elif neg_count > pos_count:
        return "negative"
    else:
        return "neutral"


def calculate_information_gain(text: str) -> float:
    """
    Estimate information gain from response length and detail.

    More detailed responses = higher information gain.
    """
    # Basic heuristic: longer, more detailed responses = more information
    word_count = len(text.split())

    if word_count < 10:
        return 0.2
    elif word_count < 30:
        return 0.4
    elif word_count < 60:
        return 0.6
    elif word_count < 100:
        return 0.8
    else:
        return 1.0


# ============================================================================
# TRAINING PIPELINE
# ============================================================================


async def prepare_training_data(
    df: pd.DataFrame,
    split: str = "workforce",
    max_samples: int = None,
) -> List[Tuple[InterviewSession, InterviewResult]]:
    """
    Prepare training data from DataFrame.

    Args:
        df: DataFrame with interview transcripts
        split: Dataset split name
        max_samples: Maximum number of samples to process

    Returns:
        List of (input_session, expected_result) tuples
    """
    training_data = []

    # Determine which column contains the transcript
    transcript_col = None
    for col in ["transcript", "text", "interview", "content"]:
        if col in df.columns:
            transcript_col = col
            break

    if transcript_col is None:
        # Use first text column
        transcript_col = df.columns[0]
        print(f"⚠ Transcript column not found, using: {transcript_col}")

    print(f"\nProcessing {len(df)} transcripts from '{transcript_col}' column...")

    if max_samples:
        df = df.head(max_samples)

    for idx, row in df.iterrows():
        try:
            transcript = row[transcript_col]

            if pd.isna(transcript) or len(str(transcript)) < 50:
                continue

            # Extract metadata
            metadata = {
                "participant_name": row.get("name", f"Participant_{idx}"),
                "role": row.get("role", "Professional"),
                "experience": row.get("experience", "intermediate"),
            }

            # Process transcript
            session, training_examples = process_transcript_to_session(
                str(transcript), metadata, split
            )

            # Create expected result
            expected_result = InterviewResult(
                session=session,
                prd=None,  # Will be generated
                conversation=[],
                confidence_score=0.8,  # From real interview
                questions_asked=len(training_examples),
            )

            training_data.append((session, expected_result))

            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(df)} transcripts...")

        except Exception as e:
            print(f"  ⚠ Error processing transcript {idx}: {e}")
            continue

    print(f"✓ Prepared {len(training_data)} training samples")
    return training_data


async def train_interview_agent(
    training_data: List[Tuple[InterviewSession, InterviewResult]],
    epochs: int = 5,
    model: str = "anthropic/claude-sonnet-4",
):
    """
    Train the Interview Agent using OMEGA optimization.

    Args:
        training_data: Prepared training examples
        epochs: Number of training epochs
        model: Language model to use
    """
    print("\n" + "=" * 70)
    print("TRAINING INTERVIEW AGENT")
    print("=" * 70)

    # 1. Setup models
    print("\n1. Initializing models...")
    lm = synalinks.LanguageModel(model=model)
    embedding_model = synalinks.EmbeddingModel(model="openai/text-embedding-3-large")

    # 2. Setup knowledge base
    print("2. Setting up knowledge base...")
    kb = synalinks.KnowledgeBase(
        uri="neo4j://localhost:7687",
        entity_models=[User, Requirement, UseCase, Technology, Goal],
        embedding_model=embedding_model,
        wipe_on_start=False,  # Keep historical data
    )

    # 3. Create interview agent
    print("3. Creating interview agent...")
    interview_agent = InterviewAgent(
        language_model=lm,
        knowledge_base=kb,
        max_questions=20,
    )

    # 4. Create program
    print("4. Building program...")
    session_input = synalinks.Input(shape=(InterviewSession,))
    interview_result = interview_agent(session_input)

    program = synalinks.Program(
        inputs=session_input,
        outputs=interview_result,
        name="InterviewAgentTraining",
    )

    # 5. Define reward function
    print("5. Defining reward function...")

    # Custom reward that evaluates interview quality
    class InterviewQualityReward(synalinks.rewards.Reward):
        """Reward based on interview quality metrics."""

        async def call(self, prediction, ground_truth):
            """
            Evaluate interview quality.

            Higher reward for:
            - More requirements discovered
            - Higher information gain per question
            - Good question type distribution
            - Higher confidence scores
            """
            pred_result = prediction
            true_result = ground_truth

            score = 0.0

            # Requirement discovery (40%)
            pred_reqs = len(pred_result.session.requirements)
            true_reqs = len(true_result.session.requirements)
            if true_reqs > 0:
                score += 0.4 * min(1.0, pred_reqs / true_reqs)

            # Efficiency (30%) - fewer questions, more info
            if pred_result.questions_asked > 0:
                efficiency = pred_reqs / pred_result.questions_asked
                score += 0.3 * min(1.0, efficiency / 0.5)  # Target: 0.5 reqs/question

            # Confidence (30%)
            score += 0.3 * pred_result.confidence_score

            return score

    reward = InterviewQualityReward()

    # 6. Compile with OMEGA optimizer
    print("6. Compiling with OMEGA optimizer...")
    program.compile(
        optimizer=synalinks.optimizers.OMEGA(
            language_model=lm,
            embedding_model=embedding_model,
        ),
        reward=reward,
    )

    # 7. Prepare training split
    print(f"7. Preparing training data ({len(training_data)} samples)...")
    x_train = [x for x, _ in training_data]
    y_train = [y for _, y in training_data]

    # 8. Train!
    print(f"\n8. Training for {epochs} epochs...")
    print("=" * 70)

    history = await program.fit(
        x=x_train,
        y=y_train,
        validation_split=0.2,
        batch_size=1,
        epochs=epochs,
        callbacks=[
            synalinks.callbacks.CSVLogger("training_log.csv"),
            synalinks.callbacks.ProgressBar(),
        ],
    )

    print("\n" + "=" * 70)
    print("✓ Training complete!")
    print("=" * 70)

    # 9. Save trained model
    print("\n9. Saving trained model...")
    program.save("trained_interview_agent.json")
    program.save_variables("trained_interview_agent_vars.json")
    print("✓ Model saved to trained_interview_agent.json")

    return program, history


# ============================================================================
# EVALUATION
# ============================================================================


async def evaluate_agent(
    program: synalinks.Program,
    test_data: List[Tuple[InterviewSession, InterviewResult]],
):
    """Evaluate trained agent on test data."""
    print("\n" + "=" * 70)
    print("EVALUATING AGENT")
    print("=" * 70)

    results = {
        "avg_requirements": 0.0,
        "avg_questions": 0.0,
        "avg_confidence": 0.0,
        "avg_efficiency": 0.0,
    }

    for session, expected in test_data[:10]:  # Test on first 10
        result = await program(session)

        results["avg_requirements"] += len(result.session.requirements)
        results["avg_questions"] += result.questions_asked
        results["avg_confidence"] += result.confidence_score

        if result.questions_asked > 0:
            efficiency = len(result.session.requirements) / result.questions_asked
            results["avg_efficiency"] += efficiency

    # Average
    n = min(10, len(test_data))
    for key in results:
        results[key] /= n

    print(f"\nTest Results (n={n}):")
    print(f"  Average requirements discovered: {results['avg_requirements']:.1f}")
    print(f"  Average questions asked: {results['avg_questions']:.1f}")
    print(f"  Average confidence: {results['avg_confidence']:.0%}")
    print(f"  Average efficiency: {results['avg_efficiency']:.2f} reqs/question")

    return results


# ============================================================================
# MAIN
# ============================================================================


async def main():
    """
    Main training pipeline.

    Steps:
    1. Load Anthropic dataset
    2. Process transcripts
    3. Train agent with OMEGA
    4. Evaluate performance
    5. Save trained model
    """
    print("=" * 70)
    print("INTERVIEW AGENT TRAINING PIPELINE")
    print("Anthropic Interviewer Dataset")
    print("=" * 70)

    # Configuration
    SPLIT = "workforce"  # Choose: workforce, creatives, scientists
    MAX_SAMPLES = 50  # Limit for testing (set to None for all)
    EPOCHS = 3

    try:
        # 1. Load dataset
        df = load_anthropic_dataset(split=SPLIT)

        # 2. Prepare training data
        training_data = await prepare_training_data(
            df, split=SPLIT, max_samples=MAX_SAMPLES
        )

        if len(training_data) == 0:
            print("❌ No valid training data found. Check dataset format.")
            return

        # 3. Train agent
        program, history = await train_interview_agent(
            training_data=training_data,
            epochs=EPOCHS,
        )

        # 4. Evaluate
        test_split = int(len(training_data) * 0.8)
        test_data = training_data[test_split:]

        if test_data:
            results = await evaluate_agent(program, test_data)

        print("\n✅ Training pipeline completed successfully!")
        print("\nNext steps:")
        print("  1. Load trained model: program = synalinks.Program.load('trained_interview_agent.json')")
        print("  2. Use for interviews: result = await program(new_session)")
        print("  3. Monitor performance and retrain as needed")

    except FileNotFoundError:
        print("\n❌ Dataset not found. Install huggingface_hub:")
        print("   pip install huggingface_hub")
        print("\nOr download manually from:")
        print("   https://huggingface.co/datasets/Anthropic/AnthropicInterviewer")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

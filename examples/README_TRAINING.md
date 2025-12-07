# Training the Interview Agent

This guide explains how to train the Interview Agent using the Anthropic Interviewer dataset.

## Quick Start

### 1. Install Dependencies

```bash
pip install synalinks pandas huggingface_hub
```

### 2. Setup Graph Database (Optional but Recommended)

```bash
# Using Docker - Neo4j
docker run -d \
  --name neo4j \
  -p 7687:7687 \
  -p 7474:7474 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest

# Or MemGraph
docker run -d \
  --name memgraph \
  -p 7687:7687 \
  memgraph/memgraph
```

### 3. Explore the Dataset

First, understand the dataset structure:

```bash
python explore_anthropic_dataset.py
```

This will show you:
- Dataset structure and columns
- Number of interviews
- Question type distribution
- Common topics and patterns
- Training recommendations

### 4. Train the Agent

```bash
python train_interview_agent.py
```

This will:
1. Load the Anthropic Interviewer dataset
2. Parse interview transcripts
3. Extract requirements and patterns
4. Train the agent using OMEGA optimization
5. Save the trained model

**Training Configuration:**
- Default: 50 samples, 3 epochs (for testing)
- Full training: Set `MAX_SAMPLES = None` and `EPOCHS = 5-10`

**Expected Output:**
```
Training for 3 epochs...
Epoch 1/3: avg_reward=0.65 ...
Epoch 2/3: avg_reward=0.72 ...
Epoch 3/3: avg_reward=0.78 ...
✓ Model saved to trained_interview_agent.json
```

### 5. Use the Trained Agent

```bash
# Interactive mode
python use_trained_agent.py

# Batch mode (for testing)
python use_trained_agent.py batch
```

## Training Details

### Dataset: Anthropic Interviewer

The training uses real interview transcripts from Anthropic's interviewer dataset:
- **workforce**: Professional workers (developers, managers, etc.)
- **creatives**: Creative professionals (designers, writers, etc.)
- **scientists**: Research scientists and academics

Load with:
```python
import pandas as pd

splits = {
    'workforce': 'interview_transcripts/workforce_transcripts.csv',
    'creatives': 'interview_transcripts/creatives_transcripts.csv',
    'scientists': 'interview_transcripts/scientists_transcripts.csv'
}

df = pd.read_csv("hf://datasets/Anthropic/AnthropicInterviewer/" + splits["workforce"])
```

### Training Pipeline

```
Raw Transcripts
    ↓
Parse into Q&A pairs
    ↓
Extract requirements (heuristic)
    ↓
Create InterviewSession objects
    ↓
OMEGA Optimization
    ↓
Trained Interview Agent
```

### What the Agent Learns

1. **Effective Question Patterns**
   - When to ask open-ended vs. specific questions
   - How to follow up on interesting points
   - When to clarify vs. when to explore new areas

2. **Requirement Extraction**
   - Identifying pain points from responses
   - Recognizing implicit needs
   - Categorizing requirements appropriately

3. **Interview Flow**
   - Optimal question sequencing
   - When to deep-dive vs. move on
   - How to conclude effectively

4. **Context Awareness**
   - Using previous answers to inform new questions
   - Building on discovered requirements
   - Adapting to user type (experience level, industry)

### Reward Function

The agent is trained with a custom reward that optimizes for:

```python
InterviewQualityReward:
  - Requirement discovery (40%): More requirements discovered = higher reward
  - Efficiency (30%): Requirements per question ratio
  - Confidence (30%): Overall confidence in extracted information
```

### OMEGA Optimizer

Uses in-context reinforcement learning to optimize:
- Question generation prompts
- Response analysis instructions
- Follow-up strategy selection

No model weights are changed - only the prompts and strategies.

## Training Configurations

### Quick Test (5 minutes)
```python
MAX_SAMPLES = 10
EPOCHS = 1
```

### Standard Training (30 minutes)
```python
MAX_SAMPLES = 50
EPOCHS = 3
```

### Full Training (2-3 hours)
```python
MAX_SAMPLES = None  # All samples
EPOCHS = 5
VALIDATION_SPLIT = 0.2
```

### Advanced Training
```python
MAX_SAMPLES = None
EPOCHS = 10
BATCH_SIZE = 1
VALIDATION_SPLIT = 0.2

# Custom reward with domain-specific weights
reward = InterviewQualityReward(
    requirement_weight=0.5,  # Emphasize requirement discovery
    efficiency_weight=0.3,
    confidence_weight=0.2,
)
```

## Evaluation Metrics

After training, the agent is evaluated on:

1. **Requirements Discovered**: Average number of requirements per interview
2. **Questions Asked**: Average questions needed
3. **Efficiency**: Requirements per question ratio
4. **Confidence**: Average confidence score

**Example Results:**
```
Test Results (n=10):
  Average requirements discovered: 8.3
  Average questions asked: 14.2
  Average confidence: 78%
  Average efficiency: 0.58 reqs/question
```

## Customization

### Train on Specific Domain

```python
# Load only workforce data
df_workforce = load_anthropic_dataset(split="workforce")
training_data = await prepare_training_data(df_workforce)

# Train domain-specific agent
program = await train_interview_agent(training_data, epochs=5)
```

### Custom Requirement Extraction

Edit `extract_requirements_from_response()` in `train_interview_agent.py`:

```python
def extract_requirements_from_response(response_text, question_text):
    """Add domain-specific patterns."""
    requirements = []

    # Add your custom patterns
    custom_patterns = [
        r"(?:automate|automation).{10,80}",
        r"(?:integrate|integration).{10,80}",
        # ... your patterns
    ]

    # ... extraction logic
    return requirements
```

### Fine-tune on Your Data

```python
# Load your interview transcripts
your_df = pd.read_csv("your_interviews.csv")

# Process into training format
training_data = await prepare_training_data(
    your_df,
    split="custom",
    transcript_column="interview_text"
)

# Train
program = await train_interview_agent(training_data)
```

## Monitoring Training

Training logs are saved to `training_log.csv`:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load training history
history = pd.read_csv("training_log.csv")

# Plot reward over time
plt.plot(history['epoch'], history['reward'])
plt.xlabel('Epoch')
plt.ylabel('Reward')
plt.title('Training Progress')
plt.show()
```

## Troubleshooting

### Dataset Not Found
```bash
pip install huggingface_hub
# OR download manually from:
# https://huggingface.co/datasets/Anthropic/AnthropicInterviewer
```

### Out of Memory
```python
# Reduce batch size or max samples
MAX_SAMPLES = 20
BATCH_SIZE = 1
```

### Low Reward Scores
```python
# Train longer
EPOCHS = 10

# Or adjust reward weights
reward = InterviewQualityReward(
    requirement_weight=0.6,  # Increase if agent finds few requirements
    efficiency_weight=0.2,
    confidence_weight=0.2,
)
```

### Graph Database Connection Issues
```python
# Test connection
from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    "neo4j://localhost:7687",
    auth=("neo4j", "password")
)
driver.verify_connectivity()
```

## Best Practices

1. **Start Small**: Test with 10-20 samples first
2. **Monitor Progress**: Check `training_log.csv` after each epoch
3. **Validate Often**: Use 20% validation split
4. **Save Checkpoints**: Training saves model after completion
5. **Evaluate Before Deploying**: Test on held-out data

## Advanced Topics

### Multi-Split Training

Train on all splits (workforce + creatives + scientists):

```python
all_data = []
for split in ["workforce", "creatives", "scientists"]:
    df = load_anthropic_dataset(split)
    data = await prepare_training_data(df, split=split)
    all_data.extend(data)

program = await train_interview_agent(all_data, epochs=5)
```

### Transfer Learning

Start from a trained model and fine-tune:

```python
# Load pre-trained
program = synalinks.Program.load("trained_interview_agent.json")
program.load_variables("trained_interview_agent_vars.json")

# Fine-tune on new data
history = await program.fit(
    x=new_training_data,
    y=new_expected_results,
    epochs=3,
    validation_split=0.2,
)
```

### Ensemble Training

Train multiple agents with different configurations:

```python
# Train 3 agents with different strategies
agents = []
for i in range(3):
    agent = await train_interview_agent(
        training_data=training_data,
        epochs=5,
    )
    agents.append(agent)

# Use voting or averaging for final decisions
```

## Next Steps

After training:

1. **Evaluate**: Run `use_trained_agent.py batch` to test
2. **Deploy**: Integrate into your application
3. **Monitor**: Track real-world performance
4. **Retrain**: Periodically retrain with new data
5. **A/B Test**: Compare trained vs. base agent

## References

- [Anthropic Interviewer Dataset](https://huggingface.co/datasets/Anthropic/AnthropicInterviewer)
- [Synalinks Documentation](https://github.com/synalinks/synalinks)
- [OMEGA Optimizer Paper](https://arxiv.org/abs/...)
- [Interview Agent Architecture](../docs/interview_agent_architecture.md)

## Support

For issues or questions:
- Check the examples in `examples/`
- Review the architecture doc in `docs/interview_agent_architecture.md`
- Open an issue on GitHub

## License

Same as Synalinks main project.

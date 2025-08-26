# FlowPlan-G2P: Graph-Based Paragraph Planning for Structured Patent Description Generation

A Python implementation of the FlowPlan-G2P framework that converts scientific papers into patent descriptions using a three-stage graph-based approach.

## Overview

FlowPlan-G2P implements the methodology from the paper "FlowPlan-G2P: Graph-Based Paragraph Planning for Structured Patent Description Generation". The framework uses three sequential stages:

1. **Concept Graph Induction**: Extract key technical concepts and relationships using structured reasoning
2. **Paragraph Planning**: Organize concepts into coherent patent sections with proper ordering  
3. **Graph-Conditioned Generation**: Generate patent-style text conditioned on structured subgraphs

## Architecture

```
Scientific Paper → [Stage 1] → Concept Graph → [Stage 2] → Section Plan → [Stage 3] → Patent Description
```

### Stage 1: Concept Graph Induction
- 9-step structured reasoning: Field, TechProblem, PriorArt, Novelty, Solution, Implementation, Effects, Embodiment, Figure
- Generates multiple candidate graphs and merges them using LLM-based consolidation
- Creates directed graph with predefined relations: solves, implements, causes, improves, validates

### Stage 2: Paragraph Planning
- Clusters nodes into patent sections: Field, Background, Summary, Detailed Description, Effects
- Multi-sampling with lightweight gating based on connectivity and consistency metrics
- Enforces patent-specific constraints and prerequisite ordering

### Stage 3: Graph-Conditioned Generation
- Generates patent text using section-specific templates and boilerplate phrases
- Conditions generation on linearized subgraph representations
- Ensures legalistic phrasing and repetitive elaboration typical of patents

## Installation

### Requirements
- Python 3.8+
- AWS account with Bedrock access
- AWS credentials configured (via AWS CLI, environment variables, or IAM roles)

### Setup
```bash
# Clone or download the implementation
cd flowplan_g2p

# Install dependencies
pip install -r requirements.txt

# Configure AWS credentials (if not already done)
aws configure
```

### Dependencies
- `boto3>=1.26.137` - AWS SDK for Bedrock integration
- `json5>=0.9.6` - JSON parsing utilities
- `typing-extensions>=4.5.0` - Type hints support

## Usage

### Command Line Interface

```bash
# Basic usage
python main.py input_paper.md output_patent.md

# With custom parameters
python main.py input_paper.md output_patent.md --candidates 5 --k 3 --tau-c 0.6

# Export intermediate concept graph
python main.py input_paper.md output_patent.md --export-graph concept_graph.json

# Validate setup only
python main.py --validate-only
```

### Parameters

**Stage 1 (Concept Induction):**
- `--candidates N`: Number of concept graph candidates to generate and merge (default: 3)

**Stage 2 (Paragraph Planning):**
- `--k N`: Number of section plan candidates to generate (default: 5)
- `--tau-c FLOAT`: Connectivity threshold for section planning (default: 0.5)
- `--tau-s FLOAT`: Semantic consistency threshold (default: 0.6)

**LLM Configuration:**
- `--max-tokens N`: Maximum tokens per generation (default: 4096)
- `--temperature FLOAT`: Generation temperature (default: 0.2)
- `--aws-region REGION`: AWS region for Bedrock (default: us-east-1)

### Python API

```python
from flowplan_g2p.pipeline import FlowPlanG2P
from flowplan_g2p.llm_client import LLMConfig

# Configure LLM
config = LLMConfig(
    max_tokens=4096,
    temperature=0.2
)

# Initialize pipeline
pipeline = FlowPlanG2P(llm_config=config)

# Process paper to patent
patent_text = pipeline.process_paper_to_patent(
    input_paper_path="paper.md",
    output_patent_path="patent.md",
    num_candidates=3,
    k=5
)

# Access intermediate results
results = pipeline.get_intermediate_results()
concept_graph = results['concept_graph']
section_plan = results['section_plan']
```

### Testing

Test the pipeline with sample data:

```bash
python test_pipeline.py
```

This will run the complete pipeline on a sample from the Pap2Pat dataset and generate:
- `test_output_patent.md` - Generated patent description
- `test_concept_graph.json` - Intermediate concept graph (if successful)

## Input/Output Format

### Input
- **Format**: Markdown (.md) files
- **Content**: Scientific paper with standard academic structure (abstract, introduction, methods, results, etc.)
- **Encoding**: UTF-8

### Output  
- **Format**: Markdown (.md) files
- **Structure**: Patent description with canonical sections:
  - Field of the Invention
  - Background of the Invention
  - Summary of the Invention  
  - Detailed Description of the Invention
  - Effects of the Invention

## AWS Configuration

### Bedrock Setup
1. Ensure AWS Bedrock service is available in your region
2. Request access to Claude Sonnet 4 model (`us.anthropic.claude-sonnet-4-20250514-v1:0`)
3. Configure appropriate IAM permissions:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel"
            ],
            "Resource": [
                "arn:aws:bedrock:*::foundation-model/us.anthropic.claude-sonnet-4-20250514-v1:0"
            ]
        }
    ]
}
```

### Authentication
Configure AWS credentials using one of these methods:
- AWS CLI: `aws configure`
- Environment variables: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`
- IAM roles (for EC2/Lambda deployment)
- AWS credentials file

## Project Structure

```
flowplan_g2p/
├── flowplan_g2p/           # Main package
│   ├── __init__.py
│   ├── graph.py            # Custom graph data structures
│   ├── llm_client.py       # AWS Bedrock Claude integration
│   ├── pipeline.py         # Main pipeline orchestration
│   └── stages/             # Pipeline stages
│       ├── __init__.py
│       ├── concept_induction.py    # Stage 1
│       ├── paragraph_planning.py   # Stage 2
│       └── graph_generation.py     # Stage 3
├── main.py                 # Command-line interface
├── test_pipeline.py        # Test script
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Performance

Typical execution times (depends on paper length and AWS region):
- **Stage 1** (Concept Induction): 30-60 seconds
- **Stage 2** (Paragraph Planning): 10-20 seconds  
- **Stage 3** (Text Generation): 20-40 seconds
- **Total**: 1-2 minutes per paper

## Troubleshooting

### Common Issues

**AWS Connection Errors:**
- Verify AWS credentials are configured correctly
- Check that your region supports Bedrock
- Ensure Claude Sonnet 4 model access is granted

**Rate Limiting:**
- The client includes automatic retry with exponential backoff
- If you encounter persistent rate limits, increase retry delays or reduce concurrency

**Memory Issues:**
- For very large papers, consider reducing `max_tokens` parameter
- Process papers in smaller chunks if necessary

**Generation Quality:**
- Adjust `temperature` parameter (lower = more conservative)
- Increase `num_candidates` for better concept graph quality
- Tune `tau_c` and `tau_s` thresholds for section planning

### Validation

Always run validation before processing:
```bash
python main.py --validate-only
```

This checks:
- AWS Bedrock connectivity
- Model access permissions
- Pipeline component initialization

## Limitations

- Requires AWS Bedrock access with Claude Sonnet 4
- Optimized for English-language scientific papers
- Patent output follows US patent format conventions
- Processing time scales with paper length
- Quality depends on input paper structure and clarity

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{flowplan_g2p_2025,
  title={FlowPlan-G2P: Graph-Based Paragraph Planning for Structured Patent Description Generation},
  author={Anonymous ACL submission},
  year={2025}
}
```

## License

This implementation is provided for research purposes. Please refer to the original paper for licensing terms.
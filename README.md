# FlowPlan-G2P: Graph-Based Paragraph Planning for Structured Patent Description Generation

A Python implementation of the FlowPlan-G2P framework that converts scientific papers into patent descriptions using a three-stage graph-based approach. Supports multiple LLM providers: OpenAI, Anthropic, and AWS Bedrock.

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
- [uv](https://docs.astral.sh/uv/) - Fast Python package and project manager

### Quick Start

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/repoofideas/flowplan_g2p.git
cd flowplan_g2p

# Create virtual environment and install dependencies
uv sync
```

### Install with Bedrock Support

If you plan to use AWS Bedrock as your LLM provider, install the optional Bedrock dependencies:

```bash
pip install flowplan-g2p[bedrock]
# or with uv
uv sync --extra bedrock
```

### Environment Variables

Copy the `.env.example` file and fill in your API keys:

```bash
cp .env.example .env
```

Set the API key for your chosen provider:

```bash
# For OpenAI (default provider)
export OPENAI_API_KEY=your-openai-api-key-here

# For Anthropic
export ANTHROPIC_API_KEY=your-anthropic-api-key-here

# For AWS Bedrock, configure AWS credentials via AWS CLI or environment variables
aws configure
```

## Usage

### Command Line Interface

**OpenAI (default):**
```bash
uv run flowplan-g2p input_paper.md output_patent.md --provider openai
```

**Anthropic:**
```bash
uv run flowplan-g2p input_paper.md output_patent.md --provider anthropic
```

**AWS Bedrock:**
```bash
uv run flowplan-g2p input_paper.md output_patent.md --provider bedrock --aws-region us-west-2
```

**With custom parameters:**
```bash
uv run flowplan-g2p input_paper.md output_patent.md --candidates 5 --k 3 --tau-c 0.6
```

**Export intermediate concept graph:**
```bash
uv run flowplan-g2p input_paper.md output_patent.md --export-graph concept_graph.json
```

**Validate setup only:**
```bash
uv run flowplan-g2p --validate-only --provider openai
```

**Using main.py directly:**
```bash
uv run python main.py input_paper.md output_patent.md --provider anthropic
```

### Parameters

**Stage 1 (Concept Induction):**
- `--candidates N`: Number of concept graph candidates to generate and merge (default: 3)

**Stage 2 (Paragraph Planning):**
- `--k N`: Number of section plan candidates to generate (default: 5)
- `--tau-c FLOAT`: Connectivity threshold for section planning (default: 0.5)
- `--tau-s FLOAT`: Semantic consistency threshold (default: 0.6)

**LLM Configuration:**
- `--provider`: LLM provider — `openai` (default), `anthropic`, or `bedrock`
- `--model`: Model identifier (e.g., `gpt-4o`, `claude-sonnet-4-20250514`); uses provider default if omitted
- `--api-key`: API key for the selected provider (falls back to environment variable if omitted)
- `--max-tokens N`: Maximum tokens per generation (default: 4096)
- `--temperature FLOAT`: Generation temperature (default: 0.2)
- `--aws-region REGION`: AWS region for Bedrock (default: us-east-1)

### Python API

```python
from flowplan_g2p import FlowPlanG2P, LLMConfig

# OpenAI (default)
config = LLMConfig(provider="openai")
pipeline = FlowPlanG2P(llm_config=config)

# Anthropic with explicit API key
config = LLMConfig(provider="anthropic", api_key="your-anthropic-key")
pipeline = FlowPlanG2P(llm_config=config)

# AWS Bedrock
config = LLMConfig(provider="bedrock", aws_region="us-west-2")
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

Run the test suite (no API keys required):

```bash
uv run pytest tests/ -v
```

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

## Provider Notes

### OpenAI
- Default model: `gpt-4o`
- Requires `OPENAI_API_KEY` environment variable or `--api-key` flag

### Anthropic
- Default model: `claude-sonnet-4-20250514`
- Requires `ANTHROPIC_API_KEY` environment variable or `--api-key` flag

### AWS Bedrock
- Default model: `us.anthropic.claude-sonnet-4-20250514-v1:0`
- Requires AWS credentials configured via AWS CLI, environment variables, or IAM roles
- Install with: `pip install flowplan-g2p[bedrock]`
- Ensure Bedrock service is available in your region and you have `bedrock:InvokeModel` permissions for your chosen model

## Project Structure

```
flowplan_g2p/
├── flowplan_g2p/           # Main package
│   ├── __init__.py         # Package public API
│   ├── cli.py              # Command-line interface
│   ├── graph.py            # Graph data structures (ConceptGraph, ConceptNode, ConceptEdge)
│   ├── llm_client.py       # Multi-provider LLM client abstraction
│   ├── pipeline.py         # Main pipeline orchestration
│   └── stages/             # Pipeline stages
│       ├── __init__.py
│       ├── concept_induction.py    # Stage 1: Concept Graph Induction
│       ├── paragraph_planning.py   # Stage 2: Paragraph Planning
│       └── graph_generation.py     # Stage 3: Graph-Conditioned Generation
├── tests/                  # Test suite (runs without API keys)
│   ├── test_graph.py
│   ├── test_llm_client.py
│   ├── test_cli.py
│   └── test_pipeline.py
├── main.py                 # Alternative CLI entry point
├── .env.example            # Environment variable template
├── pyproject.toml          # Project configuration
├── Makefile                # Development commands
├── DEVELOPMENT.md          # Development guide
└── README.md               # This file
```

## Evaluation Dataset

The Pap2Pat dataset used for evaluation is available from the upstream repository:
[https://github.com/Franck-Dernoncourt/Pap2Pat](https://github.com/Franck-Dernoncourt/Pap2Pat)

## Performance

Typical execution times (depends on paper length and provider):
- **Stage 1** (Concept Induction): 30-60 seconds
- **Stage 2** (Paragraph Planning): 10-20 seconds
- **Stage 3** (Text Generation): 20-40 seconds
- **Total**: 1-2 minutes per paper

## Troubleshooting

### API Key Issues
- Ensure the correct environment variable is set for your provider (`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`)
- You can pass the key directly with `--api-key` to bypass environment variables
- Run `flowplan-g2p --validate-only --provider <your-provider>` to test connectivity

### Provider-Specific Issues

**OpenAI:**
- Verify your API key has access to the model you're using (default: `gpt-4o`)
- Check your OpenAI account for rate limits or billing issues

**Anthropic:**
- Verify your API key is valid and has access to the model (default: `claude-sonnet-4-20250514`)
- Check Anthropic console for rate limits

**AWS Bedrock:**
- Ensure `boto3` is installed: `pip install flowplan-g2p[bedrock]`
- Verify AWS credentials are configured: `aws sts get-caller-identity`
- Check that Bedrock is available in your region and model access is granted

### Rate Limiting
- The client includes automatic retry with exponential backoff
- If you encounter persistent rate limits, increase retry delays or reduce concurrency

### Generation Quality
- Adjust `temperature` parameter (lower = more conservative)
- Increase `num_candidates` for better concept graph quality
- Tune `tau_c` and `tau_s` thresholds for section planning

### Validation

Always run validation before processing:
```bash
uv run flowplan-g2p --validate-only --provider openai
```

This checks provider connectivity, model access, and pipeline component initialization.

## Limitations

- Optimized for English-language scientific papers
- Patent output follows US patent format conventions
- Processing time scales with paper length
- Quality depends on input paper structure and clarity

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@inproceedings{pan2026flowplan,
  title={FlowPlan-G2P: Graph-Based Paragraph Planning for Structured Patent Description Generation},
  author={Pan, Kris W and Yoo, Yongmin},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2026}
}
```

## License

This implementation is provided for research purposes. Please refer to the LICENSE file for terms.

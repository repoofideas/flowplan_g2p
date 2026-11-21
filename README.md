<div align="center">

# FlowPlan-G2P

**A Structured Generation Framework for Transforming Scientific Papers into Patent Descriptions**

Kris W Pan · Yongmin Yoo<br>
*Equal contribution*

Accepted to **Findings of EACL 2027**

[![arXiv paper](https://img.shields.io/badge/arXiv-2601.02589-B31B1B?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2601.02589)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-27476E?style=flat-square)](https://arxiv.org/pdf/2601.02589)
[![Apache 2.0 license](https://img.shields.io/badge/License-Apache_2.0-28745B?style=flat-square)](LICENSE)

[Quick start](#quick-start) · [Method](#method) · [Evaluation](#evaluation) · [Citation](#citation)

</div>

FlowPlan-G2P takes an English-language scientific paper in Markdown and drafts a patent description. It extracts the technical elements and how they depend on one another, then uses that graph to plan and draft the sections. You can export the concept graph as JSON or inspect both the graph and section plan from Python.

## Quick start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/). The repository pins Python 3.11.

```bash
git clone https://github.com/repoofideas/flowplan_g2p.git
cd flowplan_g2p
uv sync
```

Save your source paper as UTF-8 Markdown. Set your provider's credentials, then run:

```bash
uv run flowplan-g2p paper.md patent_description.md --export-graph concept_graph.json
```

This command uses OpenAI by default, reads `OPENAI_API_KEY` from the environment, and writes the draft and graph to the paths shown above.

| Provider | Credentials | CLI options |
| --- | --- | --- |
| OpenAI | `OPENAI_API_KEY` | `--provider openai` |
| Anthropic | `ANTHROPIC_API_KEY` | `--provider anthropic` |
| AWS Bedrock | AWS credentials with `bedrock:InvokeModel` access | Run `uv sync --extra bedrock`, then add `--provider bedrock --aws-region <region>` |

Run `uv run flowplan-g2p --help` for all options. Use `--model` to select a model ID for your provider.

## Method

```mermaid
flowchart LR
    A["Scientific paper"] -->|"1. Concept Graph Induction"| B["Directed concept graph"]
    B -->|"2. Section-Level Planning"| C["Ordered section subgraphs"]
    C -->|"3. Graph-Conditioned Generation"| D["Patent description"]
```

1. **Concept Graph Induction.** Extract technical content across nine drafting categories. Build one rule-based graph and two LLM-generated candidates by default. Merge nodes by identifier, resolve competing edge labels by vote, and repair missing required categories.
2. **Section-Level Planning.** Sample candidate assignments to Field, Background, Summary, Detailed Description, and Effects. Use intra-section connectivity and category homogeneity as gates. Rank candidates with a score that includes drafting-rule penalties, and select the highest-scoring plan as a fallback if no candidate passes.
3. **Graph-Conditioned Generation.** Serialize the concepts and internal edges of each selected section subgraph. Draft each section with its own instructions and assemble the generated text in section order.

<details>
<summary><strong>Planning defaults and Python API</strong></summary>

| Option | Default | Controls |
| --- | ---: | --- |
| `--candidates` | 3 | Concept graph candidates |
| `--k` | 5 | Section plan candidates |
| `--tau-c` | 0.5 | Connectivity threshold |
| `--tau-s` | 0.6 | Category homogeneity threshold |
| `--max-tokens` | 4096 | Maximum tokens per generation request |

```python
from flowplan_g2p import FlowPlanG2P, LLMConfig

pipeline = FlowPlanG2P(llm_config=LLMConfig(provider="openai"))
pipeline.process_paper_to_patent(
    input_paper_path="paper.md",
    output_patent_path="patent_description.md",
)

artifacts = pipeline.get_intermediate_results()
concept_graph = artifacts["concept_graph"]
section_plan = artifacts["section_plan"]
```

</details>

## Evaluation

The paper evaluates FlowPlan-G2P on **146 curated paper–patent pairs** from Pap2Pat-EvalGold. Four patent experts rated outputs on a subset of **75 paper–patent pairs**. [Pap2Pat-EvalGold and Pat-DEVAL](https://arxiv.org/abs/2601.00166) · [Upstream PAP2PAT dataset and code](https://github.com/boschresearch/Pap2Pat)

| Measure | Baseline | FlowPlan-G2P |
| --- | --- | --- |
| Expert legal-professional compliance (1–5) | Zero-shot · 1.5 | **4.7** |
| Overall Pat-DEVAL (1–5) | Few-shot Sonnet-4.5 · 2.3 | Llama-4-Scout with FlowPlan-G2P · **4.3** |

On the full benchmark, zero-shot prompting had the highest BERTScore and ROUGE-L among the systems in the surface-metric comparison. It received the lowest expert legal-professional compliance rating on the expert-rated subset. See the [paper](https://arxiv.org/abs/2601.02589) for the full comparisons and ablations.

FlowPlan-G2P produces research drafts of patent descriptions. The pipeline does not generate claims, search for prior art, or determine patentability.

## Citation

Use the arXiv record when citing the publicly available paper.

```bibtex
@misc{pan2026flowplang2p,
  title         = {{FlowPlan-G2P}: A Structured Generation Framework for
                   Transforming Scientific Papers into Patent Descriptions},
  author        = {Pan, Kris W and Yoo, Yongmin},
  year          = {2026},
  eprint        = {2601.02589},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  doi           = {10.48550/arXiv.2601.02589},
  url           = {https://arxiv.org/abs/2601.02589}
}
```

## Development and license

Run `uv sync --extra test` followed by `uv run pytest` to run the test suite. See the [development guide](DEVELOPMENT.md) for contributor setup.

The implementation is available under the [Apache License 2.0](LICENSE).

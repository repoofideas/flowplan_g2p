#!/usr/bin/env python3
import sys
import json
from pathlib import Path
from flowplan_g2p.pipeline import FlowPlanG2P
from flowplan_g2p.llm_client import LLMConfig

def main():
    if len(sys.argv) != 3:
        print("Usage: python run_with_intermediates.py <input_paper> <output_prefix>")
        sys.exit(1)
    
    input_paper = sys.argv[1]
    output_prefix = sys.argv[2]
    
    # Configure pipeline
    config = LLMConfig(max_tokens=4096, temperature=0.2)
    pipeline = FlowPlanG2P(llm_config=config)
    
    # Run pipeline
    patent_text = pipeline.process_paper_to_patent(
        input_paper_path=input_paper,
        output_patent_path=f"{output_prefix}_patent.md"
    )
    
    # Get intermediate results
    results = pipeline.get_intermediate_results()
    
    # Save Stage 1: Concept Graph
    if 'concept_graph' in results:
        graph_data = {
            'nodes': [{'id': node.id, 'content': node.content, 'type': node.node_type} 
                     for node in results['concept_graph'].nodes],
            'edges': [{'source': edge.source, 'target': edge.target, 'relation': edge.relation} 
                     for edge in results['concept_graph'].edges]
        }
        with open(f"{output_prefix}_stage1_concept_graph.json", 'w') as f:
            json.dump(graph_data, f, indent=2)
        print(f"✓ Stage 1 output saved: {output_prefix}_stage1_concept_graph.json")
    
    # Save Stage 2: Section Plan
    if 'section_plan' in results:
        with open(f"{output_prefix}_stage2_section_plan.json", 'w') as f:
            json.dump(results['section_plan'], f, indent=2)
        print(f"✓ Stage 2 output saved: {output_prefix}_stage2_section_plan.json")
    
    # Save Stage 3: Final Patent (already saved by pipeline)
    print(f"✓ Stage 3 output saved: {output_prefix}_patent.md")

if __name__ == "__main__":
    main()

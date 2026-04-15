#!/usr/bin/env python3
"""
FlowPlan-G2P Command Line Interface

Main entry point for the FlowPlan-G2P package when installed via uv.
"""

import argparse
import sys
import os
from pathlib import Path

from .pipeline import FlowPlanG2P
from .llm_client import LLMConfig


def main():
    """Main CLI function for FlowPlan-G2P."""
    parser = argparse.ArgumentParser(
        description="FlowPlan-G2P: Convert scientific papers to patent descriptions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (OpenAI, default)
  flowplan-g2p input_paper.md output_patent.md
  
  # Use Anthropic provider
  flowplan-g2p input_paper.md output_patent.md --provider anthropic
  
  # Use a specific model with API key
  flowplan-g2p input_paper.md output_patent.md --provider openai --model gpt-4o --api-key sk-...
  
  # Use AWS Bedrock
  flowplan-g2p input_paper.md output_patent.md --provider bedrock --aws-region us-west-2
  
  # With custom parameters
  flowplan-g2p input_paper.md output_patent.md --candidates 5 --k 3 --tau-c 0.6
  
  # Export intermediate concept graph
  flowplan-g2p input_paper.md output_patent.md --export-graph concept_graph.json
        """
    )
    
    # Required arguments
    parser.add_argument("input_paper", type=str,
                       help="Path to input scientific paper (markdown format)")
    parser.add_argument("output_patent", type=str,
                       help="Path for output patent description (markdown format)")
    
    # Stage 1 parameters
    parser.add_argument("--candidates", type=int, default=3,
                       help="Number of concept graph candidates to generate and merge (default: 3)")
    
    # Stage 2 parameters  
    parser.add_argument("--k", type=int, default=5,
                       help="Number of section plan candidates to generate (default: 5)")
    parser.add_argument("--tau-c", type=float, default=0.5,
                       help="Connectivity threshold for section planning (default: 0.5)")
    parser.add_argument("--tau-s", type=float, default=0.6,
                       help="Semantic consistency threshold for section planning (default: 0.6)")
    
    # LLM configuration
    parser.add_argument("--provider", type=str, default="openai",
                       choices=["openai", "anthropic", "bedrock"],
                       help="LLM provider to use (default: openai)")
    parser.add_argument("--model", type=str, default=None,
                       help="Model identifier (e.g., gpt-4o, claude-sonnet-4-20250514); uses provider default if omitted")
    parser.add_argument("--api-key", type=str, default=None,
                       help="API key for the selected provider (falls back to environment variable if omitted)")
    parser.add_argument("--max-tokens", type=int, default=4096,
                       help="Maximum tokens for LLM generation (default: 4096)")
    parser.add_argument("--temperature", type=float, default=0.2,
                       help="Temperature for LLM generation (default: 0.2)")
    parser.add_argument("--aws-region", type=str, default="us-east-1",
                       help="AWS region for Bedrock service (default: us-east-1)")
    
    # Output options
    parser.add_argument("--export-graph", type=str, metavar="PATH",
                       help="Export concept graph to JSON file")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    
    # Validation
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate setup and connection, don't process")
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not args.validate_only and not os.path.exists(args.input_paper):
        print(f"❌ Error: Input file not found: {args.input_paper}")
        sys.exit(1)
    
    # Configure LLM
    llm_config = LLMConfig(
        provider=args.provider,
        model_id=args.model,
        api_key=args.api_key,
        aws_region=args.aws_region,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )
    
    # Initialize pipeline
    try:
        pipeline = FlowPlanG2P(llm_config=llm_config)
        
        if args.validate_only:
            print("🔍 Validating pipeline setup...")
            if pipeline.validate_setup():
                print("✅ Pipeline validation successful!")
                sys.exit(0)
            else:
                print("❌ Pipeline validation failed!")
                sys.exit(1)
        
        # Process paper to patent
        print("=" * 60)
        print("FlowPlan-G2P: Graph-Based Paragraph Planning for Patent Generation")
        print("=" * 60)
        
        patent_text = pipeline.process_paper_to_patent(
            input_paper_path=args.input_paper,
            output_patent_path=args.output_patent,
            num_candidates=args.candidates,
            k=args.k,
            tau_c=args.tau_c,
            tau_s=args.tau_s
        )
        
        # Export concept graph if requested
        if args.export_graph:
            try:
                pipeline.export_concept_graph(args.export_graph)
            except Exception as e:
                print(f"⚠️  Warning: Failed to export concept graph: {e}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("✅ SUCCESS: Patent description generated successfully!")
        print(f"📄 Output saved to: {args.output_patent}")
        if args.export_graph:
            print(f"📊 Concept graph saved to: {args.export_graph}")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
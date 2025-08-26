#!/usr/bin/env python3
"""
Test script for FlowPlan-G2P pipeline using sample data from Pap2Pat dataset.
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flowplan_g2p.pipeline import FlowPlanG2P
from flowplan_g2p.llm_client import LLMConfig


def test_with_sample_data():
    """Test pipeline with the first available sample from Pap2Pat dataset."""
    
    # Find first available sample data
    data_dir = Path("Pap2Pat/Pap2Pat/data")
    if not data_dir.exists():
        print("❌ Error: Pap2Pat data directory not found")
        print("Expected: Pap2Pat/Pap2Pat/data/")
        return False
    
    # Find first sample directory
    sample_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    if not sample_dirs:
        print("❌ Error: No sample directories found in Pap2Pat data")
        return False
    
    sample_dir = sample_dirs[0]
    paper_file = sample_dir / "paper.md"
    patent_outline_file = sample_dir / "patent_outline_long.md"
    
    if not paper_file.exists():
        print(f"❌ Error: paper.md not found in {sample_dir}")
        return False
    
    print(f"🧪 Testing with sample: {sample_dir.name}")
    print(f"📄 Input paper: {paper_file}")
    
    if patent_outline_file.exists():
        print(f"📋 Reference outline: {patent_outline_file}")
    else:
        print("⚠️  No reference patent outline found")
    
    # Configure pipeline
    llm_config = LLMConfig(
        max_tokens=2048,  # Smaller for testing
        temperature=0.3
    )
    
    # Output paths
    output_patent = Path("test_output_patent.md")
    output_graph = Path("test_concept_graph.json")
    
    try:
        # Initialize pipeline
        print("\n🔧 Initializing FlowPlan-G2P pipeline...")
        pipeline = FlowPlanG2P(llm_config=llm_config)
        
        # Validate setup
        print("🔍 Validating pipeline setup...")
        if not pipeline.validate_setup():
            print("❌ Pipeline validation failed")
            return False
        
        # Run pipeline
        print("\n🚀 Running FlowPlan-G2P pipeline...")
        patent_text = pipeline.process_paper_to_patent(
            input_paper_path=str(paper_file),
            output_patent_path=str(output_patent),
            num_candidates=2,  # Fewer candidates for faster testing
            k=3  # Fewer section candidates
        )
        
        # Export concept graph
        try:
            pipeline.export_concept_graph(str(output_graph))
        except Exception as e:
            print(f"⚠️  Warning: Could not export concept graph: {e}")
        
        # Display results
        print("\n" + "="*50)
        print("📊 TEST RESULTS")
        print("="*50)
        
        # Show performance summary
        pipeline.print_performance_summary()
        
        # Show generated patent preview
        print(f"\n📄 Generated Patent Description Preview:")
        print("-" * 50)
        preview = patent_text[:500] + "..." if len(patent_text) > 500 else patent_text
        print(preview)
        print("-" * 50)
        
        # Compare with reference if available
        if patent_outline_file.exists():
            try:
                with open(patent_outline_file, 'r', encoding='utf-8') as f:
                    reference_outline = f.read()
                
                print(f"\n📋 Reference Patent Outline Preview:")
                print("-" * 50)
                ref_preview = reference_outline[:500] + "..." if len(reference_outline) > 500 else reference_outline
                print(ref_preview)
                print("-" * 50)
                
                print(f"\n📏 Length Comparison:")
                print(f"   Generated: {len(patent_text)} characters")
                print(f"   Reference: {len(reference_outline)} characters")
                
            except Exception as e:
                print(f"⚠️  Could not read reference outline: {e}")
        
        print("\n✅ Test completed successfully!")
        print(f"📁 Output files:")
        print(f"   Patent description: {output_patent}")
        if output_graph.exists():
            print(f"   Concept graph: {output_graph}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("🧪 FlowPlan-G2P Pipeline Test")
    print("="*40)
    
    success = test_with_sample_data()
    
    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
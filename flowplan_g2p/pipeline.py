"""
Main FlowPlan-G2P Pipeline

Orchestrates the three-stage process of converting scientific papers
to patent descriptions using graph-based paragraph planning.
"""

from typing import Dict, List, Optional, Tuple
import os
import time
from .graph import ConceptGraph
from .llm_client import BedrockClaudeClient, LLMConfig
from .stages import ConceptGraphInduction, ParagraphPlanning, GraphConditionedGeneration
from .stages.paragraph_planning import SectionSubgraph


class FlowPlanG2P:
    """
    Main pipeline for FlowPlan-G2P framework.
    Coordinates the three stages: concept induction, paragraph planning, and generation.
    """
    
    def __init__(self, llm_config: Optional[LLMConfig] = None, aws_region: str = "us-east-1"):
        """
        Initialize FlowPlan-G2P pipeline.
        
        Args:
            llm_config: Configuration for LLM client
            aws_region: AWS region for Bedrock service
        """
        self.llm_config = llm_config or LLMConfig()
        self.aws_region = aws_region
        
        # Initialize LLM client
        self.llm_client = BedrockClaudeClient(self.llm_config, aws_region)
        
        # Initialize pipeline stages
        self.concept_induction = ConceptGraphInduction(self.llm_client)
        self.paragraph_planning = ParagraphPlanning(self.llm_client)
        self.graph_generation = GraphConditionedGeneration(self.llm_client)
        
        # Pipeline state tracking
        self.last_concept_graph: Optional[ConceptGraph] = None
        self.last_section_plan: Optional[List[SectionSubgraph]] = None
        self.last_patent_text: Optional[str] = None
        
        # Performance metrics
        self.execution_times: Dict[str, float] = {}
    
    def validate_setup(self) -> bool:
        """
        Validate that the pipeline is properly configured and can connect to services.
        
        Returns:
            True if setup is valid, False otherwise
        """
        try:
            # Test LLM connection
            if not self.llm_client.validate_connection():
                print("Error: Cannot connect to AWS Bedrock Claude service")
                return False
            
            print("✓ AWS Bedrock connection validated")
            
            # Test that required components are initialized
            assert self.concept_induction is not None
            assert self.paragraph_planning is not None
            assert self.graph_generation is not None
            
            print("✓ All pipeline stages initialized")
            
            return True
            
        except Exception as e:
            print(f"Error: Pipeline validation failed: {e}")
            return False
    
    def load_paper_content(self, file_path: str) -> str:
        """
        Load paper content from markdown file.
        
        Args:
            file_path: Path to the paper markdown file
            
        Returns:
            Paper content as string
            
        Raises:
            FileNotFoundError: If file doesn't exist
            IOError: If file cannot be read
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Paper file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                raise IOError(f"Paper file is empty: {file_path}")
            
            print(f"✓ Loaded paper content from {file_path} ({len(content)} characters)")
            return content
            
        except Exception as e:
            raise IOError(f"Failed to read paper file {file_path}: {e}")
    
    def save_patent_description(self, patent_text: str, output_path: str) -> None:
        """
        Save generated patent description to file.
        
        Args:
            patent_text: Generated patent description
            output_path: Output file path
            
        Raises:
            IOError: If file cannot be written
        """
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(patent_text)
            
            print(f"✓ Saved patent description to {output_path}")
            
        except Exception as e:
            raise IOError(f"Failed to save patent description to {output_path}: {e}")
    
    def run_stage1_concept_induction(self, paper_content: str, num_candidates: int = 3) -> ConceptGraph:
        """
        Run Stage 1: Concept Graph Induction.
        
        Args:
            paper_content: Full text content of the scientific paper
            num_candidates: Number of candidate graphs to generate and merge
            
        Returns:
            Generated concept graph
        """
        print("🔄 Stage 1: Concept Graph Induction...")
        start_time = time.time()
        
        try:
            concept_graph = self.concept_induction.generate_concept_graph(
                paper_content, num_candidates=num_candidates
            )
            
            execution_time = time.time() - start_time
            self.execution_times["concept_induction"] = execution_time
            
            # Store for potential reuse
            self.last_concept_graph = concept_graph
            
            print(f"✓ Generated concept graph with {len(concept_graph)} nodes and {len(concept_graph.edges)} edges")
            print(f"  Execution time: {execution_time:.2f}s")
            
            return concept_graph
            
        except Exception as e:
            print(f"❌ Stage 1 failed: {e}")
            raise
    
    def run_stage2_paragraph_planning(self, concept_graph: ConceptGraph, 
                                     k: int = 5, tau_c: float = 0.5, tau_s: float = 0.6) -> List[SectionSubgraph]:
        """
        Run Stage 2: Paragraph and Section Planning.
        
        Args:
            concept_graph: Input concept graph from Stage 1
            k: Number of candidate plans to generate
            tau_c: Connectivity threshold
            tau_s: Semantic consistency threshold
            
        Returns:
            Ordered list of section subgraphs
        """
        print("🔄 Stage 2: Paragraph Planning...")
        start_time = time.time()
        
        try:
            section_plan = self.paragraph_planning.create_paragraph_plan(
                concept_graph, k=k, tau_c=tau_c, tau_s=tau_s
            )
            
            execution_time = time.time() - start_time
            self.execution_times["paragraph_planning"] = execution_time
            
            # Store for potential reuse
            self.last_section_plan = section_plan
            
            section_names = [section.section_name for section in section_plan]
            print(f"✓ Created section plan: {', '.join(section_names)}")
            print(f"  Execution time: {execution_time:.2f}s")
            
            return section_plan
            
        except Exception as e:
            print(f"❌ Stage 2 failed: {e}")
            raise
    
    def run_stage3_generation(self, section_plan: List[SectionSubgraph]) -> str:
        """
        Run Stage 3: Graph-Conditioned Generation.
        
        Args:
            section_plan: Ordered list of section subgraphs from Stage 2
            
        Returns:
            Generated patent description text
        """
        print("🔄 Stage 3: Graph-Conditioned Generation...")
        start_time = time.time()
        
        try:
            patent_text = self.graph_generation.generate_patent_description(section_plan)
            
            execution_time = time.time() - start_time
            self.execution_times["graph_generation"] = execution_time
            
            # Store for potential reuse
            self.last_patent_text = patent_text
            
            print(f"✓ Generated patent description ({len(patent_text)} characters)")
            print(f"  Execution time: {execution_time:.2f}s")
            
            return patent_text
            
        except Exception as e:
            print(f"❌ Stage 3 failed: {e}")
            raise
    
    def process_paper_to_patent(self, input_paper_path: str, output_patent_path: str,
                              num_candidates: int = 3, k: int = 5, 
                              tau_c: float = 0.5, tau_s: float = 0.6) -> str:
        """
        Complete pipeline: Convert scientific paper to patent description.
        
        Args:
            input_paper_path: Path to input paper markdown file
            output_patent_path: Path for output patent description file
            num_candidates: Number of concept graph candidates for Stage 1
            k: Number of section plan candidates for Stage 2
            tau_c: Connectivity threshold for Stage 2
            tau_s: Semantic consistency threshold for Stage 2
            
        Returns:
            Generated patent description text
            
        Raises:
            Various exceptions if any stage fails
        """
        pipeline_start_time = time.time()
        print("🚀 Starting FlowPlan-G2P Pipeline...")
        print(f"   Input: {input_paper_path}")
        print(f"   Output: {output_patent_path}")
        
        try:
            # Validate setup
            if not self.validate_setup():
                raise RuntimeError("Pipeline validation failed")
            
            # Load input paper
            paper_content = self.load_paper_content(input_paper_path)
            
            # Stage 1: Concept Graph Induction
            concept_graph = self.run_stage1_concept_induction(paper_content, num_candidates)
            
            # Stage 2: Paragraph Planning
            section_plan = self.run_stage2_paragraph_planning(concept_graph, k, tau_c, tau_s)
            
            # Stage 3: Graph-Conditioned Generation
            patent_text = self.run_stage3_generation(section_plan)
            
            # Save output
            self.save_patent_description(patent_text, output_patent_path)
            
            # Calculate total time
            total_execution_time = time.time() - pipeline_start_time
            self.execution_times["total_pipeline"] = total_execution_time
            
            print(f"\n🎉 Pipeline completed successfully!")
            print(f"   Total execution time: {total_execution_time:.2f}s")
            self.print_performance_summary()
            
            return patent_text
            
        except Exception as e:
            total_execution_time = time.time() - pipeline_start_time
            print(f"\n❌ Pipeline failed after {total_execution_time:.2f}s: {e}")
            raise
    
    def print_performance_summary(self) -> None:
        """Print detailed performance metrics for the last pipeline run."""
        if not self.execution_times:
            print("No performance data available")
            return
        
        print("\n📊 Performance Summary:")
        for stage, time_taken in self.execution_times.items():
            if stage != "total_pipeline":
                percentage = (time_taken / self.execution_times.get("total_pipeline", 1)) * 100
                print(f"   {stage.replace('_', ' ').title()}: {time_taken:.2f}s ({percentage:.1f}%)")
    
    def get_intermediate_results(self) -> Dict[str, any]:
        """
        Get intermediate results from the last pipeline run.
        
        Returns:
            Dictionary containing intermediate results
        """
        return {
            "concept_graph": self.last_concept_graph,
            "section_plan": self.last_section_plan,
            "patent_text": self.last_patent_text,
            "execution_times": self.execution_times.copy()
        }
    
    def export_concept_graph(self, output_path: str) -> None:
        """
        Export the last generated concept graph to JSON file.
        
        Args:
            output_path: Path for JSON output file
            
        Raises:
            ValueError: If no concept graph available
            IOError: If file cannot be written
        """
        if self.last_concept_graph is None:
            raise ValueError("No concept graph available. Run Stage 1 first.")
        
        try:
            json_content = self.last_concept_graph.to_json()
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(json_content)
            print(f"✓ Exported concept graph to {output_path}")
        except Exception as e:
            raise IOError(f"Failed to export concept graph: {e}")
    
    def load_concept_graph(self, input_path: str) -> ConceptGraph:
        """
        Load concept graph from JSON file.
        
        Args:
            input_path: Path to JSON input file
            
        Returns:
            Loaded concept graph
            
        Raises:
            FileNotFoundError: If file doesn't exist
            IOError: If file cannot be read or parsed
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Concept graph file not found: {input_path}")
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                json_content = f.read()
            
            concept_graph = ConceptGraph.from_json(json_content)
            self.last_concept_graph = concept_graph
            print(f"✓ Loaded concept graph from {input_path}")
            
            return concept_graph
            
        except Exception as e:
            raise IOError(f"Failed to load concept graph from {input_path}: {e}")
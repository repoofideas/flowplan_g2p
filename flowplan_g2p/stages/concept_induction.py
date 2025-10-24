"""
Stage 1: Concept Graph Induction

Implements the structured reasoning-based concept graph induction process
as described in the FlowPlan-G2P paper.
"""

from typing import Dict, List, Optional, Tuple
import json
from ..graph import ConceptGraph, ConceptNode, ConceptEdge, NodeType, RelationType
from ..llm_client import DeepSeekClient


class ConceptGraphInduction:
    """
    Stage 1 of FlowPlan-G2P: Extract key technical concepts and relationships
    from scientific papers and construct directed concept graphs.
    """
    
    def __init__(self, llm_client: DeepSeekClient):
        """
        Initialize concept graph induction stage.
        
        Args:
            llm_client: Configured LLM client for API calls
        """
        self.llm_client = llm_client
        
        # Define the structured reasoning steps as in the paper
        self.reasoning_steps = [
            NodeType.FIELD,
            NodeType.TECH_PROBLEM, 
            NodeType.PRIOR_ART,
            NodeType.NOVELTY,
            NodeType.SOLUTION,
            NodeType.IMPLEMENTATION,
            NodeType.EFFECTS,
            NodeType.EMBODIMENT,
            NodeType.FIGURE
        ]
        
        # Patent-oriented boilerplate phrases for each step
        self.boilerplates = {
            NodeType.FIELD: "The present invention relates to",
            NodeType.TECH_PROBLEM: "However, conventional technology has the following drawbacks",
            NodeType.PRIOR_ART: "According to the prior art",
            NodeType.NOVELTY: "The novelty of the present invention lies in",
            NodeType.SOLUTION: "An object of the present invention is to provide",
            NodeType.IMPLEMENTATION: "According to one embodiment of the present invention",
            NodeType.EFFECTS: "Therefore, according to the present invention, the effect is",
            NodeType.EMBODIMENT: "In a preferred embodiment",
            NodeType.FIGURE: "Figure 1 illustrates"
        }
        
        # Predefined relation types for concept graph
        self.valid_relations = [
            RelationType.SOLVES,
            RelationType.IMPLEMENTS,
            RelationType.CAUSES,
            RelationType.IMPROVES,
            RelationType.VALIDATES
        ]
    
    def _create_reasoning_prompt(self, paper_content: str, step_type: NodeType, 
                               previous_outputs: Dict[NodeType, str]) -> str:
        """
        Create a structured reasoning prompt for a specific step.
        
        Args:
            paper_content: Full content of the scientific paper
            step_type: Type of reasoning step to perform
            previous_outputs: Results from previous reasoning steps
            
        Returns:
            Formatted prompt for the LLM
        """
        boilerplate = self.boilerplates[step_type]
        
        # Build context from previous steps
        context = ""
        if previous_outputs:
            context = "Previous analysis:\n"
            for prev_type, prev_output in previous_outputs.items():
                context += f"- {prev_type.value}: {prev_output[:200]}...\n"
            context += "\n"
        
        prompt_mapping = {
            NodeType.FIELD: f"""
            Analyze the following scientific paper and identify the technical field.
            
            {context}Paper content:
            {paper_content}
            
            Task: Identify the main technical field this research belongs to.
            Format your response starting with: "{boilerplate}..."
            Be specific about the technical domain, methodology, or application area.
            """,
            
            NodeType.TECH_PROBLEM: f"""
            Based on the scientific paper, identify the core technical problem being addressed.
            
            {context}Paper content:
            {paper_content}
            
            Task: Identify what technical problem or limitation this research aims to solve.
            Format your response starting with: "{boilerplate}..."
            Focus on technical challenges, limitations, or gaps in existing approaches.
            """,
            
            NodeType.PRIOR_ART: f"""
            Analyze the paper to identify relevant prior art and existing approaches.
            
            {context}Paper content:
            {paper_content}
            
            Task: Summarize the existing methods, technologies, or approaches mentioned in the paper.
            Format your response starting with: "{boilerplate}..."
            Include both strengths and limitations of existing approaches.
            """,
            
            NodeType.NOVELTY: f"""
            Identify the novel aspects and innovations presented in this research.
            
            {context}Paper content:
            {paper_content}
            
            Task: What makes this research novel compared to existing work?
            Format your response starting with: "{boilerplate}..."
            Highlight the key innovations, improvements, or new insights.
            """,
            
            NodeType.SOLUTION: f"""
            Extract the core solution or approach proposed in this research.
            
            {context}Paper content:
            {paper_content}
            
            Task: Describe the main solution, method, or approach proposed by the authors.
            Format your response starting with: "{boilerplate}..."
            Focus on the key technical contribution and how it addresses the problem.
            """,
            
            NodeType.IMPLEMENTATION: f"""
            Describe how the proposed solution is implemented or realized.
            
            {context}Paper content:
            {paper_content}
            
            Task: Explain the specific implementation details, algorithms, or procedures.
            Format your response starting with: "{boilerplate}..."
            Include technical specifications, parameters, or system architecture details.
            """,
            
            NodeType.EFFECTS: f"""
            Identify the effects, benefits, or results achieved by the proposed solution.
            
            {context}Paper content:
            {paper_content}
            
            Task: What positive effects or improvements does this solution provide?
            Format your response starting with: "{boilerplate}..."
            Include performance improvements, advantages, or beneficial outcomes.
            """,
            
            NodeType.EMBODIMENT: f"""
            Identify specific embodiments, examples, or implementations described in the paper.
            
            {context}Paper content:
            {paper_content}
            
            Task: Describe specific examples, use cases, or embodiments of the invention.
            Format your response starting with: "{boilerplate}..."
            Include variations, specific applications, or example implementations.
            """,
            
            NodeType.FIGURE: f"""
            Identify and describe any figures, diagrams, or visual elements referenced in the paper.
            
            {context}Paper content:
            {paper_content}
            
            Task: List and describe figures, tables, or diagrams that illustrate the invention.
            Format your response starting with: "{boilerplate}..."
            Describe what each figure shows and its relevance to the invention.
            """
        }
        
        return prompt_mapping[step_type].strip()
    
    def _perform_structured_reasoning(self, paper_content: str) -> Dict[NodeType, str]:
        """
        Execute the 6-step structured reasoning process.
        
        Args:
            paper_content: Full content of the scientific paper
            
        Returns:
            Dictionary mapping each reasoning step to its output
        """
        reasoning_outputs = {}
        
        for step_type in self.reasoning_steps:
            prompt = self._create_reasoning_prompt(paper_content, step_type, reasoning_outputs)
            
            try:
                response = self.llm_client.generate(prompt)
                reasoning_outputs[step_type] = response.strip()
                
            except Exception as e:
                print(f"Warning: Failed to generate reasoning for {step_type.value}: {e}")
                reasoning_outputs[step_type] = f"Error generating {step_type.value}: {str(e)}"
        
        return reasoning_outputs
    
    def _construct_concept_graph(self, reasoning_outputs: Dict[NodeType, str]) -> ConceptGraph:
        """
        Construct concept graph from reasoning outputs.
        
        Args:
            reasoning_outputs: Results from structured reasoning
            
        Returns:
            ConceptGraph with nodes and edges
        """
        graph = ConceptGraph()
        
        # Add nodes for each reasoning output
        for node_type, content in reasoning_outputs.items():
            if content and not content.startswith("Error"):
                node = ConceptNode(
                    id=f"{node_type.value.lower()}_1",
                    node_type=node_type,
                    content=content
                )
                graph.add_node(node)
        
        # Add predefined edges based on patent logic
        edge_rules = [
            (NodeType.TECH_PROBLEM, NodeType.SOLUTION, RelationType.SOLVES),
            (NodeType.SOLUTION, NodeType.IMPLEMENTATION, RelationType.IMPLEMENTS),
            (NodeType.IMPLEMENTATION, NodeType.EFFECTS, RelationType.CAUSES),
            (NodeType.SOLUTION, NodeType.EFFECTS, RelationType.IMPROVES),
            (NodeType.EMBODIMENT, NodeType.IMPLEMENTATION, RelationType.VALIDATES),
            (NodeType.FIGURE, NodeType.EMBODIMENT, RelationType.VALIDATES),
            (NodeType.NOVELTY, NodeType.SOLUTION, RelationType.IMPROVES)
        ]
        
        for source_type, target_type, relation_type in edge_rules:
            source_id = f"{source_type.value.lower()}_1"
            target_id = f"{target_type.value.lower()}_1"
            
            if source_id in graph and target_id in graph:
                edge = ConceptEdge(
                    source_id=source_id,
                    target_id=target_id,
                    relation_type=relation_type
                )
                graph.add_edge(edge)
        
        return graph
    
    def _create_graph_construction_prompt(self, reasoning_outputs: Dict[NodeType, str]) -> str:
        """
        Create prompt for LLM-based graph construction with relations.
        
        Args:
            reasoning_outputs: Results from structured reasoning
            
        Returns:
            Formatted prompt for relation extraction
        """
        # Format reasoning outputs for the prompt
        concepts_text = ""
        for node_type, content in reasoning_outputs.items():
            if content and not content.startswith("Error"):
                concepts_text += f"\n- {node_type.value}: {content[:300]}...\n"
        
        prompt = f"""
        Given the following patent concepts extracted from a scientific paper, create a concept graph
        by identifying relationships between them.
        
        Concepts:
        {concepts_text}
        
        Task: Create a JSON representation of the concept graph with nodes and edges.
        
        Available relation types: {[rel.value for rel in self.valid_relations]}
        
        Rules:
        1. Use only the provided relation types
        2. Each node should have an ID and type
        3. Edges should specify source, target, and relation type
        4. Focus on logical patent-style relationships
        
        Output format:
        ```json
        {{
            "nodes": [
                {{"id": "problem_1", "type": "TechProblem", "content": "..."}},
                {{"id": "solution_1", "type": "Solution", "content": "..."}}
            ],
            "edges": [
                {{"source": "problem_1", "target": "solution_1", "relation": "solves"}}
            ]
        }}
        ```
        
        Generate the complete graph:
        """
        
        return prompt.strip()
    
    def _construct_graph_with_llm(self, reasoning_outputs: Dict[NodeType, str]) -> ConceptGraph:
        """
        Use LLM to construct concept graph with better relation inference.
        
        Args:
            reasoning_outputs: Results from structured reasoning
            
        Returns:
            ConceptGraph constructed by LLM
        """
        prompt = self._create_graph_construction_prompt(reasoning_outputs)
        
        try:
            response_data = self.llm_client.generate_structured(prompt)
            
            # Create graph from LLM response
            graph = ConceptGraph()
            
            # Add nodes
            for node_data in response_data.get("nodes", []):
                node_type_str = node_data.get("type", "")
                try:
                    node_type = NodeType(node_type_str)
                    node = ConceptNode(
                        id=node_data["id"],
                        node_type=node_type,
                        content=node_data["content"]
                    )
                    graph.add_node(node)
                except ValueError:
                    print(f"Warning: Invalid node type {node_type_str}, skipping node")
            
            # Add edges
            for edge_data in response_data.get("edges", []):
                relation_str = edge_data.get("relation", "")
                try:
                    relation_type = RelationType(relation_str)
                    edge = ConceptEdge(
                        source_id=edge_data["source"],
                        target_id=edge_data["target"],
                        relation_type=relation_type
                    )
                    graph.add_edge(edge)
                except (ValueError, KeyError) as e:
                    print(f"Warning: Invalid edge {edge_data}, skipping: {e}")
            
            return graph
            
        except Exception as e:
            print(f"Warning: LLM graph construction failed: {e}")
            # Fall back to rule-based construction
            return self._construct_concept_graph(reasoning_outputs)
    
    def _verify_graph(self, graph: ConceptGraph) -> ConceptGraph:
        """
        Verify and repair graph according to patent constraints.
        
        Args:
            graph: Input concept graph
            
        Returns:
            Verified and potentially modified graph
        """
        # Ensure all required node types are present
        required_types = [NodeType.FIELD, NodeType.TECH_PROBLEM, NodeType.SOLUTION]
        
        for required_type in required_types:
            nodes_of_type = graph.get_nodes_by_type(required_type)
            if not nodes_of_type:
                # Add placeholder node if missing
                placeholder_node = ConceptNode(
                    id=f"{required_type.value.lower()}_placeholder",
                    node_type=required_type,
                    content=f"[Placeholder for {required_type.value}]"
                )
                graph.add_node(placeholder_node)
        
        # Remove isolated nodes (nodes with no connections)
        isolated_nodes = []
        for node_id in graph.nodes:
            if (len(graph.get_neighbors(node_id)) == 0 and 
                len(graph.get_predecessors(node_id)) == 0):
                isolated_nodes.append(node_id)
        
        for node_id in isolated_nodes:
            # Try to connect to related nodes before removing
            node = graph.get_node(node_id)
            if node and node.node_type == NodeType.EFFECTS:
                # Connect effects to solution if available
                solution_nodes = graph.get_nodes_by_type(NodeType.SOLUTION)
                if solution_nodes:
                    edge = ConceptEdge(
                        source_id=solution_nodes[0].id,
                        target_id=node_id,
                        relation_type=RelationType.CAUSES
                    )
                    graph.add_edge(edge)
        
        return graph
    
    def generate_concept_graph(self, paper_content: str, num_candidates: int = 3) -> ConceptGraph:
        """
        Main method to generate concept graph from paper content.
        
        Args:
            paper_content: Full text content of the scientific paper
            num_candidates: Number of candidate graphs to generate and merge
            
        Returns:
            Final merged and verified concept graph
        """
        # Step 1: Perform structured reasoning
        reasoning_outputs = self._perform_structured_reasoning(paper_content)
        
        # Step 2: Generate multiple candidate graphs
        candidate_graphs = []
        
        for i in range(num_candidates):
            try:
                if i == 0:
                    # First candidate uses rule-based approach
                    graph = self._construct_concept_graph(reasoning_outputs)
                else:
                    # Subsequent candidates use LLM-based approach
                    graph = self._construct_graph_with_llm(reasoning_outputs)
                
                candidate_graphs.append(graph)
                
            except Exception as e:
                print(f"Warning: Failed to generate candidate graph {i+1}: {e}")
                # Create minimal graph as fallback
                fallback_graph = ConceptGraph()
                for node_type, content in reasoning_outputs.items():
                    if content and not content.startswith("Error"):
                        node = ConceptNode(
                            id=f"{node_type.value.lower()}_{i+1}",
                            node_type=node_type,
                            content=content
                        )
                        fallback_graph.add_node(node)
                candidate_graphs.append(fallback_graph)
        
        # Step 3: Merge candidate graphs
        if len(candidate_graphs) > 1:
            merged_graph = candidate_graphs[0].merge_graphs(candidate_graphs[1:])
        else:
            merged_graph = candidate_graphs[0] if candidate_graphs else ConceptGraph()
        
        # Step 4: Verify and repair graph
        final_graph = self._verify_graph(merged_graph)
        
        return final_graph
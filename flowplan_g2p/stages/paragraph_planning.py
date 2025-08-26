"""
Stage 2: Paragraph and Section Planning

Implements the paragraph planning stage that organizes concept graphs into
semantically coherent subgraphs and determines their global ordering.
"""

from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
import math
from ..graph import ConceptGraph, ConceptNode, NodeType, RelationType
from ..llm_client import BedrockClaudeClient


@dataclass 
class SectionSubgraph:
    """Represents a section with its associated subgraph."""
    section_name: str
    node_ids: Set[str]
    subgraph: ConceptGraph
    boilerplate: str
    order_priority: int = 0


@dataclass
class PlanningCandidate:
    """Represents a candidate paragraph/section plan."""
    sections: List[SectionSubgraph]
    connectivity_score: float
    consistency_score: float
    penalty_score: float
    total_score: float


class ParagraphPlanning:
    """
    Stage 2 of FlowPlan-G2P: Organize concept graph into semantically coherent
    subgraphs and determine global ordering for patent section structure.
    """
    
    def __init__(self, llm_client: BedrockClaudeClient):
        """
        Initialize paragraph planning stage.
        
        Args:
            llm_client: Configured LLM client for API calls
        """
        self.llm_client = llm_client
        
        # Canonical patent sections and their boilerplates
        self.section_definitions = {
            "Field": {
                "boilerplate": "The present invention relates to",
                "node_types": [NodeType.FIELD],
                "order": 1
            },
            "Background": {
                "boilerplate": "According to the prior art",
                "node_types": [NodeType.TECH_PROBLEM, NodeType.PRIOR_ART],
                "order": 2
            },
            "Summary": {
                "boilerplate": "An object of the present invention is to provide",
                "node_types": [NodeType.SOLUTION, NodeType.NOVELTY],
                "order": 3
            },
            "Detailed Description": {
                "boilerplate": "According to one embodiment of the present invention",
                "node_types": [NodeType.IMPLEMENTATION, NodeType.EMBODIMENT, NodeType.FIGURE],
                "order": 4
            },
            "Effects": {
                "boilerplate": "Therefore, according to the present invention, the effect is",
                "node_types": [NodeType.EFFECTS],
                "order": 5
            }
        }
        
        # Default hyperparameters from the paper
        self.default_k = 5  # Number of candidate structures to sample
        self.default_tau_c = 0.5  # Connectivity threshold
        self.default_tau_s = 0.6  # Semantic consistency threshold
    
    def _create_section_induction_prompt(self, graph: ConceptGraph) -> str:
        """
        Create prompt for LLM-based section clustering.
        
        Args:
            graph: Input concept graph
            
        Returns:
            Formatted prompt for section induction
        """
        # Format graph information
        nodes_text = ""
        for node in graph.nodes.values():
            nodes_text += f"- {node.id} ({node.node_type.value}): {node.content[:150]}...\n"
        
        edges_text = ""
        for edge in graph.edges.values():
            edges_text += f"- {edge.source_id} -> {edge.target_id} ({edge.relation_type.value})\n"
        
        prompt = f"""
        Given the following concept graph from a scientific paper, organize the concepts into
        patent sections according to standard patent structure.
        
        Nodes:
        {nodes_text}
        
        Edges:
        {edges_text}
        
        Standard Patent Sections:
        1. Field of the Invention - Technical domain and scope
        2. Background - Prior art and problems 
        3. Summary - Solution and novelty
        4. Detailed Description - Implementation details and embodiments
        5. Effects - Benefits and advantages
        
        Task: Group the concepts into appropriate patent sections. Each concept should belong 
        to exactly one section. Consider semantic relationships and patent document structure.
        
        Rules:
        - Field concepts go to "Field"
        - Problems and prior art go to "Background" 
        - Solutions and novelty go to "Summary"
        - Implementation, embodiments, figures go to "Detailed Description"
        - Effects and benefits go to "Effects"
        
        Output format:
        ```json
        {{
            "sections": {{
                "Field": ["field_1"],
                "Background": ["techproblem_1", "priorart_1"],
                "Summary": ["solution_1", "novelty_1"],
                "Detailed Description": ["implementation_1", "embodiment_1", "figure_1"],
                "Effects": ["effects_1"]
            }}
        }}
        ```
        
        Generate the section assignments:
        """
        
        return prompt.strip()
    
    def _perform_section_induction(self, graph: ConceptGraph) -> Dict[str, List[str]]:
        """
        Use LLM to cluster nodes into patent sections.
        
        Args:
            graph: Input concept graph
            
        Returns:
            Dictionary mapping section names to lists of node IDs
        """
        prompt = self._create_section_induction_prompt(graph)
        
        try:
            response_data = self.llm_client.generate_structured(prompt)
            sections = response_data.get("sections", {})
            
            # Validate that all nodes are assigned
            assigned_nodes = set()
            for node_ids in sections.values():
                assigned_nodes.update(node_ids)
            
            # Assign unassigned nodes using fallback logic
            for node_id, node in graph.nodes.items():
                if node_id not in assigned_nodes:
                    fallback_section = self._get_fallback_section(node.node_type)
                    if fallback_section not in sections:
                        sections[fallback_section] = []
                    sections[fallback_section].append(node_id)
            
            return sections
            
        except Exception as e:
            print(f"Warning: LLM section induction failed: {e}")
            return self._fallback_section_assignment(graph)
    
    def _get_fallback_section(self, node_type: NodeType) -> str:
        """
        Get fallback section assignment for a node type.
        
        Args:
            node_type: Type of the concept node
            
        Returns:
            Section name for fallback assignment
        """
        for section_name, definition in self.section_definitions.items():
            if node_type in definition["node_types"]:
                return section_name
        return "Detailed Description"  # Default fallback
    
    def _fallback_section_assignment(self, graph: ConceptGraph) -> Dict[str, List[str]]:
        """
        Deterministic fallback section assignment based on node types.
        
        Args:
            graph: Input concept graph
            
        Returns:
            Dictionary mapping section names to lists of node IDs
        """
        sections = {name: [] for name in self.section_definitions.keys()}
        
        for node_id, node in graph.nodes.items():
            section = self._get_fallback_section(node.node_type)
            sections[section].append(node_id)
        
        return sections
    
    def _calculate_connectivity_score(self, graph: ConceptGraph, sections: Dict[str, List[str]]) -> float:
        """
        Calculate intra-section connectivity score.
        
        Args:
            graph: Input concept graph
            sections: Section assignments
            
        Returns:
            Overall connectivity score (0-1)
        """
        total_connectivity = 0.0
        section_count = 0
        
        for section_name, node_ids in sections.items():
            if len(node_ids) <= 1:
                continue
                
            section_count += 1
            
            # Count internal edges
            internal_edges = 0
            for edge in graph.edges.values():
                if edge.source_id in node_ids and edge.target_id in node_ids:
                    internal_edges += 1
            
            # Maximum possible internal edges
            max_edges = len(node_ids) * (len(node_ids) - 1)
            
            # Calculate connectivity for this section
            if max_edges > 0:
                section_connectivity = internal_edges / max_edges
                total_connectivity += section_connectivity
        
        return total_connectivity / max(1, section_count)
    
    def _calculate_consistency_score(self, graph: ConceptGraph, sections: Dict[str, List[str]]) -> float:
        """
        Calculate intra-section semantic consistency score.
        
        Args:
            graph: Input concept graph
            sections: Section assignments
            
        Returns:
            Overall consistency score (0-1)
        """
        # Simplified consistency based on node type homogeneity within sections
        total_consistency = 0.0
        section_count = 0
        
        for section_name, node_ids in sections.items():
            if len(node_ids) <= 1:
                continue
                
            section_count += 1
            
            # Get node types in this section
            node_types = [graph.get_node(node_id).node_type for node_id in node_ids if graph.get_node(node_id)]
            
            if not node_types:
                continue
            
            # Calculate type diversity (lower diversity = higher consistency for sections)
            type_counts = {}
            for node_type in node_types:
                type_counts[node_type] = type_counts.get(node_type, 0) + 1
            
            # Entropy-based consistency (lower entropy = higher consistency)
            total_nodes = len(node_types)
            entropy = 0.0
            for count in type_counts.values():
                prob = count / total_nodes
                entropy -= prob * math.log2(prob) if prob > 0 else 0
            
            max_entropy = math.log2(len(type_counts)) if len(type_counts) > 1 else 0
            consistency = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0
            
            total_consistency += consistency
        
        return total_consistency / max(1, section_count)
    
    def _calculate_penalty_score(self, graph: ConceptGraph, sections: Dict[str, List[str]]) -> float:
        """
        Calculate rule-based penalty score.
        
        Args:
            graph: Input concept graph
            sections: Section assignments
            
        Returns:
            Penalty score (0 = no penalties, higher = more violations)
        """
        penalty = 0.0
        
        # Check for missing required types
        required_types = [NodeType.FIELD, NodeType.TECH_PROBLEM, NodeType.SOLUTION]
        all_assigned_nodes = set()
        for node_ids in sections.values():
            all_assigned_nodes.update(node_ids)
        
        for req_type in required_types:
            type_nodes = [node_id for node_id in all_assigned_nodes 
                         if graph.get_node(node_id) and graph.get_node(node_id).node_type == req_type]
            if not type_nodes:
                penalty += 1.0
        
        # Check for illegal type pairings within sections
        illegal_pairings = [
            (NodeType.FIELD, NodeType.EFFECTS),  # Field shouldn't be with Effects
            (NodeType.PRIOR_ART, NodeType.SOLUTION),  # Prior art shouldn't be directly with solution
        ]
        
        for section_name, node_ids in sections.items():
            section_types = [graph.get_node(node_id).node_type for node_id in node_ids 
                           if graph.get_node(node_id)]
            
            for type1, type2 in illegal_pairings:
                if type1 in section_types and type2 in section_types:
                    penalty += 0.5
        
        # Check section ordering constraints
        section_order = list(sections.keys())
        expected_order = ["Field", "Background", "Summary", "Detailed Description", "Effects"]
        
        for i, expected_section in enumerate(expected_order):
            if expected_section in section_order:
                actual_pos = section_order.index(expected_section)
                if actual_pos < i:  # Section appears before it should
                    penalty += 0.3
        
        return penalty
    
    def _create_section_subgraphs(self, graph: ConceptGraph, sections: Dict[str, List[str]]) -> List[SectionSubgraph]:
        """
        Create section subgraphs from node assignments.
        
        Args:
            graph: Input concept graph
            sections: Section assignments
            
        Returns:
            List of SectionSubgraph objects
        """
        section_subgraphs = []
        
        for section_name, node_ids in sections.items():
            if not node_ids:  # Skip empty sections
                continue
                
            # Create subgraph
            subgraph = graph.get_subgraph(set(node_ids))
            
            # Get section definition
            section_def = self.section_definitions.get(section_name, {
                "boilerplate": f"Regarding {section_name.lower()}",
                "order": 10
            })
            
            section_subgraph = SectionSubgraph(
                section_name=section_name,
                node_ids=set(node_ids),
                subgraph=subgraph,
                boilerplate=section_def["boilerplate"],
                order_priority=section_def.get("order", 10)
            )
            
            section_subgraphs.append(section_subgraph)
        
        return section_subgraphs
    
    def _generate_candidate_plan(self, graph: ConceptGraph) -> PlanningCandidate:
        """
        Generate a single candidate planning structure.
        
        Args:
            graph: Input concept graph
            
        Returns:
            PlanningCandidate with scores
        """
        # Perform section induction
        sections = self._perform_section_induction(graph)
        
        # Calculate scores
        connectivity = self._calculate_connectivity_score(graph, sections)
        consistency = self._calculate_consistency_score(graph, sections)
        penalty = self._calculate_penalty_score(graph, sections)
        
        # Create section subgraphs
        section_subgraphs = self._create_section_subgraphs(graph, sections)
        
        # Sort sections by order priority
        section_subgraphs.sort(key=lambda s: s.order_priority)
        
        # Calculate total score
        total_score = connectivity + consistency - penalty
        
        return PlanningCandidate(
            sections=section_subgraphs,
            connectivity_score=connectivity,
            consistency_score=consistency,
            penalty_score=penalty,
            total_score=total_score
        )
    
    def _select_best_candidate(self, candidates: List[PlanningCandidate], 
                             tau_c: float, tau_s: float) -> PlanningCandidate:
        """
        Select the best candidate using gating criteria.
        
        Args:
            candidates: List of planning candidates
            tau_c: Connectivity threshold
            tau_s: Consistency threshold
            
        Returns:
            Best valid candidate or fallback
        """
        # Filter candidates that pass gating criteria
        valid_candidates = []
        for candidate in candidates:
            if (candidate.connectivity_score >= tau_c and 
                candidate.consistency_score >= tau_s and 
                candidate.penalty_score == 0):
                valid_candidates.append(candidate)
        
        if valid_candidates:
            # Return candidate with highest total score
            return max(valid_candidates, key=lambda c: c.total_score)
        else:
            # Return candidate with highest total score (even if it doesn't pass gates)
            return max(candidates, key=lambda c: c.total_score) if candidates else None
    
    def create_paragraph_plan(self, graph: ConceptGraph, k: Optional[int] = None, 
                            tau_c: Optional[float] = None, tau_s: Optional[float] = None) -> List[SectionSubgraph]:
        """
        Main method to create paragraph/section planning from concept graph.
        
        Args:
            graph: Input concept graph
            k: Number of candidate structures to generate
            tau_c: Connectivity threshold
            tau_s: Consistency threshold
            
        Returns:
            Ordered list of section subgraphs representing the final plan
        """
        k = k or self.default_k
        tau_c = tau_c or self.default_tau_c  
        tau_s = tau_s or self.default_tau_s
        
        # Generate multiple candidate plans
        candidates = []
        for i in range(k):
            try:
                candidate = self._generate_candidate_plan(graph)
                candidates.append(candidate)
            except Exception as e:
                print(f"Warning: Failed to generate candidate plan {i+1}: {e}")
        
        if not candidates:
            print("Warning: No valid candidates generated, creating fallback plan")
            # Create minimal fallback plan
            fallback_sections = self._fallback_section_assignment(graph)
            fallback_subgraphs = self._create_section_subgraphs(graph, fallback_sections)
            fallback_subgraphs.sort(key=lambda s: s.order_priority)
            return fallback_subgraphs
        
        # Select best candidate
        best_candidate = self._select_best_candidate(candidates, tau_c, tau_s)
        
        if best_candidate is None:
            # Create minimal fallback plan
            fallback_sections = self._fallback_section_assignment(graph)
            fallback_subgraphs = self._create_section_subgraphs(graph, fallback_sections)
            fallback_subgraphs.sort(key=lambda s: s.order_priority)
            return fallback_subgraphs
        
        return best_candidate.sections
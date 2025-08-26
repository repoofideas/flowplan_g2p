"""
Custom graph data structures for FlowPlan-G2P concept graphs.
Implements nodes, edges, and graph operations for patent concept modeling.
"""

from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import json


class NodeType(Enum):
    """Enumeration of concept node types in patent graphs."""
    FIELD = "Field"
    TECH_PROBLEM = "TechProblem" 
    PRIOR_ART = "PriorArt"
    NOVELTY = "Novelty"
    SOLUTION = "Solution"
    IMPLEMENTATION = "Implementation"
    EFFECTS = "Effects"
    EMBODIMENT = "Embodiment"
    FIGURE = "Figure"


class RelationType(Enum):
    """Enumeration of edge relation types in concept graphs."""
    SOLVES = "solves"
    IMPLEMENTS = "implements"
    CAUSES = "causes"
    IMPROVES = "improves"
    VALIDATES = "validates"


@dataclass
class ConceptNode:
    """Represents a concept node in the patent graph."""
    
    id: str
    node_type: NodeType
    content: str
    attributes: Dict[str, str] = None
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}
    
    def to_dict(self) -> Dict:
        """Convert node to dictionary representation."""
        return {
            "id": self.id,
            "type": self.node_type.value,
            "content": self.content,
            "attributes": self.attributes
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ConceptNode':
        """Create node from dictionary representation."""
        return cls(
            id=data["id"],
            node_type=NodeType(data["type"]),
            content=data["content"],
            attributes=data.get("attributes", {})
        )


@dataclass
class ConceptEdge:
    """Represents a directed edge between concept nodes."""
    
    source_id: str
    target_id: str
    relation_type: RelationType
    weight: float = 1.0
    attributes: Dict[str, str] = None
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}
    
    def to_dict(self) -> Dict:
        """Convert edge to dictionary representation."""
        return {
            "source": self.source_id,
            "target": self.target_id,
            "relation": self.relation_type.value,
            "weight": self.weight,
            "attributes": self.attributes
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ConceptEdge':
        """Create edge from dictionary representation."""
        return cls(
            source_id=data["source"],
            target_id=data["target"],
            relation_type=RelationType(data["relation"]),
            weight=data.get("weight", 1.0),
            attributes=data.get("attributes", {})
        )


class ConceptGraph:
    """
    Custom directed graph implementation for patent concept modeling.
    Supports nodes of different types and typed relations between them.
    """
    
    def __init__(self):
        self.nodes: Dict[str, ConceptNode] = {}
        self.edges: Dict[Tuple[str, str], ConceptEdge] = {}
        self._adjacency_list: Dict[str, Set[str]] = {}
        self._reverse_adjacency_list: Dict[str, Set[str]] = {}
    
    def add_node(self, node: ConceptNode) -> None:
        """Add a concept node to the graph."""
        self.nodes[node.id] = node
        if node.id not in self._adjacency_list:
            self._adjacency_list[node.id] = set()
        if node.id not in self._reverse_adjacency_list:
            self._reverse_adjacency_list[node.id] = set()
    
    def add_edge(self, edge: ConceptEdge) -> None:
        """Add a directed edge between two nodes."""
        if edge.source_id not in self.nodes:
            raise ValueError(f"Source node {edge.source_id} not found in graph")
        if edge.target_id not in self.nodes:
            raise ValueError(f"Target node {edge.target_id} not found in graph")
        
        edge_key = (edge.source_id, edge.target_id)
        self.edges[edge_key] = edge
        self._adjacency_list[edge.source_id].add(edge.target_id)
        self._reverse_adjacency_list[edge.target_id].add(edge.source_id)
    
    def get_node(self, node_id: str) -> Optional[ConceptNode]:
        """Get a node by its ID."""
        return self.nodes.get(node_id)
    
    def get_edge(self, source_id: str, target_id: str) -> Optional[ConceptEdge]:
        """Get an edge between two nodes."""
        return self.edges.get((source_id, target_id))
    
    def get_neighbors(self, node_id: str) -> Set[str]:
        """Get all nodes that this node has edges to."""
        return self._adjacency_list.get(node_id, set()).copy()
    
    def get_predecessors(self, node_id: str) -> Set[str]:
        """Get all nodes that have edges to this node."""
        return self._reverse_adjacency_list.get(node_id, set()).copy()
    
    def get_nodes_by_type(self, node_type: NodeType) -> List[ConceptNode]:
        """Get all nodes of a specific type."""
        return [node for node in self.nodes.values() if node.node_type == node_type]
    
    def get_edges_by_relation(self, relation_type: RelationType) -> List[ConceptEdge]:
        """Get all edges of a specific relation type."""
        return [edge for edge in self.edges.values() if edge.relation_type == relation_type]
    
    def remove_node(self, node_id: str) -> None:
        """Remove a node and all its associated edges."""
        if node_id not in self.nodes:
            return
        
        # Remove all edges involving this node
        edges_to_remove = []
        for edge_key, edge in self.edges.items():
            if edge.source_id == node_id or edge.target_id == node_id:
                edges_to_remove.append(edge_key)
        
        for edge_key in edges_to_remove:
            del self.edges[edge_key]
        
        # Update adjacency lists
        for neighbor in self._adjacency_list.get(node_id, set()):
            self._reverse_adjacency_list[neighbor].discard(node_id)
        
        for predecessor in self._reverse_adjacency_list.get(node_id, set()):
            self._adjacency_list[predecessor].discard(node_id)
        
        # Remove node
        del self.nodes[node_id]
        del self._adjacency_list[node_id]
        del self._reverse_adjacency_list[node_id]
    
    def has_cycle(self) -> bool:
        """Check if the graph has any cycles using DFS."""
        WHITE, GRAY, BLACK = 0, 1, 2
        colors = {node_id: WHITE for node_id in self.nodes}
        
        def dfs(node_id: str) -> bool:
            if colors[node_id] == GRAY:
                return True  # Back edge found, cycle exists
            if colors[node_id] == BLACK:
                return False  # Already processed
            
            colors[node_id] = GRAY
            for neighbor in self._adjacency_list[node_id]:
                if dfs(neighbor):
                    return True
            colors[node_id] = BLACK
            return False
        
        for node_id in self.nodes:
            if colors[node_id] == WHITE:
                if dfs(node_id):
                    return True
        return False
    
    def topological_sort(self) -> List[str]:
        """Return a topological ordering of nodes (if no cycles exist)."""
        if self.has_cycle():
            raise ValueError("Cannot perform topological sort on graph with cycles")
        
        visited = set()
        stack = []
        
        def dfs(node_id: str):
            visited.add(node_id)
            for neighbor in self._adjacency_list[node_id]:
                if neighbor not in visited:
                    dfs(neighbor)
            stack.append(node_id)
        
        for node_id in self.nodes:
            if node_id not in visited:
                dfs(node_id)
        
        return stack[::-1]  # Reverse to get correct order
    
    def get_subgraph(self, node_ids: Set[str]) -> 'ConceptGraph':
        """Create a subgraph containing only the specified nodes and edges between them."""
        subgraph = ConceptGraph()
        
        # Add nodes
        for node_id in node_ids:
            if node_id in self.nodes:
                subgraph.add_node(self.nodes[node_id])
        
        # Add edges
        for edge_key, edge in self.edges.items():
            if edge.source_id in node_ids and edge.target_id in node_ids:
                subgraph.add_edge(edge)
        
        return subgraph
    
    def to_dict(self) -> Dict:
        """Convert graph to dictionary representation."""
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges.values()]
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ConceptGraph':
        """Create graph from dictionary representation."""
        graph = cls()
        
        # Add nodes
        for node_data in data["nodes"]:
            node = ConceptNode.from_dict(node_data)
            graph.add_node(node)
        
        # Add edges
        for edge_data in data["edges"]:
            edge = ConceptEdge.from_dict(edge_data)
            graph.add_edge(edge)
        
        return graph
    
    def to_json(self) -> str:
        """Convert graph to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ConceptGraph':
        """Create graph from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def merge_graphs(self, other_graphs: List['ConceptGraph']) -> 'ConceptGraph':
        """Merge multiple graphs using the rules from the paper."""
        merged = ConceptGraph()
        all_graphs = [self] + other_graphs
        
        # Collect all unique nodes
        all_nodes = {}
        for graph in all_graphs:
            for node_id, node in graph.nodes.items():
                if node_id not in all_nodes:
                    all_nodes[node_id] = node
                # TODO: Handle semantic merging of overlapping nodes
        
        # Add all nodes to merged graph
        for node in all_nodes.values():
            merged.add_node(node)
        
        # Collect edges with conflict resolution
        edge_counts = {}
        for graph in all_graphs:
            for edge_key, edge in graph.edges.items():
                if edge_key not in edge_counts:
                    edge_counts[edge_key] = []
                edge_counts[edge_key].append(edge)
        
        # Add edges using majority voting for conflicts
        for edge_key, edge_list in edge_counts.items():
            if len(edge_list) == 1:
                merged.add_edge(edge_list[0])
            else:
                # Use majority voting for relation type
                relation_counts = {}
                for edge in edge_list:
                    rel = edge.relation_type
                    relation_counts[rel] = relation_counts.get(rel, 0) + 1
                
                majority_relation = max(relation_counts, key=relation_counts.get)
                
                # Use the first edge with majority relation
                for edge in edge_list:
                    if edge.relation_type == majority_relation:
                        merged.add_edge(edge)
                        break
        
        return merged
    
    def __len__(self) -> int:
        """Return number of nodes in the graph."""
        return len(self.nodes)
    
    def __contains__(self, node_id: str) -> bool:
        """Check if node exists in graph."""
        return node_id in self.nodes
    
    def __str__(self) -> str:
        """String representation of the graph."""
        return f"ConceptGraph(nodes={len(self.nodes)}, edges={len(self.edges)})"
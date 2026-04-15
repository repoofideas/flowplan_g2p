"""Unit tests for ConceptGraph, ConceptNode, and ConceptEdge."""

import json
import pytest

from flowplan_g2p.graph import (
    ConceptGraph,
    ConceptNode,
    ConceptEdge,
    NodeType,
    RelationType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_node(id: str, ntype: NodeType = NodeType.SOLUTION, content: str = "test") -> ConceptNode:
    return ConceptNode(id=id, node_type=ntype, content=content)


def _simple_graph() -> ConceptGraph:
    """A -> B -> C  (linear DAG)."""
    g = ConceptGraph()
    g.add_node(_make_node("A", NodeType.FIELD, "field"))
    g.add_node(_make_node("B", NodeType.SOLUTION, "solution"))
    g.add_node(_make_node("C", NodeType.EFFECTS, "effects"))
    g.add_edge(ConceptEdge("A", "B", RelationType.SOLVES))
    g.add_edge(ConceptEdge("B", "C", RelationType.CAUSES))
    return g


# ---------------------------------------------------------------------------
# ConceptNode
# ---------------------------------------------------------------------------

class TestConceptNode:
    def test_to_dict(self):
        node = ConceptNode("n1", NodeType.NOVELTY, "novel idea", {"key": "val"})
        d = node.to_dict()
        assert d["id"] == "n1"
        assert d["type"] == "Novelty"
        assert d["content"] == "novel idea"
        assert d["attributes"] == {"key": "val"}

    def test_from_dict_round_trip(self):
        node = ConceptNode("n1", NodeType.IMPLEMENTATION, "impl", {"x": "1"})
        restored = ConceptNode.from_dict(node.to_dict())
        assert restored.id == node.id
        assert restored.node_type == node.node_type
        assert restored.content == node.content
        assert restored.attributes == node.attributes

    def test_default_attributes(self):
        node = ConceptNode("n1", NodeType.FIELD, "f")
        assert node.attributes == {}


# ---------------------------------------------------------------------------
# ConceptEdge
# ---------------------------------------------------------------------------

class TestConceptEdge:
    def test_to_dict(self):
        edge = ConceptEdge("a", "b", RelationType.IMPLEMENTS, weight=0.8, attributes={"k": "v"})
        d = edge.to_dict()
        assert d["source"] == "a"
        assert d["target"] == "b"
        assert d["relation"] == "implements"
        assert d["weight"] == 0.8
        assert d["attributes"] == {"k": "v"}

    def test_from_dict_round_trip(self):
        edge = ConceptEdge("a", "b", RelationType.VALIDATES, weight=0.5)
        restored = ConceptEdge.from_dict(edge.to_dict())
        assert restored.source_id == edge.source_id
        assert restored.target_id == edge.target_id
        assert restored.relation_type == edge.relation_type
        assert restored.weight == edge.weight

    def test_default_weight_and_attributes(self):
        edge = ConceptEdge("a", "b", RelationType.SOLVES)
        assert edge.weight == 1.0
        assert edge.attributes == {}


# ---------------------------------------------------------------------------
# ConceptGraph – basic operations
# ---------------------------------------------------------------------------

class TestConceptGraphBasic:
    def test_add_and_get_node(self):
        g = ConceptGraph()
        node = _make_node("x")
        g.add_node(node)
        assert g.get_node("x") is node
        assert "x" in g
        assert len(g) == 1

    def test_get_node_missing(self):
        g = ConceptGraph()
        assert g.get_node("missing") is None

    def test_add_and_get_edge(self):
        g = ConceptGraph()
        g.add_node(_make_node("a"))
        g.add_node(_make_node("b"))
        edge = ConceptEdge("a", "b", RelationType.SOLVES)
        g.add_edge(edge)
        assert g.get_edge("a", "b") is edge

    def test_add_edge_missing_source(self):
        g = ConceptGraph()
        g.add_node(_make_node("b"))
        with pytest.raises(ValueError, match="Source node"):
            g.add_edge(ConceptEdge("a", "b", RelationType.SOLVES))

    def test_add_edge_missing_target(self):
        g = ConceptGraph()
        g.add_node(_make_node("a"))
        with pytest.raises(ValueError, match="Target node"):
            g.add_edge(ConceptEdge("a", "b", RelationType.SOLVES))

    def test_get_edge_missing(self):
        g = ConceptGraph()
        assert g.get_edge("x", "y") is None

    def test_get_neighbors(self):
        g = _simple_graph()
        assert g.get_neighbors("A") == {"B"}
        assert g.get_neighbors("B") == {"C"}
        assert g.get_neighbors("C") == set()

    def test_get_predecessors(self):
        g = _simple_graph()
        assert g.get_predecessors("A") == set()
        assert g.get_predecessors("B") == {"A"}
        assert g.get_predecessors("C") == {"B"}

    def test_get_nodes_by_type(self):
        g = _simple_graph()
        fields = g.get_nodes_by_type(NodeType.FIELD)
        assert len(fields) == 1
        assert fields[0].id == "A"

    def test_get_edges_by_relation(self):
        g = _simple_graph()
        solves = g.get_edges_by_relation(RelationType.SOLVES)
        assert len(solves) == 1
        assert solves[0].source_id == "A"


# ---------------------------------------------------------------------------
# ConceptGraph – remove_node
# ---------------------------------------------------------------------------

class TestRemoveNode:
    def test_remove_node_cleans_edges(self):
        g = _simple_graph()  # A->B->C
        g.remove_node("B")
        assert "B" not in g
        assert g.get_edge("A", "B") is None
        assert g.get_edge("B", "C") is None
        # A and C still exist
        assert "A" in g
        assert "C" in g

    def test_remove_nonexistent_node_is_noop(self):
        g = _simple_graph()
        g.remove_node("Z")  # should not raise
        assert len(g) == 3


# ---------------------------------------------------------------------------
# ConceptGraph – cycle detection
# ---------------------------------------------------------------------------

class TestHasCycle:
    def test_dag_has_no_cycle(self):
        g = _simple_graph()
        assert g.has_cycle() is False

    def test_cycle_detected(self):
        g = ConceptGraph()
        g.add_node(_make_node("A"))
        g.add_node(_make_node("B"))
        g.add_edge(ConceptEdge("A", "B", RelationType.SOLVES))
        g.add_edge(ConceptEdge("B", "A", RelationType.CAUSES))
        assert g.has_cycle() is True

    def test_self_loop_detected(self):
        g = ConceptGraph()
        g.add_node(_make_node("A"))
        g.add_edge(ConceptEdge("A", "A", RelationType.SOLVES))
        assert g.has_cycle() is True


# ---------------------------------------------------------------------------
# ConceptGraph – topological sort
# ---------------------------------------------------------------------------

class TestTopologicalSort:
    def test_valid_dag(self):
        g = _simple_graph()  # A->B->C
        order = g.topological_sort()
        assert order.index("A") < order.index("B") < order.index("C")

    def test_raises_on_cycle(self):
        g = ConceptGraph()
        g.add_node(_make_node("A"))
        g.add_node(_make_node("B"))
        g.add_edge(ConceptEdge("A", "B", RelationType.SOLVES))
        g.add_edge(ConceptEdge("B", "A", RelationType.CAUSES))
        with pytest.raises(ValueError, match="cycles"):
            g.topological_sort()


# ---------------------------------------------------------------------------
# ConceptGraph – subgraph
# ---------------------------------------------------------------------------

class TestGetSubgraph:
    def test_subgraph_contains_only_requested_nodes(self):
        g = _simple_graph()
        sub = g.get_subgraph({"A", "B"})
        assert "A" in sub
        assert "B" in sub
        assert "C" not in sub
        assert sub.get_edge("A", "B") is not None
        assert sub.get_edge("B", "C") is None

    def test_subgraph_ignores_unknown_ids(self):
        g = _simple_graph()
        sub = g.get_subgraph({"A", "Z"})
        assert len(sub) == 1


# ---------------------------------------------------------------------------
# ConceptGraph – merge_graphs (majority voting)
# ---------------------------------------------------------------------------

class TestMergeGraphs:
    def test_merge_disjoint_graphs(self):
        g1 = ConceptGraph()
        g1.add_node(_make_node("A"))
        g1.add_node(_make_node("B"))
        g1.add_edge(ConceptEdge("A", "B", RelationType.SOLVES))

        g2 = ConceptGraph()
        g2.add_node(_make_node("C"))
        g2.add_node(_make_node("D"))
        g2.add_edge(ConceptEdge("C", "D", RelationType.CAUSES))

        merged = g1.merge_graphs([g2])
        assert len(merged) == 4
        assert merged.get_edge("A", "B") is not None
        assert merged.get_edge("C", "D") is not None

    def test_merge_majority_voting(self):
        """When 3 graphs disagree on edge relation, majority wins."""
        # Build three graphs with same nodes and same edge key (A,B)
        # Two say SOLVES, one says CAUSES → SOLVES should win
        graphs = []
        for rel in [RelationType.SOLVES, RelationType.SOLVES, RelationType.CAUSES]:
            g = ConceptGraph()
            g.add_node(_make_node("A"))
            g.add_node(_make_node("B"))
            g.add_edge(ConceptEdge("A", "B", rel))
            graphs.append(g)

        base = graphs[0]
        merged = base.merge_graphs(graphs[1:])
        edge = merged.get_edge("A", "B")
        assert edge is not None
        assert edge.relation_type == RelationType.SOLVES


# ---------------------------------------------------------------------------
# ConceptGraph – serialization round-trip
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_to_dict_from_dict(self):
        g = _simple_graph()
        d = g.to_dict()
        restored = ConceptGraph.from_dict(d)
        assert len(restored) == len(g)
        assert len(restored.edges) == len(g.edges)
        for nid in g.nodes:
            assert nid in restored

    def test_to_json_from_json(self):
        g = _simple_graph()
        j = g.to_json()
        restored = ConceptGraph.from_json(j)
        assert len(restored) == len(g)
        assert len(restored.edges) == len(g.edges)

    def test_json_is_valid(self):
        g = _simple_graph()
        parsed = json.loads(g.to_json())
        assert "nodes" in parsed
        assert "edges" in parsed

    def test_round_trip_preserves_attributes(self):
        g = ConceptGraph()
        g.add_node(ConceptNode("n1", NodeType.FIGURE, "fig", {"page": "3"}))
        g.add_node(ConceptNode("n2", NodeType.EMBODIMENT, "emb"))
        g.add_edge(ConceptEdge("n1", "n2", RelationType.VALIDATES, weight=0.7, attributes={"note": "ok"}))
        restored = ConceptGraph.from_json(g.to_json())
        assert restored.get_node("n1").attributes == {"page": "3"}
        edge = restored.get_edge("n1", "n2")
        assert edge.weight == 0.7
        assert edge.attributes == {"note": "ok"}


# ---------------------------------------------------------------------------
# ConceptGraph – dunder methods
# ---------------------------------------------------------------------------

class TestDunderMethods:
    def test_len(self):
        g = _simple_graph()
        assert len(g) == 3

    def test_contains(self):
        g = _simple_graph()
        assert "A" in g
        assert "Z" not in g

    def test_str(self):
        g = _simple_graph()
        s = str(g)
        assert "3" in s  # nodes
        assert "2" in s  # edges

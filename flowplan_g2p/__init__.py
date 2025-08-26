"""
FlowPlan-G2P: Graph-Based Paragraph Planning for Structured Patent Description Generation

A Python implementation of the FlowPlan-G2P framework for converting scientific papers
into patent descriptions using a three-stage approach:
1. Concept Graph Induction
2. Paragraph Planning  
3. Graph-Conditioned Generation
"""

__version__ = "1.0.0"
__author__ = "FlowPlan-G2P Implementation"

from .graph import ConceptGraph, ConceptNode, ConceptEdge
from .llm_client import BedrockClaudeClient
from .pipeline import FlowPlanG2P
from .stages import ConceptGraphInduction, ParagraphPlanning, GraphConditionedGeneration

__all__ = [
    "ConceptGraph",
    "ConceptNode", 
    "ConceptEdge",
    "BedrockClaudeClient",
    "FlowPlanG2P",
    "ConceptGraphInduction",
    "ParagraphPlanning", 
    "GraphConditionedGeneration"
]
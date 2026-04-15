"""
Stage 3: Graph-Conditioned Generation

Implements the graph-conditioned generation stage that produces patent-style
description paragraphs by conditioning on structured subgraph representations.
"""

from typing import Dict, List, Optional
from ..graph import ConceptGraph
from ..llm_client import BaseLLMClient
from .paragraph_planning import SectionSubgraph


class GraphConditionedGeneration:
    """
    Stage 3 of FlowPlan-G2P: Generate patent-style description paragraphs
    by conditioning on structured subgraph representations.
    """
    
    def __init__(self, llm_client: BaseLLMClient):
        """
        Initialize graph-conditioned generation stage.
        
        Args:
            llm_client: Configured LLM client for API calls
        """
        self.llm_client = llm_client
        
        # Enhanced patent instruction template from the paper
        self.patent_instruction_template = """
        Using the following concepts and relations, generate a patent Description section.
        Follow the standard patent structure and adopt explicit patent-style boilerplate phrases.
        
        For each section, use these specific patterns:
        - Field: Begin with "The present invention relates to..."
        - Background: Include phrases such as "According to the prior art..." and 
          highlight drawbacks with "However, such technology has the following problems..."
        - Summary: State "An object of the present invention is to provide..."
        - Detailed Description: Elaborate with "According to one embodiment of the present invention..."
          and "In another embodiment...". Repeat key ideas across embodiments and explicitly 
          refer to figures as "Figure 1 illustrates..."
        - Effects: Conclude with "Therefore, according to the present invention, the effect is..."
        
        Ensure repetitive elaboration across embodiments, provide alternative embodiments,
        and use figure references consistently. Use formal, technical language appropriate
        for patent documentation.
        """
        
        # Section-specific generation strategies
        self.section_strategies = {
            "Field": self._generate_field_section,
            "Background": self._generate_background_section,
            "Summary": self._generate_summary_section,
            "Detailed Description": self._generate_detailed_description_section,
            "Effects": self._generate_effects_section
        }
    
    def _linearize_subgraph(self, subgraph: ConceptGraph) -> str:
        """
        Convert subgraph to linear text representation for LLM input.
        
        Args:
            subgraph: Section subgraph to linearize
            
        Returns:
            Linear text representation of the subgraph
        """
        linearized_text = "Concepts:\n"
        
        # Add nodes
        for node in subgraph.nodes.values():
            linearized_text += f"- {node.id} ({node.node_type.value}): {node.content}\n"
        
        linearized_text += "\nRelations:\n"
        
        # Add edges  
        if subgraph.edges:
            for edge in subgraph.edges.values():
                linearized_text += f"- {edge.source_id} {edge.relation_type.value} {edge.target_id}\n"
        else:
            linearized_text += "- No explicit relations in this section\n"
        
        return linearized_text
    
    def _create_generation_prompt(self, section_subgraph: SectionSubgraph) -> str:
        """
        Create generation prompt for a specific section.
        
        Args:
            section_subgraph: Section with its subgraph and metadata
            
        Returns:
            Formatted prompt for patent text generation
        """
        linearized_subgraph = self._linearize_subgraph(section_subgraph.subgraph)
        
        base_prompt = f"""
        {self.patent_instruction_template}
        
        Section: {section_subgraph.section_name}
        Required Opening: "{section_subgraph.boilerplate}..."
        
        {linearized_subgraph}
        
        Task: Generate the {section_subgraph.section_name} section of a patent description.
        Requirements:
        1. Start with the required boilerplate phrase
        2. Use formal patent language and terminology
        3. Ensure technical accuracy and completeness
        4. Include all concepts from the subgraph
        5. Maintain logical flow and coherence
        """
        
        return base_prompt.strip()
    
    def _generate_field_section(self, section_subgraph: SectionSubgraph) -> str:
        """
        Generate Field of the Invention section.
        
        Args:
            section_subgraph: Field section subgraph
            
        Returns:
            Generated field section text
        """
        prompt = self._create_generation_prompt(section_subgraph)
        prompt += """
        
        Additional requirements for Field section:
        - Be concise and specific about the technical domain
        - Avoid detailed explanations (save for other sections)
        - Focus on the scope and area of the invention
        - Use technical terminology appropriate to the field
        """
        
        try:
            response = self.llm_client.generate(prompt)
            return self._post_process_section(response, "Field")
        except Exception as e:
            return f"# FIELD OF THE INVENTION\n\n{section_subgraph.boilerplate} [Error generating content: {e}]"
    
    def _generate_background_section(self, section_subgraph: SectionSubgraph) -> str:
        """
        Generate Background section.
        
        Args:
            section_subgraph: Background section subgraph
            
        Returns:
            Generated background section text
        """
        prompt = self._create_generation_prompt(section_subgraph)
        prompt += """
        
        Additional requirements for Background section:
        - Describe the current state of technology
        - Explain problems with existing approaches
        - Use phrases like "However, such technology has the following problems..."
        - Provide context for why the invention is needed
        - Reference prior art and existing methods
        """
        
        try:
            response = self.llm_client.generate(prompt)
            return self._post_process_section(response, "Background")
        except Exception as e:
            return f"# BACKGROUND OF THE INVENTION\n\n{section_subgraph.boilerplate} [Error generating content: {e}]"
    
    def _generate_summary_section(self, section_subgraph: SectionSubgraph) -> str:
        """
        Generate Summary of the Invention section.
        
        Args:
            section_subgraph: Summary section subgraph
            
        Returns:
            Generated summary section text
        """
        prompt = self._create_generation_prompt(section_subgraph)
        prompt += """
        
        Additional requirements for Summary section:
        - Clearly state the objective and solution
        - Explain what the invention achieves
        - Highlight novel aspects and improvements
        - Use clear, direct language
        - Avoid excessive technical detail (save for Detailed Description)
        """
        
        try:
            response = self.llm_client.generate(prompt)
            return self._post_process_section(response, "Summary")
        except Exception as e:
            return f"# SUMMARY OF THE INVENTION\n\n{section_subgraph.boilerplate} [Error generating content: {e}]"
    
    def _generate_detailed_description_section(self, section_subgraph: SectionSubgraph) -> str:
        """
        Generate Detailed Description section.
        
        Args:
            section_subgraph: Detailed Description section subgraph
            
        Returns:
            Generated detailed description section text
        """
        prompt = self._create_generation_prompt(section_subgraph)
        prompt += """
        
        Additional requirements for Detailed Description section:
        - Provide comprehensive technical details
        - Include multiple embodiments if applicable
        - Use phrases like "In another embodiment..." for variations
        - Reference figures explicitly: "Figure 1 illustrates..."
        - Repeat key concepts across different embodiments
        - Include specific technical parameters, methods, and implementations
        - Use detailed technical language and specifications
        """
        
        try:
            response = self.llm_client.generate(prompt)
            return self._post_process_section(response, "Detailed Description")
        except Exception as e:
            return f"# DETAILED DESCRIPTION OF THE INVENTION\n\n{section_subgraph.boilerplate} [Error generating content: {e}]"
    
    def _generate_effects_section(self, section_subgraph: SectionSubgraph) -> str:
        """
        Generate Effects section.
        
        Args:
            section_subgraph: Effects section subgraph
            
        Returns:
            Generated effects section text
        """
        prompt = self._create_generation_prompt(section_subgraph)
        prompt += """
        
        Additional requirements for Effects section:
        - Clearly describe the benefits and advantages
        - Explain how the invention solves the stated problems
        - Quantify improvements where possible
        - Connect effects back to the technical solution
        - Use conclusive language showing causation
        """
        
        try:
            response = self.llm_client.generate(prompt)
            return self._post_process_section(response, "Effects")
        except Exception as e:
            return f"# EFFECTS OF THE INVENTION\n\n{section_subgraph.boilerplate} [Error generating content: {e}]"
    
    def _post_process_section(self, generated_text: str, section_name: str) -> str:
        """
        Post-process generated section text.
        
        Args:
            generated_text: Raw generated text from LLM
            section_name: Name of the section
            
        Returns:
            Post-processed section text with proper formatting
        """
        # Clean up the text
        text = generated_text.strip()
        
        # Add section header if not present
        section_headers = {
            "Field": "# FIELD OF THE INVENTION",
            "Background": "# BACKGROUND OF THE INVENTION",
            "Summary": "# SUMMARY OF THE INVENTION", 
            "Detailed Description": "# DETAILED DESCRIPTION OF THE INVENTION",
            "Effects": "# EFFECTS OF THE INVENTION"
        }
        
        header = section_headers.get(section_name, f"# {section_name.upper()}")
        
        if not text.startswith("#"):
            text = f"{header}\n\n{text}"
        
        # Ensure proper paragraph breaks
        text = text.replace("\n\n\n", "\n\n")  # Remove excessive line breaks
        
        # Ensure text ends with proper spacing
        if not text.endswith("\n"):
            text += "\n"
        
        return text
    
    def generate_patent_description(self, section_subgraphs: List[SectionSubgraph]) -> str:
        """
        Main method to generate complete patent description from section subgraphs.
        
        Args:
            section_subgraphs: Ordered list of section subgraphs
            
        Returns:
            Complete patent description text
        """
        patent_description = "# DESCRIPTION\n\n"
        
        for section_subgraph in section_subgraphs:
            try:
                # Use section-specific generation strategy if available
                if section_subgraph.section_name in self.section_strategies:
                    section_text = self.section_strategies[section_subgraph.section_name](section_subgraph)
                else:
                    # Use generic generation for unknown sections
                    section_text = self._generate_generic_section(section_subgraph)
                
                patent_description += section_text + "\n\n"
                
            except Exception as e:
                print(f"Warning: Failed to generate {section_subgraph.section_name} section: {e}")
                # Add error placeholder
                patent_description += f"# {section_subgraph.section_name.upper()}\n\n"
                patent_description += f"{section_subgraph.boilerplate} [Error generating content]\n\n"
        
        return patent_description.strip()
    
    def _generate_generic_section(self, section_subgraph: SectionSubgraph) -> str:
        """
        Generate a generic section for unknown section types.
        
        Args:
            section_subgraph: Section subgraph to generate
            
        Returns:
            Generated section text
        """
        prompt = self._create_generation_prompt(section_subgraph)
        
        try:
            response = self.llm_client.generate(prompt)
            return self._post_process_section(response, section_subgraph.section_name)
        except Exception as e:
            return f"# {section_subgraph.section_name.upper()}\n\n{section_subgraph.boilerplate} [Error generating content: {e}]"
    
    def generate_section_batch(self, section_subgraphs: List[SectionSubgraph]) -> Dict[str, str]:
        """
        Generate multiple sections in batch for efficiency.
        
        Args:
            section_subgraphs: List of section subgraphs to generate
            
        Returns:
            Dictionary mapping section names to generated text
        """
        section_texts = {}
        
        # Prepare prompts for batch processing
        prompts = []
        section_names = []
        
        for section_subgraph in section_subgraphs:
            prompt = self._create_generation_prompt(section_subgraph)
            prompts.append(prompt)
            section_names.append(section_subgraph.section_name)
        
        # Generate responses in batch
        try:
            responses = self.llm_client.batch_generate(prompts)
            
            for i, (section_name, response) in enumerate(zip(section_names, responses)):
                if response.startswith("ERROR:"):
                    section_texts[section_name] = f"# {section_name.upper()}\n\n[{response}]"
                else:
                    section_texts[section_name] = self._post_process_section(response, section_name)
                    
        except Exception as e:
            print(f"Warning: Batch generation failed: {e}")
            # Fall back to individual generation
            for section_subgraph in section_subgraphs:
                try:
                    if section_subgraph.section_name in self.section_strategies:
                        text = self.section_strategies[section_subgraph.section_name](section_subgraph)
                    else:
                        text = self._generate_generic_section(section_subgraph)
                    section_texts[section_subgraph.section_name] = text
                except Exception as e2:
                    section_texts[section_subgraph.section_name] = f"# {section_subgraph.section_name.upper()}\n\n[Error: {e2}]"
        
        return section_texts
CATEGORY_EXTRACTION_PROMPT = """
Your job is to extract important information from a Computer Science article and organize it into specific categories.

Instructions:

Use only the information from the provided document.

Make sure every category is filled in if the information exists in the text.

skip References

Extract and organize into these categories:

High-Level Extraction
"

Title of Article,

Keywords.

Name of the Tool,

Frameworks Used,

Specific Frameworks,

VA Evaluations,

Datasets,

data Tasks Identified/Addressed,

Medical Tasks Identified/Addressed,


user Interactions , 

data Visualization ,

Analytics methods,

Multi-scale and Multidimensionality ,

Explainability of Visualization,

Data Integration Method,

Human Decision-Making on Data,

Implications for SDOH Visualization Research,

Visualization Technique used,

Application User / Target Population,

Objective,

Data Used,

Medical Doctor as Asuthor (Yes/No),

Technological Stack for Development,

Evaluation Details,

Link
"

Document context to use and extract above data:"
{context_str}
"

    
"""

CATEGORY_EXTRACTION_PROMPT = """
Your job is to extract important information from a Computer Science article and organize it into specific categories.
ensure to read the figure descriptions.


Instructions:

Use only the information from the provided document.

Make sure every category is filled in if the information exists in the text.

Extract accurate information and organize percisly into these categories:

High-Level Extraction
"

Title of Article,

Keywords.

Name of the Tool,

Frameworks Used,

Specific Frameworks,

VA Evaluations,

Datasets,

Tasks Identified/Addressed,

medical Tasks Identified/Addressed,

user Interactions implimented , 

data Visualization used ,

Analytics methods (ml/stats),

Multi-scale and Multidimensionality ,

explanable Visualization approach,

Data Integration Method details,

Human Decision-Making supports,

Implications for SDOH Visualization Research,

data Visualization structure used,

Application User / Target Population,

Objective,

Data Used,

Medical Doctor/MD as author (Yes/No),

Technological Stack for Development used,

Study Evaluation used,

Link to paper (first doi)
"

Document context to use and extract above data:"
{context_str}
"

    
"""

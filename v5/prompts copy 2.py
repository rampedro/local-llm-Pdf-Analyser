CATEGORY_EXTRACTION_PROMPT = """
  instructions":
    "Ensure all required fields are extracted for each document. If any category is missing or not applicable to the document, mark it as 'Not Available' or 'Not Applicable'.",
  
  "fields": 
    "Year": "Year of publication",
    "ID": "Unique identifier or reference for the paper/article",
    "Title of Computer Science Article": "Full title of the article",
    "Inclusion Criteria": "Criteria used for selecting articles or studies for inclusion",
    "Name of Tool": "Name of any tool mentioned",
    "Frameworks": "Frameworks mentioned in the paper",
    "Uses Specific Frameworks": "Does the paper use specific frameworks? Mention them if so.",
    "VA Evaluations": "Details related to Visual Analytics (VA) evaluations",
    "Datasets": "Datasets mentioned in the context",
    "Tasks Identified and Addressed": "Specific tasks identified and addressed",
    "Interaction Type": "List of interactions like annotating, filtering, searching, etc.",
    "Visualization Type": "Type of visualization discussed (e.g., charts, maps, etc.)",
    "Analytic (ML/Stat)": "Mention any ML or statistical methods used in the paper",
    "Multi-scale and Multidimensionality Addressed": "How multi-scale and multidimensionality are addressed",
    "Explanability of Visualization": "References to how explainable or interpretable the visualization is",
    "Data Integration Method": "Methods used to integrate different types of data",
    "Human Decision-making on Data": "Information related to human decision-making on data",
    "Implication for My Research": "Implications for your research or related areas",
    "Visualization Techniques": "Specific visualization techniques discussed",
    "Application User / Target Population": "Target population or users the application is designed for",
    "Objective": "Main objective or goal of the study",
    "Data Used": "Details on the data used in the study",
    "Medical Doctor as Author": "Yes/No",
    "Patterns Used": 
      "Primary Structures": "List of primary structures used in the paper (e.g., Fusion, Token, etc.)",
      "Substrate Structures": "List of substrate structures used (e.g., Coordinate, Cell, etc.)",
      "Relational Structures": "List of relational structures used (e.g., Branch, Cycle, etc.)"
    
    "Featured Interaction": "Main or featured type of user interaction",
    "Technical Stack for Development": "Technical stack or tools used for development",
    "Evaluation Details": "Details on how the study or tool was evaluated",
    "Link to the Paper": "Direct link to the paper (if available)"
  

  "document_context": "{context_str}"
"""

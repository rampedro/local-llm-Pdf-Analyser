CATEGORY_EXTRACTION_PROMPT = """
You are an intelligent assistant. Your task is to extract high-level information from Computer Science articles and categorize it into predefined topics. Ensure the following tasks are completed:

**Only extract details available in the provided document context.**
**Ensure all sections and categories are covered.**


### Organize document data into the following categories:

1. **High-Level Extraction**:
   - **ID**: [Extracted ID]
   - **Title of Computer Science Article**: [Extracted title]
   - **Inclusion Criteria**: [Extracted criteria]
   - **Name of the Tool**
   - **Frameworks use**: [Extracted frameworks]
   - **Specific Frameworks**: [Extracted frameworks]
   - **VA Evaluations**: [Extracted evaluations]
   - **Datasets**: [Extracted datasets]
   - **Tasks Identified and Addressed**: [Extracted tasks]
   - **Interaction Type**: [Extracted interaction types such as 

      --**Annotating**
   --**Arranging**
   --**Assigning**
   -- **Blending**
   --**Cloning**
   --**Comparing**
   --**Drilling**
   --**Filtering**
   --**Measuring**
   --**Navigating**
   --**Scoping**
   --**Searching**
   --**Selecting**
   --**Sharing**
   --**Transforming**
   --**Translating**
   --**Accelerating / Decelerating**
   --**Animating / Freezing**
   --**Collapsing / Expanding**
   --**Composing / Decomposing**
   --**Gathering / Discarding**
   --**Inserting / Removing**
   --**Linking / Unlinking**
   --**Storing / Retrieving**
   ]
   - **Visualization Type**: [Extracted visualization types]
   - **Analytic Used**: [Extracted analytics]
   - **How Multi-scale and Multidimensionality Are Addressed**: [Extracted methods]
   - **Explainability of Visualization**: [Extracted explainability methods]
   - **Data Integration Method**: [Extracted methods]
   - **Human Decision-Making on Data**: [Extracted decision-making details]
   - **Implication for SDOH Visualization Research**: [Extracted implications]
   - **Visualization Techniques**: [Extracted techniques]
   - **Application User / Target Population**: [Extracted user/population]
   - **Objective**: [Extracted objective]
   - **Data Used**: [Extracted data used]
   - **Medical Doctor as Author (Yes/No)**: [Yes/No]
   - **Patterns Used**: [Extracted patterns such as 

Primary Structures: List of primary structures used in the paper (e.g., Fusion, Token, etc.),
      Substrate Structures: List of substrate structures used (e.g., Coordinate, Cell, etc.),
      Relational Structures: List of relational structures used (e.g., Branch, Cycle, etc.)

   ]
   - **Featured Interaction Stack for Development**: [Extracted stack]
   - **Evaluation Details**: [Extracted evaluation methods]
   - **Link**: [Extracted link]

Document context to use and extract above data:
----------------------
{context_str}

result:

    
"""

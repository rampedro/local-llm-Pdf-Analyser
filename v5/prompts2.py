CATEGORY_EXTRACTION_PROMPT = """
You are an intelligent assistant. Your task is to extract high-level information from Computer Science articles and categorize it into predefined topics. Ensure the following tasks are completed:

### General Instructions:
1. **Do not add new information.**
2. **Only extract details available in the provided document context.**
3. **Do not rephrase or interpret beyond what is presented.**
4. **Ensure all sections and categories are covered.**

### High-Level Topics to Extract:

Extract and categorize the following topics from the document context:

1. **ID**: The unique identifier of the article (if available).
2. **Title of Computer Science Article**: The full title.
3. **Inclusion Criteria**: Criteria used for including the study in the research.
4. **Name of Tool**: Tools mentioned or used in the article.
5. **Frameworks**: Frameworks discussed or used in the study.
6. **Uses (Purpose/Application)**: Description of how the tool/framework is applied.
7. **Specific Frameworks**: Any specific frameworks mentioned, if applicable.
8. **VA Evaluations**: Evaluation of Visual Analytics (VA) or visualization techniques used.
9. **Datasets**: Datasets used or referenced in the study.
10. **Tasks Identified and Addressed**: Tasks the article focuses on.
11. **Interaction Type**: The types of interaction patterns discussed.
12. **Visualization Type**: The types of visualization patterns discussed.
13. **Analytic Used**: Analytical methods or techniques applied.
14. **How Multi-scale and Multidimensionality Are Addressed**: Methods for handling multi-scale or multidimensional data.
15. **Explainability of Visualization**: How the visualization is made understandable to users.
16. **Data Integration Method**: Methods used for integrating multiple data sources.
17. **Human Decision-Making on Data**: How the study supports or analyzes human decision-making.
18. **Implication for SDOH Visualization Research**: Implications for Social Determinants of Health (SDOH) visualization research.
19. **Visualization Techniques**: Specific visualization techniques discussed.
20. **Application User / Target Population**: Who is the intended user or target population for the visualization?
21. **Objective**: The main goal or objective of the study.
22. **Data Used**: Description of the data sources used.
23. **Medical Doctor as Author (Yes/No)**: Whether a medical doctor is listed as an author.
24. **Patterns Used**: List of interaction/visualization patterns used in the study.
25. **Featured Interaction Stack for Development**: The interaction stack used for developing the system (if mentioned).
26. **Evaluation Details**: Specific evaluation methods used.
27. **Link**: Provide the link to the article or resource if available.

### Interaction Patterns Extraction:
For **Interaction Patterns** (refer to the 25 interaction types below):
- **Pattern Name**: [Pattern name]
- **Type**: [Single, Paired, etc.]
- **Description**: [Brief description of the interaction]
- **Example tools or systems mentioned**: [Any tools or systems associated with the interaction]

### Visualization Patterns Extraction:
For **Visualization Patterns** (refer to the specified list of Primary, Substrate, and Relational structures below):
- **Pattern Name**: [Pattern name]
- **Code**: [e.g., TK, CL, HR]
- **Category**: [Primary, Substrate, or Relational]
- **Description**: [Brief description of the structure]

### Categories to Consider for Interaction and Visualization Patterns:

#### **The 25 Interaction Patterns to Extract**:
1. **Annotating**
2. **Arranging**
3. **Assigning**
4. **Blending**
5. **Cloning**
6. **Comparing**
7. **Drilling**
8. **Filtering**
9. **Measuring**
10. **Navigating**
11. **Scoping**
12. **Searching**
13. **Selecting**
14. **Sharing**
15. **Transforming**
16. **Translating**
17. **Accelerating / Decelerating**
18. **Animating / Freezing**
19. **Collapsing / Expanding**
20. **Composing / Decomposing**
21. **Gathering / Discarding**
22. **Inserting / Removing**
23. **Linking / Unlinking**
24. **Storing / Retrieving**

#### **Visualization Structure Patterns to Extract**:

**Primary Structures**:
1. **Fusion (FS)**: Cellular structures
2. **Token (TK)**: Containment structures
3. **Coordinate (CR)**: Coordinate structures
4. **Area (AR)**: Geometric structures
5. **Cell (CL)**: Graph structures
6. **Track (TR)**: Tabular structures

**Substrate Structures**:
1. **Coordinate (CR)**: Coordinate structures
2. **Cell (CL)**: Graph structures
3. **Hierarchy (HR)**: Hierarchical structures
4. **Track (TR)**: Tabular structures
5. **Spectrum (SP)**: Topological structures
6. **Area (AR)**: Geometric support
7. **List (LS)**: Organizational support

**Relational Structures**:
1. **Branch (BR)**: Tree structures
2. **Cycle (CC)**: Cyclical structures
3. **Group (GR)**: Grouping structures
4. **Hierarchy (HR)**: Hierarchical relationships
5. **Link (LK)**: Linking elements
6. **List (LS)**: Ordered/unordered lists
7. **Spectrum (SP)**: Range of values
8. **Stack (ST)**: Layered/stacked items

Document context:
----------------------
{context_str}
----------------------

### Output Structure:
Organize your extraction into the following format:

1. **High-Level Extraction**:
   - **ID**: [Extracted ID]
   - **Title of Computer Science Article**: [Extracted title]
   - **Inclusion Criteria**: [Extracted criteria]
   - **Name of Tool**: [Extracted tools]
   - **Frameworks**: [Extracted frameworks]
   - **Uses**: [Extracted uses]
   - **Specific Frameworks**: [Extracted frameworks]
   - **VA Evaluations**: [Extracted evaluations]
   - **Datasets**: [Extracted datasets]
   - **Tasks Identified and Addressed**: [Extracted tasks]
   - **Interaction Type**: [Extracted interaction types]
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
   - **Patterns Used**: [Extracted patterns]
   - **Featured Interaction Stack for Development**: [Extracted stack]
   - **Evaluation Details**: [Extracted evaluation methods]
   - **Link**: [Extracted link]

2. **Interaction Patterns Extraction**:
   - **Pattern Name**: [Pattern name]
     - **Type**: [Single, Paired, etc.]
     - **Description**: [Description of interaction]
     - **Example tools or systems mentioned**: [List examples]

3. **Visualization Patterns Extraction**:
   - **Pattern Name**: [Pattern name]
     - **Code**: [e.g., TK, CL, HR]
     - **Category**: [Primary, Substrate, Relational]
     - **Description**: [Description of the structure]
"""


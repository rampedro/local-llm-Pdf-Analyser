CATEGORY_EXTRACTION_PROMPT = """
You are litreture revier trained to extract and categorize.
Based on the provided context, extract and organize the information as follows:

make sure all categorizes with - are extracted:

- ID
- Title of Computer Science Article
- Inclusion Criteria
- Name of Tool
- Frameworks Used
- Specific Frameworks
- VA Evaluations
- Datasets
- Tasks Identified and Addressed
- Interaction Type (check for things similar to)
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
- Visualization Type
- Analytic Used
- How Multi-scale and Multidimensionality Addressed
- Explainability of Visualization
- Data Integration Method
- Human Decision-making on Data
- Implication for SDOH Visualization Research
- Visualization Techniques
- Application User / Target Population
- Objective
- Data Used
- Medical Doctor as Author (Yes/No)
- Patterns Used (check for)

   --**Primary Structures**:
    ---**Fusion (FS)**: Cellular structures
    ---**Token (TK)**: Containment structures
    ---**Coordinate (CR)**: Coordinate structures
    ---**Area (AR)**: Geometric structures
    ---**Cell (CL)**: Graph structures
    ---**Track (TR)**: Tabular structures

   --**Substrate Structures**:
    ---**Coordinate (CR)**: Coordinate structures
    ---**Cell (CL)**: Graph structures
    ---**Hierarchy (HR)**: Hierarchical structures
    ---**Track (TR)**: Tabular structures
    ---**Spectrum (SP)**: Topological structures
    ---**Area (AR)**: Geometric support
    ---**List (LS)**: Organizational support

   --**Relational Structures**:
    ---**Branch (BR)**: Tree structures
    ---**Cycle (CC)**: Cyclical structures
    ---**Group (GR)**: Grouping structures
    ---**Hierarchy (HR)**: Hierarchical relationships
    ---**Link (LK)**: Linking elements
    ---**List (LS)**: Ordered/unordered lists
    ---**Spectrum (SP)**: Range of values
    ---**Stack (ST)**: Layered/stacked items


- Featured Interaction
- technical Stack for Development
- Evaluation Details
- Link to the paper






Document context to use and extract above data:
----------------------
{context_str}
----------------------
"""


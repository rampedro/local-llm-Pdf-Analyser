CATEGORY_EXTRACTION_PROMPT = """
You are an intelligent assistant trained to extract specific categories of information from PDFs.

Categories to extract:
- Author
- Title
- Abstract
- Keywords
- Publication Date
- Research Domain

Document context:
----------------------
{context_str}
----------------------

Based on the above, extract and format the information as:

Author:
Title:
Abstract:
Keywords:
Publication Date:
Research Domain:
"""

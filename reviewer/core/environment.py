"""Environment for reading and parsing scientific papers."""

import re
from typing import List, Dict, Optional
# import fitz  # PyMuPDF
from pydantic import BaseModel


class Section(BaseModel):
    """A section in the paper."""

    name: str
    page_start: int
    page_end: Optional[int] = None
    content: str = ""


class PaperEnvironment:
    """Environment for interacting with a scientific paper."""

    def __init__(self, paper: str):
        """Initialize the environment with a paper.
        TODO: Add support for other formats, e.g. pdf.

        Args:
            paper: Paper content in latex or markdown format.
        """
        # self.paper_path = paper_path
        # self.doc = fitz.open(paper_path)
        # with open(paper_path, "r", encoding="utf-8") as f:
            # self.paper = f.readlines()[0]
        self.paper = paper
        # Convert escaped newlines to actual newlines for parsing
        self.paper = self.paper.replace('\\n', '\n')
        self.sections: Dict[str, Section] = {}
        self.section_to_subsections: Dict[str, List[str]] = {}  # Maps parent section to its subsections
        self._extract_sections()
        self._extract_toc()
        self._build_subsection_map()

    def _extract_toc(self) -> None:
        """Extract table of contents from the paper."""
        self.toc = [section.name for section in self.sections.values()]

    def _build_subsection_map(self) -> None:
        """Build a mapping from parent sections to their subsections.

        For example, "2 related work" -> ["2.1 video multimodal", "2.2 text analysis"]
        """
        section_keys = list(self.sections.keys())

        for section_key in section_keys:
            # Extract the section number prefix (e.g., "2" from "2 related work" or "2." from "2. methods")
            # Pattern matches: "2 ", "2. ", "2.1 ", etc.
            match = re.match(r'^(\d+)(?:\.|\s)', section_key)
            if not match:
                continue

            section_num = match.group(1)
            subsections = []

            # Find all subsections that start with this section number followed by a dot
            # e.g., for section "2", find "2.1", "2.2", etc.
            subsection_pattern = f"^{section_num}\\."
            for other_key in section_keys:
                if other_key != section_key and re.match(subsection_pattern, other_key):
                    subsections.append(other_key)

            if subsections:
                self.section_to_subsections[section_key] = subsections

    def _detect_format(self) -> str:
        """Detect whether the paper is in LaTeX or Markdown format.
        
        Returns:
            'latex' or 'markdown'
        """
        # Check for Markdown indicators first (##, ###, etc. at start of line or string)
        # Do this before LaTeX detection to avoid false positives when paper body
        # contains quoted LaTeX snippets (e.g., \section{...} in an example table).
        markdown_pattern = r'(?:^|\n)#{1,6}\s+'
        if re.search(markdown_pattern, self.paper):
            return 'markdown'

        # Check for LaTeX indicators (only reached if no markdown headers found)
        latex_indicators = [
            r'\\title\{',
            r'\\begin\{abstract\}',
            r'\\section\{',
            r'\\subsection\{',
            r'\\documentclass',
        ]
        for pattern in latex_indicators:
            if re.search(pattern, self.paper):
                return 'latex'
        
        # Default to None
        return None

    def _extract_sections(self) -> None:
        """Extract sections based on detected format."""
        format_type = self._detect_format()
        if format_type == 'markdown':
            self._extract_sections_markdown()
        elif format_type == 'latex':
            self._extract_sections_latex()
        else:
            raise ValueError(f"Unsupported format: {format_type}. Please check the paper format.")

    def _extract_sections_markdown(self) -> None:
        """Extract section content from Markdown paper.
        
        Supports formats like:
        - # Title or Title: ... (title can be markdown heading or plain text with "Title:" prefix)
        - ## 1 Introduction (high-level sections)
        - ### 2.1 Video multimodal (subsections, extracted up to this level)
        - ###### Abstract (level 6 headers, hardcoded) or ## Abstract (level 2 headers, hardcoded)
        """
        # First, check for "Title: ..." format at the beginning of the document
        title_prefix_pattern = r'^Title:\s*(.+?)(?:\n|$)'
        title_match = re.match(title_prefix_pattern, self.paper, re.IGNORECASE)
        if title_match:
            title_content = title_match.group(1).strip()
            self.sections["title"] = Section(
                name="Title",
                page_start=0,
                page_end=None,
                content=title_content
            )
        
        # Pattern to match markdown headers: # Title, ## Section, ### Subsection, ###### Abstract or ## Abstract
        # Captures: level (number of #), optional number prefix, and title
        header_pattern = r'(?:^|\n)(#{1,6})\s+(.+?)(?:\n|$)'
        header_matches = list(re.finditer(header_pattern, self.paper))
        
        if not header_matches:
            return
        
        # First, extract abstract with hardcoded ###### Abstract (level 6) or ## Abstract (level 2)
        for match in header_matches:
            level = len(match.group(1))
            full_title = match.group(2).strip()
            if (level == 6 or level == 2) and 'abstract' in full_title.lower():
                start_pos = match.end()
                # Find end position (next header or end of document)
                next_match = None
                for next_m in header_matches:
                    if next_m.start() > match.start():
                        next_match = next_m
                        break
                if next_match:
                    end_pos = next_match.start()
                else:
                    end_pos = len(self.paper)
                content = self.paper[start_pos:end_pos].strip()
                self.sections["abstract"] = Section(
                    name="Abstract",
                    page_start=0,
                    page_end=None,
                    content=content
                )
                break
        
        # Extract sections up to level 3 (###) only
        for i, match in enumerate(header_matches):
            level = len(match.group(1))  # Number of # symbols
            full_title = match.group(2).strip()
            start_pos = match.end()
            
            # Skip level 1 (title) - handle separately, but only if not already extracted from "Title:" format
            if level == 1:
                if "title" not in self.sections:
                    self.sections["title"] = Section(
                        name="Title",
                        page_start=0,
                        page_end=None,
                        content=full_title
                    )
                continue
            
            # Skip levels 4, 5, and 6 (except abstract which is handled above)
            if level > 3:
                continue
            
            # Skip abstract if it's level 2 or 6 (already extracted above)
            if 'abstract' in full_title.lower():
                continue
            
            # Skip References section
            if 'references' in full_title.lower():
                continue
            
            display_name = full_title
            section_key = display_name.lower()
            
            # For level 2 sections (##), find all content including subsections (###)
            if level == 2:
                # Find the end position: next level 2 header or end of document
                end_pos = len(self.paper)
                for j in range(i + 1, len(header_matches)):
                    next_level = len(header_matches[j].group(1))
                    if next_level == 2:  # Next high-level section
                        end_pos = header_matches[j].start()
                        break
                
                # Extract all content including subsections
                content = self.paper[start_pos:end_pos].strip()
                
                # Create Section object with all subsections included
                self.sections[section_key] = Section(
                    name=display_name,
                    page_start=0,
                    page_end=None,
                    content=content
                )
            elif level == 3:
                # For level 3 (###), extract only this subsection's content
                # Find end position: next header at level 2 or 3, or end of document
                end_pos = len(self.paper)
                for j in range(i + 1, len(header_matches)):
                    next_level = len(header_matches[j].group(1))
                    if next_level <= 3:  # Next section or subsection
                        end_pos = header_matches[j].start()
                        break
                
                # Extract content between headers
                content = self.paper[start_pos:end_pos].strip()
                
                # Create Section object
                self.sections[section_key] = Section(
                    name=display_name,
                    page_start=0,
                    page_end=None,
                    content=content
                )

    def _extract_sections_latex(self) -> None:
        """Extract section content from LaTeX paper."""
        # Extract title
        title_match = re.search(r'\\title\{([^}]+)\}', self.paper)
        if title_match:
            self.sections["title"] = Section(
                name="Title",
                page_start=0,
                page_end=None,
                content=title_match.group(1)
            )

        # Extract abstract
        abstract_match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', self.paper, re.DOTALL)
        if abstract_match:
            self.sections["abstract"] = Section(
                name="Abstract",
                page_start=0,
                page_end=None,
                content=abstract_match.group(1).strip()
            )

        # Find all section positions
        section_pattern = r'\\(section|subsection|subsubsection)(\*?)\{([^}]+)\}'
        section_matches = list(re.finditer(section_pattern, self.paper))

        counters = [0, 0, 0]  # section, subsection, subsubsection

        for i, match in enumerate(section_matches):
            sec_type = match.group(1)
            is_starred = match.group(2) == '*'
            name = match.group(3)
            start_pos = match.end()

            # Determine numbering
            display_name = name
            if not is_starred:
                if sec_type == 'section':
                    counters[0] += 1
                    counters[1] = 0
                    counters[2] = 0
                    display_name = f"{counters[0]}. {name}"
                elif sec_type == 'subsection':
                    counters[1] += 1
                    counters[2] = 0
                    display_name = f"{counters[0]}.{counters[1]} {name}"
                elif sec_type == 'subsubsection':
                    counters[2] += 1
                    display_name = f"{counters[0]}.{counters[1]}.{counters[2]} {name}"

            section_key = display_name.lower()

            # Determine end position (start of next section or end of document)
            if i + 1 < len(section_matches):
                end_pos = section_matches[i + 1].start()
            else:
                # Look for bibliography or end of document
                bib_match = re.search(r'\\begin\{thebibliography\}|\\bibliographystyle|\\bibliography\{', self.paper[start_pos:])
                if bib_match:
                    end_pos = start_pos + bib_match.start()
                else:
                    end_pos = len(self.paper)

            # Extract content between sections
            content = self.paper[start_pos:end_pos].strip()

            # Create Section object
            self.sections[section_key] = Section(
                name=display_name,
                page_start=0,  # Page numbers not available in raw LaTeX
                page_end=None,
                content=content
            )

    def get_section_dict(self) -> Dict[str, str]:
        """Get a dictionary mapping section names to their content.

        Returns:
            Dictionary with section names as keys and content as values
        """
        return {name: section.content for name, section in self.sections.items()}

    def get_section_names(self) -> List[str]:
        """Get list of all section names.

        Returns:
            List of section names
        """
        return list(self.sections.keys())

    def get_subsections(self, section_name: str) -> List[str]:
        """Get list of subsections covered by a parent section.

        Args:
            section_name: Name of the parent section

        Returns:
            List of subsection names (empty if no subsections)
        """
        section_key = section_name.lower()
        return self.section_to_subsections.get(section_key, [])

    def read_section(self, section_name: str) -> Optional[str]:
        """Read the content of a specific section.

        Args:
            section_name: Name of the section to read (supports fuzzy matching)

        Returns:
            Section content or None if not found
        """
        # Normalize Unicode apostrophes/quotes to ASCII before matching
        _UNICODE_APOS = str.maketrans("\u2019\u2018\u02bc\u0060\u00b4", "'''''" )
        section_name = section_name.lower().strip().translate(_UNICODE_APOS)

        # Try exact match first
        section = self.sections.get(section_name, None)

        # Fuzzy match: find section that ends with or contains the query
        if section is None:
            for key in self.sections:
                norm_key = key.translate(_UNICODE_APOS)
                # Match "experiments" to "3 experiments" or "3. experiments"
                # Strip numbers and punctuation from the beginning for comparison
                key_stripped = re.sub(r'^[\d\.\s]+', '', norm_key).strip()
                if key_stripped == section_name or section_name in norm_key or norm_key.endswith(section_name):
                    section = self.sections[key]
                    break

        if section is None:
            return None
        else:
            return f"[{section.name}]:\n{section.content}"

    def get_full_text(self) -> str:
        """Get the full text of the paper.

        Returns:
            Full paper text
        """
        return self.paper

    def search_paper(self, query: str, context_chars: int = 200) -> List[Dict]:
        """Search for a query string across all sections.

        Args:
            query: Search term (case-insensitive)
            context_chars: Characters of context around each match

        Returns:
            List of matches: [{section, snippet, match_count}, ...]
        """
        query_lower = query.lower()
        results = []

        for section_name, section in self.sections.items():
            content = section.content
            content_lower = content.lower()

            # Find all occurrences
            matches = []
            start = 0
            while True:
                idx = content_lower.find(query_lower, start)
                if idx == -1:
                    break
                # Extract snippet with context
                snippet_start = max(0, idx - context_chars)
                snippet_end = min(len(content), idx + len(query) + context_chars)
                snippet = content[snippet_start:snippet_end]
                if snippet_start > 0:
                    snippet = "..." + snippet
                if snippet_end < len(content):
                    snippet = snippet + "..."
                matches.append(snippet)
                start = idx + 1

            if matches:
                results.append({
                    "section": section_name,
                    "match_count": len(matches),
                    "snippets": matches[:3]  # Limit to top 3 snippets per section
                })

        return results
    
    
if __name__ == "__main__":
    # Test LaTeX format
    print("=" * 60)
    print("Testing LaTeX Format")
    print("=" * 60)
    
    paper_content_latex = r"""
\title{Hierarchical Paper}
\begin{abstract}
This is the abstract.
\end{abstract}

\section{Introduction}
Intro content.

\section{Methods}
Methods overview.

\subsection{Data Collection}
We collected data.

\subsubsection{Preprocessing}
We cleaned the data.

\subsection{Model Architecture}
We used a transformer.

\section{Results}
We got 99% accuracy.

\section*{Acknowledgments}
Thanks to everyone.
"""
    # Note: You would need to create a test.tex file or modify to test inline
    # paper = PaperEnvironment(paper_path="test.tex")
    
    # Test Markdown format
    print("\n" + "=" * 60)
    print("Testing Markdown Format")
    print("=" * 60)
    
    paper_content_markdown = """# My Research Paper

###### Abstract
This is the abstract of the paper.

## 1 Introduction
This is the introduction section with some content about the research motivation.

## 2 Related Work
Previous work in this area includes many studies.

### 2.1 Video Multimodal
Video multimodal learning has gained significant attention.

### 2.2 Text Analysis  
Text analysis methods have evolved over time.

#### 2.2.1 Deep Learning Approaches
Deep learning has revolutionized text analysis.

## 3 Methods
We propose a novel approach.

### 3.1 Data Collection
We collected data from multiple sources.

### 3.2 Model Architecture
Our model consists of several components.

## 4 Results
We achieved state-of-the-art results.

## 5 Conclusion
In this paper, we presented our approach.

## Acknowledgments
Thanks to everyone who helped.
"""
    
    # Write test markdown file as single line with escaped newlines
    # This matches the format when extracting paper content from JSON
    with open("/tmp/test_paper.md", "w") as f:
        # Convert actual newlines to escaped \n (stored as single line)
        single_line = paper_content_markdown.replace('\n', '\\n')
        f.write(single_line)
    
    paper = PaperEnvironment(paper=paper_content_markdown)
    
    print("\nDetected format:", paper._detect_format())
    print("\n" + "-" * 40)
    print("Table of Contents:")
    print("-" * 40)
    for section in paper.toc:
        print(f"  - {section}")
        
    print("\n" + "-" * 40)
    print("Reading Specific Sections:")
    print("-" * 40)
    
    sections_to_test = [
        "title",
        "abstract",
        "1 introduction",
        "2.1 video multimodal",
        "3 methods",
        "acknowledgments"
    ]
    
    for section_name in sections_to_test:
        content = paper.read_section(section_name)
        print(f"\n{content[:200]}..." if len(content) > 200 else f"\n{content}")

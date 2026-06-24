#!/usr/bin/env python3
r"""
LaTeX Concatenator - Recursively concatenates LaTeX files

This module provides functionality to:
1. Find main LaTeX files in a directory
2. Recursively resolve and inline \input{} and \include{} commands
3. Handle edge cases like circular references and missing files
4. Preserve all LaTeX commands intact

Usage:
    from utils.latex_concatenator import LaTeXConcatenator, LaTeXValidator

    validator = LaTeXValidator()
    main_file = validator.find_main_file("/path/to/latex/dir")

    concatenator = LaTeXConcatenator()
    content, metadata = concatenator.concatenate(main_file)
"""

import os
import re
from pathlib import Path
from typing import Tuple, List, Set, Optional, Dict
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LaTeXValidator:
    """Validates LaTeX directory structure and finds main files."""

    # Common main file names in order of priority
    MAIN_FILE_CANDIDATES = [
        'main.tex',
        'arxiv.tex',
        'paper.tex',
        'iclr2025_conference.tex',
        'neurips_2023.tex',
        'neurips_2024.tex',
    ]

    def find_main_file(self, latex_dir: str) -> Optional[str]:
        r"""
        Find the main .tex file in a directory.

        Strategy:
        1. Try common main file names (main.tex, arxiv.tex, etc.)
        2. Find .tex file with \documentclass command
        3. Return None if no main file found (user must check manually)

        Args:
            latex_dir: Path to LaTeX directory

        Returns:
            Path to main .tex file, or None if not found
        """
        latex_path = Path(latex_dir)

        if not latex_path.exists():
            logger.error(f"Directory does not exist: {latex_dir}")
            return None

        # Strategy 1: Try common main file names
        for candidate in self.MAIN_FILE_CANDIDATES:
            candidate_path = latex_path / candidate
            if candidate_path.exists():
                logger.debug(f"Found main file by name: {candidate}")
                return str(candidate_path)

        # Strategy 2: Find file with \documentclass
        tex_files = list(latex_path.glob('*.tex'))
        for tex_file in tex_files:
            try:
                with open(tex_file, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        if line.strip().startswith('\\documentclass'):
                            logger.debug(f"Found main file by \\documentclass: {tex_file.name}")
                            return str(tex_file)
            except Exception as e:
                logger.debug(f"Error reading {tex_file}: {e}")
                continue

        # No main file found - return None to mark as failed
        logger.error(f"Could not find main .tex file in {latex_dir}")
        if tex_files:
            logger.error(f"Found {len(tex_files)} .tex files but none have \\documentclass or match common names")
        return None

    def validate_directory(self, latex_dir: str) -> bool:
        """
        Validate that directory exists and contains .tex files.

        Args:
            latex_dir: Path to LaTeX directory

        Returns:
            True if valid, False otherwise
        """
        latex_path = Path(latex_dir)

        if not latex_path.exists():
            logger.error(f"Directory does not exist: {latex_dir}")
            return False

        if not latex_path.is_dir():
            logger.error(f"Not a directory: {latex_dir}")
            return False

        tex_files = list(latex_path.glob('**/*.tex'))
        if not tex_files:
            logger.error(f"No .tex files found in {latex_dir}")
            return False

        return True


class LaTeXConcatenator:
    r"""Recursively concatenates LaTeX files by resolving \input and \include commands."""

    # Regex patterns for matching \input and \include commands
    INPUT_PATTERNS = [
        re.compile(r'\\input\{([^}]+)\}'),        # \input{filename}
        re.compile(r'\\input\s+([^\s]+)'),        # \input filename
        re.compile(r'\\include\{([^}]+)\}'),      # \include{filename}
        re.compile(r'\\subfile\{([^}]+)\}'),      # \subfile{filename}
    ]

    # Commands to skip (don't inline these)
    SKIP_COMMANDS = [
        '\\bibliography',
        '\\includegraphics',
        '\\usepackage',
        '\\documentclass',
    ]

    # Common subdirectories to search
    SEARCH_DIRS = [
        '.',
        'sections',
        'chapters',
        'tables',
        'texts',
        'figTexts',
        'programs',
    ]

    def __init__(self, max_depth: int = 10):
        """
        Initialize concatenator.

        Args:
            max_depth: Maximum recursion depth to prevent infinite loops
        """
        self.max_depth = max_depth
        self.warnings: List[str] = []
        self.files_included: List[str] = []

    def concatenate(self, main_tex_path: str) -> Tuple[str, Dict]:
        """
        Concatenate LaTeX files starting from main file.

        Args:
            main_tex_path: Path to main .tex file

        Returns:
            Tuple of (concatenated_content, metadata)
            metadata includes: files_included, warnings, total_lines, etc.
        """
        # Reset state
        self.warnings = []
        self.files_included = []

        main_path = Path(main_tex_path)
        if not main_path.exists():
            raise FileNotFoundError(f"Main file not found: {main_tex_path}")

        base_dir = main_path.parent
        visited: Set[str] = set()

        # Start recursive concatenation
        content = self._concatenate_recursive(
            main_path,
            base_dir,
            visited,
            depth=0
        )

        # Generate metadata
        metadata = {
            'main_file': main_path.name,
            'files_included': self.files_included,
            'warnings': self.warnings,
            'total_lines': content.count('\n') + 1,
            'total_chars': len(content),
            'concatenated_at': datetime.now().isoformat()
        }

        return content, metadata

    def _concatenate_recursive(
        self,
        tex_file: Path,
        base_dir: Path,
        visited: Set[str],
        depth: int
    ) -> str:
        """
        Recursively concatenate LaTeX files.

        Args:
            tex_file: Current .tex file to process
            base_dir: Base directory for resolving relative paths
            visited: Set of already visited files (for cycle detection)
            depth: Current recursion depth

        Returns:
            Concatenated content
        """
        # Check recursion depth
        if depth > self.max_depth:
            warning = f"Max recursion depth ({self.max_depth}) reached at {tex_file.name}"
            logger.warning(warning)
            self.warnings.append(warning)
            return ""  # Return empty string to avoid adding comments

        # Check for circular references
        tex_file_str = str(tex_file.resolve())
        if tex_file_str in visited:
            warning = f"Circular reference detected: {tex_file.name}"
            logger.warning(warning)
            self.warnings.append(warning)
            return ""  # Return empty string to avoid adding comments

        visited.add(tex_file_str)
        self.files_included.append(str(tex_file.relative_to(base_dir)))

        # Read file with encoding fallback
        content = self._read_file_safe(tex_file)
        if content is None:
            warning = f"Failed to read file: {tex_file.name}"
            self.warnings.append(warning)
            return ""  # Return empty string to avoid adding comments

        # Process each line (no delimiters for cleaner output)
        result = []
        lines = content.split('\n')
        for line in lines:
            # Check if line should be inlined
            if self._should_inline(line):
                # Extract filename from \input or \include command
                filename = self._extract_filename(line)
                if filename:
                    # Try to resolve file path
                    resolved_path = self._resolve_path(filename, tex_file.parent, base_dir)

                    if resolved_path and resolved_path.exists():
                        # Recursively concatenate
                        inlined_content = self._concatenate_recursive(
                            resolved_path,
                            base_dir,
                            visited.copy(),  # Copy to allow same file in different branches
                            depth + 1
                        )
                        result.append(inlined_content)
                    else:
                        # Keep original command and log warning
                        warning = f"File not found: {filename} (referenced in {tex_file.name})"
                        logger.debug(warning)
                        self.warnings.append(warning)
                        result.append(f"{line}\n")
                else:
                    # Couldn't extract filename, keep original
                    result.append(f"{line}\n")
            else:
                # Keep original line
                result.append(f"{line}\n")

        return ''.join(result)

    def _should_inline(self, line: str) -> bool:
        r"""
        Check if line contains \input or \include command that should be inlined.

        Args:
            line: Line of LaTeX code

        Returns:
            True if should inline, False otherwise
        """
        stripped = line.strip()

        # Skip comments
        if stripped.startswith('%'):
            return False

        # Skip certain commands
        for skip_cmd in self.SKIP_COMMANDS:
            if skip_cmd in line:
                return False

        # Check for input/include patterns
        for pattern in self.INPUT_PATTERNS:
            if pattern.search(line):
                return True

        return False

    def _extract_filename(self, line: str) -> Optional[str]:
        r"""
        Extract filename from \input or \include command.

        Args:
            line: Line containing \input or \include command

        Returns:
            Filename or None if not found
        """
        for pattern in self.INPUT_PATTERNS:
            match = pattern.search(line)
            if match:
                filename = match.group(1).strip()
                return filename

        return None

    def _resolve_path(
        self,
        filename: str,
        current_dir: Path,
        base_dir: Path
    ) -> Optional[Path]:
        """
        Resolve file path by trying multiple strategies.

        Strategy:
        1. Try exact path from current directory
        2. Try with .tex extension
        3. Search in common subdirectories
        4. Try from base directory

        Args:
            filename: Filename to resolve
            current_dir: Directory of current .tex file
            base_dir: Base directory of LaTeX project

        Returns:
            Resolved Path or None if not found
        """
        # Strategy 1: Exact path from current directory
        candidate = current_dir / filename
        if candidate.exists():
            return candidate

        # Strategy 2: Try with .tex extension
        if not filename.endswith('.tex'):
            candidate = current_dir / f"{filename}.tex"
            if candidate.exists():
                return candidate

        # Strategy 3: Search in common subdirectories from current dir
        for search_dir in self.SEARCH_DIRS:
            candidate = current_dir / search_dir / filename
            if candidate.exists():
                return candidate

            if not filename.endswith('.tex'):
                candidate = current_dir / search_dir / f"{filename}.tex"
                if candidate.exists():
                    return candidate

        # Strategy 4: Try from base directory
        candidate = base_dir / filename
        if candidate.exists():
            return candidate

        if not filename.endswith('.tex'):
            candidate = base_dir / f"{filename}.tex"
            if candidate.exists():
                return candidate

        # Strategy 5: Search subdirectories from base
        for search_dir in self.SEARCH_DIRS:
            candidate = base_dir / search_dir / filename
            if candidate.exists():
                return candidate

            if not filename.endswith('.tex'):
                candidate = base_dir / search_dir / f"{filename}.tex"
                if candidate.exists():
                    return candidate

        return None

    def _read_file_safe(self, file_path: Path) -> Optional[str]:
        """
        Read file with encoding fallback.

        Tries UTF-8 first, then latin-1.

        Args:
            file_path: Path to file

        Returns:
            File content or None if failed
        """
        encodings = ['utf-8', 'latin-1', 'iso-8859-1']

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except (UnicodeDecodeError, UnicodeError):
                logger.debug(f"Failed to read {file_path} with {encoding}")
                continue
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
                return None

        logger.error(f"Failed to read {file_path} with all encodings")
        return None


# Convenience function
def concatenate_latex(latex_dir: str) -> Tuple[Optional[str], Optional[Dict]]:
    """
    Convenience function to concatenate LaTeX files in a directory.

    Args:
        latex_dir: Path to LaTeX directory

    Returns:
        Tuple of (content, metadata) or (None, None) if failed
    """
    try:
        validator = LaTeXValidator()
        main_file = validator.find_main_file(latex_dir)

        if not main_file:
            logger.error(f"Could not find main .tex file in {latex_dir}")
            return None, None

        concatenator = LaTeXConcatenator()
        content, metadata = concatenator.concatenate(main_file)

        return content, metadata

    except Exception as e:
        logger.error(f"Error concatenating LaTeX in {latex_dir}: {e}")
        return None, None


if __name__ == '__main__':
    # Simple CLI for testing
    import sys

    if len(sys.argv) < 2:
        print("Usage: python latex_concatenator.py <latex_dir>")
        sys.exit(1)

    latex_dir = sys.argv[1]
    content, metadata = concatenate_latex(latex_dir)

    if content:
        print(f"Successfully concatenated {metadata['files_included']} files")
        print(f"Total lines: {metadata['total_lines']}")
        print(f"Total chars: {metadata['total_chars']}")
        if metadata['warnings']:
            print(f"Warnings: {len(metadata['warnings'])}")
            for warning in metadata['warnings'][:5]:
                print(f"  - {warning}")
    else:
        print("Failed to concatenate LaTeX")
        sys.exit(1)

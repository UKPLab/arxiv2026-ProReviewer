"""PDF processing with MinerU and PyMuPDF fallback.

This module provides functionality to convert PDFs to markdown using:
1. MinerU (magic-pdf) - preferred, better quality
2. PyMuPDF (fitz) - fallback, simple text extraction
"""

import subprocess
import logging
import shutil
from pathlib import Path
from typing import Optional

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    fitz = None


logger = logging.getLogger(__name__)


class MinerUProcessor:
    """PDF to Markdown converter using MinerU with PyMuPDF fallback."""

    def __init__(self, mineru_path: str = "magic-pdf"):
        """Initialize the PDF processor.

        Args:
            mineru_path: Path to MinerU executable (default: "magic-pdf")
        """
        self.mineru_path = mineru_path
        self._mineru_available = None

    def is_mineru_available(self) -> bool:
        """Check if MinerU is installed and accessible.

        Returns:
            True if MinerU is available, False otherwise
        """
        if self._mineru_available is not None:
            return self._mineru_available

        # Check if the command exists
        mineru_cmd = shutil.which(self.mineru_path)
        if mineru_cmd is None:
            logger.info(
                f"MinerU not found at '{self.mineru_path}'. "
                "Install with: pip install magic-pdf"
            )
            self._mineru_available = False
            return False

        # Try running it to verify it works
        try:
            result = subprocess.run(
                [self.mineru_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            self._mineru_available = result.returncode == 0
            if self._mineru_available:
                logger.info(f"MinerU is available: {self.mineru_path}")
            else:
                logger.warning(
                    f"MinerU found but not working: {result.stderr}"
                )
        except Exception as e:
            logger.warning(f"Error checking MinerU availability: {e}")
            self._mineru_available = False

        return self._mineru_available

    def convert_pdf_to_markdown(
        self,
        pdf_path: Path,
        output_dir: Path
    ) -> Optional[str]:
        """Convert PDF to markdown.

        Tries MinerU first, falls back to PyMuPDF if that fails.

        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to write output files

        Returns:
            Markdown content as string, or None if both methods fail
        """
        # Try MinerU first
        if self.is_mineru_available():
            markdown = self._convert_with_mineru(pdf_path, output_dir)
            if markdown is not None:
                logger.info(
                    f"Successfully converted {pdf_path.name} using MinerU"
                )
                return markdown
            else:
                logger.warning(
                    f"MinerU conversion failed for {pdf_path.name}, "
                    "trying PyMuPDF fallback"
                )

        # Fallback to PyMuPDF
        if PYMUPDF_AVAILABLE:
            markdown = self._convert_with_pymupdf(pdf_path)
            if markdown is not None:
                logger.info(
                    f"Successfully converted {pdf_path.name} using PyMuPDF"
                )
                return markdown

        logger.error(f"Failed to convert {pdf_path.name} with any method")
        return None

    def _convert_with_mineru(
        self,
        pdf_path: Path,
        output_dir: Path
    ) -> Optional[str]:
        """Convert PDF to markdown using MinerU.

        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to write output files

        Returns:
            Markdown content as string, or None if conversion fails
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Converting {pdf_path.name} with MinerU")

        try:
            # Run MinerU
            # magic-pdf -p {pdf_path} -o {output_dir} -m auto
            result = subprocess.run(
                [
                    self.mineru_path,
                    "-p", str(pdf_path),
                    "-o", str(output_dir),
                    "-m", "auto"
                ],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode != 0:
                logger.warning(
                    f"MinerU conversion failed for {pdf_path.name}: "
                    f"{result.stderr}"
                )
                return None

            # Find the generated markdown file
            # MinerU typically creates files in output_dir/{pdf_name}/auto/
            pdf_stem = pdf_path.stem
            possible_paths = [
                output_dir / pdf_stem / "auto" / f"{pdf_stem}.md",
                output_dir / pdf_stem / f"{pdf_stem}.md",
                output_dir / f"{pdf_stem}.md",
            ]

            # Also try glob search
            md_files = list(output_dir.glob("**/*.md"))

            # Try specific paths first
            for md_path in possible_paths:
                if md_path.exists():
                    return md_path.read_text(encoding='utf-8', errors='ignore')

            # Try glob results
            if md_files:
                # Use the first or largest markdown file
                md_path = max(md_files, key=lambda f: f.stat().st_size)
                logger.debug(f"Using markdown file: {md_path}")
                return md_path.read_text(encoding='utf-8', errors='ignore')

            logger.warning(
                f"MinerU completed but no markdown file found for {pdf_path.name}"
            )
            return None

        except subprocess.TimeoutExpired:
            logger.error(f"MinerU conversion timed out for {pdf_path.name}")
            return None

        except Exception as e:
            logger.error(
                f"Error during MinerU conversion of {pdf_path.name}: {e}"
            )
            return None

    def _convert_with_pymupdf(self, pdf_path: Path) -> Optional[str]:
        """Convert PDF to text using PyMuPDF.

        This is a simple fallback that extracts raw text. Not as good as
        MinerU but works when MinerU is unavailable.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text as string, or None if extraction fails
        """
        if not PYMUPDF_AVAILABLE:
            logger.error(
                "PyMuPDF not available. Install with: pip install pymupdf"
            )
            return None

        logger.debug(f"Converting {pdf_path.name} with PyMuPDF")

        try:
            doc = fitz.open(pdf_path)

            # Extract text from all pages
            text_parts = []
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text()
                if text.strip():
                    # Add page separator
                    text_parts.append(f"# Page {page_num}\n\n{text}")

            doc.close()

            if not text_parts:
                logger.warning(f"No text extracted from {pdf_path.name}")
                return None

            markdown = "\n\n".join(text_parts)
            logger.debug(
                f"Extracted {len(markdown)} characters from {pdf_path.name}"
            )

            return markdown

        except Exception as e:
            logger.error(
                f"Error during PyMuPDF conversion of {pdf_path.name}: {e}"
            )
            return None

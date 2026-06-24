# Paper Fetching System

This module provides tools to fetch paper content from arXiv and OpenReview for ICLR papers.

## Features

- **arXiv Integration**: Search papers by title, download LaTeX sources
- **OpenReview Integration**: Download PDFs using paper IDs
- **Smart Fallback**: Tries arXiv first (preferred), falls back to OpenReview
- **Date Validation**: Only uses arXiv sources posted before rebuttal deadlines
- **PDF Processing**: Converts PDFs to markdown using MinerU or PyMuPDF
- **Caching**: Avoids redundant downloads, tracks failures
- **Concurrent Processing**: Fetches multiple papers in parallel
- **Progress Tracking**: Visual progress bar for batch processing

## Installation

Install required dependencies:

```bash
uv pip install arxiv aiohttp python-dateutil
```

Optional: Install MinerU for better PDF conversion quality:

```bash
pip install magic-pdf
```

If MinerU is not installed, the system will automatically fall back to PyMuPDF.

## Usage

### CLI Script

The main way to use this system is through the `scripts/fetch_papers.py` CLI:

#### Process a single dataset

```bash
python scripts/fetch_papers.py \
    --input datasets/normal-review/iclr2024.init.json \
    --output datasets/normal-review-enriched/iclr2024.json \
    --conference "ICLR 2024"
```

#### Process all datasets in a directory

```bash
python scripts/fetch_papers.py \
    --input-dir datasets/normal-review/ \
    --output-dir datasets/normal-review-enriched/
```

The conference will be auto-detected from filenames like `iclr2024.init.json`.

#### Advanced options

```bash
# Force re-download (ignore cache)
python scripts/fetch_papers.py \
    --input datasets/normal-review/iclr2024.init.json \
    --output datasets/normal-review-enriched/iclr2024.json \
    --force

# Only try arXiv (skip OpenReview fallback)
python scripts/fetch_papers.py \
    --input datasets/normal-review/iclr2024.init.json \
    --output datasets/normal-review-enriched/iclr2024.json \
    --arxiv-only

# Use custom arXiv cutoff date (override conference rebuttal deadline)
python scripts/fetch_papers.py \
    --input datasets/normal-review/iclr2024.init.json \
    --output datasets/normal-review-enriched/iclr2024.json \
    --arxiv-cutoff-date 2024-02-01

# Adjust concurrency
python scripts/fetch_papers.py \
    --input datasets/normal-review/iclr2024.init.json \
    --output datasets/normal-review-enriched/iclr2024.json \
    --max-concurrent 10

# Enable verbose logging
python scripts/fetch_papers.py \
    --input datasets/normal-review/iclr2024.init.json \
    --output datasets/normal-review-enriched/iclr2024.json \
    --verbose
```

### Programmatic Usage

You can also use the components directly in Python:

```python
import asyncio
from pathlib import Path
from utils.paper_fetcher import (
    ArxivClient,
    OpenReviewClient,
    MinerUProcessor,
    CacheManager,
    PaperEnricher
)

# Initialize components
arxiv_client = ArxivClient()
openreview_client = OpenReviewClient()
pdf_processor = MinerUProcessor()
cache_manager = CacheManager(Path(".cache/papers"))

# Create enricher
enricher = PaperEnricher(
    arxiv_client,
    openreview_client,
    pdf_processor,
    cache_manager,
    conference="ICLR 2024"
)

# Enrich a single paper
paper_data = {"id": "HE9eUQlAvo", "title": "Example Paper"}
enriched = asyncio.run(enricher.enrich_paper(paper_data))

# Enrich a dataset
asyncio.run(enricher.enrich_dataset(
    input_json_path=Path("input.json"),
    output_json_path=Path("output.json")
))
```

## Output Format

The enriched JSON files contain all original metadata plus new fields:

```json
{
  "id": "HE9eUQlAvo",
  "title": "Example Paper",
  "rating": "3;5;6;6;6",
  ... (all original fields),

  "paper_content": "\\documentclass{article}...",
  "content_source": "arxiv",
  "arxiv_id": "2401.12345v2",
  "arxiv_version": 2,
  "content_format": "latex",
  "content_fetched_at": "2026-01-08T10:30:00Z"
}
```

Note: `arxiv_id` includes the version number (e.g., "2401.12345v2"), and `arxiv_version` provides the numeric version (2).

For failed papers:

```json
{
  "id": "xyz123",
  "title": "Failed Paper",
  ... (original fields),

  "paper_content": null,
  "content_source": null,
  "fetch_error": "Not found on arXiv or OpenReview",
  "content_fetched_at": "2026-01-08T10:35:00Z"
}
```

## How It Works

### Fetching Pipeline

For each paper:

1. **Check cache** - Return cached content if available
2. **Try arXiv**:
   - Search by paper title (fuzzy matching)
   - Validate publication date < rebuttal deadline
   - Download LaTeX source (.tar.gz)
   - Extract and return LaTeX content
3. **Fallback to OpenReview** (if arXiv fails):
   - Download PDF using paper ID
   - Convert to markdown (MinerU or PyMuPDF)
   - Return markdown content
4. **Cache result** - Save for future use

### arXiv Version Selection

arXiv papers can have multiple versions (v1, v2, v3, etc.). The system automatically:

1. **Finds all versions** of a paper on arXiv
2. **Checks each version's upload date** against the cutoff
3. **Selects the latest version** posted before the cutoff that is also **no more than 6 months old** from the cutoff date
4. **Downloads that specific version**

This ensures you get the paper as it was before the rebuttal phase, even if later versions exist. The 6-month threshold ensures that the selected version is recent enough to be relevant for the review process.

### Date Validation

By default, the system uses hardcoded rebuttal start dates:

- ICLR 2024: February 17, 2024
- ICLR 2025: February 13, 2025
- ICLR 2026: February 15, 2026

You can override these with `--arxiv-cutoff-date`:

```bash
python scripts/fetch_papers.py \
    --input datasets/normal-review/iclr2024.init.json \
    --output datasets/normal-review-enriched/iclr2024.json \
    --arxiv-cutoff-date 2024-02-01
```

This will select only arXiv versions posted before February 1, 2024.

### Cache Structure

```
.cache/papers/
├── content/
│   ├── HE9eUQlAvo.tex      # LaTeX from arXiv
│   └── pszewhybU9.md       # Markdown from OpenReview
├── pdfs/
│   └── pszewhybU9.pdf      # Intermediate PDFs
├── latex/
│   └── 2401.12345/         # Extracted LaTeX sources
└── metadata.json           # Cache state
```

## Integration with Existing Code

The enriched format is backward-compatible with `run_review.py`. The `load_paper()` function now checks for `paper_content` first:

```python
# NEW: Check for enriched format first
if "paper_content" in paper_data and paper_data["paper_content"]:
    paper_content = paper_data["paper_content"]
# EXISTING: Backward compatibility
elif "cf_paper" in paper_data and "md" in paper_data["cf_paper"]:
    paper_content = paper_data["cf_paper"]["md"]
...
```

## Performance

Expected timing (with 5 concurrent workers):

- arXiv search: ~2-3 seconds per paper
- LaTeX download: ~5-10 seconds per paper
- OpenReview PDF download: ~3-5 seconds per paper
- MinerU PDF conversion: ~10-30 seconds per paper

For ICLR 2024 (~7,000 papers):
- First run: ~4-6 hours
- Subsequent runs: Instant (cache hits)

## Troubleshooting

### MinerU not available

If you see warnings about MinerU not being available:

```bash
pip install magic-pdf
```

The system will automatically use PyMuPDF as a fallback if MinerU is not installed.

### arXiv rate limiting

If you encounter rate limit errors, the client automatically waits 3 seconds between requests. You can reduce `--max-concurrent` if needed:

```bash
python scripts/fetch_papers.py ... --max-concurrent 3
```

### Timezone errors

If you see datetime comparison errors, ensure your system timezone is set correctly. The code uses UTC for all comparisons.

### Network errors

The system includes retry logic with exponential backoff for network failures. If a paper consistently fails, check:
- Paper ID is correct
- Paper is publicly available on OpenReview
- arXiv search term is accurate

## Components

- **arxiv_client.py**: arXiv API integration
- **openreview_client.py**: OpenReview API integration
- **pdf_processor.py**: PDF to markdown conversion
- **conference_config.py**: Rebuttal deadline configuration
- **cache_manager.py**: Download caching
- **enricher.py**: Main orchestration logic

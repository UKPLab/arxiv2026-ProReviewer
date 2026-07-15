#!/usr/bin/env python3
"""CLI wrapper for ReviewLog — provides persistent state via JSON serialization.

Usage:
    python review_cli.py init paper.pdf              # creates review_<name>/ with paper.md + review_log.json
    python review_cli.py add_claim --text "..." --section "§1" --claim_type empirical
    python review_cli.py finalize                    # export review.md, print review + output path

The agent reads paper.md directly using its Read/Grep tools.
State is stored in an output directory (auto-detected or specified via --dir).
"""

import argparse
import json
import os
import re
import sys

# reviewer_memory.py sits next to this file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reviewer_memory import ReviewLog, CONFERENCE_SCALES, DEFAULT_CONFERENCE, format_score_with_scale, get_valid_scores


STATE_FILE = "review_log.json"
PAPER_FILE = "paper.md"
REVIEW_FILE = "review.md"


def get_output_dir(args) -> str:
    """Get output directory from --dir flag. Required for all commands except init."""
    d = getattr(args, "dir", None)
    if not d:
        print("Error: --dir is required. Pass the output directory created by 'init'.", file=sys.stderr)
        sys.exit(1)
    return os.path.abspath(d)


def load_log(output_dir: str) -> ReviewLog:
    path = os.path.join(output_dir, STATE_FILE)
    if not os.path.exists(path):
        print(f"Error: {STATE_FILE} not found in {output_dir}. Run 'init' first.", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        return ReviewLog.model_validate_json(f.read())


def _strip_rl_fields(data: dict) -> dict:
    """Remove RL-training-specific fields from serialized ReviewLog.

    These fields (step counters, section_visits, search_history, etc.) are used
    by the RL training pipeline but are always null/empty in CLI skill use.
    """
    # Top-level ReviewLog fields
    for key in ("section_visits", "search_history", "current_iteration"):
        data.pop(key, None)

    # Per-entry step fields
    for claim in data.get("claims", []):
        claim.pop("step", None)
        claim.pop("status_updated_step", None)
    for question in data.get("questions", []):
        question.pop("step", None)
    for note in data.get("notes", []):
        note.pop("step", None)

    # ReviewOutline fields
    outline = data.get("review_outline", {})
    outline.pop("summary_step", None)
    outline.pop("overall_score_step", None)
    for section in ("strengths", "weaknesses", "questions"):
        for item in outline.get(section, []):
            item.pop("step", None)
            item.pop("human_point", None)

    return data


def save_log(log: ReviewLog, output_dir: str):
    path = os.path.join(output_dir, STATE_FILE)
    data = log.model_dump()
    data = _strip_rl_fields(data)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def split_list(value: str) -> list:
    """Split comma-separated string into list, stripping whitespace."""
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def convert_pdf_to_text(pdf_path: str) -> str:
    """Convert PDF to markdown text.

    Tries pymupdf4llm (best quality), falls back to pymupdf/fitz (basic).
    """
    try:
        import pymupdf4llm
        return pymupdf4llm.to_markdown(pdf_path)
    except ImportError:
        pass

    try:
        import pymupdf
        doc = pymupdf.open(pdf_path)
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n\n".join(pages)
    except ImportError:
        pass

    try:
        import fitz
        doc = fitz.open(pdf_path)
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n\n".join(pages)
    except ImportError:
        pass

    print(
        "Error: No PDF library found. Install one:\n"
        "  pip install pymupdf4llm   # recommended — produces markdown\n"
        "  pip install pymupdf       # basic text extraction",
        file=sys.stderr,
    )
    sys.exit(1)


def resolve_tex_inputs(tex_path: str) -> str:
    r"""Recursively resolve \input{...} and \include{...} in a .tex file.

    Replaces each \input{path} or \include{path} with the contents of the
    referenced file (adding .tex extension if missing). Resolves paths
    relative to the directory of the file containing the directive.
    Handles nested \input statements up to 10 levels deep.

    Args:
        tex_path: Absolute or relative path to the root .tex file.

    Returns:
        The fully expanded LaTeX source as a single string.
    """
    tex_path = os.path.abspath(tex_path)

    # Track files to prevent infinite recursion
    _seen = set()

    def _resolve(filepath: str, depth: int = 0) -> str:
        if depth > 10:
            return f"% [review_cli] max include depth exceeded: {filepath}\n"

        filepath = os.path.abspath(filepath)
        if filepath in _seen:
            return f"% [review_cli] circular include skipped: {filepath}\n"
        _seen.add(filepath)

        if not os.path.isfile(filepath):
            return f"% [review_cli] file not found: {filepath}\n"

        with open(filepath, encoding="utf-8", errors="replace") as f:
            text = f.read()

        base_dir = os.path.dirname(filepath)

        # Match \input{...} and \include{...} (ignoring commented-out lines)
        pattern = re.compile(r'^(?!%)\s*\\(?:input|include)\{([^}]+)\}', re.MULTILINE)

        def _replace(match):
            ref = match.group(1).strip()
            # Add .tex extension if not present
            if not os.path.splitext(ref)[1]:
                ref += ".tex"
            child_path = os.path.join(base_dir, ref)
            return _resolve(child_path, depth + 1)

        return pattern.sub(_replace, text)

    return _resolve(tex_path)


def make_output_dir_name(paper_path: str) -> str:
    """Derive output directory name from paper filename.

    e.g. 'my_paper.pdf' -> 'review_my_paper'
    """
    base = os.path.splitext(os.path.basename(paper_path))[0]
    # Clean up: replace spaces/special chars with underscores
    clean = re.sub(r'[^\w\-]', '_', base).strip('_')
    # Truncate long names
    if len(clean) > 60:
        clean = clean[:60].rstrip('_')
    return f"review_{clean}"


def format_review(log: ReviewLog) -> str:
    """Format the review outline as a clean markdown review."""
    outline = log.review_outline
    parts = []

    if outline.summary:
        parts.append("## Summary\n")
        parts.append(outline.summary)
        parts.append("")

    if outline.strengths:
        parts.append("## Strengths\n")
        for item in outline.strengths:
            parts.append(item.text)
        parts.append("")

    if outline.weaknesses:
        parts.append("## Weaknesses\n")
        for item in outline.weaknesses:
            parts.append(item.text)
        parts.append("")

    if outline.questions:
        parts.append("## Questions for Authors\n")
        for item in outline.questions:
            parts.append(item.text)
        parts.append("")

    if outline.overall_score is not None:
        parts.append("## Overall Score\n")
        parts.append(format_score_with_scale(outline.overall_score, log.conference))
        parts.append("")

    return "\n".join(parts)


# --- Commands ---

def cmd_init(args):
    input_path = args.paper
    if not os.path.exists(input_path):
        print(f"Error: path not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # If a directory is given, look for main.tex inside it
    if os.path.isdir(input_path):
        candidates = ["main.tex", "paper.tex", "manuscript.tex"]
        found = None
        for c in candidates:
            p = os.path.join(input_path, c)
            if os.path.isfile(p):
                found = p
                break
        if found is None:
            # Fall back to any .tex file
            tex_files = [f for f in os.listdir(input_path) if f.endswith(".tex")]
            if len(tex_files) == 1:
                found = os.path.join(input_path, tex_files[0])
            elif len(tex_files) > 1:
                print(
                    f"Error: directory contains multiple .tex files: {tex_files}\n"
                    f"  Pass the root .tex file explicitly, e.g.:\n"
                    f"  python review_cli.py init {input_path}/main.tex",
                    file=sys.stderr,
                )
                sys.exit(1)
            else:
                print(f"Error: no .tex file found in {input_path}", file=sys.stderr)
                sys.exit(1)
        input_path = found
        print(f"Found root tex file: {input_path}")

    # Create output directory
    explicit_dir = getattr(args, "dir", None)
    if explicit_dir:
        output_dir = os.path.abspath(explicit_dir)
    else:
        output_dir = os.path.join(os.getcwd(), make_output_dir_name(input_path))

    state_path = os.path.join(output_dir, STATE_FILE)
    if os.path.exists(state_path) and not args.force:
        print(f"{state_path} already exists. Use --force to overwrite.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    ext = os.path.splitext(input_path)[1].lower()

    if ext == ".pdf":
        print(f"Converting {input_path} to markdown...")
        text = convert_pdf_to_text(input_path)
    elif ext == ".tex":
        print(f"Resolving \\input/\\include in {input_path}...")
        text = resolve_tex_inputs(input_path)
    elif ext in (".md", ".txt", ".markdown"):
        with open(input_path, encoding="utf-8") as f:
            text = f.read()
    else:
        print(f"Error: unsupported format '{ext}'. Use .pdf, .tex, or .md", file=sys.stderr)
        sys.exit(1)

    # Save paper text
    paper_path = os.path.join(output_dir, PAPER_FILE)
    with open(paper_path, "w", encoding="utf-8") as f:
        f.write(text)

    # Create ReviewLog with conference setting
    conf = getattr(args, "conference", DEFAULT_CONFERENCE).lower()
    if conf not in CONFERENCE_SCALES:
        print(f"Error: unknown conference '{conf}'. Supported: {', '.join(CONFERENCE_SCALES.keys())}", file=sys.stderr)
        sys.exit(1)

    log = ReviewLog(conference=conf)
    log.review_outline.conference = conf
    save_log(log, output_dir)

    scale = CONFERENCE_SCALES[conf]
    valid = scale["scores"]

    print(f"Output directory → {output_dir}")
    print(f"Paper saved → {os.path.join(output_dir, PAPER_FILE)} ({len(text)} chars)")
    print(f"ReviewLog initialized → {os.path.join(output_dir, STATE_FILE)}")
    print(f"Conference: {scale['name']}  (score scale: {valid})")
    print(f"\nRead {os.path.join(output_dir, PAPER_FILE)} with your Read tool to begin reviewing.")


def cmd_add_claim(args):
    output_dir = get_output_dir(args)
    log = load_log(output_dir)
    issues = split_list(args.issues) if args.issues else None
    claim_id = log.add_claim(
        text=args.text,
        section=args.section,
        claim_type=args.claim_type,
        issues=issues,
    )
    if claim_id is None:
        print("Skipped: duplicate claim.")
    else:
        save_log(log, output_dir)
        claim = log.get_claim(claim_id)
        print(f"Added {claim_id}: {claim.to_prompt_str()}")


def cmd_add_question(args):
    output_dir = get_output_dir(args)
    log = load_log(output_dir)
    related = split_list(args.related_claims) if args.related_claims else None
    question_id = log.add_question(
        question=args.text,
        source_section=args.section,
        question_type=args.type or "clarification",
        related_claims=related,
    )
    if question_id is None:
        print("Skipped: duplicate question.")
    else:
        save_log(log, output_dir)
        q = log.get_question(question_id)
        print(f"Added {question_id}: {q.to_prompt_str()}")


def cmd_add_note(args):
    output_dir = get_output_dir(args)
    log = load_log(output_dir)
    tag = split_list(args.tag) if args.tag else None
    note_id = log.add_note(
        text=args.text,
        section=args.section,
        tag=tag,
    )
    if note_id is None:
        print("Skipped: duplicate note.")
    else:
        save_log(log, output_dir)
        note = log.get_note(note_id)
        print(f"Added {note_id}: {note.to_prompt_str()}")


def cmd_update_claim(args):
    output_dir = get_output_dir(args)
    log = load_log(output_dir)
    cross_refs = split_list(args.cross_refs) if args.cross_refs else None
    ok = log.update_claim_status(
        claim_id=args.id,
        status=args.status,
        reason=args.reason,
        cross_refs=cross_refs,
    )
    if not ok:
        print(f"Error: claim {args.id} not found.", file=sys.stderr)
        sys.exit(1)
    save_log(log, output_dir)
    claim = log.get_claim(args.id)
    print(f"Updated {args.id}: {claim.to_prompt_str(detailed=True)}")


def cmd_resolve_question(args):
    output_dir = get_output_dir(args)
    log = load_log(output_dir)
    sections = split_list(args.sections) if args.sections else []
    ok = log.resolve_question(
        question_id=args.id,
        answer=args.answer,
        answer_sections=sections,
        status=args.status or "resolved",
    )
    if not ok:
        print(f"Error: question {args.id} not found.", file=sys.stderr)
        sys.exit(1)
    save_log(log, output_dir)
    q = log.get_question(args.id)
    print(f"Resolved {args.id}: {q.to_prompt_str(detailed=True)}")


def cmd_outline(args):
    output_dir = get_output_dir(args)
    log = load_log(output_dir)
    section = args.section

    if section == "overall_score":
        raw = float(args.content)
        score = int(raw) if raw == int(raw) else raw
        try:
            log.update_outline(section="overall_score", content=score)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        save_log(log, output_dir)
        print(f"Set overall_score: {format_score_with_scale(score, log.conference)}")
        return

    if section == "summary":
        log.update_outline(section="summary", content=args.content)
        save_log(log, output_dir)
        print(f"Set summary: {args.content[:80]}...")
        return

    # strengths, weaknesses, questions — require evidence tags
    claims = split_list(args.claims) if args.claims else []
    questions = split_list(args.questions) if args.questions else []
    notes = split_list(args.notes) if args.notes else []

    try:
        result = log.update_outline(
            section=section,
            content=args.content,
            related_claims=claims,
            related_questions=questions,
            related_notes=notes,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if result == "duplicate_skipped":
        print("Skipped: duplicate outline item.")
    else:
        save_log(log, output_dir)
        items = getattr(log.review_outline, section)
        latest = items[-1]
        print(f"Added {section}: {latest.to_prompt_str()}")


def cmd_set_conference(args):
    output_dir = get_output_dir(args)
    log = load_log(output_dir)
    conf = args.conference.lower()
    if conf not in CONFERENCE_SCALES:
        print(f"Error: unknown conference '{conf}'. Supported: {', '.join(CONFERENCE_SCALES.keys())}", file=sys.stderr)
        sys.exit(1)

    old_score = log.review_outline.overall_score
    log.conference = conf
    log.review_outline.conference = conf

    # If a score was already set, check it against the new scale
    if old_score is not None:
        valid = CONFERENCE_SCALES[conf]["scores"]
        if old_score not in valid:
            print(f"Warning: existing score {old_score} is not valid for {CONFERENCE_SCALES[conf]['name']}.")
            print(f"  Valid scores: {valid}")
            print(f"  Score has been cleared. Use 'outline --section overall_score' to set a new one.")
            log.review_outline.overall_score = None

    save_log(log, output_dir)
    scale = CONFERENCE_SCALES[conf]
    print(f"Conference set to: {scale['name']}")
    print(f"Score scale: {scale['scores']}")
    for score_val, label in sorted(scale["labels"].items(), reverse=True):
        print(f"  {score_val}: {label}")


def cmd_show(args):
    output_dir = get_output_dir(args)
    log = load_log(output_dir)
    print(log.build_context(detailed=args.detailed))


def cmd_export(args):
    output_dir = get_output_dir(args)
    log = load_log(output_dir)
    review_text = format_review(log)
    out_path = os.path.join(output_dir, args.output or REVIEW_FILE)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(review_text)
    print(f"Review exported → {out_path}")


def cmd_finalize(args):
    """Export review.md, print the review content, and show the output folder."""
    output_dir = get_output_dir(args)
    log = load_log(output_dir)

    # Export review.md
    review_text = format_review(log)
    review_path = os.path.join(output_dir, REVIEW_FILE)
    with open(review_path, "w", encoding="utf-8") as f:
        f.write(review_text)

    # Print the review content
    print(review_text)

    # Print output folder summary
    print("---")
    print(f"Output folder: {output_dir}")
    print(f"  {PAPER_FILE}         — converted paper")
    print(f"  {STATE_FILE}   — investigation log (claims, questions, notes)")
    print(f"  {REVIEW_FILE}        — final review")


def main():
    parser = argparse.ArgumentParser(
        description="ReviewLog CLI — persistent evidence-based review state"
    )
    parser.add_argument("--dir", help="Output directory (auto-detected if omitted)")
    sub = parser.add_subparsers(dest="command", required=True)

    # init
    p = sub.add_parser("init", help="Convert paper and create output directory")
    p.add_argument("paper", help="Path to paper file (.pdf, .tex, .md) or directory containing main.tex")
    p.add_argument("--force", action="store_true", help="Overwrite existing state")
    p.add_argument("--conference", default=DEFAULT_CONFERENCE,
                    choices=list(CONFERENCE_SCALES.keys()),
                    help=f"Conference rating scale (default: {DEFAULT_CONFERENCE})")

    # add_claim
    p = sub.add_parser("add_claim", help="Add a claim")
    p.add_argument("--text", required=True, help="The author's assertion")
    p.add_argument("--section", required=True, help="Paper section (e.g. §Abstract)")
    p.add_argument("--claim_type", required=True, help="Type: empirical, novelty, scope, etc.")
    p.add_argument("--issues", help="Comma-separated issues to flag")

    # add_question
    p = sub.add_parser("add_question", help="Add a question")
    p.add_argument("--text", required=True, help="The question to investigate")
    p.add_argument("--section", required=True, help="Section where question arose")
    p.add_argument("--type", help="Question type (default: clarification)")
    p.add_argument("--related_claims", help="Comma-separated related claim IDs")

    # add_note
    p = sub.add_parser("add_note", help="Add a note")
    p.add_argument("--text", required=True, help="Your observation")
    p.add_argument("--section", required=True, help="Relevant section")
    p.add_argument("--tag", help="Comma-separated tags")

    # update_claim
    p = sub.add_parser("update_claim", help="Update a claim's status")
    p.add_argument("--id", required=True, help="Claim ID (e.g. C1)")
    p.add_argument("--status", required=True, choices=["supported", "weak", "invalid"])
    p.add_argument("--reason", required=True, help="Reasoning with specific evidence")
    p.add_argument("--cross_refs", help="Comma-separated sections providing evidence")

    # resolve_question
    p = sub.add_parser("resolve_question", help="Resolve a question")
    p.add_argument("--id", required=True, help="Question ID (e.g. Q1)")
    p.add_argument("--answer", required=True, help="What you found")
    p.add_argument("--sections", help="Comma-separated sections where answer was found")
    p.add_argument("--status", choices=["resolved", "partially_answered"], help="Default: resolved")

    # outline
    p = sub.add_parser("outline", help="Add to the review outline")
    p.add_argument("--section", required=True,
                    choices=["summary", "strengths", "weaknesses", "questions", "overall_score"])
    p.add_argument("--content", required=True, help="The text content (or score for overall_score)")
    p.add_argument("--claims", help="Comma-separated claim IDs (required for strengths/weaknesses/questions)")
    p.add_argument("--questions", help="Comma-separated question IDs")
    p.add_argument("--notes", help="Comma-separated note IDs")

    # set_conference
    p = sub.add_parser("set_conference", help="Change the conference rating scale")
    p.add_argument("conference", choices=list(CONFERENCE_SCALES.keys()),
                    help="Conference rating scale to use")

    # show
    p = sub.add_parser("show", help="Print current log state")
    p.add_argument("--detailed", action="store_true", help="Show full details")

    # export
    p = sub.add_parser("export", help="Export the review outline as a clean markdown file")
    p.add_argument("--output", help=f"Output file path (default: {REVIEW_FILE})")

    # finalize
    p = sub.add_parser("finalize", help="Export review, print content, and show output folder")

    args = parser.parse_args()

    commands = {
        "init": cmd_init,
        "add_claim": cmd_add_claim,
        "add_question": cmd_add_question,
        "add_note": cmd_add_note,
        "update_claim": cmd_update_claim,
        "resolve_question": cmd_resolve_question,
        "outline": cmd_outline,
        "set_conference": cmd_set_conference,
        "show": cmd_show,
        "export": cmd_export,
        "finalize": cmd_finalize,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()

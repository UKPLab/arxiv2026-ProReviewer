"""Build the interactive ProReviewer demo from recorded review sessions.

Reads one or more session directories written by the ProReviewer skill and emits
a single self-contained HTML file: all CSS, JavaScript and data inlined, no
network access, openable straight from the filesystem.

Sessions are named explicitly so nothing is ever published by accident.

Usage:
    python build_demo.py review_0A4Uf88pog
    python build_demo.py review_0A4Uf88pog review_0ACUx9pMWJ --out demo/index.html
    python build_demo.py review_0A4Uf88pog --on-mismatch skip_trajectory

A session directory is expected to contain review_log.json (required) and, where
available, paper.md, review.md and trajectory.jsonl. Only sessions with an
archived transcript can show the step-by-step replay; the rest still contribute
their review log, evidence graph and final review.

The build fails if a session's recorded run does not replay to its own
review_log.json, because animating a run the agent did not perform would be
worse than showing no run at all. Use --on-mismatch to downgrade that.
"""

import argparse
import logging
import sys

from demo.builder import DemoBuildError, build_demo

logger = logging.getLogger("build_demo")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the ProReviewer demo page from recorded sessions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("sessions", nargs="+",
                        help="Session directories, e.g. review_0A4Uf88pog")
    parser.add_argument("--out", default="demo/index.html",
                        help="Output HTML file (default: demo/index.html)")
    parser.add_argument("--on-mismatch", default="raise",
                        choices=["raise", "skip_trajectory", "warn"],
                        help="What to do when a replay does not match its review log "
                             "(default: raise)")
    parser.add_argument("--title", default=None, help="Override the page title")
    parser.add_argument("--quiet", action="store_true", help="Only report problems")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    kwargs = {"out": args.out, "on_replay_mismatch": args.on_mismatch}
    if args.title:
        kwargs["title"] = args.title

    try:
        path = build_demo(args.sessions, **kwargs)
    except DemoBuildError as exc:
        logger.error("%s", exc)
        return 1

    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

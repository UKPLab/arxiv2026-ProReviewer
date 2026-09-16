"""Build a self-contained interactive demo from recorded ProReviewer sessions.

    from demo import build_demo
    build_demo(["review_0A4Uf88pog"], out="demo/index.html")

`build_demo` is resolved lazily so that importing a single submodule (for
example demo.replay in the test suite) does not pull in the whole builder.
"""

__all__ = ["build_demo"]


def __getattr__(name):
    if name == "build_demo":
        from .builder import build_demo
        return build_demo
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

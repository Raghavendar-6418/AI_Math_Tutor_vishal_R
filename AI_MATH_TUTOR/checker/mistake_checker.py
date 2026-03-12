"""Simple mistake detection heuristics for algebra problems.

This module inspects the original equation and the sympy-processed forms
to suggest common mistakes like distribution errors, sign errors, and
transposition issues.
"""
from sympy import simplify, Eq, expand
import logging

logger = logging.getLogger(__name__)


def detect_mistakes(latex_str: str, sym_obj):
    """Return a list of detected mistakes (strings)."""
    mistakes = []
    try:
        if isinstance(sym_obj, Eq):
            L = sym_obj.lhs
            R = sym_obj.rhs
            # Check distribution: if parentheses on L or R, compare expansion
            if any(str(x).find("(") >= 0 for x in [L, R]):
                try:
                    if expand(L) != L and simplify(expand(L) - L) != 0:
                        mistakes.append("Possible distribution mistake on left side (check expansion)")
                    if expand(R) != R and simplify(expand(R) - R) != 0:
                        mistakes.append("Possible distribution mistake on right side (check expansion)")
                except Exception:
                    pass

            # Sign/transposition: move all terms left and see if simplified matches
            try:
                combined = simplify(L - R)
                # If combined is not simplified properly to expected polynomial form, warn
                s = str(combined)
                if s.count("-") > s.count("+") + 2:
                    mistakes.append("Check sign handling during transposition")
            except Exception:
                pass

        else:
            # For expressions, check trivial arithmetic contradictions
            try:
                s = simplify(sym_obj)
                _ = str(s)
            except Exception:
                mistakes.append("Expression parsing may be incorrect; check OCR/LaTeX conversion")

    except Exception as e:
        logger.exception("Mistake detection failed: %s", e)
    return mistakes
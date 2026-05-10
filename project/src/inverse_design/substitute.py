"""Element substitution utilities for ABO3 perovskite inverse design.

`substitute_cif`: read a base ABO3 CIF, swap A/B cations to a new pair, write a new CIF.
`substitute_text`: word-boundary regex substitution of A/B element symbols inside a
  Robocrystallographer description string. NEVER touches O, since 'O' is the anion
  shared by every ABO3 sample and substring matching on 'O' would corrupt words like
  "Octahedra" or "tetrahedral".
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

from pymatgen.core import Element, Structure


def substitute_cif(
    base_cif_path: str,
    base_a: str,
    base_b: str,
    new_a: str,
    new_b: str,
    out_path: str,
) -> str:
    """Swap A and B cations in a base ABO3 CIF and write a new CIF.

    Uses ``pymatgen.Structure.replace_species`` for an in-place atomic swap. Does NOT
    relax the resulting structure — geometry is the parent template's geometry with
    only the chemical identity of A/B atoms changed. This is fine as a surrogate input
    to the trained forward model (which encodes structure via 8 Å neighbor graph and
    text via Robocrystallographer); it is NOT a DFT-quality starting point.

    The mapping is applied as a single ``replace_species`` call so that swaps like
    {A: B, B: A} (rare but legal) don't corrupt themselves through sequential edits.

    Args:
        base_cif_path: Path to a CIF file (any pymatgen-readable structure).
        base_a, base_b: Element symbols currently at A/B sites.
        new_a, new_b: Replacement element symbols.
        out_path: Where to write the new CIF.

    Returns:
        Path to the written CIF (str equal to ``out_path``).

    Raises:
        ValueError: if any element symbol is not a valid pymatgen Element.
        Exception: any pymatgen I/O error from reading/writing CIF.
    """
    structure = Structure.from_file(base_cif_path)

    mapping: Dict[Element, Element] = {}
    if base_a != new_a:
        mapping[Element(base_a)] = Element(new_a)
    if base_b != new_b:
        mapping[Element(base_b)] = Element(new_b)

    if mapping:
        structure.replace_species(mapping)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    structure.to(filename=out_path, fmt="cif")
    return out_path


def substitute_text(
    base_text: str,
    base_a: str,
    base_b: str,
    new_a: str,
    new_b: str,
) -> str:
    """Word-boundary swap of A/B element symbols in a Robocrystallographer string.

    Uses a single regex pass with alternation, so a swap like {La: Sr, Fe: La}
    won't double-substitute (the new La introduced by the Fe→La swap is not
    re-processed). The pattern matches the element symbol surrounded by ``\\b``
    word boundaries, which preserves ``Element(N)`` tokens like ``Li(1)`` (the
    closing of ``Li`` before ``(`` is a word boundary) but does not match
    ``Li`` inside ``Lithium`` (no word boundary inside the word).

    Refuses O substitution: O appears in every ABO3 sample and as a substring
    of words like "Octahedra"/"tetrahedral". The ``\\b`` regex already protects
    most of those, but we still bail out to avoid silent confusion in callers.

    Args:
        base_text: Robocrystallographer description.
        base_a, base_b: Element symbols currently at A/B.
        new_a, new_b: Replacement symbols.

    Returns:
        Substituted string (or the input, if base==new on both sites).

    Raises:
        ValueError: if any of the four symbols is "O".
    """
    if "O" in {base_a, base_b, new_a, new_b}:
        raise ValueError(
            "Substituting 'O' is not allowed: oxygen is the conserved ABO3 anion."
        )

    mapping: Dict[str, str] = {}
    if base_a != new_a:
        mapping[base_a] = new_a
    if base_b != new_b:
        mapping[base_b] = new_b
    if not mapping:
        return base_text

    pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in mapping) + r")\b")
    return pattern.sub(lambda m: mapping[m.group(1)], base_text)

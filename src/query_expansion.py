"""
query_expansion.py — Domain-Specific Query Expansion & Medical Acronym Normalization.

Key Features:
1. Medical & Hardware Acronym Normalization: Expands common medical device acronyms
   (e.g., PSU, OVP, HV, PCB, CCU, NIBP, SDI, DSP) to ensure high lexical recall.
2. Error Code Variant Generation: Detects patterns like 'E205', 'E-205', 'Error 205',
   or 'ERR_205' and normalizes them into canonical search formats.
3. Enhanced Keyword Signal: Enriches technical search terms before passing to BM25 and Dense search.
"""

from __future__ import annotations

import re
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Medical & Technical Acronym Dictionary
# ---------------------------------------------------------------------------
ACRONYM_MAP: dict[str, str] = {
    # Power & Electrical
    "psu": "power supply unit",
    "ovp": "overvoltage protection",
    "uvp": "undervoltage protection",
    "ocp": "overcurrent protection",
    "otp": "overtemperature protection",
    "hv": "high voltage",
    "lv": "low voltage",
    "pcb": "printed circuit board",
    "vac": "volts alternating current",
    "vdc": "volts direct current",
    "ac": "alternating current",
    "dc": "direct current",
    "gnd": "ground",
    
    # Video, Optics & Processing
    "ccu": "camera control unit",
    "sdi": "serial digital interface",
    "dsp": "digital signal processor",
    "fpga": "field programmable gate array",
    "lcd": "liquid crystal display",
    "led": "light emitting diode",
    "oled": "organic light emitting diode",
    "hdmi": "high definition multimedia interface",
    "fov": "field of view",
    "dvi": "digital visual interface",
    
    # Medical & Diagnostic Systems
    "nibp": "non-invasive blood pressure",
    "spo2": "oxygen saturation pulse oximetry",
    "ecg": "electrocardiogram",
    "eeg": "electroencephalogram",
    "co2": "carbon dioxide",
    "ins": "insufflator insufflation",
    "hf": "high frequency electrosurgery",
    "rf": "radio frequency",
    "ivd": "in vitro diagnostic",
    
    # Maintenance & Standards
    "emc": "electromagnetic compatibility",
    "esd": "electrostatic discharge",
    "pm": "preventive maintenance",
    "cal": "calibration",
    "p/n": "part number",
    "pn": "part number",
    "s/n": "serial number",
    "sn": "serial number",
    "rev": "revision",
    "o-ring": "sealing ring gasket",
}


class ExpandedQuery(NamedTuple):
    original_query: str
    expanded_query: str
    expanded_terms: list[str]
    detected_error_codes: list[str]


def extract_and_expand_error_codes(query: str) -> list[str]:
    """Find error code mentions in query and produce all common format variants.

    Examples:
        'E205' -> ['E-205', 'E205', 'Error 205', 'ERR-205', 'ERR 205']
        'Error 101' -> ['E-101', 'E101', 'Error 101', 'ERR-101']
        '0x8004' -> ['0x8004', '8004']
    """
    variants: set[str] = set()

    # Pattern: E-205, E205, F-102, F102, Err-309, ERR_309, ERR 309, Error 205
    pattern_letter_num = re.finditer(
        r"\b(?:error|err|code|fault|e|f|a|b|c|d)[\s\-_:]*([0-9]{2,5})\b",
        query,
        re.IGNORECASE,
    )
    for match in pattern_letter_num:
        num = match.group(1)
        variants.add(f"E-{num}")
        variants.add(f"E{num}")
        variants.add(f"ERR-{num}")
        variants.add(f"ERR_{num}")
        variants.add(f"Error {num}")
        variants.add(f"Code {num}")
        variants.add(f"Fault {num}")

    # Hex error pattern: 0x8004, 0x1A2F
    pattern_hex = re.finditer(r"\b(0x[0-9a-fA-F]{3,8})\b", query)
    for match in pattern_hex:
        hex_code = match.group(1)
        variants.add(hex_code)
        variants.add(hex_code.lower())
        variants.add(hex_code.upper())

    return sorted(list(variants))


def expand_query(query: str) -> ExpandedQuery:
    """Enrich a technical query with medical acronym expansions and error code variants.

    Parameters
    ----------
    query : str
        User's raw query (e.g. "What causes E205 on the PSU?")

    Returns
    -------
    ExpandedQuery
        Contains the original query, enriched search query string, and list of added terms.
    """
    if not query or not query.strip():
        return ExpandedQuery(
            original_query=query,
            expanded_query=query,
            expanded_terms=[],
            detected_error_codes=[],
        )

    added_terms: list[str] = []
    
    # 1. Expand error code variations
    error_variants = extract_and_expand_error_codes(query)
    for v in error_variants:
        if v.lower() not in query.lower():
            added_terms.append(v)

    # 2. Expand medical/technical acronyms
    words = re.findall(r"\b[a-zA-Z0-9/_-]+\b", query)
    for word in words:
        clean_word = word.lower().strip()
        if clean_word in ACRONYM_MAP:
            expansion = ACRONYM_MAP[clean_word]
            # Avoid redundant duplicate additions
            if expansion not in added_terms and expansion.lower() not in query.lower():
                added_terms.append(expansion)

    # 3. Construct the expanded query string
    if added_terms:
        # Append expanded synonyms while keeping original query in primary focus
        expansion_suffix = " " + " ".join(added_terms)
        enriched_query = f"{query}{expansion_suffix}"
    else:
        enriched_query = query

    return ExpandedQuery(
        original_query=query,
        expanded_query=enriched_query,
        expanded_terms=added_terms,
        detected_error_codes=error_variants,
    )

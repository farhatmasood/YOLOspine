"""
Pfirrmann disc degeneration grading utilities.

Pfirrmann Classification (Grades 1-5):
    1 - Normal: homogeneous, bright white, normal height
    2 - Inhomogeneous with horizontal band, white, normal height
    3 - Inhomogeneous, grey, normal-to-slightly-decreased height
    4 - Inhomogeneous, grey-to-black, moderately decreased height
    5 - Inhomogeneous, black, collapsed disc space
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


def load_metadata(metadata_path: str) -> dict:
    """Load Pfirrmann metadata from JSON file."""
    with open(metadata_path, "r") as f:
        return json.load(f)


def get_pfirrmann_grade(metadata: dict, filename: str,
                        disc_level: Optional[str] = None):
    """
    Retrieve Pfirrmann grade(s) for an image file.

    Args:
        metadata: Loaded Pfirrmann JSON content.
        filename: Image filename (e.g. ``"T1_0001_S8.png"``).
        disc_level: Optional specific level (``"D3"``, ``"D4"``, ``"D5"``).

    Returns:
        Dict of grades ``{"D5": 2, "D4": 5, "D3": 2}`` or a single int
        if ``disc_level`` is specified. ``None`` if not found.
    """
    stem = filename.rsplit(".", 1)[0]
    file_grades = metadata.get("file_to_grades", {})
    if stem not in file_grades:
        return None
    grades = file_grades[stem]
    if disc_level:
        return grades.get(disc_level)
    return grades


def is_degenerated(grade: int, threshold: int = 3) -> bool:
    """
    Whether a disc is clinically degenerated.

    Clinical interpretation:
        Grades 1-2: normal to early changes.
        Grades 3-5: moderate to severe degeneration.

    Args:
        grade: Pfirrmann grade (1-5).
        threshold: Grade cutoff (default 3).
    """
    return grade >= threshold


def get_grade_statistics(metadata: dict) -> dict:
    """Return grade distribution counts from metadata."""
    return metadata.get("grade_distribution", {})


def get_patients_by_grade(metadata: dict, disc_level: str,
                          grade: int) -> List[str]:
    """List image filenames with a specific grade at a disc level."""
    file_grades = metadata.get("file_to_grades", {})
    return [
        f"{fn}.png"
        for fn, grades in file_grades.items()
        if grades.get(disc_level) == grade
    ]


def get_degeneration_severity_score(grades: Dict[str, int]) -> float:
    """
    Weighted severity score across disc levels.

    L5-S1 (D5) receives highest weight as the most commonly affected level.

    Args:
        grades: ``{"D5": g5, "D4": g4, "D3": g3}`` grade dict.

    Returns:
        Float in [1.0, 5.0].
    """
    weights = {"D5": 0.4, "D4": 0.35, "D3": 0.25}
    score = sum(grades.get(d, 1) * w for d, w in weights.items())
    return round(score, 2)

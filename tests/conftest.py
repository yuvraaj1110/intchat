"""Shared pytest fixtures.

`sample_docs` is a tiny corpus that mirrors the real normalized schema. It is
deliberately small so embedding-backed tests stay fast.
"""

import pytest


@pytest.fixture
def sample_docs():
    """Six docs: includes one A/B overlap pair and one exact-term doc."""
    return [
        {
            "id": "dataset_A__opt_guidance__0",
            "text": "OPT allows F-1 students to work in their field for up to 12 months.",
            "category": "Practical Training and Employment",
            "topic": "Optional Practical Training (OPT)",
            "type": "paragraph",
            "source": "dataset_A",
            "metadata": {},
        },
        {
            "id": "dataset_B__opt_optional_practical_training__summary",
            "text": "OPT allows F-1 students to engage in temporary employment "
                    "related to their major. File Form I-765 to apply.",
            "category": "OPT",
            "topic": "Optional Practical Training (OPT)",
            "type": "summary",
            "source": "dataset_B",
            "metadata": {"required_forms": ["I-20", "I-765"]},
        },
        {
            "id": "dataset_A__sevis_overview__0",
            "text": "SEVIS is the system DHS uses to track F-1 and M-1 students.",
            "category": "SEVIS System Overview",
            "topic": "SEVIS Purpose and Management",
            "type": "paragraph",
            "source": "dataset_A",
            "metadata": {},
        },
        {
            "id": "mentorstyle__p1_meaning_001",
            "text": "Q: What does it mean to be an international student?\n"
                    "A: You are a guest student here to study on a visa.",
            "category": "What Being an International Student Really Means",
            "topic": "What does it mean to be an international student?",
            "type": "qa",
            "source": "mentorstyle",
            "metadata": {},
        },
        {
            "id": "dataset_B__ssn__summary",
            "text": "F-1 students with authorized employment may apply for an SSN.",
            "category": "SSN",
            "topic": "Social Security Number (SSN)",
            "type": "summary",
            "source": "dataset_B",
            "metadata": {},
        },
        {
            "id": "dataset_A__travel_entry__0",
            "text": "Enter the U.S. no earlier than 30 days before your program start.",
            "category": "Travel and Entry",
            "topic": "Entering the United States",
            "type": "paragraph",
            "source": "dataset_A",
            "metadata": {},
        },
    ]

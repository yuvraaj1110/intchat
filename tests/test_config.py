# tests/test_config.py
from app import config


def test_config_has_expected_constants():
    assert config.EMBEDDING_MODEL == "all-MiniLM-L6-v2"
    # Preferred model is first in the preference list (self-healing fallback)
    assert config.GROQ_MODEL_PREFERENCES[0] == "llama-3.3-70b-versatile"
    assert len(config.GROQ_MODEL_PREFERENCES) >= 2
    assert config.TOP_K == 5
    assert config.SEMANTIC_K == 8
    assert config.MEMORY_WINDOW == 10
    assert config.COLLECTION_NAME == "intchat_knowledge"
    # Paths are pathlib.Path objects
    assert config.NORMALIZED_DATASET.name == "normalized_dataset.json"
    assert config.CHROMA_DIR.name == "chroma_db"


def test_dedup_exclusions_are_the_eight_overlap_docs():
    assert config.DEDUP_EXCLUDE_IDS == {
        "dataset_A__cpt_guidance__0",
        "dataset_A__cpt_guidance__1",
        "dataset_A__opt_guidance__0",
        "dataset_A__opt_guidance__1",
        "dataset_A__stem_opt__0",
        "dataset_A__stem_opt__1",
        "dataset_A__ssn_guidance__0",
        "dataset_A__ssn_guidance__1",
    }

from evaluation.run import term_coverage


def test_term_coverage_ignores_punctuation_and_case():
    assert (
        term_coverage(
            "The value is h = 8 parallel attention.", ["h=8", "parallel attention"]
        )
        == 1
    )


def test_term_coverage_is_none_without_terms():
    assert term_coverage("anything", []) is None


def test_term_coverage_accepts_documented_synonyms():
    assert term_coverage("divide by sqrt of dimension", ["sqrt|square root"]) == 1

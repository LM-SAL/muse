from muse.utils.documentation import _SkipAwareFileNameSortKey


def test_scripts_sort_ignoring_skip_prefix_with_unnumbered_skip_scripts_first():
    key = _SkipAwareFileNameSortKey("unused_src_dir")
    files = [
        "08_create_ci_response.py",
        "skip_03_prepare_chianti_line_lists.py",
        "aia_94_response.py",
        "skip_prepare_chianti_line_lists.py",
    ]
    assert sorted(files, key=key) == [
        "skip_prepare_chianti_line_lists.py",
        "skip_03_prepare_chianti_line_lists.py",
        "08_create_ci_response.py",
        "aia_94_response.py",
    ]

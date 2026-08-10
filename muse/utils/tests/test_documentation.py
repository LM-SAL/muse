from muse.utils.documentation import _SkipAwareFileNameSortKey


def test_numbered_scripts_interleave_ignoring_skip_prefix():
    key = _SkipAwareFileNameSortKey("unused_src_dir")
    files = [
        "08_create_ci_response.py",
        "skip_03_prepare_chianti_line_lists.py",
        "02_vdem_to_muse_fov.py",
        "skip_01_create_vdem.py",
    ]
    assert sorted(files, key=key) == [
        "skip_01_create_vdem.py",
        "02_vdem_to_muse_fov.py",
        "skip_03_prepare_chianti_line_lists.py",
        "08_create_ci_response.py",
    ]


def test_unnumbered_skip_scripts_sort_first():
    key = _SkipAwareFileNameSortKey("unused_src_dir")
    files = [
        "aia_94_response.py",
        "skip_prepare_chianti_line_lists.py",
        "eis_fe_xii_synthesis.py",
    ]
    assert sorted(files, key=key) == [
        "skip_prepare_chianti_line_lists.py",
        "aia_94_response.py",
        "eis_fe_xii_synthesis.py",
    ]

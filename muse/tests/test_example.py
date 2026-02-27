def test_primes_c():
    from muse.example_c import primes as primes_c  # noqa: PLC0415

    assert primes_c(10) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]


def test_primes():
    from muse.example_mod import primes  # noqa: PLC0415

    assert primes(10) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

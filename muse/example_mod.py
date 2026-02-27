__all__ = ["do_primes", "primes"]


def primes(imax):
    """
    Returns prime numbers up to imax.

    Parameters
    ----------
    imax : `int`
        The number of primes to return. This should be less or equal to 10000.

    Returns
    -------
    result : `list`
        The list of prime numbers.
    """
    p = list(range(10000))
    result = []
    k = 0
    n = 2

    if imax > 10000:
        msg = "imax should be <= 10000"
        raise ValueError(msg)

    while len(result) < imax:
        i = 0
        while i < k and n % p[i] != 0:
            i = i + 1
        if i == k:
            p[k] = n
            k = k + 1
            result.append(n)
            if k > 10000:
                break
        n = n + 1

    return result


def do_primes(n, usecython=False):
    if usecython:
        from .example_c import primes as cprimes  # noqa: PLC0415

        print("Using cython-based primes")  # noqa: T201
        return cprimes(n)

    print("Using pure python primes")  # noqa: T201
    return primes(n)


def main(args=None):

    from time import time  # noqa: PLC0415

    from astropy.utils.compat import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "-c", "--use-cython", dest="cy", action="store_true", help="Use the Cython-based Prime number generator."
    )
    parser.add_argument("-t", "--timing", dest="time", action="store_true", help="Time the Fibonacci generator.")
    parser.add_argument("-p", "--print", dest="prnt", action="store_true", help="Print all of the Prime numbers.")
    parser.add_argument("n", metavar="N", type=int, help="Get Prime numbers up to this number.")

    res = parser.parse_args(args)

    pre = time()
    primes = do_primes(res.n, res.cy)
    post = time()

    print(f"Found {len(primes)} prime numbers")  # noqa: T201
    print(f"Largest prime: {primes[-1]}")  # noqa: T201

    if res.time:
        print(f"Running time: {post - pre} s")  # noqa: T201

    if res.prnt:
        print(f"Primes: {primes}")  # noqa: T201

# -*- coding: utf-8 -*-

##  Part of the pyprimes.py package.
##
##  Copyright © 2014 Steven D'Aprano.
##  See the file __init__.py for the licence terms for this software.

"""Return or yield the prime factors of an integer.

The ``factors(n)`` generator yields distinct prime factors and the
number of times they are repeated:

>>> list(factors(37*37*109))
[(37, 2), (109, 1)]

The ``factorise(n)`` function returns a list of prime factors:

>>> factorise(37*37*109)
[37, 37, 109]


To get a list of distinct factors:

>>> [p for p,x in factors(37*37*109)]
[37, 109]


To efficiently calculate ω(n), the number of distinct factors:

>>> sum(1 for x in factors(37*37*109))
2

For more information, see:

    http://mathworld.wolfram.com/DistinctPrimeFactors.html
    http://oeis.org/A001221

"""

__all__ = ['factors', 'factorise']


if __debug__:
    # Set _EXTRA_CHECKS to True to enable potentially expensive assertions
    # in the factors() and factorise() functions. This is only defined or
    # checked when assertions are enabled.
    _EXTRA_CHECKS = False


def factorise(n):
    """factorise(integer) -> [list of factors]

    Returns a list of the (mostly) prime factors of integer n. For negative
    integers, -1 is included as a factor. If n is 0, 1 or -1, [n] is
    returned as the only factor. Otherwise all the factors will be prime.

    >>> factorise(-693)
    [-1, 3, 3, 7, 11]
    >>> factorise(55614)
    [2, 3, 13, 23, 31]

    """
    result = []
    for p, count in factors(n):
        result.extend([p]*count)
    if __debug__:
        # The following test only occurs if assertions are on.
        if _EXTRA_CHECKS:
            prod = 1
            for x in result:
                prod *= x
            assert prod == n, ('factors(%d) failed multiplication test' % n)
    return result


def factors(n):
    """factors(integer) -> yield factors of integer lazily

    >>> list(factors(3*7*7*7*11))
    [(3, 1), (7, 3), (11, 1)]

    Yields tuples of (factor, count) where each factor is unique and usually
    prime, and count is an integer 1 or larger.

    The factors are prime, except under the following circumstances: if the
    argument n is negative, -1 is included as a factor; if n is 0 or 1, it
    is given as the only factor. For all other integer n, all of the factors
    returned are prime.
    """
    if n in (0, 1, -1):
        yield (n, 1)
        return
    elif n < 0:
        yield (-1, 1)
        n = -n
    assert n >= 2
    from pyprimes import primes
    for p in primes():
        if p*p > n: break
        count = 0
        while n % p == 0:
            count += 1
            n //= p
        if count:
            yield (p, count)
    if n != 1:
        if __debug__:
            # The following test only occurs if assertions are on.
            if _EXTRA_CHECKS:
                from pyprimes import is_prime
                assert is_prime(n), ('final factor %d is not prime' % n)
        yield (n, 1)


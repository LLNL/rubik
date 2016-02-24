# -*- coding: utf-8 -*-

##  Part of the pyprimes.py package.
##
##  Copyright © 2014 Steven D'Aprano.
##  See the file __init__.py for the licence terms for this software.

"""Generate prime numbers using a sieve."""

import itertools

from pyprimes.compat23 import compress, next, range

__all__ = ['best_sieve', 'cookbook', 'croft', 'erat', 'sieve', 'wheel']


def erat(n):
    """Return a list of primes up to and including n.

    This is a fixed-size version of the Sieve of Eratosthenes, using an
    adaptation of the traditional algorithm.

    >>> erat(30)
    [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

    """
    if n < 2:
        return []
    # Generate a fixed array of integers.
    arr = list(range(n+1))  # A list is faster than an array.
    # Cross out 0 and 1 since they aren't prime.
    arr[0] = arr[1] = None
    i = 2
    while i*i <= n:
        # Cross out all the multiples of i starting from i**2.
        for p in range(i*i, n+1, i):
            arr[p] = None
        # Advance to the next number not crossed off.
        i += 1
        while i <= n and arr[i] is None:
            i += 1
    return list(filter(None, arr))


def sieve():
    """Yield prime integers using the Sieve of Eratosthenes.

    This recursive algorithm is modified to generate the primes lazily
    rather than the traditional version which operates on a fixed size
    array of integers.
    """
    # This is based on a paper by Melissa E. O'Neill, with an implementation
    # given by Gerald Britton:
    # http://mail.python.org/pipermail/python-list/2009-January/1188529.html
    innersieve = sieve()
    prevsq = 1
    table  = {}
    i = 2
    while True:
        # This explicit test is slightly faster than using
        # prime = table.pop(i, None) and testing for None.
        if i in table:
            prime = table[i]
            del table[i]
            nxt = i + prime
            while nxt in table:
                nxt += prime
            table[nxt] = prime
        else:
            yield i
            if i > prevsq:
                j = next(innersieve)
                prevsq = j**2
                table[prevsq] = j
        i += 1


def cookbook():
    """Yield prime integers lazily using the Sieve of Eratosthenes.

    Another version of the algorithm, based on the Python Cookbook,
    2nd Edition, recipe 18.10, variant erat2.
    """
    # http://onlamp.com/pub/a/python/excerpt/pythonckbk_chap1/index1.html?page=2
    table = {}
    yield 2
    # Iterate over [3, 5, 7, 9, ...]. The following is equivalent to, but
    # faster than, (2*i+1 for i in itertools.count(1))
    for q in itertools.islice(itertools.count(3), 0, None, 2):
        # Note: this explicit test is marginally faster than using
        # table.pop(i, None) and testing for None.
        if q in table:
            p = table[q]; del table[q]  # Faster than pop.
            x = p + q
            while x in table or not (x & 1):
                x += p
            table[x] = p
        else:
            table[q*q] = q
            yield q


def croft():
    """Yield prime integers using the Croft Spiral sieve.

    This is a variant of wheel factorisation modulo 30.
    """
    # Implementation is based on erat3 from here:
    #   http://stackoverflow.com/q/2211990
    # and this website:
    #   http://www.primesdemystified.com/
    # Memory usage increases roughly linearly with the number of primes seen.
    # dict ``roots`` stores an entry p**2:p for every prime p.
    for p in (2, 3, 5):
        yield p
    roots = {9: 3, 25: 5}  # Map d**2 -> d.
    primeroots = frozenset((1, 7, 11, 13, 17, 19, 23, 29))
    selectors = (1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0)
    for q in compress(
            # Iterate over prime candidates 7, 9, 11, 13, ...
            itertools.islice(itertools.count(7), 0, None, 2),
            # Mask out those that can't possibly be prime.
            itertools.cycle(selectors)
            ):
        # Using dict membership testing instead of pop gives a
        # 5-10% speedup over the first three million primes.
        if q in roots:
            p = roots[q]
            del roots[q]
            x = q + 2*p
            while x in roots or (x % 30) not in primeroots:
                x += 2*p
            roots[x] = p
        else:
            roots[q*q] = q
            yield q


def wheel():
    """Generate prime numbers using wheel factorisation modulo 210."""
    for i in (2, 3, 5, 7, 11):
        yield i
    # The following constants are taken from the paper by O'Neill.
    spokes = (2, 4, 2, 4, 6, 2, 6, 4, 2, 4, 6, 6, 2, 6, 4, 2, 6, 4, 6,
        8, 4, 2, 4, 2, 4, 8, 6, 4, 6, 2, 4, 6, 2, 6, 6, 4, 2, 4, 6, 2,
        6, 4, 2, 4, 2, 10, 2, 10)
    assert len(spokes) == 48
    # This removes about 77% of the composites that we would otherwise
    # need to divide by.
    found = [(11, 121)]  # Smallest prime we care about, and its square.
    for incr in itertools.cycle(spokes):
        i += incr
        for p, p2 in found:
            if p2 > i:  # i must be a prime.
                found.append((i, i*i))
                yield i
                break
            elif i % p == 0:  # i must be composite.
                break
        else:  # This should never happen.
            raise RuntimeError("internal error: ran out of prime divisors")


# This is the preferred way of generating prime numbers. Set this to the
# fastest/best generator. It *must* be a generator.
best_sieve = croft


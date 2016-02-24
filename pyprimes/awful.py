# -*- coding: utf-8 -*-

##  Part of the pyprimes.py package.
##
##  Copyright © 2014 Steven D'Aprano.
##  See the file __init__.py for the licence terms for this software.


"""\
===================
Algorithms to avoid
===================

This is a collection of awful and naive prime-related functions, supplied
for educational purposes, as toys, curios, or as terrible warnings on
what **not** to do.

None of these methods have acceptable performance in practice; they are
barely tolerable even for the first 100 primes.

Do not use these in production. These functions are subject to change
without warning.

"""

from __future__ import division

import itertools
import re

from pyprimes.compat23 import all, filter, next, range

from pyprimes.utilities import isqrt


# === Prime number testing ===

def isprime(n):
    """Naive primality test using naive and unoptimized trial division.

    >>> isprime(17)
    True
    >>> isprime(18)
    False

    Naive, slow but thorough test for primality using unoptimized trial
    division. This function does far too much work, and consequently is
    very slow. Nevertheless, it is guaranteed to give the right answer.
    Eventually.
    """
    if n == 2:
        return True
    if n < 2 or n % 2 == 0:
        return False
    for i in range(3, isqrt(n)+1, 2):
        if n % i == 0:
            return False
    return True


def isprime_regex(n):
    """Slow primality test using a regular expression.

    >>> isprime_regex(11)
    True
    >>> isprime_regex(15)
    False

    Unsurprisingly, this is not efficient, and should be treated as a
    novelty rather than a serious implementation. It is O(N**2) in time
    and O(N) in memory: in other words, slow and expensive.
    """
    # For a Perl or Ruby version of this, see here:
    # http://montreal.pm.org/tech/neil_kandalgaonkar.shtml
    # http://www.noulakaz.net/weblog/2007/03/18/a-regular-expression-to-check-for-prime-numbers/
    return not re.match(r'^1?$|^(11+?)\1+$', '1'*n)


# === Generating prime numbers by trial division ===

def primes0():
    """Generate prime numbers by trial division extremely slowly.

    >>> p = primes0()
    >>> [next(p) for _ in range(10)]
    [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

    This is about as naive an implementation of trial division as you can
    get. Not even the most obvious and trivial optimizations are used:

    - it uses all numbers as potential primes, whether odd or even,
      instead of skipping even numbers;
    - it checks for primality by dividing against every number less
      than the candidate, instead of stopping early;
    - even when it finds a factor, it stupidly keeps on going.

    """
    i = 2
    yield i
    while True:
        i += 1
        composite = False
        for p in range(2, i):
            if i%p == 0:
                composite = True
        if not composite:  # It must be a prime.
            yield i


def primes1():
    """Generate prime numbers by trial division very slowly.

    >>> p = primes1()
    >>> [next(p) for _ in range(10)]
    [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

    This adds a single optimization to ``primes0``, using a short-circuit
    test for primality: as soon as a factor is found, the candidate is
    rejected immediately.
    """
    i = 2
    yield i
    while True:
        i += 1
        if all(i%p != 0 for p in range(2, i)):
            yield i


def primes2():
    """Generate prime numbers by trial division very slowly.

    >>> p = primes2()
    >>> [next(p) for _ in range(10)]
    [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

    This is an incremental improvement over ``primes1`` by only testing
    odd numbers as potential primes and factors.
    """
    yield 2
    i = 3
    yield i
    while True:
        i += 2
        if all(i%p != 0 for p in range(3, i, 2)):
            yield i


def primes3():
    """Generate prime numbers by trial division slowly.

    >>> p = primes3()
    >>> [next(p) for _ in range(10)]
    [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

    This is an incremental improvement over ``primes2`` by only testing
    potential factors up to the square root of the candidate. For small
    primes below 50000 or so, this may be slightly faster than ``primes4``.

    """
    yield 2
    i = 3
    yield i
    while True:
        i += 2
        if all(i%p != 0 for p in range(3, isqrt(i)+1, 2)):
            yield i


def primes4():
    """Generate prime numbers by trial division slowly.

    >>> p = primes4()
    >>> [next(p) for _ in range(10)]
    [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

    This trial division implementation includes all obvious optimizations:

    - bail out of the test as soon as you find a factor and can be
      sure that the candidate cannot be prime;
    - only test odd numbers as possible candidates;
    - only test against the prime factors already seen;
    - stop checking at the square root of the number being tested.

    With these four optimizations, we get asymptotic behaviour of
    O(N*sqrt(N)/(log N)**2) where N is the number of primes found.

    Despite these optimizations, this is still unacceptably slow for
    generating large numbers of primes.
    """
    yield 2
    seen = []  # Odd primes only.
    # Add a few micro-optimizations to shave off a microsecond or two.
    takewhile = itertools.takewhile
    append = seen.append
    all_ = all
    # And now we search for primes.
    i = 3
    while True:
        it = takewhile(lambda p, i=i: p*p <= i, seen)
        if all_(i%p != 0 for p in it):
            append(i)
            yield i
        i += 2


# === Non-trial division algorithms ===

def turner():
    """Generate prime numbers very slowly using Euler's sieve.

    >>> p = turner()
    >>> [next(p) for _ in range(10)]
    [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

    The function is named for David Turner, who developed this implementation
    in a paper in 1975. Due to its simplicity, it has become very popular,
    particularly in Haskell circles where it is usually implemented as some
    variation of::

        primes = sieve [2..]
        sieve (p : xs) = p : sieve [x | x <- xs, x `mod` p > 0]

    This algorithm is sometimes wrongly described as the Sieve of
    Eratosthenes, but it is not, it is a version of Euler's Sieve.

    Although simple, it is extremely slow and inefficient, with
    asymptotic behaviour of O(N**2/(log N)**2) which is worse than
    trial division, and only marginally better than ``primes0``.

    In her paper http://www.cs.hmc.edu/~oneill/papers/Sieve-JFP.pdf
    O'Neill calls this the "Sleight on Eratosthenes".
    """
    # See also:
    #   http://en.wikipedia.org/wiki/Sieve_of_Eratosthenes
    #   http://en.literateprograms.org/Sieve_of_Eratosthenes_(Haskell)
    #   http://www.haskell.org/haskellwiki/Prime_numbers
    nums = itertools.count(2)
    while True:
        prime = next(nums)
        yield prime
        nums = filter(lambda v, p=prime: (v % p) != 0, nums)



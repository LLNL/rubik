# -*- coding: utf-8 -*-

##  Package pyprimes.py
##
##  Copyright © 2015 Steven D'Aprano.
##
##  Permission is hereby granted, free of charge, to any person obtaining
##  a copy of this software and associated documentation files (the
##  "Software"), to deal in the Software without restriction, including
##  without limitation the rights to use, copy, modify, merge, publish,
##  distribute, sublicense, and/or sell copies of the Software, and to
##  permit persons to whom the Software is furnished to do so, subject to
##  the following conditions:
##
##  The above copyright notice and this permission notice shall be
##  included in all copies or substantial portions of the Software.
##
##  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
##  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
##  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
##  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
##  CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
##  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
##  SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""\
================
Package pyprimes
================

This package includes functions for generating prime numbers, primality
testing, and factorising numbers into prime factors.


Definitions
-----------

"Prime numbers" are positive integers with no factors other than themselves
and 1. The first few primes are 2, 3, 5, 7, 11, ... . There is only one
even prime number, namely 2, and an infinite number of odd prime numbers.

"Composite numbers" are positive integers which have factors other than
themselves and 1. Composite numbers can be uniquely factorised into the
product of two or more (possibly repeated) primes, e.g. 18 = 2*3*3.

Both 0 and 1 are considered by mathematicians to be neither prime nor
composite.


Generating prime numbers
------------------------

To generate an unending stream of prime numbers, use the ``primes``
function with no arguments:

    >>> p = primes()
    >>> [next(p) for _ in range(10)]
    [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

Or give start and end arguments for just a subset of the primes:

    >>> list(primes(180, 225))
    [181, 191, 193, 197, 199, 211, 223]


The ``next_prime`` and ``prev_prime`` functions return the next, or
previous, prime from the given argument:

    >>> next_prime(20)
    23
    >>> prev_prime(20)
    19

    NOTE:: For large prime numbers p, the *average* distance between p and
           the next (or previous) prime is proportional to log p. If p
           is large enough, it may take some considerable time to calculate
           the next or previous prime from p.


Primality testing
-----------------

To test whether an integer is prime or not, use ``is_prime``:

    >>> is_prime(10)
    False
    >>> is_prime(11)
    True

For extremely large values ``is_prime`` may be probabilistic. That is,
if it reports a number is prime, it may be only "almost certainly prime",
with a very small chance that the number is actually composite. If it
returns False, the number is certainly composite. For more details on the
probabilistic nature of primality testing, see the ``probabilistic`` module.

The ``trial_division`` function also returns True for primes and False for
non-primes, unlike ``is_prime`` it is always an exact test and never
deterministic. However, it may be very slow and require large amounts of
memory for very large values.

    >>> trial_division(15)
    False
    >>> trial_division(17)
    True


Number theory convenience functions
-----------------------------------

``pyprimes`` offers a few convenience functions from the Number Theory
branch of mathematics.

    nprimes:
        Return the first n primes.

    nth_prime:
        Return the nth prime.

    prime_count:
        Return the number of primes less than n.

    prime_sum:
        Return the sum of primes less than n.

    prime_partial_sums:
        Yield the running sums of primes less than n.

See the individual functions for further details.


Sub-modules
-----------

The ``pyprimes`` package also includes the following public sub-modules:

    awful:
        Simple but inefficient, slow or otherwise awful algorithms for
        generating primes and testing for primality. This module is
        provided only for pedagogical purposes (mostly as a lesson in
        what *not* to do).

    factors:
        Factorise small numbers into the product of primes.

    probabilistic:
        Generate and test for primes using probabilistic methods.

    sieves:
        Generate prime numbers using sieving algorithms.

    strategic:
        Various prime generating and testing functions implemented
        with the Strategy design pattern.

    utilities:
        Assorted utility functions.

plus the following private sub-modules:

    compat23:
        Internal compatibility layer, to support multiple versions of
        Python.

    tests:
        Unit and regression tests for the package.

The contents of the private sub-modules are subject to change or removal
without notice.

"""

from __future__ import division

import itertools
import warnings

import pyprimes.strategic
from pyprimes.compat23 import next
from pyprimes.utilities import filter_between


# Module metadata.
__version__ = "0.2.2a"
__date__ = "2015-01-19"
__author__ = "Steven D'Aprano"
__author_email__ = "steve+python@pearwood.info"


__all__ = ['is_prime', 'MaybeComposite', 'next_prime', 'nprimes',
           'nth_prime', 'prev_prime', 'prime_count', 'prime_partial_sums',
           'prime_sum', 'primes', 'trial_division',
          ]


class MaybeComposite(RuntimeWarning):
    """Warning raised when a primality test is probabilistic."""
    pass


# === Generate prime numbers ===

def primes(start=None, end=None):
    """Yield primes, optionally between ``start`` and ``end``.

    If ``start`` or ``end`` arguments are given, they must be integers.
    Only primes between ``start`` and ``end`` will be yielded:

    >>> list(primes(start=115, end=155))
    [127, 131, 137, 139, 149, 151]

    ``start`` is inclusive, and ``end`` is exclusive:

    >>> list(primes(5, 31))
    [5, 7, 11, 13, 17, 19, 23, 29]
    >>> list(primes(5, 32))
    [5, 7, 11, 13, 17, 19, 23, 29, 31]

    If ``start`` is not given, or is None, there is no lower limit and the
    first prime yielded will be 2. If ``end`` is not given or is None,
    there is no upper limit.
    """
    from pyprimes.sieves import best_sieve
    return pyprimes.strategic.primes(best_sieve, start, end)


def next_prime(n):
    """Return the first prime number strictly greater than n.

    >>> next_prime(97)
    101

    For sufficiently large n, over approximately 341 trillion, the result
    may be only probably prime rather than certainly prime.
    """
    return pyprimes.strategic.next_prime(is_prime, n)


def prev_prime(n):
    """Return the first prime number strictly less than n.

    >>> prev_prime(100)
    97

    If there are no primes less than n, raises ValueError.

    For sufficiently large n, over approximately 341 trillion, the result
    may be only probably prime rather than certainly prime.
    """
    return pyprimes.strategic.prev_prime(is_prime, n)


# === Primality testing ===

def is_prime(n):
    """Return True if n is probably a prime number, and False if it is not.

    >>> is_prime(103)
    True
    >>> is_prime(105)
    False


    For sufficiently large numbers, ``is_prime`` may be probabilistic rather
    than deterministic. If that is the case, ``is_prime`` will raise a
    ``MaybeComposite`` warning if ``n`` is only probably prime rather than
    certainly prime. The probability of a randomly choosen value being
    mistakenly identified as prime when it is actually composite is less
    than 1e-18 (1 chance in a million million million).

    If ``is_prime`` returns False, the number is certainly composite.
    """
    from pyprimes.probabilistic import is_probable_prime
    flag = pyprimes.strategic.is_prime(is_probable_prime, n)
    assert flag in (0, 1, 2)
    if flag == 2:
        message = "%d is only only probably prime" % n
        import warnings
        warnings.warn(message, MaybeComposite)
    return bool(flag)


def trial_division(n):
    """trial_division(n) -> True|False

    An exact but slow primality test using trial division by primes only.
    It returns True if the argument is a prime number, otherwise False.

    >>> trial_division(11)
    True
    >>> trial_division(12)
    False

    For large values of n, this may be slow or run out of memory.
    """
    return pyprimes.strategic.trial_division(primes, n)


# === Number theory convenience functions ===

def nprimes(n):
    """Convenience function that yields the first n primes only.

    >>> list(nprimes(10))
    [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

    """
    return itertools.islice(primes(), n)


def nth_prime(n):
    """nth_prime(n) -> int

    Return the nth prime number, starting counting from 1. Equivalent to
    p[n] (p subscript n) in standard maths notation.

    >>> nth_prime(1)  # First prime is 2.
    2
    >>> nth_prime(5)
    11
    >>> nth_prime(50)
    229

    """
    # http://oeis.org/A000040
    if n < 1:
        raise ValueError('argument must be a positive integer')
    return next(itertools.islice(primes(), n-1, n))


def prime_count(n):
    """prime_count(n) -> int

    Returns the number of prime numbers less than or equal to n.
    It is also known as the Prime Counting Function, π(x) or pi(x).
    (Not to be confused with the constant pi π = 3.1415....)

    >>> prime_count(20)
    8
    >>> prime_count(10780)
    1312
    >>> prime_count(30075)
    3251

    The number of primes less than n is approximately n/(ln n - 1).
    """
    # Values for pi(x) taken from here: http://primes.utm.edu/nthprime/
    # See also:  http://primes.utm.edu/howmany.shtml
    # http://mathworld.wolfram.com/PrimeCountingFunction.html
    # http://oeis.org/A000720
    return sum(1 for p in primes(end=n+1))


def prime_sum(n):
    """prime_sum(n) -> int

    prime_sum(n) returns the sum of the first n primes.

    >>> prime_sum(9)
    100
    >>> prime_sum(49)
    4888

    The sum of the first n primes is approximately n**2*(ln n)/2.
    """
    # See:  http://mathworld.wolfram.com/PrimeSums.html
    # http://oeis.org/A007504
    if n < 1:
        return 0
    return sum(nprimes(n))


def prime_partial_sums():
    """Yield the partial sums of the prime numbers.

    >>> p = prime_partial_sums()
    >>> [next(p) for _ in range(6)]  # primes 2, 3, 5, 7, 11, ...
    [0, 2, 5, 10, 17, 28]

    """
    # http://oeis.org/A007504
    n = 0
    for p in primes():
        yield n
        n += p


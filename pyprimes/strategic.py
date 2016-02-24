# -*- coding: utf-8 -*-

##  Part of the pyprimes.py package.
##
##  Copyright © 2014 Steven D'Aprano.
##  See the file __init__.py for the licence terms for this software.

"""The module implements various prime generating and testing functions using
the Strategy design pattern, allowing the caller to easily experiment with
different algorithms and implementations.

The functions in this module will take at least one mandatory argument,
usually named either ``strategy`` or ``prover``.

    strategy:

        The ``strategy`` argument is used to delegate to a prime generator.
        It must be a function which takes no arguments and returns an
        iterator that yields primes. (A generator function is a convenient
        way to manage this.)

        This module makes no check that the strategy function actually
        yields prime numbers. It is the caller's responsibility to ensure
        that is the case.

    prover:

        The ``prover`` argument is used to delegate to a primality testing
        function. It must be a function which takes a single argument, an
        integer, and returns one of the following flags:

            0 or False      Number is definitely nonprime.
            1 or True       Number is definitely prime.
            2               Number is a probable prime or pseudoprime.

        Any other result will raise TypeError or ValueError.

        This module makes no check to confirm that the prover function
        actually tests for primality. It is the caller's responsibility to
        ensure that is the case.

"""

from __future__ import division

from pyprimes.compat23 import next


__all__ = ['is_prime', 'next_prime', 'prev_prime', 'primes',
           'trial_division',
          ]



# === Primality testing ===

def is_prime(prover, n):
    """Perform a primality test on n using the given prover.

    See the docstring for this module for specifications for
    the ``prover`` function.

    >>> import pyprimes.awful
    >>> is_prime(pyprimes.awful.isprime, 103)
    True
    >>> is_prime(pyprimes.awful.isprime, 105)
    False

    """
    flag = prover(n)
    if flag is True or flag is False:
        return flag
    # Check for actual ints, not subclasses. Gosh this takes me back to
    # Python 1.5 days...
    if type(flag) is int:
        if flag in (0, 1, 2):
            return flag
        raise ValueError('prover returned invalid int flag %d' % flag)
    raise TypeError('expected bool or int but prover returned %r' % flag)


def trial_division(strategy, n):
    """Perform a trial division primality test using the given strategy.

    See this module's docstring for specifications for the ``strategy``
    function.

    This performs an exact but slow primality test using trial division
    by dividing by primes only. It returns True if the argument is a
    prime number, otherwise False.

    >>> import pyprimes.awful
    >>> trial_division(pyprimes.awful.primes0, 11)
    True
    >>> trial_division(pyprimes.awful.primes0, 12)
    False

    For large values of n, this may be slow or run out of memory.
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    limit = n**0.5  # FIXME: should use exact isqrt
    for divisor in strategy():
        if divisor > limit: break
        if n % divisor == 0: return False
    return True


# === Prime generators ===

def primes(strategy, start=None, end=None):
    """Yield primes using the given strategy function.

    See this module's docstring for specifications for the ``strategy``
    function.

    If the optional arguments ``start`` and ``end`` are given, they must be
    either None or an integer. Only primes in the half-open range ``start``
    (inclusive) to ``end`` (exclusive) are yielded. If ``start`` is None,
    the range begins at the lowest prime (namely 2), if ``end`` is None,
    the range has no upper limit.

    >>> from pyprimes.awful import turner
    >>> list(primes(turner, 6, 30))
    [7, 11, 13, 17, 19, 23, 29]

    """
    #return filter_between(gen(), start, end)
    it = strategy()
    p = next(it)
    if start is not None:
        # Drop the primes below start as fast as possible, then yield.
        while p < start:
            p = next(it)
    assert start is None or p >= start
    if end is not None:
        while p < end:
            yield p
            p = next(it)
    else:
        while True:
            yield p
            p = next(it)
    # Then yield until end.


def next_prime(prover, n):
    """Return the first prime number strictly greater than n.

    See the docstring for this module for specifications for
    the ``prover`` function.

    >>> import pyprimes.awful
    >>> next_prime(pyprimes.awful.isprime, 97)
    101

    """
    if n < 2:
        return 2
    # Advance to the next odd number.
    if n % 2 == 0:  n += 1
    else:  n += 2
    assert n%2 == 1
    while not is_prime(prover, n):
        n += 2
    return n


def prev_prime(prover, n):
    """Return the first prime number strictly less than n.

    See the docstring for this module for specifications for
    the ``prover`` function.

    >>> import pyprimes.awful
    >>> prev_prime(pyprimes.awful.isprime, 100)
    97

    If there are no primes less than n, raises ValueError.
    """
    if n <= 2:
        raise ValueError('There are no smaller primes than 2.')
    # Retreat to the previous odd number.
    if n % 2 == 1:  n -= 2
    else:  n -= 1
    assert n%2 == 1
    while not is_prime(prover, n):
        n -= 2
    return n


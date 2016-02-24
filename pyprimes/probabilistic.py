# -*- coding: utf-8 -*-

##  Part of the pyprimes.py package.
##
##  Copyright © 2014 Steven D'Aprano.
##  See the file __init__.py for the licence terms for this software.

"""\
===============================
Probabilistic primality testing
===============================

Unless otherwise stated, the functions in this module are probabilistic
primality tests, with a small chance of "false positive" errors, that is,
wrongly reporting a number as prime when it is actually composite. "False
negative" errors, where a prime is reported as composite, cannot occur.

Unless otherwise stated, the functions in this module return a three-state
flag for the primality of the given argument:

    ------  ------------------------------------------------
    State   Meaning
    ------  ------------------------------------------------
    0       Argument is definitely composite or non-prime.
    1       Argument is definitely prime.
    2       Argument may be prime.
    ------  ------------------------------------------------

The interpretation of a return result of 2 will depend on the primality
test being used, but it generally means that the argument is probably
prime, although with some (hopefully small) chance of being composite.
See the specific function for details.

The recommended probabilistic test is the ``is_probable_prime`` function,
which uses a combination of deterministic and multiple strong probabilistic
tests to return a certain result as often as possible, otherwise a
probabilistic result with a very high degree of confidence:

    >>> is_probable_prime(35184372088891)  # Definitely prime.
    1
    >>> is_probable_prime(35184372088893)  # Composite 3*31*4493*84203557.
    0


Other primality tests included are:

    is_fermat_probable_prime(n [, base])
        A weak probabilistic test.

        >>> is_fermat_probable_prime(12400009)  # Actually is prime.
        2
        >>> is_fermat_probable_prime(12400013)  # Composite 23*443*1217.
        0


    is_miller_rabin_probable_prime(n [, base])
        A strong probabilistic test.

        >>> is_miller_rabin_probable_prime(14008957)  # Actually is prime.
        2
        >>> is_miller_rabin_probable_prime(14008971)  # Composite 3*947*4931.
        0


    is_miller_rabin_definite_prime(n)
        Performs a deterministic set of Miller-Rabin tests.

        >>> is_miller_rabin_definite_prime(282311353)  # Prime.
        True
        >>> is_miller_rabin_definite_prime(282311367)  # 3*17*949*5833.
        False

"""

from __future__ import division

import random

import pyprimes.utilities


__all__ = ['is_probable_prime', 'is_fermat_probable_prime',
           'is_miller_rabin_probable_prime', 'primes',
           ]


try:
    _Int = (int, long)
except NameError:
    _Int = int


# === Instrumented probable prime test ===

class IsProbablePrime(object):
    """Callable class to perform probabilistic primality tests.

    When called, instances return a three-state flag (0, 1 or 2) giving
    the primality of the integer argument. By default, calls are
    instrumented, so you can tell which primality tests are passing.

    Instead of using this class directly, normally you would use the top
    level function ``is_probable_prime``, which is a bound method of this
    class. See the ``__call__`` method for additional details.

    To use this class with instrumentation, instantiate it and call the

    Instrumentation
    ---------------

    Instrumentation of instances can be controlled with three methods:

        enable_instrumentation
        disable_instrumentation



    Return result
    -------------

        0:      argument is definitely composite or non-prime.
        1:      argument is definitely prime.
        2:      argument is a probable prime or strong pseudoprime.


    Usage
    -----

    Normally you would use the instance ``is_probable_prime``:


    For ``n`` below approximately 3.8 million trillion, the return result
    will always be 0 or 1 (definitely non-prime, or definitely prime).
    For larger arguments, the result may be 2, which indicates that the
    argument is only probably a prime number. The probability that a
    composite number is wrongly identified as prime is vanishingly small
    (less than once in 18 thousand years of continual usage).


    Examples
    --------

    >>> is_probable_prime(982451651)  # 16811*58441
    0
    >>> is_probable_prime(982451653)
    1

    >>> is_probable_prime(58477*72817)
    0
    >>> is_probable_prime(4258119781)
    1



    For larger values of ``n``, a result of 0 is definitive (there are
    no false negatives), but a result of 2 means only that ``n`` may be
    prime, with an extremely small probability that it is actually
    composite (a false positive result).

    In this example, p is the twelth Mersenne prime, which was discovered
    by Édouard Lucas in 1876. (http://oeis.org/A000668)

    >>> p = 170141183460469231731687303715884105727
    >>> assert p == 2**127 - 1
    >>> is_probable_prime(p)
    2
    >>> is_probable_prime(p + 2)
    0

    Use the pre-allocated instance:

    >>> is_probable_prime(97)
    1

    or instantiate your own:

    >>> ipp = IsProbablePrime()
    >>> ipp(97)
    1

    """

    # Table of small primes.
    primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    # Default number of random Miller-Rabin tests to perform.
    count = 30

    def __init__(self):
        # Allocate instrumentation.
        self.instrument = pyprimes.utilities.Instrument(self, self._methods)

    def _trial_division(self, n):
        """Deterministic but limited primality test using trial division
        by the first few small primes.

        Returns 0, 1 or 2 for definitely non-prime, definitely prime,
        or unsure (may be prime).
        """
        # For speed, we deal with some simple deterministic cases first,
        # and use a quick trial division round to eliminate factors of
        # the smaller primes. This eliminates most small candidate primes
        # without the expense of the Miller-Rabin tests.
        assert n > 1
        for p in self.primes:
            if n == p:
                return 1  # Certainly prime.
            elif n % p == 0:
                return 0  # Certainly composite.
        # When doing trial division, we can stop checking for prime factors
        # at the square root of n. If the last prime factor we checked is
        # larger than sqrt(n), n must be definitely prime.
        assert p == self.primes[-1]
        if p**2 > n:
            return 1
        assert n >= 367  # The smallest number not handled above.
        return 2  # Unsure.

    def _determistic_miller_rabin(self, n):
        """Deterministic but limited primality test using Miller-Rabin.

        Returns 0, 1 or 2 for definitely non-prime, definitely prime,
        or unsure (maybe prime).
        """
        assert (n > 2) and (n % 2 == 1)
        witnesses = _get_miller_rabin_witnesses(n)
        if witnesses is None:
            flag = 2
        else:
            flag = is_miller_rabin_probable_prime(n, witnesses)
            if flag == 2:
                flag = 1
        return flag

    def _prime_miller_rabin(self, n):
        """Probabilistic primality test using Miller-Rabin
        with small primes as bases.

        Returns 0, 1 or 2 for definitely non-prime, definitely prime,
        or unsure (maybe prime).
        """
        assert n > 1
        # Before doing the randomised Miller-Rabin test, we do a small
        # number of tests using small primes as witnesses, which is
        # cheaper than the randomised test. This should satisfy all values
        # below 2**64.
        assert len(self.primes) >= 12
        return is_miller_rabin_probable_prime(n, self.primes)

    def _randomized_miller_rabin(self, n, count=None):
        """Probabilistic primality test using Miller-Rabin
        with random bases.

        Returns 0, 1 or 2 for definitely non-prime, definitely prime,
        or unsure (maybe prime).
        """
        # For sufficiently large n, currently greater than ~3.8e18, there
        # is no known minimal set of Miller-Rabin witnesses which will
        # definitively prove primality. We can still get a guaranteed
        # result by exhaustive testing with every witness in the range
        # bases 2...⌊(2*(ln n)**2)⌋ (assuming the extended Riemann
        # hypothesis), which for n of this size, is at least 3661.
        #
        # Fortunately, in practical terms that would be hugely overkill:
        # for far fewer number of rounds, we can get a result which (while
        # technically probabilistic) is as close to certain as we like.
        #
        # For large n, with randomly chosen bases in the range 1...(n-1),
        # the probability of declaring a composite number "probably prime"
        # (a false positive result) after k tests is 1/4**k on average. We
        # can choose a sufficiently large k so as to make false positives
        # vanishingly rare:
        #
        # With k = 10, the probability of a false positive is < 1e-8;
        # With k = 20, the probability of a false positive is < 1e-12;
        # With k = 30, the probability of a false positive is < 1e-18,
        #   and if we could test a million numbers a second, it would take
        #   an average of 18000+ years to come across a single such error.
        #
        # We can ignore 1 as a witness, because it always claims that n is
        # a probable prime, regardless of n.
        assert n > 1
        if count is None:
            count = self.count
        bases = tuple([random.randint(2, n-1) for _ in range(count)])
        return is_miller_rabin_probable_prime(n, bases)

    def _simple(self, n):
        """Handle the simple cases."""
        if n < 2:
            return 0  # Certainly non-prime.
        return 2  # Maybe prime.

    # Don't change the order of these (unbound) methods.
    _methods = (_simple,
                _trial_division,
                _determistic_miller_rabin,
                _prime_miller_rabin,
                _randomized_miller_rabin,
                )

    def _check_primality(self, n):
        if not isinstance(n, _Int):
            raise TypeError
        for method in self._methods:
            flag = method(self, n)
            assert flag in (0, 1, 2)
            if flag in (0, 1):
                # Certainly prime or not prime.
                return (method.__name__, flag)
        assert flag == 2  # Unsure.
        return (method.__name__, flag)

    def is_probable_prime(self, n):
        """xxx
        """
        return self._check_primality(n)[1]

    def __call__(self, n):
        """Instrumented version of the ``is_probable_prime`` method."""
        name, flag = self._check_primality(n)
        instrument = self.instrument
        if instrument is not None:
            instrument.update(name, n, flag)
        return flag


is_probable_prime = IsProbablePrime().is_probable_prime



# === Specific primality tests ===


# http://en.wikipedia.org/wiki/Fermat_primality_test
def is_fermat_probable_prime(n, base=2):
    """is_fermat_probable_prime(n [, base]) -> 0|1|2

    Return a three-state flag (either 0, 1 or 2) that integer ``n `` is
    either a prime or Fermat pseudoprime, as witnessed by one or more
    integer bases.

    Arguments
    ---------

        n       Integer to be tested for primality.
        base    Optional integer base, or tuple of bases. (Defaults to 2.)

    Return result
    -------------

        0       Number is definitely non-prime.
        1       Number is definitely prime.
        2       Number is a weak probable prime or pseudoprime.


    ``is_fermat_probable_prime`` performs the Fermat primality test,
    which is a weak probabilistic test. If a number fails this test,
    it is definitely composite (there are no false negative tests):

    >>> is_fermat_probable_prime(99, 7)
    0

    However, if the number passes the test, it is provisional evidence
    that it may be a prime:

    >>> is_fermat_probable_prime(29, 7)  # 29 actually is prime.
    2

    In this case we can state that "7 is a witness that 29 may be prime".

    As the Fermat test is probabilistic, composite numbers will sometimes
    pass a test, or even repeated tests. We call them pseudoprimes to
    some base:

    >>> is_fermat_probable_prime(3*11, 10)  # 33 is a pseudoprime to base 10.
    2

    and we call 10 a "Fermat liar" for 33.

    A single passed test is not very convincing, but with more tests, we
    can gain more confidence. ``base`` must be a positive int between 1
    and n-1 inclusive, or a tuple of such bases. 1 is permitted, but not
    very useful: it is a witness for all numbers. By default, base=2.

    >>> is_fermat_probable_prime(33, (10, 7))
    0

    It may take an arbitrary number of Fermat tests to definitively
    prove a number is composite:

    >>> is_fermat_probable_prime(7*11*13*41, (17, 23, 356, 359))
    2
    >>> is_fermat_probable_prime(7*11*13*41, (17, 23, 356, 359, 363))
    0

    Unfortunately, there are some numbers which are composite but still
    pass *all* Fermat tests. These pseudoprime numbers are called the
    Carmichael numbers, and ``is_fermat_probable_prime`` cannot
    distinguish them from actual primes no matter how many tests you
    perform.

    For large enough ``n``, if a number passes ``k`` randomly chosen and
    independent Fermat tests, we can conclude that the probability that
    it is either prime or a Carmichael number is (on average) at least
    ``1 - (1/2**k)``.
    """
    if not isinstance(base, tuple):
        base = (base,)
    # Deal with the simple deterministic cases first.
    if n < 2:
        return 0  # Certainly composite (or unity, or zero).
    elif n == 2:
        return 1  # Certainly prime.
    elif n % 2 == 0:
        return 0
    # Now the Fermat test proper.
    for a in base:
        if pow(a, n-1, n) != 1:
            return 0  # n is certainly composite.
    return 2  # All of the bases are witnesses for n being prime.


# http://en.wikipedia.org/wiki/Miller%E2%80%93Rabin_primality_test
def is_miller_rabin_probable_prime(n, base=2):
    """is_miller_rabin_probable_prime(n [, base]) -> 0|1|2

    Return a three-state flag (either 0, 1 or 2) that integer ``n `` is
    either a prime or strong pseudoprime, as witnessed by one or more
    integer bases.

    Arguments
    ---------

        n       Integer to be tested for primality.
        base    Optional integer base, or tuple of bases. (Defaults to 2.)

    Return result
    -------------

        0       Number is definitely non-prime.
        1       Number is definitely prime.
        2       Number is a probable prime or pseudoprime.


    ``is_miller_rabin_probable_prime`` performs the Miller-Rabin primality
    test, which is a strong probabilistic test. If a number fails this
    test, it is definitely composite (there are no false negative tests):

    >>> is_miller_rabin_probable_prime(99, 7)
    0

    However, if the number passes the test, it is provisional evidence
    that it may be a prime:

    >>> is_miller_rabin_probable_prime(29, 7)  # 29 actually is prime.
    2

    In this case we can state that "7 is a witness that 29 may be prime".

    As the Miller-Rabin test is probabilistic, composite numbers will
    sometimes pass a test, or even repeated tests. Such numbers are
    known as pseudoprimes to some base:

    >>> assert 561 == 3*11*17  # Actually composite.
    >>> is_miller_rabin_probable_prime(561, 103)
    2

    A single passed test is not very convincing, but with more tests, we
    can gain more confidence. ``base`` must be a positive int between 1
    and n-1 inclusive, or a tuple of such bases. 1 is permitted, but not
    very useful: it is a witness for all numbers. By default, base=2.

    >>> is_miller_rabin_probable_prime(561, (103, 7))
    0

    It may take an arbitrary number of Miller-Rabin tests to definitively
    prove a number is composite:

    >>> assert 41041 == 7*11*13*41  # Actually composite.
    >>> is_miller_rabin_probable_prime(41041, (16, 92, 100, 256))
    2
    >>> is_miller_rabin_probable_prime(41041, (16, 92, 100, 256, 288))
    0

    For large enough ``n``, if a number passes ``k`` randomly chosen and
    independent Miller-Rabin tests, we can conclude that the probability
    that it is either prime or a strong pseudoprime is (on average) at
    least ``1 - (1/4**k)``.
    """
    if not isinstance(base, tuple):
        base = (base,)
    # Deal with the trivial cases.
    if n < 2:
        return 0  # Certainly composite (or unity, or zero).
    if n == 2:
        return 1  # Certainly prime.
    elif n % 2 == 0:
        return 0
    # Now perform the Miller-Rabin test proper.
    # Start by writing n-1 as 2**s * d.
    d, s = _factor2(n-1)
    for a in base:
        if _is_composite(a, d, s, n):
            return 0  # n is definitely composite.
    # If we get here, all of the bases are witnesses for n being prime.
    return 2  # Maybe prime.


def _factor2(n):
    """Factorise positive integer n as d*2**i, and return (d, i).

    >>> _factor2(768)
    (3, 8)
    >>> _factor2(18432)
    (9, 11)

    Private function used internally by the Miller-Rabin primality test.
    """
    assert n > 0
    i = 0
    d = n
    while 1:
        q, r = divmod(d, 2)
        if r == 1:
            break
        i += 1
        d = q
    assert d%2 == 1
    assert d*2**i == n
    return (d, i)


def _is_composite(b, d, s, n):
    """_is_composite(b, d, s, n) -> True|False

    Tests base b to see if it is a witness for n being composite. Returns
    True if n is definitely composite, otherwise False if it *may* be prime.

    >>> _is_composite(4, 3, 7, 385)
    True
    >>> _is_composite(221, 3, 7, 385)
    False

    Private function used internally by the Miller-Rabin primality test.
    """
    assert d*2**s == n-1
    if pow(b, d, n) == 1:
        return False
    for i in range(s):
        if pow(b, 2**i * d, n) == n-1:
            return False
    return True


def is_miller_rabin_definite_prime(n):
    """Deterministic but limited primality test using Miller-Rabin.

    Returns True for definite primes, False for definite non-primes, and
    raises ValueError for numbers which are unsure.
    """
    if n <= 1:
        return False
    elif n == 2:
        return True
    elif n % 2 == 0:
        return False
    witnesses = _get_miller_rabin_witnesses(n)
    if witnesses is None:
        msg = 'no definite Miller-Rabin test is available for %d' % n
        raise ValueError(msg)
    return is_miller_rabin_probable_prime(n, witnesses) != 0


def _get_miller_rabin_witnesses(n):
    """Return a tuple of definitive Miller-Rabin witnesses for n."""
    # We can always get a guaranteed (determistic, non-probabilistic)
    # result from Miller-Rabin by exhaustively testing with every
    # potential witness in the inclusive range 1...sqrt(n). If the
    # extended Riemann hypothesis is correct, that upper bound can be
    # further reduced to min(n-1, floor(2*(ln n)**2)).
    #
    # However, for sufficiently small n, it is possible to get a
    # deterministic answer from a mere handful of witnesses.
    #
    # Pomerance, Selfridge and Wagstaff (1980), and Jaeschke (1993)
    # have found small sets of bases which conclusively determine
    # primality for all values of n up to some upper limit, currently
    # around 3.8 million trillion (3.8e18).
    #
    # References:
    #   [1] http://oeis.org/A014233
    #   [2] http://mathworld.wolfram.com/Rabin-MillerStrongPseudoprimeTest.html
    #   [3] http://primes.utm.edu/prove/prove2_3.html
    #   [4] http://en.wikipedia.org/wiki/Miller%E2%80%93Rabin_primality_test
    #
    assert (n > 2) and (n%2 == 1)
    if n < 2047:
        # References: [1], [2], [4]
        witnesses = (2,)
    elif n < 1373653:  # ~1.3 million
        # References: [1], [2], [3], [4]
        witnesses = (2, 3)
    elif n < 9080191:  # ~9.0 million
        # References: [3], [4]
        witnesses = (31, 73)
    elif n < 25326001:  # ~25.3 million
        # References: [1], [2], [3], [4]
        witnesses = (2, 3, 5)
    elif n < 3215031751:  # ~3.2 billion
        # References: [1], [2], [3], [4]
        witnesses = (2, 3, 5, 7)
    elif n < 4759123141:  # ~4.7 billion
        # References: [3], [4]
        witnesses = (2, 7, 61)
    # elif n < 1122004669633:  # ~1.2 trillion
    ## We don't bother with these witnesses, since the next block
    ## uses almost as few and has better references.
    #     bases = (2, 13, 23, 1662803)  # References: [4]
    elif n < 2152302898747:  # ~2.1 trillion
        # References: [1], [2], [3], [4]
        witnesses = (2, 3, 5, 7, 11)
    elif n < 3474749660383:  # ~3.4 trillion
        # References: [1], [2], [3], [4]
        witnesses = (2, 3, 5, 7, 11, 13)
    elif n < 341550071728321:  # ~341.5 trillion
        # References: [1], [2], [3], [4]
        witnesses = (2, 3, 5, 7, 11, 13, 17)
    elif n < 3825123056546413051:  # ~3.8 million trillion
        # References: [1], [4]
        witnesses = (2, 3, 5, 7, 11, 13, 17, 19, 23)
    elif n <= 2**64:
        # The first 12 primes make up a set of deterministic witnesses
        # for all numbers below 2**64.
        # References: FIXME
        witnesses = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
        # This page: http://miller-rabin.appspot.com/
        # claims that these make up a minimal set:
        # witnesses = (2, 325, 9375, 28178, 450775, 9780504, 1795265022)
        # but until that result is confirmed I'm sticking to the primes.
    else:
        witnesses = None
    return witnesses


def primes():
    """Generate prime numbers by testing rather than sieving."""
    warned = False
    yield 2
    i = 3
    while True:
        flag = is_probable_prime(i)
        if flag == 2 and not warned:
            import warnings
            from pyprimes import MaybeComposite
            warnings.warn("values are now only probably prime", MaybeComposite)
            warned = True
        if flag:
            yield i
        i += 2



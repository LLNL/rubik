# -*- coding: utf-8 -*-

##  Part of the pyprimes.py package.
##
##  Copyright © 2014 Steven D'Aprano.
##  See the file __init__.py for the licence terms for this software.


"""Unit test suites for the pyprimes package.

This module is considered a private implementation detail and is subject
to change without notice.
"""

from __future__ import division

import doctest
import inspect
import itertools
import operator
import random
import sys
import unittest


# Conditionally hack the PYTHONPATH.
if __name__ == '__main__':
    import os
    path = os.path.dirname(__file__)
    parent, here = os.path.split(path)
    sys.path.append(parent)


# Modules being tested:
import pyprimes
import pyprimes.awful as awful
import pyprimes.compat23 as compat23
import pyprimes.factors as factors
import pyprimes.probabilistic as probabilistic
import pyprimes.sieves as sieves
import pyprimes.strategic as strategic
import pyprimes.utilities as utilities

# Support Python 2.4 through 3.x
from pyprimes.compat23 import next, range, reduce


# First 100 primes from here:
# http://en.wikipedia.org/wiki/List_of_prime_numbers
PRIMES = [2,   3,   5,   7,   11,  13,  17,  19,  23,  29,
          31,  37,  41,  43,  47,  53,  59,  61,  67,  71,
          73,  79,  83,  89,  97,  101, 103, 107, 109, 113,
          127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
          179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
          233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
          283, 293, 307, 311, 313, 317, 331, 337, 347, 349,
          353, 359, 367, 373, 379, 383, 389, 397, 401, 409,
          419, 421, 431, 433, 439, 443, 449, 457, 461, 463,
          467, 479, 487, 491, 499, 503, 509, 521, 523, 541,
          ]
assert len(PRIMES) == 100


# Skipping tests is only supported in Python 2.7 and up. For older versions,
# we define a quick and dirty decorator which more-or-less does the same.
try:
    skip = unittest.skip
except AttributeError:
    # Python version is too old, there is no skip decorator, so we make
    # our own basic version that silently replaces the test method with
    # a do-nothing function.
    def skip(reason):
        def decorator(method):
            return lambda self: None
        return decorator


# === Helper functions ===

try:
    isgeneratorfunction = inspect.isgeneratorfunction
except AttributeError:
    # Python 2.4 through 2.6?
    def isgeneratorfunction(obj):
        # Magic copied from inspect.py in Python 2.7.
        CO_GENERATOR = 0x20
        if inspect.isfunction(obj):
            return obj.func_code.co_flags & CO_GENERATOR
        else:
            return False

def product(values):
    """Return the product of multiplying all the values.

    >>> product([3, 4, 5, 10])
    600
    >>> product([])
    1

    """
    return reduce(operator.mul, values, 1)


# === Tests ====

# This is a magic function which automatically loads doctests and
# creates unit tests from them. It only works in Python 2.7 or better,
# older versions will ignore it.
def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite())
    tests.addTests(doctest.DocTestSuite(pyprimes))
    tests.addTests(doctest.DocTestSuite(awful))
    tests.addTests(doctest.DocTestSuite(compat23))
    tests.addTests(doctest.DocTestSuite(factors))
    tests.addTests(doctest.DocTestSuite(probabilistic))
    tests.addTests(doctest.DocTestSuite(sieves))
    tests.addTests(doctest.DocTestSuite(strategic))
    tests.addTests(doctest.DocTestSuite(utilities))
    return tests


class TestHelpers(unittest.TestCase):
    def test_product(self):
        self.assertEqual(product([2, 2, 7, 11, 31]), 9548)

    def test_isgeneratorfunction(self):
        self.assertFalse(isgeneratorfunction(None))
        self.assertFalse(isgeneratorfunction(lambda x: None))
        def gen(): yield None
        self.assertTrue(isgeneratorfunction(gen))


class PrimesMixin:
    def check_primes_are_prime(self, prime_checker):
        """Check that primes are detected as prime."""
        for n in PRIMES:
            self.assertTrue(prime_checker(n))

    def check_composites_are_not_prime(self, prime_checker):
        """Check that composites are not detected as prime."""
        composites = set(range(-100, max(PRIMES)+1)) - set(PRIMES)
        for n in composites:
            self.assertFalse(prime_checker(n))

    def check_against_known_prime_list(self, prime_maker):
        """Check that generator produces the first 100 primes."""
        it = prime_maker()
        primes = [next(it) for _ in range(100)]
        self.assertEqual(primes, PRIMES)

    def check_is_generator(self, func):
        """Check that func is a generator function."""
        self.assertTrue(isgeneratorfunction(func))
        it = func()
        self.assertTrue(it is iter(it))
        try:
            isgenerator = inspect.isgenerator
        except AttributeError:
            pass
        else:
            self.assertTrue(isgenerator(it))


class TestMetadata(unittest.TestCase):
    """Check metadata for the pyprimes package."""
    private_message = "private implementation detail"

    def test_module_docstrings(self):
        # Test that modules define a docstring.
        modules = [pyprimes, awful, factors, sieves, probabilistic]
        if __name__ == '__main__':
            # Include this test module when being run as the main module.
            import __main__
            modules.append(__main__)
        for module in modules:
            assert hasattr(module, '__doc__')
            self.assertTrue(isinstance(module.__doc__, str),
                            'module %s has no docstring' % module.__name__)

    def test_private_modules_are_documented_as_such(self):
        self.assertTrue(self.private_message in __doc__)
        private_modules = (compat23,)
        for module in private_modules:
            self.assertTrue(self.private_message in module.__doc__)

    def check_contents_of_all(self, module):
        """Check everything in __all__ exists and is public."""
        for name in module.__all__:
            # No private names in __all__:
            self.assertFalse(name.startswith("_"),
                'private name "%s" in %s.__all__' % (name, module.__name__)
                )
            # And anything in __all__ must exist:
            self.assertTrue(hasattr(module, name),
                'missing name "%s" in %s.__all__' % (name, module.__name__)
                )

    def check_existence_of_metadata(self, module, expected):
        for meta in expected:
            self.assertTrue(hasattr(module, meta),
                    "%s not present in module %s" % (meta, module.__name__))

    def test_meta(self):
        # Test the existence of metadata.
        for module in (pyprimes, factors, sieves, probabilistic):
            self.check_existence_of_metadata(module, ['__all__'])
            self.check_contents_of_all(module)
        additional_metadata = ["__version__", "__date__", "__author__",
                             "__author_email__"]
        self.check_existence_of_metadata(pyprimes, additional_metadata)


class Compat23_Test(unittest.TestCase):
    """Test suite for the compatibility module."""

    def check_returns_iterator(self, func, *args, **kw):
        """Check that func(*args, **kw) returns an iterator, not a list."""
        obj = func(*args, **kw)
        nm = func.__name__
        self.assertFalse(isinstance(obj, list), "%s returns a list" % nm)
        self.assertTrue(obj is iter(obj))

    def test_next(self):
        # Test basic use of next.
        next = compat23.next
        it = iter([1, 'a'])
        self.assertEqual(next(it), 1)
        self.assertEqual(next(it), 'a')
        self.assertRaises(StopIteration, next, it)

    def test_next_default(self):
        # Test next with a default argument.
        next = compat23.next
        it = iter([2, 'b'])
        self.assertEqual(next(it, -99), 2)
        self.assertEqual(next(it, -99), 'b')
        self.assertEqual(next(it, -99), -99)
        self.assertRaises(TypeError, next, it, -1, -2)

    def test_range(self):
        range = compat23.range
        self.assertEqual(list(range(5)), [0, 1, 2, 3, 4])
        self.assertEqual(list(range(5, 10)), [5, 6, 7, 8, 9])
        self.assertEqual(list(range(5, 15, 3)), [5, 8, 11, 14])
        if sys.version_info[0] < 3:
            self.assertTrue(range is xrange)

    def test_filter(self):
        self.check_returns_iterator(compat23.filter, None, [1, 2, 3])
        result = compat23.filter(lambda x: x > 100, [1, 2, 101, 102, 3, 103])
        self.assertEqual(list(result), [101, 102, 103])

    def test_zip(self):
        self.check_returns_iterator(compat23.zip, "abc", [1, 2, 3])
        result = compat23.zip("xyz", [10, 11, 12])
        self.assertEqual(list(result), [('x', 10), ('y', 11), ('z', 12)])

    def test_all(self):
        self.assertTrue(compat23.all([1, 2, 3, 4]))
        self.assertTrue(compat23.all([]))
        self.assertFalse(compat23.all([1, 2, 0, 4]))

    def test_compress(self):
        self.check_returns_iterator(compat23.compress, "abc", [1, 0, 1])
        actual = ' '.join(compat23.compress('ABCDEF', [1,0,1,0,1,1]))
        self.assertEqual(actual, "A C E F")

    def test_reduce(self):
        values = [1, 2, 3, 3, 6, 8, 9, 9, 12, 15, 0, 2, 7]
        self.assertEqual(reduce(operator.add, values), sum(values))


class Awful_Test(unittest.TestCase, PrimesMixin):
    """Test suite for the pyprimes.awful module."""

    def test_isprime(self):
        self.check_primes_are_prime(awful.isprime)
        self.check_composites_are_not_prime(awful.isprime)

    def test_isprime_regex(self):
        self.check_primes_are_prime(awful.isprime_regex)
        self.check_composites_are_not_prime(awful.isprime_regex)

    def test_primes0(self):
        f = awful.primes0
        self.check_is_generator(f)
        self.check_against_known_prime_list(f)

    def test_primes1(self):
        f = awful.primes1
        self.check_is_generator(f)
        self.check_against_known_prime_list(f)

    def test_primes2(self):
        f = awful.primes2
        self.check_is_generator(f)
        self.check_against_known_prime_list(f)

    def test_primes3(self):
        f = awful.primes3
        self.check_is_generator(f)
        self.check_against_known_prime_list(f)

    def test_turner(self):
        f = awful.turner
        self.check_is_generator(f)
        self.check_against_known_prime_list(f)

    def test_primes4(self):
        f = awful.primes4
        self.check_is_generator(f)
        self.check_against_known_prime_list(f)


class Sieves_Test(unittest.TestCase, PrimesMixin):
    """Test suite for the pyprimes.sieves module."""

    def primes_below(self, n):
        """Return prime numbers from PRIMES global up to and including n."""
        for i,p in enumerate(PRIMES):
            if p > n:
                return PRIMES[:i]
        return PRIMES

    def test_erat_returns_list(self):
        self.assertTrue(isinstance(sieves.erat(10), list))

    def test_erat(self):
        for i in range(2, 544):
            self.assertEqual(sieves.erat(i), self.primes_below(i))

    def test_erat_empty(self):
        # Check that erat() returns an empty list for values below 2.
        for i in (1, 0, -1, -17):
            self.assertEqual(sieves.erat(i), [])

    def test_best_sieve(self):
        f = sieves.best_sieve
        self.check_is_generator(f)
        self.check_against_known_prime_list(f)

    def test_cookbook(self):
        f = sieves.cookbook
        self.check_is_generator(f)
        self.check_against_known_prime_list(f)

    def test_croft(self):
        f = sieves.croft
        self.check_is_generator(f)
        self.check_against_known_prime_list(f)

    def test_sieve(self):
        f = sieves.sieve
        self.check_is_generator(f)
        self.check_against_known_prime_list(f)

    def test_wheel(self):
        f = sieves.wheel
        self.check_is_generator(f)
        self.check_against_known_prime_list(f)


class Probabilistic_Mixin:
    """Mixin class for testing probabilistic primality tests.

    The tests in this class should only supply a single argument
    to the probabilistic test function. E.g.:

        is_miller_rabin_probable_prime(n)         # Yes.
        is_miller_rabin_probable_prime(n, bases)  # No!

    For tests with the second argument, see Probabilistic_Extra_Mixin.
    """
    def get_primality_test(self):
        """Returns the primality test function to be tested."""
        raise NotImplementedError("override this in subclasses")

    def test_below_two_are_nonprime(self):
        # Test that values of n below 2 are non-prime.
        isprime = self.get_primality_test()
        for n in range(-7, 2):
            self.assertEqual(isprime(n), 0)

    def test_two_is_prime(self):
        # Test that 2 is definitely prime.
        isprime = self.get_primality_test()
        self.assertEqual(isprime(2), 1)

    def test_primes_are_not_nonprime(self):
        # Test the primality test with primes.
        isprime = self.get_primality_test()
        self.check_primes_are_prime(isprime)

    def test_with_composites(self):
        # Composites should return 0 or 2 but never 1.
        isprime = self.get_primality_test()
        for _ in range(10):
            factors = self.get_factors()
            n = product(factors)
            self.assertTrue(isprime(n) != 1,
                            "composite %d detected as definitely prime" % n)

    def get_factors(self):
        factors = PRIMES*4
        random.shuffle(factors)
        return factors[:10]


class Probabilistic_Extra_Mixin:
    """Mixin class for probabilistic tests involving a second argument."""

    def test_below_two_are_nonprime_with_bases(self):
        # Test that values of n below 2 are non-prime.
        isprime = self.get_primality_test()
        for n in range(-7, 2):
            self.assertEqual(isprime(n, 4), 0)
            self.assertEqual(isprime(n, (3, 5)), 0)

    def test_primes_are_not_nonprime_with_bases(self):
        # Since there are no false negatives, primes will never test as
        # composite, no matter what bases are used.
        isprime = self.get_primality_test()
        for p in PRIMES[1:]:  # Skip prime 2.
            bases = list(range(1, p))
            random.shuffle(bases)
            bases = tuple(bases[:10])
            self.assertEqual(isprime(p, bases), 2)

    def test_with_composites_with_bases(self):
        # Composites should return 0 or 2 but never 1.
        isprime = self.get_primality_test()
        errmsg = "%d detected as definitely prime with %r"
        for _ in range(10):
            factors = PRIMES[51:]*3
            random.shuffle(factors)
            n = product(factors[:8])
            bases = tuple([random.randint(2, n-1) for _ in range(5)])
            self.assertTrue(isprime(n, bases) != 1, errmsg % (n, bases))


class Fermat_Test(
            unittest.TestCase, Probabilistic_Mixin,
            Probabilistic_Extra_Mixin, PrimesMixin
            ):
    """Test the Fermat primality test function."""

    def get_primality_test(self):
        return probabilistic.is_fermat_probable_prime


class Miller_Rabin_Probable_Test(
            unittest.TestCase, Probabilistic_Mixin,
            Probabilistic_Extra_Mixin, PrimesMixin
            ):
    """Test the Miller-Rabin probabilistic primality test."""

    def get_primality_test(self):
        return probabilistic.is_miller_rabin_probable_prime

    def test_composites_with_known_liars(self):
        # These values have come from this email:
        # https://gmplib.org/list-archives/gmp-discuss/2005-May/001652.html
        isprime = self.get_primality_test()
        N = 1502401849747176241
        # Composite, but 2 through 11 are all M-R liars.
        # Lowest witness is 12.
        self.assertEqual(isprime(N, tuple(range(2, 12))), 2)
        self.assertEqual(isprime(N, 12), 0)
        N = 341550071728321
        # Composite, but 2 through 22 are all M-R liars.
        # Lowest witness is 23.
        self.assertEqual(isprime(N, tuple(range(2, 23))), 2)
        self.assertEqual(isprime(N, 23), 0)


class Miller_Rabin_Definite_Test(
            unittest.TestCase, Probabilistic_Mixin, PrimesMixin):
    """Test the Miller-Rabin definite primality test.

    Even though this is a definite test and non-probabilistic, we can
    use the Probabilistic_Mixin tests.
    """

    def get_primality_test(self):
        return probabilistic.is_miller_rabin_definite_prime

    def get_factors(self):
        factors = super(Miller_Rabin_Definite_Test, self).get_factors()
        return factors[:6]

    def test_failure(self):
        # Test the definitive version of M-R raises when it cannot be sure.
        isprime = self.get_primality_test()
        self.assertRaises(ValueError, isprime, 2**100-1)


class Is_Probable_Test(unittest.TestCase, PrimesMixin):
    """Test the is_probable_prime function."""

    def setUp(self):
        # Prepare a randomized collection of moderately-sized primes.
        # Source: http://primes.utm.edu/lists/small/millions/
        primes = [104395303, 122949823, 122949829, 141650939, 141650963,
                  160481183, 160481219, 179424673, 179424691, 198491317,
                  198491329, 217645177, 217645199, 236887691, 236887699,
                  256203161, 256203221, 275604541, 275604547, 295075147,
                  295075153, 314606869, 314606891, 334214459, 334214467,
                  353868013, 353868019, 373587883, 373587911, 393342739,
                  393342743, 413158511, 413158523, 433024223, 433024253,
                  452930459, 452930477, 472882027, 472882049, 492876847,
                  492876863, 512927357, 512927377, 533000389, 533000401,
                  553105243, 553105253, 573259391, 573259433, 593441843,
                  593441861, 613651349, 613651369, 633910099, 633910111,
                  654188383, 654188429, 674506081, 674506111, 694847533,
                  694847539, 715225739, 715225741, 735632791, 735632797,
                  756065159, 756065179, 776531401, 776531419, 797003413,
                  797003437, 817504243, 817504253, 838041641, 838041647,
                  858599503, 858599509, 879190747, 879190841, 899809343,
                  899809363, 920419813, 920419823, 941083981, 941083987,
                  961748927, 961748941, 982451653, ]
        assert len(primes) > 75  # More than enough for the tests we do.
        random.shuffle(primes)
        self.primes = iter(primes)

    def test_is_probable_prime(self):
        # Basic tests for is_probable_prime.
        ipp = probabilistic.is_probable_prime
        self.check_primes_are_prime(ipp)
        self.check_composites_are_not_prime(ipp)

    def test_moderate_primes(self):
        # Test is_probable_prime with a few moderate-sized primes.
        for p in itertools.islice(self.primes, 15):
            self.assertEqual(probabilistic.is_probable_prime(p), 1)
        # FIXME we should have at least one detailed test case to cover
        # each of the if...elif blocks in the is_probable_prime function.
        # That involves having detailed knowledge of what blocks are
        # called with what input arguments.

    def test_moderate_composites(self):
        # Test is_probable_prime with moderate-sized composites.
        for i in range(10):
            # We should not run out of primes here. If we do, it's a bug
            # in the test.
            p, q = next(self.primes), next(self.primes)
            n = p*q
            assert n < 2**60, "n not in deterministic range for i_p_p"
            self.assertEqual(probabilistic.is_probable_prime(n), 0)

    def test_primes(self):
        # Test the prime generator based on is_probable_prime.
        f = probabilistic.primes
        self.check_is_generator(f)
        self.check_against_known_prime_list(f)


class Factors_Test(unittest.TestCase):
    """Test suite for the factors module."""

    def test_factors_basic(self):
        # Basic test for factors.factorise.
        self.assertEqual(factors.factorise(2*7*31*31*101), [2, 7, 31, 31, 101])

    def test_factors_random(self):
        # Test the factors.factorise function with a random number.
        numfactors = random.randint(1, 8)
        values = [random.choice(PRIMES) for _ in range(numfactors)]
        values.sort()
        n = product(values)
        self.assertEqual(factors.factorise(n), values)

    def test_factors_special(self):
        # Test the factors.factorise function with special values.
        self.assertEqual(factors.factorise(0), [0])
        self.assertEqual(factors.factorise(1), [1])
        self.assertEqual(factors.factorise(-1), [-1])

    def test_factors_negative(self):
        # Test the factors.factorise function with negative values.
        f = factors.factorise
        for n in range(40, 50):
            assert n != -1
            self.assertEqual(f(-n), [-1] + f(n))

    def test_factors(self):
        values = [2, 3, 3, 3, 3, 29, 29, 31, 101, 137]
        n = product(values)
        expected = [(2, 1), (3, 4), (29, 2), (31, 1), (101, 1), (137, 1)]
        actual = list(factors.factors(n))
        self.assertEqual(actual, expected)

    def test_factors_extra(self):
        # Test factorise with _EXTRA_CHECKS enabled.
        if __debug__:
            self.assertTrue(hasattr(factors, '_EXTRA_CHECKS'))
            # Monkey-patch the factors module to ensure the extra checks
            # are exercised.
            save = factors._EXTRA_CHECKS
            factors._EXTRA_CHECKS = True
        else:
            self.assertFalse(hasattr(factors, '_EXTRA_CHECKS'))
        try:
            values = [2, 2, 2, 3, 17, 17, 29, 31, 61, 61, 103, 227, 227]
            n = product(values)
            expected = [(2, 3), (3, 1), (17, 2), (29, 1), (31, 1),
                        (61, 2), (103, 1), (227, 2)]
            actual = list(factors.factors(n))
            self.assertEqual(actual, expected)
        finally:
            if __debug__:
                factors._EXTRA_CHECKS = save


class PyPrimesTest(unittest.TestCase, PrimesMixin):
    """Test suite for the __init__ module."""

    def test_primes_basic(self):
        # Basic tests for the prime generator.
        self.check_against_known_prime_list(pyprimes.primes)

    def test_primes_start(self):
        # Test the prime generator with start argument only.
        expected = [211, 223, 227, 229, 233, 239, 241, 251,
                    257, 263, 269, 271, 277, 281, 283, 293]
        assert len(expected) == 16
        it = pyprimes.primes(200)
        values = [next(it) for _ in range(16)]
        self.assertEqual(values, expected)

    def test_primes_end(self):
        # Test the prime generator with end argument only.
        expected = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        it = pyprimes.primes(end=50)
        self.assertEqual(list(it), expected)

    def test_primes_start_is_inclusive(self):
        # Start argument to primes() is inclusive.
        n = 211
        assert pyprimes.is_prime(n)
        it = pyprimes.primes(start=n)
        self.assertEqual(next(it), n)

    def test_primes_end_is_exclusive(self):
        # End argument to primes() is exclusive.
        n = 211
        assert pyprimes.is_prime(n)
        it = pyprimes.primes(end=n)
        values = list(it)
        self.assertEqual(values[-1], 199)
        assert pyprimes.next_prime(199) == n

    def test_primes_end_none(self):
        # Check that None is allowed as an end argument.
        it = pyprimes.primes(end=None)
        self.assertEqual(next(it), 2)

    def test_primes_start_end(self):
        # Test the prime generator with both start and end arguments.
        expected = [401, 409, 419, 421, 431, 433, 439, 443, 449,
                    457, 461, 463, 467, 479, 487, 491, 499]
        values = list(pyprimes.primes(start=400, end=500))
        self.assertEqual(values, expected)

    def test_is_prime(self):
        # Basic tests for is_prime.
        self.check_primes_are_prime(pyprimes.is_prime)
        self.check_composites_are_not_prime(pyprimes.is_prime)

    def test_trial_division(self):
        # Basic tests for trial_division.
        self.check_primes_are_prime(pyprimes.trial_division)
        self.check_composites_are_not_prime(pyprimes.trial_division)

    def test_next_prime(self):
        self.assertEqual(pyprimes.next_prime(122949823), 122949829)
        self.assertEqual(pyprimes.next_prime(961748927), 961748941)

    def test_prev_prime(self):
        self.assertEqual(pyprimes.prev_prime(122949829), 122949823)
        self.assertEqual(pyprimes.prev_prime(961748941), 961748927)
        # self.assertEqual(pyprimes.prev_prime(3), 2)
        self.assertRaises(ValueError, pyprimes.prev_prime, 2)

    def test_nprimes(self):
        it = pyprimes.nprimes(100)
        self.assertTrue(it is iter(it))
        self.assertEqual(list(it), PRIMES)

    def test_nth_primes(self):
        self.assertEqual(pyprimes.nth_prime(100), PRIMES[-1])
        self.assertRaises(ValueError, pyprimes.nth_prime, 0)
        self.assertRaises(ValueError, pyprimes.nth_prime, -1)

    def test_prime_count(self):
        self.assertEqual(pyprimes.prime_count(PRIMES[-1]), len(PRIMES))
        # Table of values from http://oeis.org/A000720
        # plus one extra 0 to adjust for Python's 0-based indexing.
        expected = [
            0, 0, 1, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 8, 8,
            8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12,
            12, 12, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 16, 16,
            16, 16, 16, 16, 17, 17, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19,
            20, 20, 21, 21, 21, 21, 21, 21]
        for i, count in enumerate(expected):
            self.assertEqual(pyprimes.prime_count(i), count)

    def test_prime_count_tens(self):
        # Test prime_count function with powers of ten.
        # Values come from:
        #   http://mathworld.wolfram.com/PrimeCountingFunction.html
        #   http://oeis.org/A006880
        expected = [0, 4, 25, 168, 1229, 9592, 78498]
        for i, count in enumerate(expected):
            self.assertEqual(pyprimes.prime_count(10**i), count)

    def test_prime_partial_sums(self):
        it = pyprimes.prime_partial_sums()
        self.assertTrue(it is iter(it))
        # Table of values from http://oeis.org/A007504
        expected = [
            0, 2, 5, 10, 17, 28, 41, 58, 77, 100, 129, 160, 197, 238, 281,
            328, 381, 440, 501, 568, 639, 712, 791, 874, 963, 1060, 1161,
            1264, 1371, 1480, 1593, 1720, 1851, 1988, 2127, 2276, 2427,
            2584, 2747, 2914, 3087, 3266, 3447, 3638, 3831, 4028, 4227,
            4438, 4661, 4888]
        actual = [next(it) for _ in range(len(expected))]
        self.assertEqual(actual, expected)

    def test_prime_sum(self):
        # Test the prime_sum function by comparing it to prime_partial_sums.
        it = pyprimes.prime_partial_sums()
        for i in range(100):
            expected = next(it)
            actual = pyprimes.prime_sum(i)
            self.assertEqual(actual, expected)


class StrategicTest:  ###### FIXME     (unittest.TestCase, PrimesMixin):
    """Test suite for the strategic module."""

    def test_primes_basic(self):
        # Basic tests for the prime generator.
        self.check_against_known_prime_list(strategic.primes)
        self.check_is_generator(strategic.primes)

    def test_primes_start(self):
        # Test the prime generator with start argument only.
        expected = [211, 223, 227, 229, 233, 239, 241, 251,
                    257, 263, 269, 271, 277, 281, 283, 293]
        assert len(expected) == 16
        it = pyprimes.primes(200)
        values = [next(it) for _ in range(16)]
        self.assertEqual(values, expected)

    def test_primes_end(self):
        # Test the prime generator with end argument only.
        expected = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        it = pyprimes.primes(end=50)
        self.assertEqual(list(it), expected)

    def test_primes_start_is_inclusive(self):
        # Start argument to primes() is inclusive.
        n = 211
        assert pyprimes.is_prime(n)
        it = pyprimes.primes(start=n)
        self.assertEqual(next(it), n)

    def test_primes_end_is_exclusive(self):
        # End argument to primes() is exclusive.
        n = 211
        assert pyprimes.is_prime(n)
        it = pyprimes.primes(end=n)
        values = list(it)
        self.assertEqual(values[-1], 199)
        assert pyprimes.next_prime(199) == n

    def test_primes_end_none(self):
        # Check that None is allowed as an end argument.
        it = pyprimes.primes(end=None)
        self.assertEqual(next(it), 2)

    def test_primes_start_end(self):
        # Test the prime generator with both start and end arguments.
        expected = [401, 409, 419, 421, 431, 433, 439, 443, 449,
                    457, 461, 463, 467, 479, 487, 491, 499]
        values = list(pyprimes.primes(start=400, end=500))
        self.assertEqual(values, expected)

    def test_primes_with_generator(self):
        # Test the prime generator with a custom generator.
        # These aren't actually primes.
        def gen():
            yield 3; yield 3; yield 5; yield 9; yield 0
        it = pyprimes.primes(strategy=gen)
        self.assertEqual(list(it), [3, 3, 5, 9, 0])

    def test_is_prime(self):
        # Basic tests for is_prime.
        self.check_primes_are_prime(pyprimes.is_prime)
        self.check_composites_are_not_prime(pyprimes.is_prime)

    def test_trial_division(self):
        # Basic tests for trial_division.
        self.check_primes_are_prime(pyprimes.trial_division)
        self.check_composites_are_not_prime(pyprimes.trial_division)

    def test_next_prime(self):
        self.assertEqual(pyprimes.next_prime(122949823), 122949829)
        self.assertEqual(pyprimes.next_prime(961748927), 961748941)

    def test_prev_prime(self):
        self.assertEqual(pyprimes.prev_prime(122949829), 122949823)
        self.assertEqual(pyprimes.prev_prime(961748941), 961748927)
        # self.assertEqual(pyprimes.prev_prime(3), 2)
        self.assertRaises(ValueError, pyprimes.prev_prime, 2)


def skip_if_too_expensive(testcase):
    """ Decorator to skip a testcase if deemed too expensive for this run. """
    want_expensive = '--do-expensive-tests' in sys.argv
    if want_expensive:
        # import pdb ; pdb.set_trace()
        do_func = testcase
    else:
        do_func = skip("Too expensive for default run")(testcase)
    return do_func


class ExpensiveTests(unittest.TestCase):
    """Test cases that consume a lot of CPU time.

    By default, these tests are not run. To run them, pass:

        --do-expensive-tests

    on the command line.

    BE WARNED THAT THESE TESTS MAY TAKE MANY HOURS (DAYS?) TO RUN.
    """

    @skip_if_too_expensive
    def test_prime_count_tens_big(self):
        # See also PyPrimesTest.test_prime_count_tens.
        self.assertEqual(pyprimes.prime_count(10**7), 664579)
        self.assertEqual(pyprimes.prime_count(10**8), 5761455)

    @skip_if_too_expensive
    def test_bertelsen(self):
        # http://mathworld.wolfram.com/BertelsensNumber.html
        result = pyprimes.prime_count(10**9)
        self.assertNotEqual(result, 50847478,
            "prime_count returns the erronous Bertelsen's Number")
        self.assertEqual(result, 50847534)


class UtilitiesTests(unittest.TestCase):
    """Test suite for the pyprimes.utilities module."""
    # FIXME: add tests for isqrt or instrumentation.

    def test_filter_between(self):
        filter_between = utilities.filter_between
        values = [1, 2, 3, 4, 3, 3, 2, 5, 6, 7, 8, 9, 0]
        it = filter_between(values)
        self.assertEqual(list(it), values)
        it = filter_between(values, start=4)  # No end.
        self.assertEqual(list(it), [4, 3, 3, 2, 5, 6, 7, 8, 9, 0])
        it = filter_between(values, end=6)  # No start.
        self.assertEqual(list(it), [1, 2, 3, 4, 3, 3, 2, 5])
        it = filter_between(values, start=3, end=7)  # Both start and end.
        self.assertEqual(list(it), [3, 4, 3, 3, 2, 5, 6])


class RegressionTests(unittest.TestCase):
    """Regression tests for fixed bugs."""

    @skip("FIXME -- this hangs")
    def test_prev_prime_from_3(self):
        # Regression test for the case of prev_prime(3) --> 2.
        for prover in ():
            self.assertEqual(strategic.prev_prime(prover, 3), 2)
        self.assertEqual(pyprimes.prev_prime(3), 2)


if __name__ == '__main__':
    if '--do-expensive-tests' in sys.argv:
        # By this point the functions to skip have all been decorated.
        # Remove it from sys.argv, otherwise unittest will complain.
        sys.argv.remove('--do-expensive-tests')
    unittest.main()


# Evil Miller-Rabin value?
# 8038374574536394912570796143419421081388376882875581458374889175222974273765333652186502336163960045457915042023603208766569966760987284043965408232928738791850869166857328267761771029389697739470167082304286871099974399765441448453411558724506334092790222752962294149842306881685404326457534018329786111298960644845216191652872597534901
# https://gmplib.org/list-archives/gmp-discuss/2005-May/001651.html
#
# => N-1 = 2**2 * 3**4 * 5**2 * 641 * 12107 * M
# => M+1 = 2**4 * 3**2 * 307 * 4817 * K
# => K-1 = 2 * 37 * 53 * ...

# -*- coding: utf-8 -*-

##  Part of the pyprimes.py package.
##
##  Copyright © 2014 Steven D'Aprano.
##  See the file __init__.py for the licence terms for this software.

from __future__ import division

import itertools
import operator

try:
    # If it is available, we prefer to use partial for speed.
    from functools import partial
except ImportError:
    partial = None


def filter_between(iterable, start=None, end=None):
    """Yield items from iterable in the range(start, end).

    Returns an iterator from the given iterable, optionally filtering
    items at the beginning and end. If ``start`` is not None, values are
    silently dropped from the beginning of the iterator until the first
    item which equals or exceeds ``start``:

    >>> it = filter_between("aaabbbcdeabcd", start="c")
    >>> list(it)
    ['c', 'd', 'e', 'a', 'b', 'c', 'd']

    If ``end`` is not None, the first value which equals or exceeds ``end``
    halts the iterator:

    >>> it = filter_between("aaabbbcdeabcd", end="c")
    >>> list(it)
    ['a', 'a', 'a', 'b', 'b', 'b']


    FIXME
    If ``start`` or ``end``
    are not None, then the iterator silently drops all values until the
    first value that equals or exceeds ``start``, and halts at the first
    value that exceeds ``end``. The ``start`` value is inclusive, the
    ``end`` value is exclusive.

    For example:

    >>> it = filter_between("abcdefgh", start="c", end="g")
    >>> list(it)
    ['c', 'd', 'e', 'f']



    """
    iterator = iter(iterable)
    if start is not None:
        # Drop values strictly less than start.
        if partial is None:
            drop = lambda p: p < start  # Bite me, PEP 8.
        else:
            # We want to skip over any values "v < start", but since
            # partial assigns operands from the left, we have to write
            # that as "start > p".
            drop = partial(operator.gt, start)
        iterator = itertools.dropwhile(drop, iterator)
    if end is not None:
        # Take values strictly less than end.
        if partial is None:
            take = lambda p: p < end  # Bite me, PEP 8.
        else:
            # We want to halt at the first value "v >= end", which means
            # we take values "p < end". Since partial assigns operands
            # from the left, we write that as "end > p".
            take = partial(operator.gt, end)
        iterator = itertools.takewhile(take, iterator)
    return iterator


# Every integer between 0 and MAX_EXACT inclusive
MAX_EXACT = 9007199254740991


# Get the number of bits needed to represent an int in binary.
try:
    _bit_length = int.bit_length
except AttributeError:
    def _bit_length(n):
        if n == 0:
            return 0
        elif n < 0:
            n = -n
        assert n >= 1
        numbits = 0
        # Accelerator for larger values of n.
        while n > 2**64:
            numbits += 64; n >>= 64
        while n:
            numbits += 1; n >>= 1
        return numbits


def isqrt(n):
    """Return the integer square root of n.

    >>> isqrt(48)
    6
    >>> isqrt(49)
    7
    >>> isqrt(9500)
    97

    Equivalent to floor(sqrt(x)).
    """
    if n < 0:
        raise ValueError('square root not defined for negative numbers')
    elif n <= MAX_EXACT:
        # For speed, we use floating point maths.
        return int(n**0.5)
    return _isqrt(n)

def _isqrt(n):
    # Tested every value of n in the following ranges:
    #   - range(0, 9394201554) (took about ~12.5 hours)
    #   - range(9007154720172961, 9007154885883381)
    #
    if n == 0:
        return 0
    bits = _bit_length(n)
    a, b = divmod(bits, 2)
    x = 2**(a+b)
    while True:
        y = (x + n//x)//2
        if y >= x:
            return x
        x = y


# === Instrumentation used by the probabilistic module ===


try:
    from collections import namedtuple
except ImportError:
    # Probably Python 2.4. If namedtuple is not available, just use a
    # regular tuple instead.
    def namedtuple(name, fields):
        # Ignore the arguments and just hope for the best.
        return tuple


class MethodStats(object):
    """Statistics for individual methods of ``is_probably_prime``.

    Instances are intended to be mapped to a method name in a dict, where
    they record how often the method was able to conclusively determine
    the primality of its argument (that is, by returning 0 or 1 rather
    than 2).

    Instances record three pieces of data:

        hits:
            The number of times the method conclusively determined
            the primality of the argument.

        low:
            The smallest argument that the method has determined
            the primality of the argument.

        high:
            The largest argument that the method has determined
            the primality of the argument.

    E.g. given a mapping ``{"frob": MethodStats(250, 357, 993)}``,
    that indicates that the method ``frob`` determined the primality of its
    argument 250 times (not necessarily distinct arguments), with the
    smallest such argument being 357 and the largest being 993.

    """
    def __init__(self, hits=0, low=None, high=None):
        self.hits = hits
        self.low = low
        self.high = high

    def __repr__(self):
        template = "%s(hits=%d, low=%r, high=%r)"
        name = type(self).__name__
        return template % (name, self.hits, self.low, self.high)

    def update(self, value):
        self.hits += 1
        a, b = self.low, self.high
        if a is None: a = value
        else: a = min(a, value)
        if b is None: b = value
        else: b = max(b, value)
        self.low, self.high = a, b


class Instrument(object):
    """Instrumentation for ``is_probable_prime``.

    Instrument objects have four public attributes recording total counts:

        calls:
            The total number of times the function is successfully called.

        notprime:
            The total number of non-prime results returned.

        prime:
            The total number of definitely prime results returned.

        uncertain:
            The total number of possibly prime results returned.

    """
    def __init__(self, owner, methods):
        self.calls = 0
        self.uncertain = 0
        self.prime = 0
        self.notprime = 0
        self._owner = owner
        self._stats = {}
        for function in methods:
            self._stats[function.__name__] = MethodStats()

    def display(self):
        """Display a nicely formatted version of the instrumentation."""
        print(str(self))

    def __str__(self):
        template = (
            'Instrumentation for %s\n'
            '  - definitely not prime:  %d\n'
            '  - definitely prime:      %d\n'
            '  - probably prime:        %d\n'
            '  - total:                 %d\n'
            '%s\n'
            )
        items = sorted(self._stats.items())
        items = ['%s: %s' % item for item in items if item[1].hits != 0]
        items = '\n'.join(items)
        args = (self._owner, self.notprime, self.prime, self.uncertain, self.calls, items)
        return template % args

    def update(self, name, n, flag):
        assert flag in (0, 1, 2)
        self._stats[name].update(n)
        self.calls += 1
        if flag == 0:
            self.notprime += 1
        elif flag == 1:
            self.prime += 1
        else:
            self.uncertain += 1


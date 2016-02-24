# -*- coding: utf-8 -*-

##  Part of the pyprimes.py package.
##
##  Copyright © 2014 Steven D'Aprano.
##  See the file __init__.py for the licence terms for this software.


"""Python 2 and 3 compatibility layer for the pyprimes package.

This module is considered a private implementation detail and is subject
to change without notice.
"""

from __future__ import division


try:
    import builtins  # Python 3.x.
except ImportError:
    # We're probably running Python 2.x.
    import __builtin__ as builtins


try:
    next = builtins.next
except AttributeError:
    # No next() builtin, so we're probably running Python 2.4 or 2.5.
    def next(iterator, *args):
        if len(args) > 1:
            n = len(args) + 1
            raise TypeError("next expected at most 2 arguments, got %d" % n)
        try:
            return iterator.next()
        except StopIteration:
            if args:
                return args[0]
            else:
                raise


try:
    range = builtins.xrange
except AttributeError:
    # No xrange built-in, so we're probably running Python3
    # where range is already a lazy iterator.
    assert type(builtins.range(3)) is not list
    range = builtins.range


try:
    from itertools import ifilter as filter, izip as zip
except ImportError:
    # Python 3, where filter and zip are already lazy.
    assert type(builtins.filter(None, [1, 2])) is not list
    assert type(builtins.zip("ab", [1, 2])) is not list
    filter = builtins.filter
    zip = builtins.zip


try:
    all = builtins.all
except AttributeError:
    # Likely Python 2.4.
    def all(iterable):
        for element in iterable:
            if not element:
                return False
        return True


try:
    from itertools import compress
except ImportError:
    # Must be Python 2.x, so we need to roll our own.
    def compress(data, selectors):
        """compress('ABCDEF', [1,0,1,0,1,1]) --> A C E F"""
        return (d for d, s in zip(data, selectors) if s)


try:
    reduce = builtins.reduce
except AttributeError:
    from functools import reduce



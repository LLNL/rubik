#!/usr/bin/env python
################################################################################
# Copyright (c) 2012, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory
# Written by Todd Gamblin et al. <tgamblin@llnl.gov>
# LLNL-CODE-599252
# All rights reserved.
#
# This file is part of Rubik. For details, see http://scalability.llnl.gov.
# Please read the LICENSE file for further information.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the disclaimer below.
#
#     * Redistributions in binary form must reproduce the above copyright notice,
#       this list of conditions and the disclaimer (as noted below) in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of the LLNS/LLNL nor the names of its contributors may be
#       used to endorse or promote products derived from this software without
#       specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC, THE
# U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
################################################################################
from rubik.zorder import *
import unittest

class TestZorder(unittest.TestCase):

    def encode_and_decode_shape(self, shape):
        """ Construct a Z encoder for the given shape, then run tests on it."""
        zencoder = ZEncoder.for_shape(shape)
        print zencoder
        for point in np.ndindex(*shape):
            code = zencoder.encode(point)
            decoded = zencoder.decode(code)
            self.assertEqual(
                point, decoded,
                "Error for code got %s but expected %s for code %s" % (decoded, point, b(code)))

    def test_1_dimension(self):
        self.encode_and_decode_shape([1024])

    def test_2_dimensions(self):
        self.encode_and_decode_shape([64,32])

    def test_3_dimensions(self):
        self.encode_and_decode_shape([8,16,32])

    def test_4_dimensions(self):
        self.encode_and_decode_shape([8,4,32,128])

    def test_5_dimensions(self):
        self.encode_and_decode_shape([8,4,16,16,32])

    def test_6_dimensions(self):
        self.encode_and_decode_shape([8,2,8,32,4,4])

if __name__ == "__main__":
    unittest.main()

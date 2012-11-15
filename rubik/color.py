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
"""This module defines the Color class used by rubik.view.
"""

import colorsys

class Color(object):
    """ Really basic color class that provides methods for setting alpha.
    """
    def __init__(self, r, g, b, a=1.0):
        """Constructor with r, g, b, and alpha values."""
        self.value = [r, g, b, a]

    @property
    def r(self):
        return self.value[0]

    @property
    def g(self):
        return self.value[1]

    @property
    def b(self):
        return self.value[2]

    @property
    def a(self):
        return self.value[3]

    def __getitem__(self, index):
        return self.value[index]

    def with_alpha(self, alpha):
        return Color(self.r, self.g, self.b, alpha)

    def lighter(self, amount):
        h, s, v = colorsys.rgb_to_hsv(self.r, self.g, self.b)
        r, g, b = colorsys.hsv_to_rgb(h, s, max(0.0, v - amount))
        return Color(r, g, b, self.a)

    def darker(self, amount):
        h, s, v = colorsys.rgb_to_hsv(self.r, self.g, self.b)
        r, g, b = colorsys.hsv_to_rgb(h, s, min(1.0, v + amount))
        return Color(r, g, b, self.a)

    def mix(self, color, ratio=.5):
        mixed = [s*(1-ratio) + c*ratio for s,c in zip(self.value, color.value)]
        return Color(*mixed)

    def __repr__(self):
        return "<Color: %.2f, %.2f, %.2f, %.2f>" % tuple(self.value)

    def __iter__(self):
        """Iterator over red, green, blue, and alpha components."""
        for val in self.value:
            yield val


class ColorMapper(object):
    def __init__(self, color_list, **kwargs):
        self.colors = color_list
        self.mapping = {}
        self.next_color = kwargs.get("start_color", 0)

    def __getitem__(self, object):
        if not object in self.mapping:
            self.mapping[object] = self.colors[self.next_color]
            self.next_color += 1
            self.next_color %= len(self.colors)
        return self.mapping[object]

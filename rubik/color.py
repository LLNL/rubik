"""This module defines the Color class.
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
    def __init__(self, color_list):
        self.colors = color_list
        self.mapping = {}
        self.next_color = 0

    def __getitem__(self, object):
        if not object in self.mapping:
            self.mapping[object] = self.colors[self.next_color]
            self.next_color += 1
            self.next_color %= len(self.colors)
        return self.mapping[object]

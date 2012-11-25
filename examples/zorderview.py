#!/usr/bin/env python

import sys
import rubik.view as rv
from rubik import *

if __name__ == "__main__":
    p = box([8, 8, 8])
    rv.color(p, start_color=2)
    p.zorder()

    rv.viewbox(p)

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
"""\
Support for nicely indented GL sections in python using Python's with statement.
Author:
    Todd Gamblin, tgamblin@llnl.gov
"""
from contextlib import contextmanager
from OpenGL.GL import *

@contextmanager
def glSection(type):
    glBegin(type)
    yield
    glEnd()

@contextmanager
def glMatrix():
    glPushMatrix()
    yield
    glPopMatrix()

@contextmanager
def attributes(*glBits):
    for bit in glBits:
        glPushAttrib(bit)
    yield
    for bit in glBits:
        glPopAttrib()

@contextmanager
def enabled(*glBits):
    for bit in glBits:
        glEnable(bit)
    yield
    for bit in glBits:
        glDisable(bit)

@contextmanager
def disabled(*glBits):
    for bit in glBits:
        glDisable(bit)
    yield
    for bit in glBits:
        glEnable(bit)

class DisplayList(object):
    """Use this to turn some rendering function of yours into a DisplayList,
       without all the tedious setup.

       Suppose you have a rendering function that looks like this:
           def myRenderFunction():
               # ... Do rendering stuff ...

       And you want to use it to build a displayList:
           myList = DisplayList(myRenderFunction)

       Now to call the list, just do this:
           myList()

       If you want the render function to get called again the next time your
       display list is called, you call "update()" on the display list:
           myList.update()

       Now the next call to myList() will generate you a new display list by
       running your render function again.  After that, it just calls the
       new list.

       Note that your render function *can* take args, and you can pass them
       when you call the display list, but they're only really passed along
       when an update is needed, so it's probably best to just not make
       render functions with arguments.

       TODO: this should probably have ways to keep around active list ids so
       TODO: that they can be freed up at the end of execution.
    """
    def __init__(self, renderFunction):
        self.renderFunction = renderFunction
        self.needsUpdate = True
        self.listId = None

    def update(self):
        self.needsUpdate = True

    def __call__(self, *args):
        if self.needsUpdate:
            if self.listId:
                glDeleteLists(self.listId, 1)
            self.listId = glGenLists(1)

            glNewList(self.listId, GL_COMPILE_AND_EXECUTE)
            self.renderFunction(*args)
            glEndList()
            self.needsUpdate = False
        else:
            glCallList(self.listId)

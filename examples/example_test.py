"""
===============
Verbing a thing
===============

How to <verb> <active tense> <does something>.

The example uses <packages> to <do something> and <other package> to <do other thing>.
"""

import matplotlib.pyplot as plt
import numpy as np

##############################################################################
# This code block is executed, although it produces no output. Lines starting
# with a simple hash are code comment and get treated as part of the code
# block. To include this new comment string we started the new block with a
# long line of hashes.
#
# The sphinx-gallery parser will assume everything after this splitter and that
# continues to start with a **comment hash and space** (respecting code style)
# is text that has to be rendered in
# html format. Keep in mind to always keep your comments always together by
# comment hashes. That means to break a paragraph you still need to comment
# that line break.
#
# In this example the next block of code produces some plotable data. Code is
# executed, figure is saved and then code is presented next, followed by the
# inlined figure.

x = np.linspace(-np.pi, np.pi, 300)
xx, yy = np.meshgrid(x, x)
z = np.cos(xx) + np.cos(yy)

plt.figure()
plt.imshow(z)
plt.colorbar()
plt.xlabel("$x$")
plt.ylabel("$y$")

plt.show()

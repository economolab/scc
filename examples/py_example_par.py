import numpy as np
import numexpr as ne

a = np.arange(1000000)

# use  numexpr library
ne.evaluate('sin(a)')

# let numpy auto-paralleze
np.sin(a)


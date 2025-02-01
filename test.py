import matplotlib.pyplot as plt

import Data
M = 1
T = 1

data = Data.LinearData(M, T)
data.setM(-2.5)
data.setT(-1)

fig     = plt.figure()
ax      = plt.subplot(111)
data    = ax.plot(data.getX(), data.getY())

ax.set_xlabel("")
ax.set_ylabel("")
ax.set_title("")

ax.set_xlim(-15, 15)
ax.set_ylim(-30, 30)
ax.grid("both")


plt.show()


#Todo:
# Test time for change t vs generate new
# same for m
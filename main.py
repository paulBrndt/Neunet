import numpy as np
import matplotlib.pyplot as plt

from Data import LinearData

from Network import Network
from FCLayer import FCLayer
from ActivationLayer import ActivationLayer

from commonMath import activationFunctions, lossFunctions


data = LinearData(m = 1, t = 0, minX=-1, maxX=1)

net = Network()
net.add(FCLayer(2, 3))
net.add(ActivationLayer(activationFunctions.tanh, activationFunctions.tanhPrime))
net.add(FCLayer(3, 1))
net.add(ActivationLayer(activationFunctions.tanh, activationFunctions.tanhPrime))


net.use(lossFunctions.mse, lossFunctions.msePrime)

net.fit(data.getX(), data.getY(), epochs=100, learningRate=0.05)

out = net.predict(data.getX())
out = np.array(out).squeeze()

print(out)



fig     = plt.figure()
ax      = plt.subplot(111)
true    = ax.plot(data.getX(), data.getY(), label="true")
predict = ax.plot(data.getX(), out, label="predicted")

ax.set_xlabel("")
ax.set_ylabel("")
ax.set_title("True values vs predicted")
ax.legend()

# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
ax.grid("both")

plt.show()
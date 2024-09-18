import math
import matplotlib.pyplot as plt
l = []
for i in range(50):
    l.append(math.cos(i/50))
plt.plot(l)
plt.show()
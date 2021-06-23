import numpy as np
import matplotlib.pyplot as plt

with open("eval/eval_loss_avg.txt") as f:
    data = f.read()
    data = np.array(data.strip().split('\n')).astype(float)
    print(data)
    plt.plot(np.arange(0,len(data)),data, label = "eval loss avg")
with open("eval/train_loss_avg.txt") as f:
    data = f.read()
    data = np.array(data.strip().split('\n')).astype(float)
    plt.plot(np.arange(0,len(data)),data, label = "train loss avg")
plt.legend()
plt.show()
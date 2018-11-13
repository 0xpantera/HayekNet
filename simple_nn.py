from tensor import Tensor
import numpy as np
np.random.seed(0)

data = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), autograd=True)
target = Tensor(np.array([[0], [1], [0], [1]]), autograd=True)

w = list()
w.append(Tensor(np.random.rand(2, 3), autograd=True))
w.append(Tensor(np.random.rand(3, 1), autograd=True))

for i in range(10):

    # Predict
    pred = data.matmul(w[0]).matmul(w[1])

    # Compare
    loss = ((pred - target)*(pred - target)).sum(0)

    # Learn
    loss.backward(Tensor(np.ones_like(loss.data)))

    for w_ in w:
        w_.data -= w_.grad.data * 0.1
        w_.grad.data *= 0

    print(loss)

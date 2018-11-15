from tensor import Tensor, SGD, Sequential, Linear, MSELoss, Tanh, Sigmoid
from tensor import Embedding
import numpy as np
np.random.seed(0)

data = Tensor(np.array([1, 2, 1, 2]), autograd=True)
target = Tensor(np.array([[0], [1], [0], [1]]), autograd=True)

embed = Embedding(5, 3)
model = Sequential([embed, Tanh(), Linear(3, 1), Sigmoid()])
criterion = MSELoss()

optim = SGD(parameters=model.get_parameters(), alpha=0.5)

for i in range(10):

    # Predict
    pred = model.forward(data)

    # Compare
    loss = criterion.forward(pred, target)

    # Learn
    loss.backward()
    optim.step()

    print(loss)

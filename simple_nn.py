from tensor import Tensor, SGD, Sequential, Linear, MSELoss, Tanh, Sigmoid
from tensor import Embedding, CrossEntropyLoss
import numpy as np
np.random.seed(0)

data = Tensor(np.array([1, 2, 1, 2]), autograd=True)
target = Tensor(np.array([0, 1, 0, 1]), autograd=True)

model = Sequential([Embedding(3, 3), Tanh(), Linear(3, 4)])
criterion = CrossEntropyLoss()

optim = SGD(parameters=model.get_parameters(), alpha=0.1)

for i in range(10):

    # Predict
    pred = model.forward(data)

    # Compare
    loss = criterion.forward(pred, target)

    # Learn
    loss.backward()
    optim.step()

    print(loss)

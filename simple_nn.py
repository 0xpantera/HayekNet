from tensor import Tensor, SGD, Sequential, Linear, MSELoss
import numpy as np
np.random.seed(0)

data = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), autograd=True)
target = Tensor(np.array([[0], [1], [0], [1]]), autograd=True)

model = Sequential([Linear(2, 3), Linear(3, 1)])
criterion = MSELoss()

optim = SGD(parameters=model.get_parameters(), alpha=0.1)

for i in range(10):

    # Predict
    pred = model.forward()

    # Compare
    loss = criterion.forward(pred, target)

    # Learn
    loss.backward()
    optim.step()

    print(loss)

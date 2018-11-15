import numpy as np


class Tensor(object):

    def __init__(self, data,
                 autograd=False,
                 creators=None,
                 creation_op=None,
                 id=None):

        self.data = np.array(data)
        self.creators = creators
        self.creation_op = creation_op
        self.grad = None
        self.autograd = autograd
        self.children = {}
        if id is None:
            self.id = np.random.randint(0, 100000)
        else:
            self.id = id

        # Keep track of children
        if creators is not None:
            for c in creators:
                if self.id not in c.children:
                    c.children[self.id] = 1
                else:
                    c.children[self.id] += 1

    # Check if Tensor has received correct # of gradients from children
    def all_children_grads_accounted_for(self):
        for id, cnt in self.children.items():
            if cnt != 0:
                return False
        return True

    def backward(self, grad=None, grad_origin=None):
        if self.autograd:
            if grad is None:
                grad = Tensor(np.ones_like(self.data))
            # Check if we can backprop or are still awaiting gradients
            if grad_origin is not None:
                if self.children[grad_origin.id] == 0:
                    raise Exception("Can't backpropagate more than once")
                else:
                    self.children[grad_origin.id] -= 1

            # Accumulate gradients from several children
            if self.grad is None:
                self.grad = grad
            else:
                self.grad += grad

        # grads must not have grads of their own
            assert grad.autograd is False

        # only continue backproping if there's something to
        # backprop into and if all gradients (from children)
        # are accounted for override waiting for children if
        # backprop was called on this variable directly
            if (self.creators is not None and
                (self.all_children_grads_accounted_for() or
                 grad_origin is None)):

                if self.creation_op == "add":
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad, self)

                if self.creation_op == "sub":
                    self.creators[0].backward(Tensor(self.grad.data), self)
                    self.creators[1].backward(Tensor(self.grad
                                                     .__neg__().data), self)

                if self.creation_op == "mul":
                    new = self.grad * self.creators[1]
                    self.creators[0].backward(new, self)
                    new = self.grad * self.creators[0]
                    self.creators[1].backward(new, self)

                if self.creation_op == "matmul":
                    c0 = self.creators[0]
                    c1 = self.creators[1]
                    new = self.grad.matmul(c1.transpose())
                    c0.backward(new)
                    new = self.grad.transpose().matmul(c0).transpose()
                    c1.backward(new)

                if self.creation_op == "transpose":
                    self.creators[0].backward(self.grad.transpose())

                if "sum" in self.creation_op:
                    dim = int(self.creation_op.split("_")[1])
                    self.creators[0].backward(self
                                              .grad
                                              .expand(dim, self
                                                      .creators[0]
                                                      .data
                                                      .shape[dim]))

                if "expand" in self.creation_op:
                    dim = int(self.creation_op.split("_")[1])
                    self.creators[0].backward(self.grad.sum(dim))

                if self.creation_op == "neg":
                    self.creators[0].backward(self.grad.__neg__())

                if self.creation_op == "sigmoid":
                    ones = Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backward(self.grad *
                                              (self * (ones - self)))

                if self.creation_op == "tanh":
                    ones = Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backward(self.grad *
                                              (ones - (self * self)))

                if self.creation_op == "relu":
                    ones = Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backward(self.grad *
                                              (ones * (self > 0)))

                if self.creation_op == "index_select":
                    new_grad = np.zeros_like(self.creators[0].data)
                    indices_ = self.index_select_indices.data.flatten()
                    grad_ = grad.data.reshape(len(indices_), -1)
                    for i in range(len(indices)):
                        new_grad[indices_[i]] += grad_[i]
                    self.creators[0].backward(Tensor(new_grad))

    def __add__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data + other.data,
                          autograd=True,
                          creators=[self, other],
                          creation_op="add")
        else:
            return Tensor(self.data + other.data)

    def __neg__(self):
        if self.autograd:
            return Tensor(self.data * -1,
                          autograd=True,
                          creators=[self],
                          creation_op="neg")
        else:
            return Tensor(self.data * -1)

    def __sub__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data - other.data,
                          autograd=True,
                          creators=[self, other],
                          creation_op="sub")
        else:
            return Tensor(self.data - other.data)

    def __mul__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data * other.data,
                          autograd=True,
                          creators=[self, other],
                          creation_op="mul")
        else:
            return Tensor(self.data * other.data)

    def sum(self, dim):
        if self.autograd:
            return Tensor(self.data.sum(dim),
                          autograd=True,
                          creators=[self],
                          creation_op=f"sum_{dim}")
        else:
            return Tensor(self.data.sum(dim))

    def expand(self, dim, copies):
        trans_cmd = list(range(0, len(self.data.shape)))
        trans_cmd.insert(dim, len(self.data.shape))
        new_shape = list(self.data.shape) + [copies]
        new_data = (self.data
                    .repeat(copies)
                    .reshape(new_shape)
                    .transpose(trans_cmd))

        if self.autograd:
            return Tensor(new_data,
                          autograd=True,
                          creators=[self],
                          creation_op=f"expand_{dim}")
        else:
            return Tensor(new_data)

    def transpose(self):
        if self.autograd:
            return Tensor(self.data.transpose(),
                          autograd=True,
                          creators=[self],
                          creation_op="transpose")

        else:
            return Tensor(self.data.transpose())

    def matmul(self, other):
        if self.autograd:
            return Tensor(np.dot(self.data, other.data),
                          autograd=True,
                          creators=[self, other],
                          creation_op="matmul")
        else:
            return Tensor(np.dot(self.data, other.data))

    def sigmoid(self):
        if self.autograd:
            return Tensor(1 / (1 + np.exp(-self.data)),
                          autograd=True,
                          creators=[self],
                          creation_op="sigmoid")
        else:
            return Tensor(1 / (1 + np.exp(-self.data)))

    def tanh(self):
        if self.autograd:
            return Tensor(np.tanh(self.data),
                          autograd=True,
                          creators=[self],
                          creation_op="tanh")
        else:
            return Tensor(np.tanh(self.data))

    def relu(self):
        if self.autograd:
            return Tensor(self.data * (self.data > 0),
                          autograd=True,
                          creators=[self],
                          creation_op="relu")
        else:
            return Tensor(self.data * (self.data > 0))

    def index_select(self, indices):
        if self.autograd:
            new = Tensor(self.data[indices.data],
                         autograd=True,
                         creators=[self],
                         creation_op="index_select")
            new.index_select_indices = indices
            return new
        else:
            return Tensor(self.data[indices.data])

    def __repr__(self):
        show_shape = self.data.shape
        show_id = self.id.__repr__()
        show_autograd = self.autograd
        id_and_shape = f"Tensor {show_id}: Shape = {show_shape}, "
        is_autograd = f"autograd = {show_autograd}\n"
        return id_and_shape + is_autograd

    def __str__(self):
        return str(self.data.__str__())


class SGD(object):

    def __init__(self, parameters, alpha=0.1):
        self.parameters = parameters
        self.alpha = alpha

    def zero(self):
        for param in self.parameters:
            param.grad.data *= 0

    def step(self, zero=True):
        for param in self.parameters:
            param.data -= param.grad.data * self.alpha
            if zero:
                param.grad.data *= 0


class Layer(object):

    def __init__(self):
        self.parameters = []

    def get_parameters(self):
        return self.parameters


class Linear(Layer):

    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        W = np.random.randn(n_inputs, n_outputs) * np.sqrt(2/n_inputs)
        self.weight = Tensor(W, autograd=True)
        self.bias = Tensor(np.zeros(n_outputs), autograd=True)

        self.parameters.append(self.weight)
        self.parameters.append(self.bias)

    def forward(self, inputs):
        in_weights = inputs.matmul(self.weight)
        in_bias = self.bias.expand(0, len(inputs.data))
        return in_weights + in_bias


class Sequential(Layer):

    def __init__(self, layers=[]):
        super().__init__()

        self.layers = layers

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def get_parameters(self):
        params = []
        for layer in self.layers:
            params += layer.get_parameters()
        return params


class MSELoss(Layer):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return ((pred - target) * (pred - target)).sum(0)


class Tanh(Layer):

    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return inputs.tanh()


class Sigmoid(Layer):

    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return inputs.sigmoid()


class ReLU(Layer):

    def __init__(self):
        super().__init()

    def forward(self, inputs):
        return inputs.relu()

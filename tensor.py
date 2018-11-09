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

            if self.creation_op == "neg":
                self.creators[0].backward(self.grad.__neg__())

    def __add__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data + other.data,
                          autograd=True,
                          creators=[self, other],
                          creation_op="add")

    def __neg__(self):
        if self.autograd:
            return Tensor(self.data * -1,
                          autograd=True,
                          creators=[self],
                          creation_op="neg")
        else:
            return Tensor(self.data * -1)

    def __repr__(self):
        show_shape = self.data.shape
        show_id = self.id.__repr__()
        show_autograd = self.autograd
        id_and_shape = f"Tensor {show_id}: Shape = {show_shape}, "
        is_autograd = f"autograd = {show_autograd}\n"
        return id_and_shape + is_autograd

    def __str__(self):
        return str(self.data.__str__())

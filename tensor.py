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
        if not id:
            self.id = np.random.randint(0, 100000)
        else:
            self.id = id

        if creators:
            for c in creators:
                if self.id not in c.children:
                    c.children[self.id] = 1
                else:
                    c.children[self.id] += 1

    def all_children_grads_accounted_for(self):
        for id, cnt in self.children.items():
            if cnt != 0:
                return False
        return True

    def backward(self, grad=None, grad_origin=None):
        if self.autograd:
            if grad_origin:
                if self.children[grad_origin.id] == 0:
                    raise Exception("Can't backpropagate more than once")
                else:
                    self.children[grad_origin.id] -= 1

        if not self.grad:
            self.grad = grad
        else:
            self.grad += grad

        # grads must not have grads of their own
        assert grad.autograd is False

        # only continue backproping if there's something to
        # backprop into and if all gradients (from children)
        # are accounted for override waiting for children if
        # backprop was called on this variable directly
        if (self.creators and (self.all_children_grads_accounted_for() or
                               not grad_origin)):

            if self.creation_op == "add":
                self.creators[0].backward(self.grad, self)
                self.creators[1].backward(self.grad, self)

    def __add__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data + other.data,
                          autograd=True,
                          creators=[self, other],
                          creation_op="add")
        else:
            return Tensor(self.data + other.data)

    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())

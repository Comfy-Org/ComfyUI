import torch

class NestedTensor:
    def __init__(self, tensors):
        self.tensors = list(tensors)
        self.is_nested = True

    def _copy(self):
        return NestedTensor(self.tensors)

    def apply_operation(self, other, operation):
        o = self._copy()
        if isinstance(other, NestedTensor):
            for i, t in enumerate(o.tensors):
                o.tensors[i] = operation(t, other.tensors[i])
        else:
            for i, t in enumerate(o.tensors):
                o.tensors[i] = operation(t, other)
        return o

    def __add__(self, b):
        return self.apply_operation(b, lambda x, y: x + y)

    def __sub__(self, b):
        return self.apply_operation(b, lambda x, y: x - y)

    def __mul__(self, b):
        return self.apply_operation(b, lambda x, y: x * y)

    # def __itruediv__(self, b):
    #     return self.apply_operation(b, lambda x, y: x / y)

    def __truediv__(self, b):
        return self.apply_operation(b, lambda x, y: x / y)

    def __getitem__(self, *args, **kwargs):
        return self.apply_operation(None, lambda x, y: x.__getitem__(*args, **kwargs))

    def unbind(self):
        return self.tensors

    def to(self, *args, **kwargs):
        o = self._copy()
        for i, t in enumerate(o.tensors):
            o.tensors[i] = t.to(*args, **kwargs)
        return o

    def new_ones(self, *args, **kwargs):
        return self.tensors[0].new_ones(*args, **kwargs)

    def float(self):
        return self.to(dtype=torch.float)

    def cpu(self):
        return self.to(device="cpu")

    def chunk(self, *args, **kwargs):
        return self.apply_operation(None, lambda x, y: x.chunk(*args, **kwargs))

    def size(self):
        return self.tensors[0].size()

    @property
    def shape(self):
        return self.tensors[0].shape

    @property
    def ndim(self):
        dims = 0
        for t in self.tensors:
            dims = max(t.ndim, dims)
        return dims

    @property
    def device(self):
        return self.tensors[0].device

    @property
    def dtype(self):
        return self.tensors[0].dtype

    @property
    def layout(self):
        return self.tensors[0].layout

    def __repr__(self):
        return f"{type(self).__name__}({self.tensors!r})"


def cat_nested(tensors, *args, **kwargs):
    cated_tensors = []
    for i in range(len(tensors[0].tensors)):
        tens = []
        for j in range(len(tensors)):
            tens.append(tensors[j].tensors[i])
        cated_tensors.append(torch.cat(tens, *args, **kwargs))
    return NestedTensor(cated_tensors)

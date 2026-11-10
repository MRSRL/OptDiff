"""Vector operations for use in calculating conjugate gradient descent."""

import torch
import numpy as np
from .data_mri import to_cplx
from torch import nn

def dot(x1, x2):
    """Finds the dot product of two vectors.

    Args:
        x1 (Tensor): The first input vector.
        x2 (Tensor): The second input vector.

    Returns:
        The dot product of x1 and x2.
    """

    return torch.sum(x1*x2)

def dot_single(x):
    """Finds the dot product of a vector with itself

    Args:
        x (Tensor): The input vector.

    Returns:
        The dot product of x and x.
    """

    return dot(x, x)

def dot_batch(x1, x2):
    """Finds the dot product of two multidimensional Tensors, preserving the batch dimension.

    Args:
        x1 (Tensor): The first multidimensional Tensor.
        x2 (Tensor): The second multidimensional Tensor.

    Returns:
        The dot products along each dimension of x1 and x2.
    """

    batch = x1.shape[0]
    return torch.reshape(x1*x2, (batch, -1)).sum(1)


def dot_single_batch(x):
    """Finds the dot product of a multidimensional Tensors with itself, preserving the batch dimension.

    Args:
        x (Tensor): The multidimensional Tensor.

    Returns:
        The dot products along each non-batch dimension of x and x.
    """

    return dot_batch(x, x)

def zdot(x1, x2):
    """Finds the complex-valued dot product of two complex-valued vectors.

    Args:
        x1 (Tensor): The first input vector.
        x2 (Tensor): The second input vector.

    Returns:
        The dot product of x1 and x2, defined as sum(conj(x1) * x2)
    """

    return torch.sum(torch.conj(x1)*x2)

def zdot_single(x):
    """Finds the complex-valued dot product of a complex-valued vector with itself

    Args:
        x (Tensor): The input vector.

    Returns:
        The dot product of x and x., defined as sum(conj(x) * x)
    """

    return zdot(x, x)

def zdot_batch(x1, x2):
    """Finds the complex-valued dot product of two complex-valued multidimensional Tensors, preserving the batch dimension.

    Args:
        x1 (Tensor): The first multidimensional Tensor.
        x2 (Tensor): The second multidimensional Tensor.

    Returns:
        The dot products along each dimension of x1 and x2.
    """

    batch = x1.shape[0]
    return torch.reshape(torch.conj(x1)*x2, (batch, -1)).sum(1)


def zdot_single_batch(x):
    """Finds the complex-valued dot product of a multidimensional Tensors with itself, preserving the batch dimension.

    Args:
        x (Tensor): The multidimensional Tensor.

    Returns:
        The dot products along each non-batch dimension of x and x.
    """

    return zdot_batch(x, x)

def l2ball_proj_batch(x, eps):
    """ Performs a batch projection onto the L2 ball.

    Args:
        x (Tensor): The tensor to be projected.
        eps (Tensor): A tensor containing epsilon values for each dimension of the L2 ball.

    Returns:
        The projection of x onto the L2 ball.
    """

    #print('l2ball_proj_batch')
    reshape = (-1,) + (1,) * (len(x.shape) - 1)
    x = x.contiguous()
    q1 = torch.real(zdot_single_batch(x)).sqrt()
    #print(eps,q1)
    q1_clamp = torch.min(q1, eps)

    z = x * q1_clamp.reshape(reshape) / (1e-8 + q1.reshape(reshape))
    #q2 = torch.real(zdot_single_batch(z)).sqrt()
    #print(eps,q1,q2)
    return z


# ----------------------------------------------------------------------
# Solvers
# ----------------------------------------------------------------------

class ConjugateGradient(nn.Module):
    """
    Implementation of conjugate gradient algorithm to invert
    the linear system:
        y = A x
    where A is a symmetric, positive-definite matrix.

    In multi-coil MRI reconstruction, A is not symmetric. However, we can
    form the normal equations to get the problem in the form above:
        A^T y = A^T A x

    Based on code by Chris Sandino and Jon Tamir.
    """

    def __init__(self, A, num_iter, dbprint=False, complex_flag=True):
        super(ConjugateGradient, self).__init__()

        self.A = A
        self.num_iter = num_iter
        self.dbprint = dbprint
        self.reshape = None

        if complex_flag:
            self._dot_single_batch = lambda r: zdot_single_batch(to_cplx(r)).real
            self._dot_batch = lambda r, p: zdot_batch(to_cplx(r), to_cplx(p)).real
        else:
            self._dot_single_batch = dot_single_batch
            self._dot_batch = dot_batch

    def _update(self, iter):
        def update_fn(x, p, r, rsold):
            Ap = self.A(p)
#             pAp = self.zdot(p, Ap)
            pAp = self._dot_batch(p, Ap)
            alpha = (rsold / pAp).reshape(self.reshape)
            x = x + alpha * p
            r = r - alpha * Ap
#             rsnew = self.zdot_single(r)
            rsnew = self._dot_single_batch(r)
            beta = (rsnew / rsold).reshape(self.reshape)
            rsold = rsnew
            p = beta * p + r

            # print residual
            if self.dbprint:
                print(f'CG Iteration {iter}: {rsnew}')

            return x, p, r, rsold

        return update_fn

    def forward(self, x, y):
        # Compute residual
        r = y - self.A(x)
        rsold = self._dot_single_batch(r)
        p = r
        self.reshape = (-1,) + (1,) * (len(x.shape) - 1)

        for i in range(self.num_iter):
            x, p, r, rsold = self._update(i)(x, p, r, rsold)

        return x

class GradientDescent(nn.Module):
    """
    Implementation of conjugate gradient algorithm to invert
    the linear system:
        y = A x
    where A is a symmetric, positive-definite matrix.

    In multi-coil MRI reconstruction, A is not symmetric. However, we can
    form the normal equations to get the problem in the form above:
        A^T y = A^T A x

    Based on code by Chris Sandino and Jon Tamir.
    """

    def __init__(self, A, num_iter, alpha=None, dbprint=False, complex_flag=True):
        super(GradientDescent, self).__init__()

        self.A = A
        self.num_iter = num_iter
        self.dbprint = dbprint
        self.reshape = None
        self.alpha = alpha

        if complex_flag:
            self._dot_single_batch = lambda r: zdot_single_batch(r).real
            self._dot_batch = lambda r, p: zdot_batch(r, p).real
        else:
            self._dot_single_batch = dot_single_batch
            self._dot_batch = dot_batch

    def _update(self, iter):
        def update_fn(x, r):
            Ar = self.A(r)
            x = x + self.alpha * r
            r = r - self.alpha * Ar

            # print residual
            if self.dbprint:
                print(f'GD Iteration {iter}: {self._dot_single_batch(r)}')

            return x, r

        return update_fn

    def forward(self, x, y):
        # Compute residual
        r = y - self.A(x)
        self.reshape = (-1,) + (1,) * (len(x.shape) - 1)

        for i in range(self.num_iter):
            x, r = self._update(i)(x, r)

        return x


class PowerMethod(nn.Module):
    """
    Implementation of power method to compute singular values of batch
    of matrices.
    """
    def __init__(self, A, im_shape, dtype, num_iter=30, eps=1e-6, device=torch.device('cpu')):
        super(PowerMethod, self).__init__()

        self.num_iter = num_iter
        self.eps = eps
        self.im_shape = im_shape
        self.A = A
        self.dtype = dtype
        self.device = device

    def forward(self):
        v = torch.rand(self.im_shape, dtype=self.dtype, device=self.device)

        for _ in range(self.num_iter):
            v = self.A(v)
            eigenvals = torch.sum(torch.abs(v) ** 2,
                            dim=tuple(range(1,len(self.im_shape))), keepdim=True).sqrt()
            v = v / (eigenvals + self.eps)

        return eigenvals

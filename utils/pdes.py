import dde.deepxde as dde

class diffusion_reaction():
    def __init__(self, D = 0.01, k = 0.01):
        self.D = D
        self.k = k
    
    def __call__(self, x, y, aux):
        dy_t = dde.grad.jacobian(y, x[1], j=1)
        dy_xx = dde.grad.hessian(y, x[1], j=0)
        out = dy_t - self.D * dy_xx + self.k * y**2 - aux
        return out

class advection_diffusion_reation():
    def __call__(self, x, y, aux):
        dy_t = dde.grad.jacobian(y, x[1], j=1)
        dy_x = dde.grad.jacobian(y, x[1], j=0)
        dy_xx = dde.grad.hessian(y, x[1], j=0)
        out = dy_t + dy_x - aux * dy_xx
        return out

class anti_derivative():
    def __call__(self, x, y, aux):
        dy = dde.grad.jacobian(y, x[1], j=0)
        out = dy - aux
        return out
    
class burgers_equation():
    def __init__(self, v = 0.01):
        self.v = v
    
    def __call__(self, x, y, aux):
        dy_t = dde.grad.jacobian(y, x[1], j=1)
        dy_x = dde.grad.jacobian(y, x[1], j=0)
        dy_xx = dde.grad.hessian(y, x[1], j=0)
        out = dy_t + y * dy_x - self.v * dy_xx
        return out

class advection_equation():
    def __call__(self, x, y, aux):
        dy_t = dde.grad.jacobian(y, x[1], j=1)
        dy_x = dde.grad.jacobian(y, x[1], j=0)
        out = dy_t + aux * dy_x
        return out
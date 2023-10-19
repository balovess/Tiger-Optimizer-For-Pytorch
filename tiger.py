import torch
import re
import torch.optim as optim


class Tiger(optim.Optimizer):
    """Tiger Optimizer

    """
    def __init__(
        self,
        params,
        learning_rate=1e-3,
        beta=0.965,
        weight_decay=0.01,
        grad_accum_steps=1,
        lr_schedule={0: 1},
        shrink_ratio=0.99,
    ):
        if not 0.0 <= learning_rate:
            raise ValueError("Invalid learning_rate: {}".format(learning_rate))
        if not 0.0 <= beta <= 1.0:
            raise ValueError("Invalid beta: {}".format(beta))
        if not 0.0 <= shrink_ratio <= 1.0:
            raise ValueError("Invalid shrink_ratio: {}".format(shrink_ratio))
        
        defaults = dict(
            learning_rate=learning_rate,
            beta=beta,
            weight_decay=weight_decay,
            grad_accum_steps=grad_accum_steps,
            lr_schedule={int(i): j for i, j in lr_schedule.items()},
            shrink_ratio=shrink_ratio
        )
        super(Tiger, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Tiger, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                
                if grad.is_sparse:
                    raise RuntimeError(
                        'Tiger optimizer does not support sparse gradients'
                    )

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)

                t = state['step']
                d = group['weight_decay']
                k = group['grad_accum_steps']
                s = group['shrink_ratio']
                beta_tensor = torch.tensor(group['beta'])
                if t % k == 0:
                    b1 = beta_tensor
                else:
                    b1 = 1.0
                b2 = (1 - group['beta']) / k
                lr = group['learning_rate'] * self.piecewise_linear(t, group['lr_schedule'])
                if (t + 1) % k == 0:
                    lr = lr
                else:
                    lr = 0

                state['step'] += 1

                is_nan = torch.isnan(grad)
                b1 = torch.where(is_nan, torch.ones_like(b1), b1)
                g = torch.where(is_nan, torch.zeros_like(grad), grad)
                m = state['m']

                c = 0
                if p.name is None:
                    name = "default_name"
                else:
                    name = p.name
    
                if re.findall('bias|beta|gamma', name):
                    lr, d = lr * 0.5, 0
                    if 'gamma' in p.name:
                        c = 1
                elif 'embeddings' in name:
                    lr = lr * self.root_mean_square(p.data, axis=-1, keepdims=True)
                else:
                    lr = lr * self.root_mean_square(p.data)

                m_t = b1 * m + b2 * g
                state['m'] = m_t

                u = (torch.sign(m_t) + d * p.data) * lr
                v = torch.where(is_nan, (p.data - c) * s + c, p.data - u)
                p.data = v

        return loss

    @staticmethod
    def root_mean_square(x, axis=None, keepdims=False):
        """Root Mean Square"""
        return torch.sqrt(torch.mean(x**2, dim=axis, keepdim=keepdims))

    @staticmethod
    def piecewise_linear(t, schedule, from_zero=True):
        """Piecewise Linear Function

        """
        schedule = sorted(schedule.items())
        if from_zero and schedule[0][0] != 0:
            schedule = [(0, 0.0)] + schedule

        t = torch.tensor(t, dtype=torch.float32)
        x = torch.tensor(schedule[0][1], dtype=torch.float32)
        for i in range(len(schedule)):
            t_begin = schedule[i][0]
            x_begin = x
            if i != len(schedule) - 1:
                dx = schedule[i + 1][1] - schedule[i][1]
                dt = schedule[i + 1][0] - schedule[i][0]
                slope = 1.0 * dx / dt
                x = schedule[i][1] + slope * (t -t_begin)
            else:
                x = torch.tensor(schedule[i][1], dtype=torch.float32)
            x = torch.where(t >= t_begin, x, x_begin)

        return x



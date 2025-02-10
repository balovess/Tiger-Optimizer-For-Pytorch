import torch
import re
import torch.optim as optim


class Tiger(optim.Optimizer):
    """Tiger Optimizer (Sparse Support Version)

    This version of the Tiger optimizer attempts to add basic support for sparse gradients.
    It retains the core ideas of efficiency and simplicity while trying to be compatible
    with sparse data.  However, true sparse optimization might require more fundamental
    algorithm changes, and this version might not be as efficient or performant as
    a native sparse optimizer.

    Note: Sparse gradient support is experimental and might not be optimal for all scenarios.
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
        use_sign_grad=True,  # Option to use sign-based gradient update
    ):
        """Initializes the Tiger Optimizer (Sparse Support Version).

        Args:
            params (iterable): Iterable of parameters to optimize or dicts defining parameter groups
            learning_rate (float, optional): Learning rate (default: 1e-3)
            beta (float, optional): Coefficient for moving average of gradient (default: 0.965)
            weight_decay (float, optional): Weight decay (L2 penalty) (default: 0.01)
            grad_accum_steps (int, optional): Number of steps to accumulate gradient before optimization (default: 1)
            lr_schedule (dict, optional): Learning rate schedule as dict, keys are steps, values are lr multipliers (default: {0: 1})
            shrink_ratio (float, optional): Ratio to shrink parameter values when gradient is NaN (default: 0.99)
            use_sign_grad (bool, optional): Whether to use sign of gradient for parameter update (default: True)

        Raises:
            ValueError: If any input argument is invalid
        """
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
            shrink_ratio=shrink_ratio,
            use_sign_grad=use_sign_grad,
        )
        super(Tiger, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Tiger, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step (Sparse Support Version).

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.

        Returns:
            Optional[float]: The loss value, if provided by the closure.

        Raises:
            RuntimeError: if unexpected sparse gradient issues are encountered.  Basic sparse support is attempted.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]

                if 'step' not in state:
                    state['step'] = 0
                if 'momentum' not in state:
                    state['momentum'] = torch.zeros_like(p) # Initialize momentum as dense, compatible with both dense and sparse

                step_count = state['step']
                weight_decay = group['weight_decay']
                grad_accum_steps = group['grad_accum_steps']
                shrink_ratio = group['shrink_ratio']
                beta = group['beta']
                use_sign_grad = group['use_sign_grad']
                beta_tensor = torch.tensor(beta, device=grad.device)

                if step_count % grad_accum_steps == 0:
                    beta1 = beta_tensor
                else:
                    beta1 = 1.0
                beta2 = (1 - beta) / grad_accum_steps

                learning_rate = group['learning_rate'] * self.piecewise_linear(
                    step_count, group['lr_schedule']
                )
                if (step_count + 1) % grad_accum_steps != 0:
                    learning_rate = 0

                state['step'] += 1

                nan_mask = torch.isnan(grad) # nan_mask should work for both dense and sparse

                beta1 = torch.where(nan_mask, torch.ones_like(beta1), beta1) # where should work for both
                grad_is_sparse = grad.is_sparse # Check if grad is sparse

                if grad_is_sparse: # Sparse gradient handling
                    grad_dense = grad.to_dense() # Convert sparse grad to dense for easier operations.  Efficiency might degrade with high sparsity.
                    grad = grad_dense # Use dense grad from now on in this step

                grad = torch.where(nan_mask, torch.zeros_like(grad), grad) # where should work for dense grad now
                momentum = state['momentum']

                momentum_t = beta1 * momentum + beta2 * grad # Momentum update - dense operations are applied
                state['momentum'] = momentum_t # Momentum state is always dense


                lr_scale = 1.0
                param_name = p.name if hasattr(p, 'name') and p.name else "default_name"
                constant_lr_scale = 0

                if re.search(r'bias|beta|gamma', param_name):
                    learning_rate *= 0.5
                    weight_decay = 0
                    if 'gamma' in param_name:
                        constant_lr_scale = 1
                elif 'embeddings' in param_name:
                    lr_scale = self.root_mean_square(p.data, axis=-1, keepdims=True)
                else:
                    lr_scale = self.root_mean_square(p.data)

                effective_lr = learning_rate * lr_scale

                if use_sign_grad:
                    update = torch.sign(momentum_t) * effective_lr # sign works on dense tensor
                else:
                    update = (torch.sign(momentum_t) + weight_decay * p.data) * effective_lr # dense operations

                param_update = torch.where(nan_mask, (p.data - constant_lr_scale) * shrink_ratio + constant_lr_scale, p.data - update) # where works on dense
                p.data = param_update # Parameter p remains as its original type (sparse or dense)

        return loss

    @staticmethod
    def root_mean_square(x, axis=None, keepdims=False):
        """Root Mean Square"""
        return torch.sqrt(torch.mean(x**2, dim=axis, keepdim=keepdims))

    @staticmethod
    def piecewise_linear(t, schedule, from_zero=True):
        """Piecewise Linear Function"""
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

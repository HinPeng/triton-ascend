from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

_exact_request = ContextVar("ascend_reuse_exact_request", default=False)


@contextmanager
def exact_request_scope():
    token = _exact_request.set(True)
    try:
        yield
    finally:
        _exact_request.reset(token)


def requires_exact_request():
    return _exact_request.get()


@dataclass
class InvocationFrame:
    grid_resolved: bool = False
    grid: object = None
    launch_started: bool = False
    load_hook_started: bool = False

    def resolve_grid(self, grid, bound_args):
        if not self.grid_resolved:
            self.grid = grid(bound_args) if callable(grid) else grid
            self.grid_resolved = True
        return self.grid

    def before_launch(self):
        self.launch_started = True

    def select_grid(self, grid):
        """Install a proved candidate grid only after dispatch has selected it."""
        if self.launch_started:
            raise RuntimeError("cannot change grid after launch")
        self.grid = grid
        self.grid_resolved = True

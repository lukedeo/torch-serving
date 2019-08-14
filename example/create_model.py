import torch


class SomeJitModule(torch.jit.ScriptModule):
    def __init__(self, N, M):
        super(SomeJitModule, self).__init__()
        self.fc1 = torch.nn.Linear(N, 10000)
        self.fc2 = torch.nn.Linear(10000, 1000)
        self.fc3 = torch.nn.Linear(1000, M)

    @torch.jit.script_method
    def forward(self, x, y):
        # Note two inputs and a list of tensors as outputs. For now, we support
        # single tensors or lists of tensors for output, and any number of
        # tensors as input.
        return [self.fc3(self.fc2(self.fc1(x))) + y * 2, y]


my_script_module = SomeJitModule(2, 3)

my_script_module.save("model-example.pt")

import torch
from typing import Dict, List


class SomeJitModule(torch.nn.Module):

    def __init__(self, N, M):
        super(SomeJitModule, self).__init__()
        self.fc1 = torch.nn.Linear(N, 10000)
        self.fc2 = torch.nn.Linear(10000, 1000)
        self.fc3 = torch.nn.Linear(1000, M)

    def forward(
        self,
        d: List[torch.Tensor],
        o: torch.Tensor,
        td: Dict[str, torch.Tensor],
        n: int,
        s: str,
    ):
        x = d[0]
        y = d[1]
        return (
            {
                "out": (
                    [
                        self.fc3(self.fc2(self.fc1(x + td["x"]))) + y * 2,
                        y + o + td["y"] + n,
                    ],
                    s,
                    n,
                )
            },
            "some string output",
        )


my_script_module = torch.jit.script(SomeJitModule(2, 3))
my_script_module.save("model-example.pt")

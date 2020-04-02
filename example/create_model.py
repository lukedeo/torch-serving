import torch
from typing import Dict, List, Any


class SomeJitModule(torch.nn.Module):

    def __init__(self, N, M):
        super(SomeJitModule, self).__init__()
        self.fc1 = torch.nn.Linear(N, 10000)
        self.fc2 = torch.nn.Linear(10000, 1000)
        self.fc3 = torch.nn.Linear(1000, M)

    # # @torch.jit.script_method
    # def forward(
    #     self,
    #     d: List[torch.Tensor],
    #     o,
    #     td: Dict[str, torch.Tensor],
    #     n: int,
    #     s: str,
    # ):
    #     x = d[0]
    #     y = d[1]
    #     return {
    #         "out": (
    #             [
    #                 self.fc3(self.fc2(self.fc1(x + td["x"]))) + y * 2,
    #                 y + o + td["y"] + n,
    #             ],
    #             s,
    #             n,
    #         ),
    #         # "desc": ("foobar", ),
    #     }

    def forward(self, s: str) -> Dict[str, str]:

        # x = d[0]
        # y = d[1]
        # return {"scores": self.fc3(self.fc2(self.fc1(x)))}
        return {"scores": s}


my_script_module = torch.jit.script(SomeJitModule(2, 3))


my_script_module.save("model-example.pt")

# import torch
# from typing import Dict, List, Any, Tuple
#
#
# class SomeJitModule(torch.nn.Module):
#
#     def __init__(self, N, M):
#         super(SomeJitModule, self).__init__()
#         self.fc1 = torch.nn.Linear(N, 10000)
#         self.fc2 = torch.nn.Linear(10000, 1000)
#         self.fc3 = torch.nn.Linear(1000, M)
#
#     def forward(
#             self,
#             tensor_list: List[torch.Tensor],
#             single_tensor: torch.Tensor,
#             tensor_dict: Dict[str, torch.Tensor],
#             an_integer: int,
#             a_string: str,
#     ) -> Dict[str, Tuple[List[torch.Tensor], str, int]]:
#         x = tensor_list[0]
#         y = tensor_list[1]
#         return {
#             "out": (
#                 [
#                     self.fc3(self.fc2(self.fc1(x + tensor_dict["x"]))) + y * 2,
#                     y + single_tensor + tensor_dict["y"] + an_integer,
#                     ],
#                 a_string,
#                 an_integer,
#             ),
#         }
#
# my_script_module = torch.jit.script(SomeJitModule(2, 3))
#
#
# my_script_module.save("model-example.pt")

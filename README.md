# `torch-serving`

## Introduction

A simple, limited scope model server for [JIT compiled PyTorch models/TorchScript](https://pytorch.org/docs/stable/jit.html).

The idea is that one should be able to spin up a simple HTTP service that should *just work*, and be able to handle inbound requests to JIT compiled models. We do a few key things to make this work well:

1. Model caching - store deserialized `torch::jit::Module` pointers in a modification-threadsafe LRU cache so we don't need to pay deserialization cost every roundtrip
2. JSON marshaling for tensor types.
3. Speed!

## Building the project

### Dependencies

`torch-serving` comes with all dependencies as header-only or included as a submodule. 

* [`httplib`](https://github.com/yhirose/cpp-httplib) (MIT License)
* [`json`](https://github.com/nlohmann/json) (MIT License)
* [`spdlog`](https://github.com/gabime/spdlog) (MIT License)
* [`lrucache11`](https://github.com/mohaps/lrucache11) (BSD License)
* [`option-parser`](https://github.com/lukedeo/option-parser) (MIT License)

### Requirements

We require a C++11 compliant compiler, `CMake>=3.11`, and [`libtorch`](https://pytorch.org/cppdocs/installing.html) 1.3. The easiest way to obtain a version of `libtorch` is to install PyTorch for Python with `pip install "torch>=1.3"` or anaconda, and then point `CMake` to the install directory. We provide a utility script to do this, simply run `scripts/libtorch-root` when in a virtualenv with the `torch` you want to link against installed and it will print the appropriate directory. This is discussed in the _Installation_ section below.


### Installation

Clone the repo (and submodule) with:

```bash
git clone --recursive https://github.com/lukedeo/torch-serving
```

If you are using a Python installation of `torch` to link to `libtorch`, run: 
```bash
export LIBTORCH_ROOT=$(scripts/libtorch-root)
```

if not, and you are using a specific `libtorch`, run:

```bash
export LIBTORCH_ROOT=/path/to/your/libtorch
```

Then, build the project!

```bash
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=$LIBTORCH_ROOT ..
make
```

The executable will be `apps/torch-serving`. Go ahead and move that up a directory:

```bash
mv apps/torch-serving .. && cd ..
```

## Running an example

Now, we'll run through a full example of saving a PyTorch model in TorchScript and using `torch-serving` to serve the model.

### Saving a JIT model (from Python)

Ensure you have Python and `torch>=1.3` installed, and run:


```bash
python example/create_model.py
```

which creates a dumb big model with interesting inputs and outputs (i.e., `List[torch.Tensor]`, etc.) and converts to TorchScript, saving to `model-example.pt`.

Specifically, this model has a `forward` defined with the following types

```python
def forward(
    self,
    d: List[torch.Tensor],
    o: torch.Tensor,
    td: Dict[str, torch.Tensor],
    n: int,
    s: str,
) -> Tuple[Dict[str, Tuple[List[torch.Tensor], str, int]], str]:
``` 

to illustrate the variety of types that `torch-serving` can handle.


### Running the server & making a request.

From the repo directory (after following the build), just run `torch-serving`.

In another terminal, run a request through!

```bash
curl -s -X POST \
    -d@example/post-data.json \
    "localhost:8888/serve?servable_identifier=model-example.pt"
```

In `example/post-data.json`, note that each argument in the `forward` function is mapped to an element in an outer list defined in JSON. If `forward` has only one non-`self` argument, no outer list is required, though it can be used optionally. 


This should output something like:


```json
{
  "code": 200,
  "description": "Success",
  "message": "OK",
  "result": [
    {
      "type": "generic_dict",
      "value": {
        "out": [
          [
            {
              "data_type": "float32",
              "shape": [1, 3],
              "type": "tensor",
              "value": [21.711061477661133, 2.3672115802764893, 4.381892204284668]
            },
            {
              "data_type": "float32",
              "shape": [1, 3],
              "type": "tensor",
              "value": [60, 33, 33]
            }
          ],
          {"type": "string", "value": "Hello!"},
          {
            "data_type": "int64",
            "type": "scalar",
            "value": 30
          }
        ]
      }
    },
    {
      "type": "string",
      "value": "some string output"
    }
  ]
}
```

Note that we represent tensors *unraveled* and specify a shape, where you can do `tensor.tensor(unraveled_tensor).reshape(shape)`. Also, note that the `result` key contains a list, since we output a `Tuple[...]` from our script module. If this were not the case, and, say, we outputted only a `torch.Tensor`, it would look something like:

```json
{
  "code": 200,
  "description": "Success",
  "message": "OK",
  "result": {
    "data_type": "float32",
    "shape": [
      1,
      3
    ],
    "type": "tensor",
    "value": [
      20.413822174072266,
      -0.20658397674560547,
      0.81690514087677
    ]
  }
}
``` 

## Limitations and Marshalling

`torch-serving` supports a limited subset of input and output types, mostly dictated by the type restrictions in the TorchScript subset of the Python language. In particular, we support all input and output JSON marshalling for all composite types from [this list](https://pytorch.org/docs/stable/jit.html#supported-type) except for `Optional[T]`, `NamedTuple[T0, T1, ...]`, and custom TorchScript classes.

We map from TorchScript types to JSON and back with the following structures:

* `Tensor`: `{"type": "tensor", "shape": <s>, "value": <v>, "data_type": <dt>}`
* `Tuple[T0, T1, ...]`: `[{"type": <T0>, ...}, {"type": <T1>, ...}, ...]`
* `bool`: `{"type": "scalar", "data_type": "bool", "value": <v>}`
* `int`: `{"type": "scalar", "data_type": "int", "value": <v>}`
* `float`: `{"type": "scalar", "data_type": "float", "value": <v>}`
* `string`: `{"type": "string", "value": <v>}`
* `List[T]`: `[{"type": <T>, ...}, {"type": <T>, ...}, ...]`
* `Dict[K=str, V]`: `{"type": "generic_dict", "value": {"<key_0>": {"type": <V>, ...}, ...}}`


In terms of numeric datatypes, `torch-serving` supports:

* `bool`
* `uint8`
* `int8`
* `int16`
* `int32`
* `int64`
* `float16`
* `float32`
* `float64`


## TODOs (read: requests for contributions 🚀)

* More documentation
* __GPU & general device support & control__
* Support `type: image` from JSON (base64, most likely)
* Wire up the cache invalidation probability to the API.
* Tests!

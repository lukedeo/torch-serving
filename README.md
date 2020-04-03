# torch-serving

A simple, limited scope model server for [JIT compiled PyTorch models](https://pytorch.org/docs/stable/jit.html).

The key idea is that one should be able to spin up a simple HTTP service that should *just work*, and be able to handle inbound requests to JIT compiled models. We want to do a few key things to make this work well:

1. Model caching - store deserialized `torch::jit::Module` pointers in an (modification-threadsafe) LRU cache so we don't need to pay deserialization cost every roundtrip
2. JSON marshaling for tensor types.
3. Speed!

# Building the project

The build is suboptimal - it would be great to make the `CMake` config better. 

Clone the repo (and submodule) with:

```bash
git clone --recursive https://github.com/lukedeo/torch-serving
```

We expect you to have `libtorch` unpacked somewhere (available [here](https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip)), and CMake available (as well as a C++11 compliant compiler).

This is also achievable using a python installed version of `torch`. We provide a script to locate the path to `libtorch` in your python installation.

Run:

```bash
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
# OR
cmake -DCMAKE_PREFIX_PATH=$(../scripts/libtorch-root) ..
make
```

The executable will be `apps/torch-serving`. Go ahead and move that up a directory:

```bash
mv apps/torch-serving .. && cd ..
```

# Saving a JIT model (from Python)

Now, ensure you have Python & `torch` installed, and run


```bash
python example/create_model.py
```

which creates a dumb big model with interesting inputs and outputs (i.e., `List[torch.Tensor]`, etc.) and runs a JIT trace, saving to `model-example.pt`.


# Running the server & making a request.

From the repo directory (after following the build), just run `./torch-serving`.

In another terminal, run a request through (maybe pipe through `jq` if you've got that installed)!

```bash
curl -X POST \
    --data @example/post-data.json \
    localhost:8888/serve?servable_identifier=model-example.pt
```

which should output:

```json
{
  "type": "generic_dict",
  "value": {
    "out": [
      [
        {
          "data_type": "float32",
          "shape": [1, 3],
          "type": "tensor",
          "value": [19.27, 5.28, 3.72]
        },
        {
          "data_type": "float32",
          "shape": [1, 3],
          "type": "tensor",
          "value": [60, 33, 33]
        }
      ],
      {
        "type": "string",
        "value": "Hello!"
      },
      {
        "data_type": "int64",
        "type": "scalar",
        "value": 30
      }
    ]
  }
}
```

Note that we represent tensors *unraveled* and specify a shape, where you can do `tensor.tensor(unraveled_tensor).reshape(shape)`.


# TODOs

* Documentation
* Better build
* Support `type: image`  from JSON (base64 encoding)
* Wire up the cache invalidation probability to the API.

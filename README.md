# torch-serving

A simple, limited scope model server for [JIT compiled PyTorch models](https://pytorch.org/docs/stable/jit.html), whipped up in two days of a hackathon.

The key idea is that one should be able to spin up a simple HTTP service that should *just work*, and be able to handle inbound requests to JIT compiled models. We want to do a few key things to make this work well:

1. Model caching - store deserialized `torch::jit::Module` pointers in an (modification-threadsafe) LRU cache so we don't need to pay deserialization cost every roundtrip
2. JSON marshaling for tensor types.
3. Speed!

# Building the project

This was whipped up in two days so the build is a bit suboptimal to say the least...

Clone the repo (and submodule) with:

```bash
git clone --recursive https://github.com/lukedeo/torch-serving
```

We expect you to have `libtorch` unpacked somewhere (available [here](https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip)), and CMake available (as well as a C++11 compliant compiler).

Run:

```bash
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
# OR
cmake -DCMAKE_PREFIX_PATH=$(../scripts/libtorch-root) ..
make
```

The executable will be `torch-serving`. Go ahead and move that up a directory:

```bash
mv torch-serving .. && cd ..
```

# Saving a JIT model (from Python)

Now, ensure you have Python & `torch` installed, and run


```bash
python example/create_model.py
```

which creates a dumb big model with interesting inputs and outputs (i.e., `List[torch.Tensor]`, etc.) and runs a JIT trace, saving to `model-example.pt`.


# Running the server & making a request.

From the repo directory (after following the build), just run `torch-serving`.

In another terminal, run a request through!

```bash
curl -s -X POST \
    -d "[\
        {\"type\":\"tensor\", \"shape\": [1, 2], \"value\": [10, 1]}, \
        {\"type\":\"tensor\", \"shape\": [1, 3], \"value\": [10, 1, 1]}\
        ]" \
    "localhost:8888/v1/serve/model-example.pt"
```

which should output:

```json
{
  "code": 200,
  "description": "Success",
  "message": "OK",
  "result": [
    {
      "shape": [
        1,
        3
      ],
      "type": "tensor",
      "value": [
        20.395009994506836,
        0.9330979585647583,
        3.44999361038208
      ]
    },
    {
      "shape": [
        1,
        3
      ],
      "type": "tensor",
      "value": [
        10,
        1,
        1
      ]
    }
  ]
}
```

Note that we represent tensors *unraveled* and specify a shape, where you can do `tensor.tensor(unraveled_tensor).reshape(shape)`.


# TODOs

* Documentation
* Better build
* Move this to a library with actual linking - fully header-only for ease of getting off the ground
* GPU & general device support & control
* Support different tensor types (`float16`, etc.)
* Support `type: image` and `type: text` from JSON (base64 and plain string representation, respectively)
* Support dictionaries of tensors as input and output.
* Command line configuration
* Wire up the cache invalidation probability to the API.
* Tests!

<img src="https://github.com/onnx/onnx/blob/master/docs/ONNX_logo_main.png?raw=true" width="400">

KnetONNX lets you read an ONNX file and create a Knet Model that can be used for prediction, re-designed or re-trained.

If you are planning to move your models from PyTorch or Tensorflow to Knet, or simply desiring to play with popular pre-trained neural networks: KnetONNX provides that functionality. 

[Open Neural Network Exchange (ONNX)](https://onnx.ai/)
 is a community project created by Facebook and Microsoft. It provides a definition of an extensible computation graph model, as well as definitions of built-in operators and standard data types.
 
Operators are implemented externally to the graph, but the set of built-in operators are portable across frameworks. Every framework supporting ONNX will provide implementations of these operators on the applicable data types.

Although not all operations are implemented yet, visit ONNX's [model zoo](https://github.com/onnx/models) to download pre-trained, state-of-the-art models in the ONNX format.

Once you download the ONNX file, call ONNXtoKnet with the ONNX file's path to create the model.

## Tutorial

Here is how you create the Knet model corresponding to the ONNX file, and perform a forward pass:

```julia
using Knet; using KnetONNX;

#provide the ONNX file's path
model = ONNXtoKnet("vgg.onnx");

#dummy input for prediction
x = ones(Float32,224,224,3,10)

#call KnetModel object with the model input
model(x) #the output is a 1000×10 Array{Float32,2}
```
## Supported Operations
- [x] ReLU
- [x] LeakyReLU
- [x] Conv
- [x] MaxPool
- [x] Dropout
- [x] Flatten
- [x] Gemm
- [x] Add
- [x] BatchNormalization
- [x] ImageScaler
- [ ] RNN

## Collaboration
Here are some cool ideas if you want to collaborate:
- Adding a new [KnetLayer.](https://github.com/ekinakyurek/KnetLayers.jl)
- Adding a new [Converter:](https://github.com/egeersu/KnetONNX/blob/master/converters.jl) ONNX Graph Node -> KnetLayer
- New KnetModel functionalities. We still can't handle models with multiple paths. 
- Testing ONNX models from the [zoo](https://github.com/onnx/models) and sending bug reports. 

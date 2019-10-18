<img src="https://github.com/onnx/onnx/blob/master/docs/ONNX_logo_main.png?raw=true" width="400">

KnetONNX lets you read an ONNX file and create a Knet Model that can be used for prediction, re-designed or re-trained.

If you are planning to move your models from PyTorch or Tensorflow to Knet, or simply desiring to play with popular pre-trained neural networks: KnetONNX provides that functionality. 

[Open Neural Network Exchange (ONNX)](https://onnx.ai/)
 is a community project created by Facebook and Microsoft. It provides a definition of an extensible computation graph model, as well as definitions of built-in operators and standard data types.
 
Operators are implemented externally to the graph, but the set of built-in operators are portable across frameworks. Every framework supporting ONNX will provide implementations of these operators on the applicable data types.

## Tutorial

Here is how you create the Knet model corresponding to the ONNX file, and perform a forward pass:

```
using KnetONNX;
model = ONNXtoKnet("vgg.onnx");
x = ones(Float32,224,224,3,10)
y = model(x)
```

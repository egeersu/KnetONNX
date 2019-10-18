<img src="https://github.com/onnx/onnx/blob/master/docs/ONNX_logo_main.png?raw=true" width="400">


[Open Neural Network Exchange (ONNX)](https://onnx.ai/)
 is a community project created by Facebook and Microsoft. It provides a definition of an extensible computation graph model, as well as definitions of built-in operators and standard data types.

Each computation dataflow graph is structured as a list of nodes that form an acyclic graph. Nodes have one or more inputs and one or more outputs. Each node is a call to an operator. The graph also has metadata to help document its purpose, author, etc.

Operators are implemented externally to the graph, but the set of built-in operators are portable across frameworks. Every framework supporting ONNX will provide implementations of these operators on the applicable data types.

This package lets you read ONNX files into Knet Models. It loads the pre-trained weights so the model will be ready for prediction. 

Here is how you create the Knet model corresponding to the ONNX file, and perform a forward pass:

'''
using KnetONNX;
model = ONNXtoKnet("vgg.onnx");
x = ones(Float32,224,224,3,10)
model(x)
''''

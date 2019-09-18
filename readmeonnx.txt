<img src="https://github.com/onnx/onnx/blob/master/docs/ONNX_logo_main.png?raw=true" width="400">


[Open Neural Network Exchange (ONNX)](https://onnx.ai/)
 is a community project created by Facebook and Microsoft. It provides a definition of an extensible computation graph model, as well as definitions of built-in operators and standard data types.

Each computation dataflow graph is structured as a list of nodes that form an acyclic graph. Nodes have one or more inputs and one or more outputs. Each node is a call to an operator. The graph also has metadata to help document its purpose, author, etc.

Operators are implemented externally to the graph, but the set of built-in operators are portable across frameworks. Every framework supporting ONNX will provide implementations of these operators on the applicable data types.

This package lets you read ONNX files into [KnetModel](https://github.com/egeersu/KnetONNX.jl/blob/master/KnetModel.jl)s. It loads the pre-trained weights and the KnetModel is therefore ready for prediction. KnetModels can be modified and re-trained. 

### Supported Operators:
1) ReLU
2) Leaky ReLU
3) Conv
4) MaxPool
5) Dropout
6) Flatten
7) Gemm
8) Add
9) Batch Normalization


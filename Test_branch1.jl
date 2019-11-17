#push!(LOAD_PATH, ".")
#using KnetONNX

# create the graph
file_path = "/Users/egeersu/Desktop/KnetONNX/@test_onnx_files/branch1.onnx"
graph1 = ONNXtoGraph(file_path)
PrintGraph(graph1)

# create the KnetModel
model = KnetModel(graph1)

# figure out the input size
layer1 = model.model_layers[1]
@show size(layer1.layer.mult.weight)

# dummy input
x1 = randn(100,5)

# forward pass with x1*
model(x1)

# check tensors
# if you are not using jupyter this prints the tensors in a pretty way:
PrintModelTensors(model)

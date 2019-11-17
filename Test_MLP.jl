push!(LOAD_PATH, ".")
using KnetONNX

file_path = "/Users/egeersu/Desktop/KnetONNX/@test_onnx_files/mlp.onnx"

# look at graph
graph1 = ONNXtoGraph(file_path)
PrintGraph(graph1)

# you can init just with path
model = KnetModel(file_path)

# you can also init with graph
model2 = KnetModel(graph1)


x1 = randn(100,5)
out = model(x1)

PrintModelTensors()

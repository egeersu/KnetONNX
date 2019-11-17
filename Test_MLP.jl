#this makes sure we can find KnetONNX.jl which is the main file of the package
push!(LOAD_PATH, ".")
#using KnetONNX

file_path = "/Users/egeersu/Desktop/KnetONNX/@test_onnx_files/mlp.onnx"
graph1 = ONNXtoGraph(file_path)
PrintGraph(graph1)
model = KnetModel(graph1)

x1 = randn(100,5)
out = model(x1)
@show size(out)

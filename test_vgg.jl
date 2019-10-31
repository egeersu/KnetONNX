push!(LOAD_PATH, ".")
using KnetONNX;
using Knet;

file_path = "/Users/egeersu/Desktop/ONNX_models/vgg16.onnx"
graph = ONNXtoGraph(file_path)
PrintGraph(graph)




push!(LOAD_PATH, ".")

# go to the module: KnetONNX.jl to see which functions are exported. 
using Knet
using KnetONNX;

file_path = "/Users/egeersu/Desktop/KnetONNX/@test_onnx_files/rnn.onnx"

graph1 = ONNXtoGraph(file_path)

PrintGraph(graph1)


push!(LOAD_PATH, ".");
using Knet; using KnetONNX;

g = ONNXtoGraph("MLP.onnx")
PrintGraph(g)

model = ONNXtoKnet("MLP.onnx");

x = ones(100,50)
model(x)

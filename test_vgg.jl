push!(LOAD_PATH, pwd())

#Print Operation Types
g1 = KnetONNX.ONNXtoGraph("vgg16.onnx");

#=
for n in g1.node
    println(n.op_type)
end

rnn1 = g1.node[15].op_type
=#

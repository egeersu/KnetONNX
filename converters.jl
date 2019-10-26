KL = include("./KnetLayers/src/KnetLayers.jl")

# Functions to be exported:

# ONNX file path -> Graph
function ONNXtoGraph(file)
    f = readproto(open(file), Proto.ModelProto());
    f = convert(f).graph
end

# ONNX file path -> KnetModel
function ONNXtoKnet(file)
    g = ONNXtoGraph(file)
    KnetModel(g)
end

# Prints the Graph in a pretty format
function PrintGraph(g)
    println("model inputs: ", (x->x.name).(g.input))
    println("model outputs: ", (x->x.name).(g.output))
    for (i, node) in enumerate(g.node)
        print("(op", i, ") ", node.op_type)
        println()
        for (i, input) in enumerate(node.input)
            println("\tinput", i, ": " , input)
        end
        for (i, output) in enumerate(node.output)
            println("\toutput", i, ": " , output)
        end
    end
end

# Given a node, calls the appropriate constructor for the corresponding (args, layer, outs)
function convert(node, g)
    if node.op_type == "Gemm"; return converter_gemm(node, g); end
    if node.op_type == "Add"; return converter_add(node, g); end
end





# Converters Begin Here
# A converter's inputs: graph node and the graph
# they return 3 elements:
    # - args:  the names of the tensors that will be needed for the calculations. These are just the names: strings.
    # - layer: a KnetLayer will be constructed. If the weights are in the initializer, the layer will be modified with them.
    # - outs:  the names of the tensors that are the outputs of the calculations. These are just the names: strings.


# GEMM
function converter_gemm(node, g)
    input1 = node.input[1]
    
    #the layer is a Knet Layer
    layer = KnetONNX.KnetLayers.Linear(input=1,output=1)
    
    # use g.initializer to modify KnetLayer
    w_name = node.input[2]
    b_name = node.input[3]
    w = g.initializer[w_name]
    b = g.initializer[b_name]
    layer.bias = b
    layer.mult.weight = transpose(w)
    
    # return input tensor NAMES, it is called args: [input1, ...]
    # you can take the inputs from model.tensors using these names
    args = [input1]
    outs = [node]
   
    # returns these 3, use these to create ModelLayer
    (args, layer, node.output)
end

# ADD
struct AddLayer; end
(a::AddLayer)(x,y) = x+y

function converter_add(node, g)
    args = node.input
    outs = node.output
    layer = AddLayer()
    return (args, layer, outs)
end

# RELU
function node_to_relu(node, weightdims)
    layer = KL.Linear(input=1,output=1)
    w_name = node.input[2]
    b_name = node.input[3]
    w = g.initializer[w_name]
    b = g.initializer[b_name]
    layer.bias = b
    layer.mult.weight = transpose(w)
    layer
end

# LEAKY RELU
#Node -> KnetLayer
function node_to_leakyrelu(node, g)
    alpha = node.attribute[:alpha]
    LeakyReLU(alpha)
end

# CONV
#conv1 = KnetONNX.KnetLayers.Conv(;height=3, width=3, inout = 3=>64)
#currently treating [1,1,1,1] padding as an integer 1, same for stride
function node_to_conv(node, weightdims, g)
    dw = weightdims[node.input[2]]
    padding = 0
    strides = 0
    if :pads in keys(node.attribute); padding = node.attribute[:pads][1]; end
    if :strides in keys(node.attribute); stride = node.attribute[:strides][1]; end

    layer = KL.Conv(height=dw[1],width=dw[2],inout=dw[3]=>dw[4]; padding = padding, stride = stride)

    if length(node.input) >= 2
        w_name = node.input[2]
        w = g.initializer[w_name]
        #might cause a problem later on with different convs
        layer.weight = w

    end
    if length(node.input) >= 3
        b_name = node.input[3]
        b = g.initializer[b_name]
        layer.bias = reshape(b, 1, 1, size(b)[1], 1)
    end
    layer
end

# POOL
#currently treating [1,1,1,1] padding as an integer 1, same for stride
function node_to_pool(node)
    stride = 0
    padding = 0

    if :pads in keys(node.attribute); padding = node.attribute[:pads][1]; end
    if :strides in keys(node.attribute); stride = node.attribute[:strides][1]; end

    KL.Pool(padding=padding, stride=stride)
end

# DROPOUT
function node_to_dropout(node, weightdims)
    KL.Dropout(p = node.attribute[:ratio])
end


# FLATTEN
function node_to_flatten(node, weightdims)
    KL.Flatten()
end


# BATCHNORM
function node_to_batchnorm(node, g)
    momentum = node.attribute[:momentum]
    epsilon = node.attribute[:epsilon]
    spatial = node.attribute[:spatial]

    scale = g.initializer[node.input[2]]
    B = g.initializer[node.input[3]]
    mean = g.initializer[node.input[4]]
    variance = g.initializer[node.input[5]]

    KL.BatchNorm(length(scale); momentum=momentum, mean=mean, var=variance)
end


# IMAGE SCALER

function node_to_imagescaler(node, g)
    bias = node.attribute[:bias]
    scale = node.attribute[:scale]
    #ScalerLayer(x) = scale .* x
end


# RNN

function node_to_RNN(node, g)
    activations = node.attribute[:activations]
    hidden_size = node.attribute[:hidden_size]
end


# SQUEEZE
function node_to_squeeze(node)
    squeeze_layer(node.attribute[:axes])
end

mutable struct squeeze_layer
    axes
end

function (s::squeeze_layer)(x)
    new_size = []
    for (i, dim) in enumerate(size(x))
        if dim>1; push!(new_size, dim); end
    end
    new_size = (new_size...,)
    reshape(x, new_size)
end


# UNSQUEEZE
function node_to_unsqueeze(node)
    unsqueeze_layer(node.attribute[:axes])
end

mutable struct unsqueeze_layer
    axes
end


function (u::unsqueeze_layer)(x)
    data = [t for t in size(x)]
    axes = [a+1 for a in u.axes]
    for i in axes; insert!(data, i, 1); end
    new_size = (data...,)
    reshape(x, new_size)
end


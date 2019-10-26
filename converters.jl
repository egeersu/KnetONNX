KL = include("./KnetLayers/src/KnetLayers.jl")

# ONNX file -> Graph
function ONNXtoGraph(file)
    f = readproto(open(file), Proto.ModelProto());
    f = convert(f).graph
end

# ONNX file -> KnetModel
function ONNXtoKnet(file)
    g = ONNXtoGraph(file)
    KnetModel(g)
end

# ONNX file -> ChainModel
function ONNXtoChain(file)
    g = ONNXtoGraph(file)
    lst = GraphtoList(g)
    ChainModel(lst)
end

# prints an ONNX graph
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

# doesnt do much?
function get_weight_dims(g)
    tensordims = Dict()
    for k in keys(g.initializer)
        tensordims[k] = size(g.initializer[k])
    end
    tensordims
end

# delete this once KnetModel Works (only works for ChainModel)
function GraphtoList(g)
    weightdims = get_weight_dims(g);
    layers = []
    for node in g.node
        if node.op_type == "Relu"; push!(layers, node_to_relu(node, weightdims)); end
        if node.op_type == "LeakyReLU"; push!(layers, node_to_leakyrelu(node, g)); end
        if node.op_type == "Conv"; push!(layers, node_to_conv(node, weightdims, g)); end
        if node.op_type == "MaxPool"; push!(layers, node_to_pool(node)); end
        if node.op_type == "Dropout"; push!(layers, node_to_dropout(node, weightdims)); end
        if node.op_type == "Flatten"; push!(layers, node_to_flatten(node, weightdims)); end
        if node.op_type == "Gemm"; push!(layers, node_to_gemm(node, weightdims, g)); end
        if node.op_type == "Add"; push!(layers, node_to_add(node, g)); end
        #if node.op_type == "BatchNormalization"; push!(layers, node_to_batchnorm(node, g)); end
        if node.op_type == "ImageScaler"; push!(layers, node_to_imagescaler(node,g)); end
        if node.op_type == "RNN"; push!(layers, node_to_RNN(node,g)); end
        if node.op_type == "Squeeze"; push!(layers, node_to_squeeze(node)); end
        if node.op_type == "Unsqueeze"; push!(layers, node_to_unsqueeze(node)); end
    end
    layers
end


# graph node, graph -> ModelLayer
function node_to_layer(node, g)
    if node.op_type == "Gemm"; return node_to_gemm(node, g); end
end

#returns (names of tensors used for forward pass, KnetLayer, output tensor names)
function node_to_gemm(node, g)
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

#get weights from dictionary
function checkweight(g, w)
    if w in keys(g.initializer); g.initializer[w]; else; "weight not initialized"; end
end

#Node -> KnetLayer
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

#Node -> KnetLayer
function node_to_leakyrelu(node, g)
    alpha = node.attribute[:alpha]
    LeakyReLU(alpha)
end


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

#currently treating [1,1,1,1] padding as an integer 1, same for stride
function node_to_pool(node)
    stride = 0
    padding = 0

    if :pads in keys(node.attribute); padding = node.attribute[:pads][1]; end
    if :strides in keys(node.attribute); stride = node.attribute[:strides][1]; end

    KL.Pool(padding=padding, stride=stride)
end

function node_to_dropout(node, weightdims)
    KL.Dropout(p = node.attribute[:ratio])
end

function node_to_flatten(node, weightdims)
    KL.Flatten()
end

function node_to_Gemm(inputs)
    x = inputs[1]
end


function node_to_gemm(node, weightdims, g)
    layer = KL.Linear(input=1,output=1)
    w_name = node.input[2]
    b_name = node.input[3]
    w = g.initializer[w_name]
    b = g.initializer[b_name]
    layer.bias = b
    layer.mult.weight = transpose(w)
    layer
end

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

function node_to_imagescaler(node, g)
    bias = node.attribute[:bias]
    scale = node.attribute[:scale]
    #ScalerLayer(x) = scale .* x
end

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


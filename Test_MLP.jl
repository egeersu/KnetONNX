#this makes sure we can find KnetONNX.jl which is the main file of the package
push!(LOAD_PATH, ".")
using KnetONNX
"""
#the path to mlp.onnx which is in folder: @test_onnx_files
# it is a simple multi-layer-perceptron exported from PyTorch
file_path = "/Users/egeersu/Desktop/KnetONNX/@test_onnx_files/mlp.onnx"

# we can turn it into a graph (KnetONNX.graph)
graph1 = ONNXtoGraph(file_path)

# print the graph in a human readable way
PrintGraph(graph1)

# Call the KnetModel constructor with the graph we just crated
# it creates a the corresponding KnetModel
model = KnetModel(graph1)

# Let's see what this KnetModel is!
# You can also read the documentation by uncommenting:
# @doc KnetModel

#A KnetModel has 4 fields:
# model.model_inputs
# model.model_outputs
# model.model_layers
# model.tensors

# realize that model_inputs and model_outputs are the same as the graph we just printed.
@show model.model_inputs
@show model.model_outputs

# model.layers is a list of all the layers in our model. The order does not matter.
# When you print it things get ugly so let us just check the length of it.
# It should be 4, since we have 4 operations in our graph
@show length(model.model_layers)

# let's check out what these layers really are by looking at the first layer.
layer1 = model.model_layers[1]
@show typeof(layer1)
# turns out it's a ModelLayer!
# I will explain it later on.

# The 4th and the last field is tensors.
# It's simply a dictionary: the tensor's name => the tensor itself
# It includes the input tensors, output tensors, and all other intermediate tensors.
# Whenever we do a calculation we go to model.tensors and update the corresponding tensors.
# For our MLP model, we expect model.tensors["8"] to hold the output to our model in the end.

# Use PrintModelTensors(model) to see how the tensors are doing:
PrintModelTensors(model)
# Turns out they are all nothing. That makes sense since we did not calculate anything yet.

# Now let us look at what a ModelLayer is
#@doc ModelLayer
#@doc KnetModel







#PrintModelTensors(model)
#@show summary(model(x1))
#PrintModelTensors(model)
"""

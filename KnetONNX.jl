module KnetONNX

using Pkg
Pkg.add("ProtoBuf")
Pkg.add("MacroTools")
Pkg.add("DataFlow")
Pkg.add("Statistics")

using ProtoBuf
using MacroTools
using DataFlow
using Statistics

include("onnx_pb.jl")
include("convert.jl")
include("new_types.jl")
include("graph/graph.jl")
include("converters.jl"); export ONNXtoChain, ONNXtoGraph;
#include("KnetModel.jl"); 
end

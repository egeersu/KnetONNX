using BSON

rawproto(io::IO) = readproto(io, Proto.ModelProto())
rawproto(path::String) = open(rawproto, path)

"""
Retrieve only the useful information from a AttributeProto
object into a Dict format.
"""
function convert_model(x::Proto.AttributeProto)
    if (x._type != 0)
        field = [:f, :i, :s, :t, :g, :floats, :ints, :strings, :tensors, :graphs][x._type]
        return Symbol(x.name) => getfield(x, field)
    end
end

convert_array(as) = Dict(convert_model(a) for a in as)

"""
Convert a ValueInfoProto to  ValueInfo.
"""
function convert_model(model::Proto.ValueInfoProto)
    a = Types.ValueInfo(model.name, model.doc_string)
    return a
end

"""
Convert an OperatorSetIdProto to Dict.
"""
function convert_model(model::KnetONNX.Proto.OperatorSetIdProto)
    a = Dict{Symbol, Any}()
    fields = [:domain, :version]
    for ele in fields
        a[ele] = getfield(model, ele)
    end
    return a
end

"""
Convert a StringStringEntryProto to Dict.
"""
function convert_model(model::KnetONNX.Proto.StringStringEntryProto)
    a = Dict{Symbol, Any}()
    fields = [:key, :value]
    for ele in fields
        a[ele] = getfield(model, ele)
    end
    return a
end

"""
Get the array from a TensorProto object.
"""
function get_array(x::Proto.TensorProto)
    if (x.data_type == 1)
        if !isempty(x.float_data)
            x = reshape(reinterpret(Float32, x.float_data), reverse(x.dims)...)
        else
            x = reshape(reinterpret(Float32, x.raw_data), reverse(x.dims)...)
        end
        return x
    end
    if x.data_type == 7
        if !isempty(x.raw_data)
            x = reshape(reinterpret(Int64, x.raw_data), reverse(x.dims)...)
        else
            x = reshape(reinterpret(Int64, x.int64_data), reverse(x.dims)...)
        end
        return x
    end
    if x.data_type == 9
        x = reshape(reinterpret(Int8, x.raw_data), reverse(x.dims)...)
        return x
    end
    if x.data_type == 6
         x = reshape(reinterpret(Int32, x.raw_data), reverse(x.dims)...)
        return x
    end
    if x.data_type == 11
        if !isempty(x.raw_data)
            x = reshape(reinterpret(Float64, x.raw_data), reverse(x.dims)...)
        else
            x = Base.convert(Array{Float32, N} where N, reshape(x.double_data , reverse(x.dims)...))
        end
        return x
    end
    if x.data_type == 10
        x = reshape(reinterpret(Float16, x.raw_data), reverse(x.dims)...)
        return x
    end
end

"""
Convert a ModelProto object to a Model type.
"""
function convert(model::Proto.ModelProto)
    # conversion for opset_import
    arr1 = Array{Any, 1}()
    for ele in model.opset_import
        push!(arr1, convert_model(ele))
    end

    # conversion for stringstringentry proto
    arr2 = Array{Any, 1}()
    for ele in model.metadata_props
        push!(arr2, convert_model(ele))
    end

    m = Types.Model(model.ir_version,
                arr1, model.producer_name,
                model.producer_version,
                model.domain, model.model_version,
                model.doc_string, convert(model.graph),
                arr2)
    return m
end

"""
Convert a GraphProto object to Graph type.
"""
function convert(model::Proto.GraphProto)
    # conversion for vector of nodeproto
    arr1 = Array{Any, 1}()
    for ele in model.node
        push!(arr1, convert(ele))
    end

    # conversion for vector of tensorproto
    arr2 = Dict{Any, Any}()
    for ele in model.initializer
        arr2[ele.name] = get_array(ele)
    end

    #conversion for vector of valueinfoproto
    arr3 = Array{Types.ValueInfo ,1}()
    for ele in model.input
        push!(arr3, convert_model(ele))
    end

    arr4 = Array{Types.ValueInfo ,1}()
    for ele in model.output
        push!(arr4, convert_model(ele))
    end

    arr5 = Array{Types.ValueInfo ,1}()
    for ele in model.value_info
        push!(arr5, convert_model(ele))
    end

    m = Types.Graph(arr1,
            model.name,
            arr2, model.doc_string,
            arr3, arr4, arr5)
    return m
end

"""
Convert a Proto.NodeProto to Node type.
"""
function convert(model::Proto.NodeProto)
    # Conversion of attribute
    arr1 = convert_array(model.attribute)

    m = Types.Node(model.input,
            model.output,
            model.name,
            model.op_type,
            model.domain,
            arr1,
            model.doc_string)
    return m
end

function parent(path)
    temp = split(path, "/")
    res = ""
    for element in temp
        if (element != temp[end])
            res = res * element * "/"
        end
    end
    return res
end

mutable struct KnetModel 
    layers
end

(k::KnetModel)(x) = (for l in k.layers; x = l(x); end; x)

function (k::KnetModel)(x)
    i = 1
    println("input size: ", size(x))
    for l in k.layers
        x = l(x)
        println("layer ", i, " output size: ", size(x))
        i+=1
    end
    x
end
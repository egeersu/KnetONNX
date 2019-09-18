mutable struct KnetModel 
    layers
end

(k::KnetModel)(x) = (for l in k.layers; x = l(x); end; x)

function (k::KnetModel)(x)
    println(size(x))
    for l in k.layers
        x = l(x)
        println(size(x))
    end
    x
end
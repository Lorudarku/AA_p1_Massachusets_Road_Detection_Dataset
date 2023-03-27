
using FileIO
using Images
using ImageView
using Flux.Losses

# Documentacion en:
#   https://juliaimages.org/
#   https://juliaimages.org/latest/function_reference/

#Bucle
# Cargar una imagen
dataset = []
imagen = load("./it_c/10078660_150_0.tiff")
push!(dataset, imagen)
#Fin bucle

#Cargar el dataset como input
inputs = dataset[:,1:4];
#Transformar los input en float64
inputs = convert(Array{Float64,2},inputs);
println("Tamaño de la matriz de entradas: ", size(inputs,1), "x", size(inputs,2), " de tipo ", typeof(inputs));


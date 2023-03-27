using FileIO
using FFTW
using Random
using Flux
using Flux.Losses
using Plots
using Statistics
using DelimitedFiles
using ScikitLearn
using JLD2
using Images

const tamWindow = 15;
const saltoVentana = 5;

function imageToColorArray(image::Array{RGB{Normed{UInt8,8}},2})
    matrix = Array{Float64, 3}(undef, size(image,1), size(image,2), 3)
    matrix[:,:,1] = convert(Array{Float64,2}, red.(image));
    matrix[:,:,2] = convert(Array{Float64,2}, green.(image));
    matrix[:,:,3] = convert(Array{Float64,2}, blue.(image));
    return matrix;
end;
imageToColorArray(image::Array{RGBA{Normed{UInt8,8}},2}) = imageToColorArray(RGB.(image));

function mediaDesviacion(ventana)
    #Mirar como funcionan las funciones ##################################################################################################
    media = mean(ventana);
    desviacion = std(ventana);

    [media;desviacion]
end


#Calculamos las ventanas para cada imagen
function transformar(matrizRojo,matrizVerde,matrizAzul)

    #Calculamos la media y desviacion tipica para cada componente de color
    mDR = mediaDesviacion(matrizRojo);
    mDG = mediaDesviacion(matrizVerde);
    mDB = mediaDesviacion(matrizAzul);

    [mDR,mDG,mDB]
end

#Sacamos las caracteristicas
function estraccionCaracteristicas()
    inputs = [];
    targets = [];
    itc = readdir("./pruebas");
    gtc = readdir("./gtp");
    pixelCentro = ceil(tamWindow / 2);
    l=0;

    for images in itc

        image = load("./pruebas/"*images);
        matrix = imageToColorArray(image);
        #image = convert(Array{Float64,2}, image);
        #Bucle ventana por toda la imagen
        saltoX = 0;
        l = round(Int,sqrt(length(image)))

        for x in 1:l
            saltoY = 0;

            for y in 1:l
                #Calculamos las ventanas para cada imagen
                #println("X-->",tamWindow + saltoX)
                #println("Y-->",tamWindow + saltoY)
                windowR = matrix[x:tamWindow + saltoX, y:tamWindow + saltoY,1];
                windowG = matrix[x:tamWindow + saltoX, y:tamWindow + saltoY,2];
                windowB = matrix[x:tamWindow + saltoX, y:tamWindow + saltoY,3];
                
                
                
                #inputs[1*][2*][3*]
                #Dim 1: Dimension con la componente R con su media y desviacion tipica
                #Dim 2: Dimension con la componente G con su media y desviacion tipica
                #Dim 3: Dimension con la componente B con su media y desviacion tipica

                push!(inputs,transformar(windowR,windowG,windowB));

                y = y + saltoVentana;
                saltoY = saltoY + saltoVentana;
                if((tamWindow + saltoY)>round(Int,sqrt(length(image))))
                    break
                end

            end
            x = x + saltoVentana;
            saltoX = saltoX + saltoVentana;
            if((tamWindow + saltoX)>round(Int,sqrt(length(image))))
                break
            end
        end
        
    end

    println("Imagenes cargadas");

    inputs=hcat(inputs...);
    targets=hcat(targets...);
    [permutedims(inputs),permutedims(targets)]
end


estraccionCaracteristicas();





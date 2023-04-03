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

#Calculamos si un punto central es carretera o no
function esCarretera(matrizRojo,matrizVerde,matrizAzul)
    carretera = false;
    pixelCentro = round(Int,ceil(tamWindow / 2));
    pixelCentroUnaDimension = round(Int,tamWindow*(pixelCentro-1)+pixelCentro);

    if((matrizRojo[pixelCentroUnaDimension] == 1) & (matrizVerde[pixelCentroUnaDimension] == 1) & (matrizAzul[pixelCentroUnaDimension] == 1))
        carretera = true
        #println("Pixel: ",matrizRojo[pixelCentroUnaDimension]," ",matrizVerde[pixelCentroUnaDimension]," ",matrizAzul[pixelCentroUnaDimension])
        #println(carretera)
    end
    return carretera
end

#Sacamos las caracteristicas
function estraccionCaracteristicas()
    inputs = [];
    targets = [];
    itc = readdir("./pruebas");
    gtc = readdir("./gtp");
    tam = length(itc);
    l=0;
    i = 0;

    for images in itc

        println("Cargando Imagenes: ",(i/tam*100),"%");

        image = load("./pruebas/"*images);
        matrix = imageToColorArray(image);
        #image = convert(Array{Float64,2}, image);
        #Bucle ventana por toda la imagen
        saltoX = 0;
        posy = 1;
        posx = 1;
        l = round(Int,sqrt(length(image)));

        for x in 1:l
            posy = 1;
            saltoY = 0;

            for y in 1:l
                #Calculamos las ventanas para cada imagen
                #println("X-->",tamWindow + saltoX)
                #println("Y-->",tamWindow + saltoY)
                windowR = matrix[posx:tamWindow + saltoX, posy:tamWindow + saltoY,1];
                windowG = matrix[posx:tamWindow + saltoX, posy:tamWindow + saltoY,2];
                windowB = matrix[posx:tamWindow + saltoX, posy:tamWindow + saltoY,3];
                
                #inputs[1*][2*][3*]
                #Dim 1: Dimension con la componente R con su media y desviacion tipica
                #Dim 2: Dimension con la componente G con su media y desviacion tipica
                #Dim 3: Dimension con la componente B con su media y desviacion tipica

                push!(inputs,transformar(windowR,windowG,windowB));

                posy = posy + saltoVentana;
                saltoY = saltoY + saltoVentana;
                if((tamWindow + saltoY)>round(Int,sqrt(length(image))))
                    break
                end

            end
            posx = posx + saltoVentana;
            saltoX = saltoX + saltoVentana;
            if((tamWindow + saltoX)>round(Int,sqrt(length(image))))
                break
            end
        end
        
        i=+;
        
    end

    println("Imagenes cargadas 100%, cargando gt...");
    tam = 0;
    i = 0;

    for gts in gtc

        gt = load("./gtp/"*gts);
        matrixgt = imageToColorArray(gt);
        #image = convert(Array{Float64,2}, image);
        #Bucle ventana por toda la imagen
        saltoX = 0;
        posy = 1;
        posx = 1;
        l = round(Int,sqrt(length(gt)));

        for x in 1:l
            saltoY = 0;
            posy = 1;

            for y in 1:l
                #println("Cargando Gt $tam: ",(i/(l*l/saltoVentana/saltoVentana)*100),"%");

                #Calculamos las ventanas para cada imagen
                #println("X-->",tamWindow + saltoX)
                #println("Y-->",tamWindow + saltoY)
                windowR = matrixgt[posx:tamWindow + saltoX, posy:tamWindow + saltoY,1];
                windowG = matrixgt[posx:tamWindow + saltoX, posy:tamWindow + saltoY,2];
                windowB = matrixgt[posx:tamWindow + saltoX, posy:tamWindow + saltoY,3];
                imgsave = RGB.(windowR, windowG, windowB);

                #println("$posx","-$(tamWindow + saltoX)","y $posy","-$(tamWindow + saltoY)");

                name = "ventana_$posx.$posy.tif"
                if (esCarretera(windowR,windowG,windowB) == true)
                    #save("./positivos/"*name, imgsave)
                    push!(targets,"positivo");
        
                else
                    #save("./negativos/"*name, imgsave)
                    push!(targets,"negativo");

                end
                
                #inputs[1*][2*][3*]
                #Dim 1: Dimension con la componente R con su media y desviacion tipica
                #Dim 2: Dimension con la componente G con su media y desviacion tipica
                #Dim 3: Dimension con la componente B con su media y desviacion tipica


                posy = posy + saltoVentana;
                saltoY = saltoY + saltoVentana;
                i += 1;

                if((tamWindow + saltoY)>round(Int,sqrt(length(gt))))
                    break
                end

            end
            posx = posx + saltoVentana;
            saltoX = saltoX + saltoVentana;
            if((tamWindow + saltoX)>round(Int,sqrt(length(gt))))
                break
            end
        end
        tam=+;        
    end
    println("Gt cargado 100%.");

    inputs=hcat(inputs...);
    targets=hcat(targets...);
    [permutedims(inputs),permutedims(targets)]
end

# Dividir los arrays inputs y targets en otros 2 para entrenamiento y test
function holdOut(inputs,targets)
    in=[];
    testIn=[];
    tar=[];
    testTar=[];

    aux = bitrand(roundsize(inputs,1));
    l = length(inputs);
    trestLenght = round(Int,l*0.1)

    for i in 1:l

        if length(testIn) <= trestLenght
            if aux[i] == 1
                push!(in,inputs[i,:]);
                push!(tar,targets[i,:]);
            else
                push!(testIn,inputs[i,:]);
                push!(testTar,targets[i,:]);
            end
        else
            push!(in,inputs[i,:]);
            push!(tar,targets[i,:]);
        end
        
    end
    [in,tar,testIn,testTar]
end
# Funcion usada para transformar o array de strings nun array de targets validos para a Arn
function normalizarCaracteristicas(normalizar)
    result=[];
    p = "positivo";

    for i in 1:size(normalizar,1)
        if normalizar[i] == p
            push!(result,1.);
        else
            push!(result,0.);  
        end
  
    end;
    hcat(result...);
end


function normalizar(value,max,min)
    return ((value-min)/(max-min))
end

#  Entradas, targets , topologia, tasa de error minima y ciclos maximos
function sistemaRRNNAA(inputs,targets,topology,minerror, maxIt)

    #Randomizar y normalizar
    aux = holdOut(inputs,targets);
    trainingIn = hcat(aux[1]...);
    trainingTar = hcat(aux[2]...);
    testIn = hcat(aux[3]...);
    testTar=hcat(aux[4]...);
    ann = Chain();

    for i in 1:size(trainingIn,1)

        max = maximum(trainingIn[i,:]);
        min = minimum(trainingIn[i,:]);
        trainingIn[i,:] = normalizar(trainingIn[i,:],max,min);
        testIn[i,:] = normalizar(testIn[i,:],max,min);
    end
    #####################################################Revisado hasta aqui#############################################################################
    for i=1:size(trainingTar,1)

        max=maximum(trainingTar[i,:]);
        min=minimum(trainingTar[i,:]);
        trainingTar[i,:]=normalizar.(trainingTar[i,:],max,min);
        testTar[i,:]=normalizar.(testTar[i,:],max,min);
    end

    aux=holdOut(trainingIn',trainingTar');
    trainingIn=hcat(aux[1]...);
    trainingTar=hcat(aux[2]...);
    validationIn=hcat(aux[3]...);
    validationTar=hcat(aux[4]...);
    eTest=[];
    eTraining=[];
    eValidation=[];
    min=1;
    actual=0;
    #Creación Arn
    numInputsLayer=size(inputs,2);
    for numOutputsLayer = topology
        ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, σ) );
        numInputsLayer = numOutputsLayer;
    end;
    numOutputs=size(targets,2);
    if (numOutputs == 1)
        ann = Chain(ann..., Dense(numInputsLayer, 1, σ));
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity));
        ann = Chain(ann..., softmax);
    end;
    loss(x, y) = crossentropy(ann(x), y);
    #Entrenamento e calculo de erros
    Flux.train!(loss, params(ann), [(trainingIn, trainingTar)], ADAM(0.01));
    e=confusionMatrix(round.(ann(trainingIn))',trainingTar')
    push!(eTraining,e);
    e=confusionMatrix(round.(ann(testIn))',testTar')
    push!(eTest,e);
    e=confusionMatrix(round.(ann(validationIn))',validationTar')
    push!(eValidation,e);
    best=deepcopy(ann);
    while (e[2]>parada) & (actual<maxMellora)
        Flux.train!(loss, params(ann), [(trainingIn, trainingTar)], ADAM(0.01));
        e=confusionMatrix(round.(ann(trainingIn))',trainingTar')
        push!(eTraining,e);
        e=confusionMatrix(round.(ann(testIn))',testTar')
        push!(eTest,e);
        e=confusionMatrix(round.(ann(validationIn))',validationTar')
        push!(eValidation,e);
        if e[2]<min
            min=e[2];
            actual=0;
            best=deepcopy(ann);
        else
            actual=actual + 1;
        end
    end;
    #Devolvese a mellor Arn e os erros de cada fase do entrenamento de Test, Entrenamento e Validación
    [best,eTest,eTraining,eValidation,e]
end

caracteristicas = estraccionCaracteristicas();
caracteristicas[2] = normalizarCaracteristicas(caracteristicas[2]);

RRNNAA = RRNNAA(caracteristicas[1],caracteristicas[2],[25],0.15,100)








using FileIO
using FFTW
using Random
using Flux
using Flux.Losses
using Plots
using StatsPlots
using Statistics
using DelimitedFiles
using ScikitLearn
using JLD2
using Images

const tamWindow = 10;
const saltoVentana = 3;
const dirIt = "./p1/";
const dirGt = "./p2/";


function imageToColorArray(image::Array{RGB{Normed{UInt8,8}},2})
    matrix = Array{Float64, 3}(undef, size(image,1), size(image,2), 3)
    matrix[:,:,1] = convert(Array{Float64,2}, red.(image));
    matrix[:,:,2] = convert(Array{Float64,2}, green.(image));
    matrix[:,:,3] = convert(Array{Float64,2}, blue.(image));
    return matrix;
end;
imageToColorArray(image::Array{RGBA{Normed{UInt8,8}},2}) = imageToColorArray(RGB.(image));

function mediaDesviacion(ventana)
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

    [mDR[1],mDR[2],mDG[1],mDG[2],mDB[1],mDB[2]]
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
    itc = readdir(dirIt);
    gtc = readdir(dirGt);
    tam = length(itc);
    l=0;
    i = 0.;

    for images in itc
        v = (i/tam*100);
        i+=1;
        println("Cargando Imagenes: $v%");
        ruta = dirIt*"/"
        ruta = joinpath(dirIt,images)
        image = load(ruta);
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
        
       
    end

    println("Imagenes cargadas 100%, cargando gt...");
    i = 0.;
    v = 0.;

    for gts in gtc

        v = (i/tam*100);
        i+=1;        
        println("Cargando Imagenes: $v%");
        ruta = dirGt*"/"
        ruta = joinpath(dirGt ,gts)
        gt = load(ruta);
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

                posy = posy + saltoVentana;
                saltoY = saltoY + saltoVentana;

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

    aux = bitrand(size(inputs,1));

    l = length(aux);
    trestLenght = round(Int,l*0.2)
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

    @assert (size(in,1)==size(tar,1)) "Las matrices de entradas y
        salidas deseadas no tienen el mismo número de filas"
    @assert (size(testIn,1)==size(testTar,1)) "Las matrices de entradas y
        salidas deseadas no tienen el mismo número de filas"

    [in,tar,testIn,testTar]
end

# Funcion usada para transformar a un formato valido
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

    result=hcat(result...);
    permutedims(result)
end

function normalizar2(v,media,des)
    (v-media)/des
end

function normalizar1(v,max,min)
    (v-min)/(max-min)
end


function vPositivos(a,b)
    (a==true) & (b==true)
end

function fPositivos(a,b)
    (a==true) & (b==false)
end

function vNegativos(a,b)
    (a==false) & (b==false)
end

function fNegativos(a,b)
    (a==false) & (b==true)
end

function confusionMatrix(in,tar)
    vp = sum(vPositivos.(in,tar));
    fp = sum(fPositivos.(in,tar));
    vn = sum(vNegativos.(in,tar));
    fn = sum(fNegativos.(in,tar));

    matrizConfusion = Matrix{Int64}(undef, 2, 2)
    matrizConfusion[1,1] = vn;
    matrizConfusion[1,2] = fp;
    matrizConfusion[2,1] = fn;
    matrizConfusion[2,2] = vp;

    if((vp != 0) || (vn != 0))
        print("vp ",vp, " fp ",fp," vn ",vn, " fn ",fn);
        println();
    end

    precision = (vn+vp)/(vn+vp+fn+fp);
    tasaE = (fn+fp)/(vn+vp+fn+fp);
    sensibilidad = vp/(fn+vp);
    especificidad = vn/(fp+vn);
    valorPP = vp/(vp+fp);
    valorPN = vn/(fn+vn);
    fScore = 2/((1/valorPP)+(1/sensibilidad));

    [precision , tasaE , sensibilidad , especificidad , valorPP , valorPN , fScore , matrizConfusion]
end
function ploteable1(t)
    t[1]
end
function ploteable(t)
    t[2]
end

function normalizar(a,u)
    a>u
end

function confusionMatrix(in,tar,umbral)
    aux = normalizar.(in,umbral);
    confusionMatrix(aux,tar)
end
#  Entradas, targets , topologia, tasa de error minima y ciclos maximos
function RRNNAA(inputs,targets,topology,minerror, maxIt)

    #Randomizar y normalizar
    aux = holdOut(inputs,targets);
    trainingIn = hcat(aux[1]...);
    trainingTar = hcat(aux[2]...);
    testIn = hcat(aux[3]...);
    testTar = hcat(aux[4]...);

    println("Dimensiones :",size(trainingIn));
    println("Ventanas :",size(trainingIn,1));
    println("Inputs :",size(trainingIn,2));
    println("Sin normalizar: ",trainingIn[:,1]);
    println("Sin normalizar: ",testIn[:,1]);


    ann = Chain();

    for i in 1:size(trainingIn,1)
        max = maximum(trainingIn[i,:]);
        min = minimum(trainingIn[i,:]);
        #media = mean(trainingIn[i,:]);
        #des = std(trainingIn[i,:]);
        trainingIn[i,:] = normalizar1.(trainingIn[i,:],max,min);
        testIn[i,:] = normalizar1.(testIn[i,:],max,min);
    end

    println("Normalizado: ",trainingIn[:,1]);
    println("Normalizado: ",testIn[:,1]);

    aux = holdOut(trainingIn',trainingTar');
    
    trainingIn = hcat(aux[1]...);
    trainingTar = hcat(aux[2]...);
    validationIn = hcat(aux[3]...);
    validationTar = hcat(aux[4]...);
    ########Comprobaciones
    println("Dimensiones ':",size(trainingIn'));
    println("Inputs ':",size(trainingIn',2));
    println("Ventanas ':",size(trainingIn',1));
    

    
    ########
    errTest = [];
    errTraining = [];
    errValidation = [];
    min = 1;
    it = 0;
    #####################################################Revisado hasta aqui#############################################################################    
    #Creación Arn
    numInputsLayer = size(inputs,2);
    for numOutputsLayer = topology
        ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, σ) );
        numInputsLayer = numOutputsLayer;
    end;
    ann = Chain(ann..., Dense(numInputsLayer, 1, σ));
    println("RN: ",ann);

    loss(x, y) = binarycrossentropy(ann(x), y);

    #Entrenamiento y calculo del error
    Flux.train!(loss, Flux.params(ann), [(trainingIn, trainingTar)], ADAM(0.005));
    err = confusionMatrix(round.(ann(trainingIn))',trainingTar')
    push!(errTraining,err);

    err = confusionMatrix(round.(ann(testIn))',testTar')
    push!(errTest,err);

    err = confusionMatrix(round.(ann(validationIn))',validationTar')
    push!(errValidation,err);
    best = deepcopy(ann);
    println("Error: ",err[2] )

    while ((err[2] > minerror) && (it < maxIt))
        Flux.train!(loss, Flux.params(ann), [(trainingIn, trainingTar)], ADAM(0.005));
        err = confusionMatrix(round.(ann(trainingIn))',trainingTar')
        push!(errTraining,err);

        err = confusionMatrix(round.(ann(testIn))',testTar')
        push!(errTest,err);

        err = confusionMatrix(round.(ann(validationIn))',validationTar')
        push!(errValidation,err);
        println("Error: ",err[2] )

        if err[2] < min
            min = err[2];
            it = 0;
            best = deepcopy(ann);
        else
            it = it + 1;
            println(it);
        end
    end;
    
    println(errTest)
    [best,errTest,errTraining,errValidation,err] 
end

caracteristicas = estraccionCaracteristicas();
    
caracteristicas[2] = normalizarCaracteristicas(caracteristicas[2]);

redNeuronal = RRNNAA(caracteristicas[1],caracteristicas[2],[6 12 3],0.029,100)

# Graficar los errores
g = plot();

plot!(ploteable.(redNeuronal[2]), label="Test Error");
plot!(ploteable.(redNeuronal[3]), label="Training Error")
plot!(ploteable.(redNeuronal[4]), label="Validation Error")

display(g);

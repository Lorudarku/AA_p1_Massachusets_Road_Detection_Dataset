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
using DecisionTree
using GraphViz
using PyCall
using JLD2
using Flux
using Random
using Random:seed!

@sk_import model_selection: KFold

const tamWindow = 60;
const tamWindow2 = 7;
const tamWindow3 = 3;


const saltoVentana = 3;
const dirIt = "./datasets/inputs/";
const dirGt = "./datasets/targets/";


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
function transformar(matrizRojo,matrizVerde,matrizAzul,matrizRojo2,matrizVerde2,matrizAzul2,matrizRojo3,matrizVerde3,matrizAzul3)

    #Calculamos la media y desviacion tipica para cada componente de color
    mDR = mediaDesviacion(matrizRojo);
    mDG = mediaDesviacion(matrizVerde);
    mDB = mediaDesviacion(matrizAzul);

    mDR2 = mediaDesviacion(matrizRojo2);
    mDG2 = mediaDesviacion(matrizVerde2);
    mDB2 = mediaDesviacion(matrizAzul2);

    mDR3 = mediaDesviacion(matrizRojo3);
    mDG3 = mediaDesviacion(matrizVerde3);
    mDB3 = mediaDesviacion(matrizAzul3);
    
    [mDR[1],mDR[2],mDG[1],mDG[2],mDB[1],mDB[2], mDR2[1],mDR2[2],mDG2[1],mDG2[2],mDB2[1],mDB2[2], mDR3[1],mDR3[2],mDG3[1],mDG3[2],mDB3[1],mDB3[2]]
end

#Calculamos si un punto central es carretera o no
function esCarretera(windowC)
    carretera = false;
    pixelCentro = round(Int,ceil(tamWindow / 2));

    if(windowC[pixelCentro,pixelCentro] != 0)
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
    carretera = 0;
    noCarretera = 0;
    esCar = true
    esNoCar = true
    pixelCentro = round(Int,ceil(tamWindow / 2));
    winDiff = round(Int,ceil((tamWindow-tamWindow2) / 2))
    winDiff2 = round(Int,ceil((tamWindow-tamWindow3) / 2))

    l=0;
    i = 0.;

    for (images,gts) in zip(itc,gtc)
        v = (i/tam*100);
        i+=1;
        println("Cargando Imagenes: $v%");
        ruta = dirIt*"/"
        ruta = joinpath(dirIt,images)
        image = load(ruta);
        matrix = imageToColorArray(image);
        auxMatrix = matrix
    
        ruta2 = dirGt*"/"
        ruta2 = joinpath(dirGt ,gts)
        gt = load(ruta2);
        matrixgt = Gray.(gt)

        #image = convert(Array{Float64,2}, image);
        #Bucle ventana por toda la imagen
        saltoX = 0;
        posy = 1;
        posx = 1;
        l = round(Int,sqrt(length(image)));
        name = "Carretera.jpg"

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
                windowC= matrixgt[posx:tamWindow + saltoX, posy:tamWindow + saltoY];
                
                windowR2 = windowR[winDiff : tamWindow2 + winDiff, winDiff:tamWindow2 + winDiff];
                windowG2 = windowG[winDiff : tamWindow2 + winDiff, winDiff:tamWindow2 + winDiff];
                windowB2 = windowB[winDiff : tamWindow2 + winDiff, winDiff:tamWindow2 + winDiff];   
                
                windowR3 = windowR[winDiff2 : tamWindow3 + winDiff2, winDiff2:tamWindow3 + winDiff2];
                windowG3 = windowG[winDiff2 : tamWindow3 + winDiff2, winDiff2:tamWindow3 + winDiff2];
                windowB3 = windowB[winDiff2 : tamWindow3 + winDiff2, winDiff2:tamWindow3 + winDiff2];   

                #inputs[1:18]

                if (esCarretera(windowC) == true)
                    carretera +=1;
                    push!(inputs,transformar(windowR,windowG,windowB, windowR2,windowG2,windowB2, windowR3,windowG3,windowB3));
                    push!(targets,"positivo");
                    #=
                    auxMatrix[posx:tamWindow + saltoX, posy:tamWindow + saltoY,1] .= 0;
                    auxMatrix[posx:tamWindow + saltoX, posy:tamWindow + saltoY,2] .= 0;
                    auxMatrix[posx:tamWindow + saltoX, posy:tamWindow + saltoY,3] .= 0; 
                    =#
                    if (esCar && carretera > 250)
                        name = "esCarretera1.jpg"
                        imgsave = colorview(RGB, windowR, windowG, windowB)
                        save("./datasets/carretera/$name", imgsave)

                        name = "esCarretera2.jpg"
                        imgsave = colorview(RGB, windowR2, windowG2, windowB2)
                        save("./datasets/carretera/$name", imgsave)

                        name = "esCarretera3.jpg"
                        imgsave = colorview(RGB, windowR3, windowG3, windowB3)
                        save("./datasets/carretera/$name", imgsave)
                        esCar = false
                    end
                else
                    if(carretera*1.2 > noCarretera)
                        noCarretera +=1;
                        #save("./datasets/no_carretera/$name", imgsave)
                        push!(inputs,transformar(windowR,windowG,windowB, windowR2,windowG2,windowB2, windowR3,windowG3,windowB3));
                        push!(targets,"negativo");

                        if (esNoCar && carretera > 200)
                            name = "esNoCarretera.jpg"
                            imgsave = colorview(RGB, windowR, windowG, windowB)
                            save("./datasets/carretera/$name", imgsave)
                            esNoCar = false
                        end
                    end
                end


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
        #=
        windowR = auxMatrix[:, :, 1];
        windowG = auxMatrix[:, :, 2];
        windowB = auxMatrix[:, :, 3];   

        imgsave = colorview(RGB, windowR, windowG, windowB)
        save("./datasets/carretera/$name1", imgsave)=#

        
    end
    

    println("Imagenes cargadas 100%");

    println("Carreteras: $carretera  | No carreteras: $noCarretera")

    inputs = hcat(inputs...);
    targets = hcat(targets...);
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

function show_tree(dot_data)
    # Guardar el archivo .dot
    open("tree.dot", "w") do io
        write(io, dot_data)
    end
    dot_file = "./tree.dot"
    run(`dot -Tsvg $dot_file -o $dot_file.svg`)
    #display("image/svg+xml", read(dot_file * ".svg", String))
end

function ploteable7(t)
    t[7]
end

function ploteable3(t)
    t[3]
end

function ploteable2(t)
    t[2]
end

function ploteable1(t)
    t[1]
end

function normalizar(a,u)
    a>u
end

function predecirImagen1(rrnn)
    inputs = [];
    itc = readdir("./datasets/pruebas_input/");
    tam = length(itc);

    pixelCentro = round(Int,ceil(tamWindow / 2));

    l=0;
    i = 0.;

    for images in itc
        v = (i/tam*100);
        i+=1;
        println("Predecir Imagenes: $v%");
        ruta = "./datasets/pruebas_input/"
        ruta = joinpath(ruta,images)
        image = load(ruta);
        matrix = imageToColorArray(image);

        auxMatrix = copy(matrix)
        auxMatrix[:,:,:] .= 0

        saltoX = 0;
        posy = 1;
        posx = 1;
        l = round(Int,sqrt(length(image)));
        name = "carreteras_prediccion.jpg"

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
                push!(inputs,transformar(windowR,windowG,windowB));
                
                posy = posy + saltoVentana;
                saltoY = saltoY + saltoVentana;
                if((tamWindow + saltoY) > l)
                    break
                end
            end

            posx = posx + saltoVentana;
            saltoX = saltoX + saltoVentana;
            if((tamWindow + saltoX) > l)
                break
            end
        end

        inputs = hcat(inputs...);

        for i in 1:size(inputs,1)
            media = mean(inputs[i,:]);
            des = std(inputs[i,:]);
            inputs[i,:] = normalizar1.(inputs[i,:],media,des);
        end

        saltoX = 0;
        posy = 1;
        posx = 1;
        posy = 1;
        saltoY = 0;
        # Combine the RGB channels into a single color image
        for inp in 1:size(inputs,2)
            prediction = rrnn(inputs[:,inp])

            if(prediction[1] >= 0.60)

                auxMatrix[posx:tamWindow + saltoX, posy:tamWindow + saltoY,1] .= 1;
                auxMatrix[posx:tamWindow + saltoX, posy:tamWindow + saltoY,2] .= 1;
                auxMatrix[posx:tamWindow + saltoX, posy:tamWindow + saltoY,3] .= 1; 
            else
                auxMatrix[posx:tamWindow + saltoX, posy:tamWindow + saltoY,1] .= 0;
                auxMatrix[posx:tamWindow + saltoX, posy:tamWindow + saltoY,2] .= 0;
                auxMatrix[posx:tamWindow + saltoX, posy:tamWindow + saltoY,3] .= 0; 
            end
            
            posy = posy + saltoVentana;
            saltoY = saltoY + saltoVentana;
            if((tamWindow + saltoY) > l)
                posy = 1;
                saltoY = 0;

                posx = posx + saltoVentana;
                saltoX = saltoX + saltoVentana;
                if((tamWindow + saltoX) > l)
                    break
                end
            end
        end

        windowR = auxMatrix[:, :, 1];
        windowG = auxMatrix[:, :, 2];
        windowB = auxMatrix[:, :, 3];   

        imgsave = colorview(RGB, windowR, windowG, windowB)
        save("./datasets/carretera/$name", imgsave)
        
    end
    

    println("Predecir: 100%");

end

function confusionMatrix(in,tar,umbral)
    aux = normalizar.(in,umbral);
    confusionMatrix(aux,tar)
end

function crossvalidation(N::Int64, k::Int64)
    indices = repeat(1:k, Int64(ceil(N/k)));
    indices = indices[1:N];
    shuffle!(indices);
    return indices;
end;

#  Entradas, targets , topologia, tasa de error minima y ciclos maximos
function RRNNAA(inputs,targets,topology,minerror, maxIt, aprendizaje)
    #Cambio la tasa de error minimo a la tasa de precision minima
    precis = 100 - minerror
    errMinTraining = 100
    errMinTest = 100
    errMinVal = 100
    f1Training = 0
    precTraining = 0

    senTraining = 100
    senTest = 100
    senVal = 100

    errTest = [];
    errTraining = [];
    errValidation = [];  

    precision = []
    sensibilidad = []
    f1 = []

    errTest = []
    errTraining = []
    errValidation = []

    ann = Chain();

    #Creación Arn
    numInputsLayer = size(inputs,2);
    for numOutputsLayer = topology
        ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, σ) );
        numInputsLayer = numOutputsLayer;
    end;
    ann = Chain(ann..., Dense(numInputsLayer, 1, σ));

    loss(x, y) = binarycrossentropy(ann(x), y);
    best = deepcopy(ann);

    # Entrenar y probar en cada pliegue
    k = 10
    kf = crossvalidation(size(inputs,1),k) 

    for fold in 1:k
        trainingIndices = findall(x -> x != fold, kf);
        testIndices = findall(x -> x == fold, kf);

        # Obtener los datos de entrenamiento y validación
        trainFoldIn, trainFoldTarget = inputs[trainingIndices, :], targets[trainingIndices,:]
        testFoldIn, testFoldTarget = inputs[testIndices, :], targets[testIndices,:]
        

        #=Normalizar=#
        for i in 1:size(trainFoldIn,1)
            media = mean(trainFoldIn[i,:]);
            des = std(trainFoldIn[i,:]);
            trainFoldIn[i,:] = normalizar2.(trainFoldIn[i,:],media,des);
        end

        for i in 1:size(testFoldIn,1)
            media = mean(testFoldIn[i,:]);
            des = std(testFoldIn[i,:]);
            testFoldIn[i,:] = normalizar2.(testFoldIn[i,:],media,des);
        end

        aux = holdOut(trainFoldIn,trainFoldTarget);
        trainFoldIn = hcat(aux[1]...);
        trainFoldTarget = hcat(aux[2]...);
        validationIn = hcat(aux[3]...);
        validationTar = hcat(aux[4]...);

        testFoldIn = testFoldIn'
        testFoldTarget = testFoldTarget'

        println("#######")
        println(size(trainFoldIn))
        println(size(validationIn))
        println(size(testFoldIn))
        println("#######")

 
        ########
         
        min = 0;
        it = 0;

        #Entrenamiento y calculo del error
        Flux.train!(loss, Flux.params(ann), [(trainFoldIn, trainFoldTarget)], ADAM(aprendizaje));
        err = confusionMatrix(round.(ann(trainFoldIn))',trainFoldTarget')
        push!(errTraining,err);

        err = confusionMatrix(round.(ann(testFoldIn))',testFoldTarget')
        push!(errTest,err);

        err = confusionMatrix(round.(ann(validationIn))',validationTar')
        push!(errValidation,err);
        println("Precision: ",err[1] )

        while ((err[1] < precis) && (it < maxIt))
            Flux.train!(loss, Flux.params(ann), [(trainFoldIn, trainFoldTarget)], ADAM(aprendizaje));
            err = confusionMatrix(round.(ann(trainFoldIn))',trainFoldTarget')
            push!(errTraining,err);

            err = confusionMatrix(round.(ann(validationIn))',validationTar')
            push!(errValidation,err);

            err = confusionMatrix(round.(ann(testFoldIn))',testFoldTarget')
            push!(errTest,err);
            println("Precision: ",err[1] )

            if err[1] > min
                min = err[1];
                it = 0;
                best = deepcopy(ann);
                println(it);

                errMinTraining = last(ploteable2.(errTraining))
                errMinTest = last(ploteable2.(errTest))
                errMinVal = last(ploteable2.(errValidation))

                senTraining = last(ploteable3.(errTraining))
                senTest = last(ploteable3.(errTraining))
                senVal = last(ploteable3.(errTraining))

                precTraining = last(ploteable1.(errTraining))
                f1Training = last(ploteable7.(errTraining))
            else
                it = it + 1;
                println(it);
            end
        end

        push!(precision, precTraining)
        push!(sensibilidad, senTraining)
        push!(f1, f1Training)
    end

    precMedia = mean(precision)
    sensMedia = mean(sensibilidad)
    f1Media = mean(f1)
    
    println("Entrenamiento(Media-Kfold) -> Precision: ", precMedia, " | Sensibilidad: ", sensMedia, " | F1-Score: ", f1Media)
    println()

    [best, errTest, errTraining, errValidation, precMedia, sensMedia, f1Media] 
end

@sk_import svm: SVC

function sistemaSVM(inputs,targets,gammas, costes)
    k = 10
    precision = []
    sensibilidad = []
    f1 = []
    eTest = []

    kf = crossvalidation(size(inputs,1),k) 

    model = SVC(kernel="rbf", degree=3, gamma =gammas , C=costes);

    for fold in 1:k
        trainingIndices = findall(x -> x != fold, kf);
        testIndices = findall(x -> x == fold, kf);

        # Obtener los datos de entrenamiento y validación
        trainFoldIn, trainFoldTarget = inputs[trainingIndices, :], targets[trainingIndices,:]
        testFoldIn, testFoldTarget = inputs[testIndices, :], targets[testIndices,:]
        

        #=Normalizar=#

        for i in 1:size(trainFoldIn,1)
            media = mean(trainFoldIn[i,:]);
            des = std(trainFoldIn[i,:]);
            trainFoldIn[i,:] = normalizar2.(trainFoldIn[i,:],media,des);
        end

        for i in 1:size(testFoldIn,1)
            media = mean(testFoldIn[i,:]);
            des = std(testFoldIn[i,:]);
            testFoldIn[i,:] = normalizar2.(testFoldIn[i,:],media,des);
        end

        fit!(model, trainFoldIn, trainFoldTarget);
        
        eTraining = confusionMatrix(predict(model,trainFoldIn),trainFoldTarget);
        eTest = confusionMatrix(predict(model,testFoldIn),testFoldTarget);

        push!(precision, eTest[1])
        push!(sensibilidad, eTest[3])
        push!(f1, eTest[7])

    end
    
    precMedia = mean(precision)
    sensMedia = mean(sensibilidad)
    f1Media = mean(f1)
    
    println("Entrenamiento(Media-Kfold) -> Precision: ", precMedia, " | Sensibilidad: ", sensMedia, " | F1-Score: ", f1Media)
    println()


    [model,eTest,precMedia,sensMedia,f1Media]
   
end

@sk_import tree: DecisionTreeClassifier
@pyimport sklearn.tree as sktree

function sistemaArbol(inputs,targets, profundidad)
    k = 10
    precision = []
    sensibilidad = []
    f1 = []
    eTest = []

    kf = crossvalidation(size(inputs,1),k) 

    Armodel = DecisionTreeClassifier(max_depth=profundidad, random_state=1);

    for fold in 1:k
        trainingIndices = findall(x -> x != fold, kf);
        testIndices = findall(x -> x == fold, kf);

        # Obtener los datos de entrenamiento y validación
        trainFoldIn, trainFoldTarget = inputs[trainingIndices, :], targets[trainingIndices,:]
        testFoldIn, testFoldTarget = inputs[testIndices, :], targets[testIndices,:]
        

        #=Normalizar=#

        for i in 1:size(trainFoldIn,1)
            media = mean(trainFoldIn[i,:]);
            des = std(trainFoldIn[i,:]);
            trainFoldIn[i,:] = normalizar2.(trainFoldIn[i,:],media,des);
        end

        for i in 1:size(testFoldIn,1)
            media = mean(testFoldIn[i,:]);
            des = std(testFoldIn[i,:]);
            testFoldIn[i,:] = normalizar2.(testFoldIn[i,:],media,des);
        end

        fit!(Armodel, trainFoldIn, trainFoldTarget);
        
        eTraining = confusionMatrix(predict(Armodel,trainFoldIn),trainFoldTarget);
        eTest = confusionMatrix(predict(Armodel,testFoldIn),testFoldTarget);

        push!(precision, eTest[1])
        push!(sensibilidad, eTest[3])
        push!(f1, eTest[7])

    end
    
    precMedia = mean(precision)
    sensMedia = mean(sensibilidad)
    f1Media = mean(f1)
    
    println("Entrenamiento(Media-Kfold) -> Precision: ", precMedia, " | Sensibilidad: ", sensMedia, " | F1-Score: ", f1Media)
    println()

    [Armodel,eTest,precMedia,sensMedia,f1Media]
end


@sk_import neighbors: KNeighborsClassifier

function sistemaKNN(inputs,targets,vecinos)
    k = 10
    precision = []
    sensibilidad = []
    f1 = []
    eTest = []

    kf = crossvalidation(size(inputs,1),k) 

    KNNmodel = KNeighborsClassifier(vecinos)

    for fold in 1:k
        trainingIndices = findall(x -> x != fold, kf);
        testIndices = findall(x -> x == fold, kf);

        # Obtener los datos de entrenamiento y validación
        trainFoldIn, trainFoldTarget = inputs[trainingIndices, :], targets[trainingIndices,:]
        testFoldIn, testFoldTarget = inputs[testIndices, :], targets[testIndices,:]
        

        #=Normalizar=#

        for i in 1:size(trainFoldIn,1)
            media = mean(trainFoldIn[i,:]);
            des = std(trainFoldIn[i,:]);
            trainFoldIn[i,:] = normalizar2.(trainFoldIn[i,:],media,des);
        end

        for i in 1:size(testFoldIn,1)
            media = mean(testFoldIn[i,:]);
            des = std(testFoldIn[i,:]);
            testFoldIn[i,:] = normalizar2.(testFoldIn[i,:],media,des);
        end

        fit!(KNNmodel, trainFoldIn, trainFoldTarget);
        
        eTraining = confusionMatrix(predict(KNNmodel,trainFoldIn),trainFoldTarget);
        eTest = confusionMatrix(predict(KNNmodel,testFoldIn),testFoldTarget);

        push!(precision, eTest[1])
        push!(sensibilidad, eTest[3])
        push!(f1, eTest[7])

    end
    
    precMedia = mean(precision)
    sensMedia = mean(sensibilidad)
    f1Media = mean(f1)
    
    println("Entrenamiento(Media-Kfold) -> Precision: ", precMedia, " | Sensibilidad: ", sensMedia, " | F1-Score: ", f1Media)
    println()

    [KNNmodel,eTest, precMedia, sensMedia, f1Media]
end


function estadisticas(values, numIt)
    precision = []
    precAux = 0
    sensibilidad = []
    f1 = []

    tipoalgo = values[1]
    algoritm = []

    errorTest = []
    errorTraining = []
    errorValid = []

    for i in 1:numIt 

        if tipoalgo == "rrnnaa"
        
            algoritm = RRNNAA(values[2], values[3], values[4], values[5], values[6], values[7])
            
            push!(precision, algoritm[5])
            push!(sensibilidad, algoritm[6])
            push!(f1, algoritm[7])

            if (last(precision) > precAux)

                errorTest = algoritm[2]
                errorTraining = algoritm[3]
                errorValid = algoritm[4]

                precAux = last(precision)
            end


        elseif tipoalgo == "svm"

            algoritm = sistemaSVM(values[2], values[3], values[4], values[5])
            push!(precision, algoritm[3])
            push!(sensibilidad, algoritm[4])
            push!(f1, algoritm[5])

        elseif tipoalgo == "tree"

            algoritm = sistemaArbol(values[2], values[3], values[4])
            push!(precision, algoritm[3])
            push!(sensibilidad, algoritm[4])
            push!(f1, algoritm[5])

        elseif tipoalgo == "knn"

            algoritm = sistemaKNN(values[2], values[3], values[4])
            push!(precision, algoritm[3])
            push!(sensibilidad, algoritm[4])
            push!(f1, algoritm[5])

        end

    end

    precMedia = mean(precision)
    sensMedia = mean(sensibilidad)
    f1Media = mean(f1)
    f1DTipica = std(f1)

    println("Media para: ",numIt," iteraciones")
    println("Precision : ", precMedia)
    println("Sensibilidad: ",sensMedia)
    println("F1-Score: ",f1Media)
    println("F1-Score DTipica: ",f1DTipica)
    println()


    if tipoalgo == "rrnnaa"
        g = plot();
        plot!(ploteable2.(errorTest), label="Test Error");
        plot!(ploteable2.(errorTraining), label="Training Error")
        plot!(ploteable2.(errorValid), label="Validation Error")
        display(g);
    end


end


#==#
caracteristicas = estraccionCaracteristicas();
    
caracteristicas[2] = normalizarCaracteristicas(caracteristicas[2]);

# Graficar los errores
#=Descomentar para RRNNAA ##############################################################################################

#redNeuronal = RRNNAA(caracteristicas[1], caracteristicas[2], [36 24], 0, 150, 0.0025)

rrnnaa = ["rrnnaa", caracteristicas[1], caracteristicas[2], [24 24 12], 0, 150, 0.0025]
stats = estadisticas(rrnnaa, 1)
=#
#=
g = plot();
plot!(ploteable2.(redNeuronal[2]), label="Test Error");
plot!(ploteable2.(redNeuronal[3]), label="Training Error")
plot!(ploteable2.(redNeuronal[4]), label="Validation Error")
display(g);
=#
#=
##Guardar o cargar la mejor rrnn
rrnn = redNeuronal[1]
JLD2.save("./redes/RRNNAA.jld2", "RRNNAA", rrnn)
data = JLD2.load("./redes/RRNNAA.jld2")
rrnn = data["RRNNAA"]
predecirImagen1(rrnn)
=#

#=Descomentar para sistemaSVM ##############################################################################################

# Plotear las distancias
#SVMaux = sistemaSVM(caracteristicas[1],caracteristicas[2], 15, 40)

svm = ["svm", caracteristicas[1],caracteristicas[2], 15, 40]
stats = estadisticas(svm, 1)
=#

#=
g = scatter();
scatter!(SVMaux[4], label="Distancias")
xlabel!("Índice de la muestra")
ylabel!("Distancia al hiperplano")
display(g)=#


#=Descomentar para sistemaArbol ##############################################################################################

#Araux = sistemaArbol(caracteristicas[1],caracteristicas[2],10)
=#
tree = ["tree", caracteristicas[1], caracteristicas[2], 10]
stats = estadisticas(tree, 2)

tree = ["tree", caracteristicas[1], caracteristicas[2], 15]
stats = estadisticas(tree, 2)

tree = ["tree", caracteristicas[1], caracteristicas[2], 20]
stats = estadisticas(tree, 2)
#show_tree(Araux[4])


#=Descomentar para sistemaKNN ##############################################################################################

#KNNaux = sistemaKNN(caracteristicas[1],caracteristicas[2],vecinos)

knn = ["knn", caracteristicas[1], caracteristicas[2], 5]
stats = estadisticas(knn, 1)
=#

#=
g = plot();
plot!(vecinos, KNNaux[3], xlabel="Número de vecinos", ylabel="Error de validación", label="Curva de validación")
scatter!([argmin(KNNaux[3])], [minimum(KNNaux[3])], label="Mejor valor de vecinos", markersize=5)
display(g);
=#





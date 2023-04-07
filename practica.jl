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

const tamWindow = 7;
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
function esCarretera(windowC)
    carretera = false;
    pixelCentro = round(Int,ceil(tamWindow / 2));
    #pixelCentroUnaDimension = round(Int,tamWindow*(pixelCentro-1)+pixelCentro);

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
    positivos = [];
    negativos = [];
    itc = readdir(dirIt);
    gtc = readdir(dirGt);
    tam = length(itc);
    carretera = 0;
    noCarretera = 0;
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
                
                #inputs[1:6]

                if (esCarretera(windowC) == true)
                    carretera +=1;
                    push!(inputs,transformar(windowR,windowG,windowB));
                    #save("./positivos/"*name, imgsave)
                    push!(targets,"positivo");
        
                else
                    if(carretera*1.2 > noCarretera)
                        noCarretera +=1;
                        #save("./negativos/"*name, imgsave)
                        push!(inputs,transformar(windowR,windowG,windowB));
                        push!(targets,"negativo");
                    end
                end


            end
            posx = posx + saltoVentana;
            saltoX = saltoX + saltoVentana;
            if((tamWindow + saltoX)>round(Int,sqrt(length(image))))
                break
            end
        end
        
       
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
    #Cambio la tasa de error minimo a la tasa de precision minima
    precis = 100 - minerror
    #Randomizar y normalizar
    aux = holdOut(inputs,targets);
    trainingIn = hcat(aux[1]...);
    trainingTar = hcat(aux[2]...);
    testIn = hcat(aux[3]...);
    testTar = hcat(aux[4]...);
    println(size(trainingIn))
    println(size(testIn))

    println("Dimensiones :",size(trainingIn));
    println("Ventanas :",size(trainingIn,1));
    println("Inputs :",size(trainingIn,2));
    println("Sin normalizar: ",trainingIn[:,1]);
    println("Sin normalizar: ",testIn[:,1]);


    ann = Chain();
#==#
    for i in 1:size(trainingIn,1)
        #max = maximum(trainingIn[i,:]);
        #min = minimum(trainingIn[i,:]);
        media = mean(trainingIn[i,:]);
        des = std(trainingIn[i,:]);
        trainingIn[i,:] = normalizar2.(trainingIn[i,:],media,des);
        testIn[i,:] = normalizar2.(testIn[i,:],media,des);
    end
    println(size(trainingIn))
    println(size(testIn))
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
    min = 0;
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
    Flux.train!(loss, Flux.params(ann), [(trainingIn, trainingTar)], ADAM(0.003));
    err = confusionMatrix(round.(ann(trainingIn))',trainingTar')
    push!(errTraining,err);

    err = confusionMatrix(round.(ann(testIn))',testTar')
    push!(errTest,err);

    err = confusionMatrix(round.(ann(validationIn))',validationTar')
    push!(errValidation,err);
    best = deepcopy(ann);
    println("Precision: ",err[1] )

    while ((err[1] < precis) && (it < maxIt))
        Flux.train!(loss, Flux.params(ann), [(trainingIn, trainingTar)], ADAM(0.003));
        err = confusionMatrix(round.(ann(trainingIn))',trainingTar')
        push!(errTraining,err);

        err = confusionMatrix(round.(ann(validationIn))',validationTar')
        push!(errValidation,err);

        err = confusionMatrix(round.(ann(testIn))',testTar')
        push!(errTest,err);
        println("Precision: ",err[1] )

        if err[1] > min
            min = err[1];
            it = 0;
            best = deepcopy(ann);
            println(it);

        else
            it = it + 1;
            println(it);
        end
    end;
    
    println("Error Minimo: ",minimum(ploteable.(errTest)))
    [best,errTest,errTraining,errValidation,err] 
end

@sk_import svm: SVC

function sistemaSVM(inputs,targets,gammas, costes)
    aux=holdOut(inputs,targets);
    trainingIn = hcat(aux[1]...);
    trainingTar = hcat(aux[2]...);
    testIn = hcat(aux[3]...);
    testTar = hcat(aux[4]...);

    for i=1:size(trainingIn,1)
        media=mean(trainingIn[i,:]);
        des=std(trainingIn[i,:]);
        trainingIn[i,:] = normalizar2.(trainingIn[i,:],media,des);
        testIn[i,:]=normalizar2.(testIn[i,:],media,des);
    end

    model = SVC(kernel="rbf", degree=3, gamma =gammas , C=costes);
    fit!(model, trainingIn', trainingTar');
    
    eTraining = confusionMatrix(predict(model,trainingIn'),trainingTar');
    eTest = confusionMatrix(predict(model,testIn'),testTar');

    distances = decision_function(model, inputs);
   
    println("MatrizConfusion Test",eTest)
    println("Donde el error es: ",eTest[2])

    
    [model,eTest,eTraining,distances]
end

@sk_import tree: DecisionTreeClassifier
@pyimport sklearn.tree as sktree

function sistemaArbol(inputs,targets, profundidad)
    aux=holdOut(inputs,targets);
    trainingIn=hcat(aux[1]...);
    trainingTar=hcat(aux[2]...);
    testIn=hcat(aux[3]...);
    testTar=hcat(aux[4]...);

    for i=1:size(trainingIn,1)
        media=mean(trainingIn[i,:]);
        des=std(trainingIn[i,:]);
        trainingIn[i,:] = normalizar1.(trainingIn[i,:],media,des);
        testIn[i,:]=normalizar1.(testIn[i,:],media,des);
    end

    Armodel = DecisionTreeClassifier(max_depth=profundidad, random_state=1);
    fit!(Armodel, trainingIn', trainingTar');

    eTraining = confusionMatrix(predict(Armodel,trainingIn'),trainingTar');
    eTest = confusionMatrix(predict(Armodel,testIn'),testTar');

    dot_data = sktree.export_graphviz(Armodel, out_file=nothing)

    # Muestra el árbol de decisión en una ventana emergente

    println("MatrizConfusion Test",eTest[8])
    println("Donde el error es: ",eTest[2])

    [Armodel,eTest,eTraining,dot_data]
end


@sk_import neighbors: KNeighborsClassifier

function sistemaKNN(inputs,targets,vecinos)
    aux=holdOut(inputs,targets);
    trainingIn=hcat(aux[1]...);
    trainingTar=hcat(aux[2]...);
    testIn=hcat(aux[3]...);
    testTar=hcat(aux[4]...);

    for i=1:size(trainingIn,1)
        media=mean(trainingIn[i,:]);
        des=std(trainingIn[i,:]);
        trainingIn[i,:] = normalizar1.(trainingIn[i,:],media,des);
        testIn[i,:]=normalizar1.(testIn[i,:],media,des);
    end
#=
    KNNmodel = KNeighborsClassifier(vecinos);
    fit!(KNNmodel, trainingIn', trainingTar');
    eTraining=confusionMatrix(predict(KNNmodel,trainingIn'),trainingTar');
    eTest=confusionMatrix(predict(KNNmodel,testIn'),testTar');
=#
    KNNmodel = KNeighborsClassifier(1)
    fit!(KNNmodel, trainingIn', trainingTar')
    eTest = confusionMatrix(predict(KNNmodel, testIn'), testTar')

    error_validacion = []
    for k in vecinos
        KNNmodel = KNeighborsClassifier(k)
        fit!(KNNmodel, trainingIn', trainingTar')
        eTest = confusionMatrix(predict(KNNmodel, testIn'), testTar')
        push!(error_validacion, eTest[2])
    end

    println("Donde el error minimo es es: ",minimum(error_validacion))
    [KNNmodel,eTest,error_validacion]
end

caracteristicas = estraccionCaracteristicas();
    
caracteristicas[2] = normalizarCaracteristicas(caracteristicas[2]);

# Graficar los errores
#=Descomentar para RRNNAA
redNeuronal = RRNNAA(caracteristicas[1],caracteristicas[2],[6 3],0.22,200)
g = plot();
plot!(ploteable.(redNeuronal[2]), label="Test Error");
plot!(ploteable.(redNeuronal[3]), label="Training Error")
plot!(ploteable.(redNeuronal[4]), label="Validation Error")
display(g);
=#


#=Descomentar para sistemaSVM
# Plotear las distancias
SVMaux = sistemaSVM(caracteristicas[1],caracteristicas[2],10, 1)

g = scatter();
scatter!(SVMaux[4], label="Distancias")
xlabel!("Índice de la muestra")
ylabel!("Distancia al hiperplano")
display(g)
=#

#=Descomentar para sistemaArbol

Araux = sistemaArbol(caracteristicas[1],caracteristicas[2],19)
show_tree(Araux[4])
=#
#=Descomentar para sistemaKNN
vecinos = 1
KNNaux = sistemaKNN(caracteristicas[1],caracteristicas[2],vecinos)

g = plot();
plot!(vecinos, KNNaux[3], xlabel="Número de vecinos", ylabel="Error de validación", label="Curva de validación")
scatter!([argmin(KNNaux[3])], [minimum(KNNaux[3])], label="Mejor valor de vecinos", markersize=5)
display(g);





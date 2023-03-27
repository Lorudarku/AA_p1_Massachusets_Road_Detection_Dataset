using FileIO
using FFTW
using Random
using Flux
using Flux.Losses
using Plots
using Statistics
using DelimitedFiles
using ScikitLearn

const tamWindow = 15;
const saltoVentana = 5;

function mediaDesviacion(ventana)
    #Mirar como funcionan las funciones ##################################################################################################
    media = mean(ventana[:]);
    desviacion = std(ventana[:]);

    [media;desviacion]
end


#Calculamos las ventanas para cada imagen
function transformar(ruta)
    image = load(ruta);
    outMD = [];
    x = 1;
    salto = 0;

    #Bucle ventana por toda la imagen

    #Calculamos la ventana para la iteracion x
    window = image[x:tamWindow + salto, x:tamWindow + salto];
    pixelCentro = ceil(tamWindow / 2) + salto;
    mD = [];

    #Convertimos la ventana en un array[1:3] RGB
    #wn = convert(Array{RGB{Normed{UInt8,8}},2}, window);
    matrizRojos = red.(window);
    matrizVerde = green.(window);
    matrizAzul = blue.(window);


    #Calculamos la media y desviacion tipica para cada componente de color
    mDR = mediaDesviacion(matrizRojos);
    mDG = mediaDesviacion(matrizVerde);
    mDB = mediaDesviacion(matrizAzul);

    #Guardamos la media y desviacion tipica de cada componente en un array[1:3] de una ventana
    #md[R[m,d],G[m,d],B[m,d]]
    push!(mD,mDR);
    push!(mD,mDG);
    push!(mD,mDB);

    #Guardamos para cada ventana su media y desviacion tipica
    #outMD[md,md,md]
    push!(outMD,mD);
    print(outMD);

    salto = salto + saltoVentana; 
    x = x + saltoVentana;
    #fin bucle
    [outMD]
end

#Sacamos las caracteristicas
function estraccionCaracteristicas()
    inputs = [];
    targets = [];
    itc = readdir("./pruebas");
    gtc = readdir("./gtp");
    tam = length(itc);
    i=1;
    ant=-1;
    for c in itc
        #inputs[1*][2*][3*][4*]
        #Dim 1: Dimension con las imagenes totales
        #Dim 2: Dimension con todas las ventanas para cada imagen
        #Dim 3: Dimension con las componentes RGB con su media y desviacion tipica cada uno
        #Dim 4: Dimension con la media y desviacion tipica
        push!(inputs,transformar("./pruebas/"*c));

        porcentaxe=floor((i/tam)*100);
        if (porcentaxe!=ant)
            println(porcentaxe,"%");
            ant=porcentaxe;
        end
        i=i+1;
    end

    println("Imagenes cargadas");
    i=1;
    ant=-1;
    for c in gtc
        #targets[1*][2*]
        #Dim 1: Dimension con las imagenes totales
        #Dim 2: Dimension el valor para las ventanas de cada imagen
        push!(targets,"./gtp/"*c);
        
        porcentaxe = floor((i/tam)*100);
        if (porcentaxe!=ant)
            println(porcentaxe,"%");
            ant=porcentaxe;
        end
        i=i+1;
    end
    println("Gt cargado");

    inputs=hcat(inputs...);
    targets=hcat(targets...);
    [permutedims(inputs),permutedims(targets)]
end


estraccionCaracteristicas();





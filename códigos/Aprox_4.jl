#pIgA


#Aquí debese escribir a ruta onde se garda a BD coas carpetas de cada instrumento
rutaBD="D:/Brais/Clase/3º Carreira/AA/base_de_datos_completa";


using FileIO;
using FFTW;
using Random;
using Flux;
using Flux.Losses;
using Plots;
using Statistics;
using DelimitedFiles;
using ScikitLearn;

#Devolvese a media e desviación típica de unha señal con frecuencia Fs
function tf(senal,Fs)
    n=length(senal);
    senalFrecuencia = abs.(fft(senal));
    if (iseven(n))
        @assert(mean(abs.(senalFrecuencia[2:Int(n/2)] .- senalFrecuencia[end:-1:(Int(n/2)+2)]))<1e-8);
        senalFrecuencia = senalFrecuencia[1:(Int(n/2)+1)];
    else
        @assert(mean(abs.(senalFrecuencia[2:Int((n+1)/2)] .- senalFrecuencia[end:-1:(Int((n-1)/2)+2)]))<1e-8);
        senalFrecuencia = senalFrecuencia[1:(Int((n+1)/2))];
    end;
    m1 = Int(round(100*2*length(senalFrecuencia)/Fs));
    m2 = Int(round(500*2*length(senalFrecuencia)/Fs));
    resultado=[]

    for i in 0:49
        push!(resultado,mean(senalFrecuencia[m1*i+1:m1*(i+1)]));
        push!(resultado,std(senalFrecuencia[m1*i+1:m1*(i+1)]));
    end

    #for i in 0:4
        #push!(resultado,mean(senalFrecuencia[(m1*15+m2*i)+1:m1*15+m2*(i+1)]));
        #push!(resultado,std(senalFrecuencia[(m1*15+m2*i)+1:m1*15+m2*(i+1)]));
    #end
    #for i in 0:14
        #push!(resultado,mean(senalFrecuencia[(m1*15+m2*5+m1*i)+1:m1*15+m2*5+m1*(i+1)]));
        #push!(resultado,std(senalFrecuencia[(m1*15+m2*5+m1*i)+1:m1*15+m2*5+m1*(i+1)]));
    #end

    resultado
end

#Sacanse as caracteristicas do audio en unha ruta determinada
function transformar(ruta)
    audio = load(ruta);
    mono2 = audio[1][:,2];
    tf(mono2, audio[2])
end

#Sacanse as caracteristicas de todos os audios das carpetas "pia" e "gac" en unha ruta determinada e devolvense as matrices de inputs e targets
function estraccionCaracteristicas(ruta)
    inputs=[];
    targets=[]
    piano=readdir(ruta*"/pia");
    guitarra=readdir(ruta*"/gac");
    tam=length(piano)+length(guitarra);
    i=1;
    ant=-1;
    for c in piano
        push!(inputs,transformar(ruta*"/pia/"*c));
        push!(targets,"piano");
        porcentaxe=floor((i/tam)*100);
        if (porcentaxe!=ant)
            println(porcentaxe,"%");
            ant=porcentaxe;
        end
        i=i+1;
    end
    for c in guitarra
        push!(inputs,transformar(ruta*"/gac/"*c));
        push!(targets,"guitarra");
        porcentaxe=floor((i/tam)*100);
        if (porcentaxe!=ant)
            println(porcentaxe,"%");
            ant=porcentaxe;
        end
        i=i+1;
    end
    inputs=hcat(inputs...);
    targets=hcat(targets...);
    [permutedims(inputs),permutedims(targets)]
end

# Función auxiliar usada en normalizarStrings para devolver un array de ceros e unos.
function normalizaraux(p, a)
    if p==a
        1.
    else
        0.
    end
end

# Funcion usada para transformar o array de strings nun array de targets validos para a Arn
function normalizarStrings(tabla)
    result=[];
    for i = 1:size(tabla,2)
        p=unique(tabla[:,i])
        if length(p)<2
            push!(result, 1)
        elseif length(p)==2
            push!(result,normalizaraux.(p[1],tabla[:,i]))
        else
            for j = 1:length(p)
                push!(result,normalizaraux.(p[j],tabla[:,i]))
            end;
        end;
    end;
    hcat(result...);
end

# Funcion que divide os arrays de Inputs e Targets en dous arrays diferentes de forma aleatoria
function holdOut(in,tar)
    in1=[];
    in2=[];
    tar1=[];
    tar2=[];
    for i = 1:size(in,1)
        aux=rand();
        if aux>0.15
            push!(in1,in[i,:]);
            push!(tar1,tar[i,:]);
        else
            push!(in2,in[i,:]);
            push!(tar2,tar[i,:]);
        end
    end
    [in1,tar1,in2,tar2]
end

# Devolve a posición do maior valor de un array
function max(a)
    m=-1;
    aux=1;
    for i=1:size(a,1)
        if a[i]>m
            m=a[i];
            aux=i;
        end;
    end;
    aux
end


using Statistics;

function verdaderosPositivos(a,b)
    (a==true) & (b==true)
end

function falsosPositivos(a,b)
    (a==true) & (b==false)
end

function verdaderosNegativos(a,b)
    (a==false) & (b==false)
end

function falsosNegativos(a,b)
    (a==false) & (b==true)
end


function  confusionMatrix(o,t)
    vp=sum(verdaderosPositivos.(o,t));
    fp=sum(falsosPositivos.(o,t));
    vn=sum(verdaderosNegativos.(o,t));
    fn=sum(falsosNegativos.(o,t));
    matrizConfusion=Matrix{Int64}(undef, 2, 2)
    matrizConfusion[1,1]=vn;
    matrizConfusion[1,2]=fp;
    matrizConfusion[2,1]=fn;
    matrizConfusion[2,2]=vp;
    precision = (vn+vp)/(vn+vp+fn+fp);
    tasaE = (fn+fp)/(vn+vp+fn+fp);
    sensibilidad = vp/(fn+vp);
    especificidad = vn/(fp+vn);
    valorPP = vp/(vp+fp);
    valorPN = vn/(fn+vn);
    fScore = 2/((1/valorPP)+(1/sensibilidad));
    [precision , tasaE , sensibilidad , especificidad , valorPP , valorPN , fScore , matrizConfusion]
end

function normalizar(a,u)
    a>u
end

function  confusionMatrix(o,t,umbral)
    aux=normalizar.(o,umbral);
    confusionMatrix(aux,t)
end



function normalizar(v,media,des)
    (v-media)/des
end


function ploteable(t)
    t[2]
end



@sk_import svm: SVC

function sistemaSVM(inputs,targets)
    aux=holdOut(inputs,targets);
    trainingIn=hcat(aux[1]...);
    trainingTar=hcat(aux[2]...);
    testIn=hcat(aux[3]...);
    testTar=hcat(aux[4]...);
    for i=1:size(trainingIn,1)
        media=mean(trainingIn[i,:]);
        des=std(trainingIn[i,:]);
        trainingIn[i,:] = normalizar.(trainingIn[i,:],media,des);
        testIn[i,:]=normalizar.(testIn[i,:],media,des);
    end
    model = SVC(kernel="rbf", degree=3, gamma=2, C=1);
    fit!(model, trainingIn', trainingTar');
    eTraining=confusionMatrix(predict(model,trainingIn'),trainingTar');
    eTest=confusionMatrix(predict(model,testIn'),testTar');
    [model,eTest,eTraining]
end

@sk_import tree: DecisionTreeClassifier

function sistemaArbol(inputs,targets)
    aux=holdOut(inputs,targets);
    trainingIn=hcat(aux[1]...);
    trainingTar=hcat(aux[2]...);
    testIn=hcat(aux[3]...);
    testTar=hcat(aux[4]...);
    for i=1:size(trainingIn,1)
        media=mean(trainingIn[i,:]);
        des=std(trainingIn[i,:]);
        trainingIn[i,:] = normalizar.(trainingIn[i,:],media,des);
        testIn[i,:]=normalizar.(testIn[i,:],media,des);
    end
    Armodel = DecisionTreeClassifier(max_depth=4, random_state=1);
    fit!(Armodel, trainingIn', trainingTar');
    eTraining=confusionMatrix(predict(Armodel,trainingIn'),trainingTar');
    eTest=confusionMatrix(predict(Armodel,testIn'),testTar');
    [Armodel,eTest,eTraining]
end


@sk_import neighbors: KNeighborsClassifier

function sistemaKNN(inputs,targets)
    aux=holdOut(inputs,targets);
    trainingIn=hcat(aux[1]...);
    trainingTar=hcat(aux[2]...);
    testIn=hcat(aux[3]...);
    testTar=hcat(aux[4]...);
    for i=1:size(trainingIn,1)
        media=mean(trainingIn[i,:]);
        des=std(trainingIn[i,:]);
        trainingIn[i,:] = normalizar.(trainingIn[i,:],media,des);
        testIn[i,:]=normalizar.(testIn[i,:],media,des);
    end
    KNNmodel = KNeighborsClassifier(3);
    fit!(KNNmodel, trainingIn', trainingTar');
    eTraining=confusionMatrix(predict(KNNmodel,trainingIn'),trainingTar');
    eTest=confusionMatrix(predict(KNNmodel,testIn'),testTar');
    [KNNmodel,eTest,eTraining]
end


#Execución

#rutaBD é unha constante ao inicio do documento
caracteristicas=estraccionCaracteristicas(rutaBD)

caracteristicas[2]=normalizarStrings(caracteristicas[2])

#Os últimos tres parametros son: topoloxía, Tasa de erro aceptable e máximo de ciclos sen mellora permitidos.

SVMaux=sistemaSVM(caracteristicas[1],caracteristicas[2])
Araux=sistemaArbol(caracteristicas[1],caracteristicas[2])
KNNaux=sistemaKNN(caracteristicas[1],caracteristicas[2])

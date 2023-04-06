using Random

function holdOut()
m = []
# Crear dos arreglos ordenados
a = [1, 2, 3, 4, 5]
b = ["a", "b", "c", "d", "e"]

# Crear matriz con ambos arreglos
push!(m,[a,b])

# Shuffle de la matriz
shuffle!(m)

# Separar la matriz en los dos arreglos originales


println(m)
println(b)

end

holdOut()
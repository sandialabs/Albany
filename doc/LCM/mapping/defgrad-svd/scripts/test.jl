using DelimitedFiles

include("./update.jl")
L = [4 * pi; 0.1; 0.1]
h = 0.1
rho = [0.0; 2.0]
gamma = [0.0; 0.2]
N = 32
update(L, h, N, rho, gamma)

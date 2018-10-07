include("./lie.jl")

#
# Testing functions
#

#
# Helix deformation gradient. Adapted from: Lie-group interpolation
# and variational recovery for internal variables, Mota et al.,
# Computational Mechanics, 52:6, 1281--1299, 2013, section 6.2.
#
function helix(X, ρ, γ)
    F = zeros(3, 3)
    if (ρ == 0.0)
        x = [X[1]; X[2]; γ * X[1] + X[3]]
        F = [1 0 0; 0 1 0; γ 0 1]
        return x, F
    end
    R = 1.0 / abs(ρ)
    θ = X[1] / R
    c = cos(θ)
    s = sin(θ)
    a = R - X[2]
    b = a / R
    x = [a * s; R - a * c; γ * X[1] + X[3]]
    F = [b*c -s 0; b*s c 0; γ 0 1]
    return x, F
end

function configuration(L, h, ρ, γ)
    LX = L[1]
    LY = L[2]
    LZ = L[3]
    NX = round(Int, LX / h)
    NY = round(Int, LY / h)
    NZ = round(Int, LZ / h)

    nodes_per_cell = 8
    num_nodes = (NX + 1) * (NY + 1) * (NZ + 1)
    num_cells = NX * NY * NZ

    coord = zeros(num_nodes, 3)
    defgrad = zeros(num_nodes, 3, 3)
    conn = zeros(Int, num_cells, nodes_per_cell + 1)

    for nx = 0 : NX
        X1 = nx * LX / NX - LX / 2.0
        for ny = 0 : NY
            X2 = ny * LY / NY - LY / 2.0
            for nz = 0 : NZ
                X3 = nz * LZ / NZ - LZ / 2.0
                X = [X1 X2 X3]
                x, F = helix(X, ρ, γ)
                i = (nx * (NY + 1) + ny) * (NZ + 1) + nz + 1
                coord[i, :] = x
                defgrad[i, :, :] = F
            end
        end
    end
    
    for nx = 0 : NX - 1
        for ny = 0 : NY - 1
            for nz = 0 : NZ - 1
                n1 = (nx * (NY + 1) + ny) * (NZ + 1) + nz
                n2 = n1 + (NY + 1) * (NZ + 1)
                n3 = n2 + NZ + 1
                n4 = n1 + NZ + 1
                n5 = n1 + 1
                n6 = n2 + 1
                n7 = n3 + 1
                n8 = n4 + 1
                e = (nx * NY + ny) * NZ + nz + 1
                n = [nodes_per_cell n1 n2 n3 n4 n5 n6 n7 n8]
                conn[e, :] = n
            end
        end
    end
    return coord, conn, defgrad
end

using LinearAlgebra

function write_vtk(filename, coord, conn, u, v)
    num_cells = size(conn, 1)
    num_nodes = size(coord, 1)
    nodes_per_cell = size(conn, 2) - 1
    cell_type = ones(Int, num_cells, 1) * 12
    theta = zeros(num_nodes)
    phi = zeros(num_nodes)
    for i = 1 : num_nodes
        theta[i] = norm(u[i, :])
        phi[i] = norm(v[i, :])
    end
    open(filename, "w") do file
        print(file, "# vtk DataFile Version 3.0\n")
        print(file, "LGR\n")
        print(file, "ASCII\n")
        print(file, "DATASET UNSTRUCTURED_GRID\n")
        print(file, "POINTS ", num_nodes, " double\n")
        writedlm(file, coord, ' ')
        list_size = (nodes_per_cell + 1) * num_cells
        print(file, "CELLS ", num_cells, " ", list_size, "\n")
        writedlm(file, conn, ' ')
        print(file, "CELL_TYPES ", num_cells, "\n")
        writedlm(file, cell_type, ' ')
        print(file, "POINT DATA ", num_nodes, "\n")
        print(file, "SCALARS theta double 1\n")
        writedlm(file, theta, ' ')
        print(file, "SCALARS phi double 1\n")
        writedlm(file, phi, ' ')
    end
end

using Printf

function update(L, h, N, ρ, γ)
    LX = L[1]
    LY = L[2]
    LZ = L[3]
    NX = round(Int, LX / h)
    NY = round(Int, LY / h)
    NZ = round(Int, LZ / h)
    ρini = ρ[1]
    ρfin = ρ[2]
    γini = γ[1]
    γfin = γ[2]
    nodes_per_cell = 8
    num_nodes = (NX + 1) * (NY + 1) * (NZ + 1)
    num_cells = NX * NY * NZ
    
    u = zeros(num_nodes, 3)
    s = zeros(num_nodes, 3)
    v = zeros(num_nodes, 3)
    
    _, _, Fprev = configuration(L, h, ρini, γini)
    
    for n = 0 : N
        ξ = convert(AbstractFloat, n) / N 
        ρ = (1.0 - ξ) * ρini + ξ * ρfin
        γ = (1.0 - ξ) * γini + ξ * γfin
        coord, conn, Fcurr = configuration(L, h, ρ, γ)
        for i = 1 : num_nodes
            ΔF = Fcurr[i, :, :] * inv(Fprev[i, :, :])
            U = RTofRV(u[i, :])
            S = diagm(0 => exp.(s[i, :]))
            V = RTofRV(v[i, :])
            USV = svd(ΔF)
            ΔU = USV.U
            ΔS = diagm(0 => USV.S)
            ΔV = USV.V
            U = ΔU * U
            S = ΔS * S
            V = ΔV * U
            s[i, :] = log.(diag(S))
            utry = RVofRT(U)
            vtry = RVofRT(V)
            u[i, :] = RVcontin(utry, u[i, :]')
            v[i, :] = RVcontin(vtry, v[i, :]')
        end        
        tick = @sprintf("%04d", n)
        filename = string("helix_", tick, ".vtk")
        write_vtk(filename, coord, conn, u, v)
    end
end

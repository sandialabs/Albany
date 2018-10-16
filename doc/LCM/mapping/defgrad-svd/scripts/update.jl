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
        return x, F, F, F, 0.0
    end
    R = 1.0 / abs(ρ)
    θ = X[1] / R
    c = cos(θ)
    s = sin(θ)
    a = R - X[2]
    b = a / R
    x = [a * s; R - a * c; γ * X[1] + X[3]]
    F = [b*c -s 0; b*s c 0; γ 0 1]
    R = [c -s 0; s c 0; 0 0 1]
    U = [b 0 0; 0 1 0; γ 0 1]
    return x, F, R, U, θ
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

    rot = zeros(num_nodes, 3, 3)
    stretch = zeros(num_nodes, 3, 3)
    angle = zeros(num_nodes)

    for nx = 0 : NX
        X1 = nx * LX / NX - LX / 2.0
        for ny = 0 : NY
            X2 = ny * LY / NY - LY / 2.0
            for nz = 0 : NZ
                X3 = nz * LZ / NZ - LZ / 2.0
                X = [X1 X2 X3]
                x, F, R, U, θ = helix(X, ρ, γ)
                i = (nx * (NY + 1) + ny) * (NZ + 1) + nz + 1
                coord[i, :] = x
                defgrad[i, :, :] = F
                rot[i, :, :] = R
                stretch[i, :, :] = U
                angle[i] = θ
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
    return coord, conn, defgrad, rot, stretch, angle
end

using LinearAlgebra

function write_vtk(filename, coord, conn, u, s, v, theta)
    num_cells = size(conn, 1)
    num_nodes = size(coord, 1)
    nodes_per_cell = size(conn, 2) - 1
    cell_type = ones(Int, num_cells, 1) * 12
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
        print(file, "POINT_DATA ", num_nodes, "\n")
        print(file, "SCALARS u double 3\n")
        print(file, "LOOKUP_TABLE default\n")
        writedlm(file, u, ' ')
        print(file, "SCALARS s double 3\n")
        print(file, "LOOKUP_TABLE default\n")
        writedlm(file, s, ' ')
        print(file, "SCALARS v double 3\n")
        print(file, "LOOKUP_TABLE default\n")
        writedlm(file, v, ' ')
        print(file, "SCALARS theta double 1\n")
        print(file, "LOOKUP_TABLE default\n")
        writedlm(file, theta, ' ')
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

    ref, _, Fprev, _, _, _ = configuration(L, h, ρini, γini)

    for n = 0 : N
        ξ = convert(AbstractFloat, n) / N
        ρ = (1.0 - ξ) * ρini + ξ * ρfin
        γ = (1.0 - ξ) * γini + ξ * γfin
        coord, conn, Fcurr, rot, stretch, theta = configuration(L, h, ρ, γ)
        for i = 1 : num_nodes
            ΔF = Fcurr[i, :, :] * inv(Fprev[i, :, :])
            USV = svd(ΔF)
            ΔU = USV.U * USV.Vt
            ΔS = diagm(0 => USV.S)
            ΔV = USV.V
            detV = det(ΔV)
            if (detV < 0.0)
                ΔV *= -1.0
            end
            U = RTofRV(u[i, :])
            S = diagm(0 => exp.(s[i, :]))
            V = RTofRV(v[i, :])
            U = ΔU * U
            S = ΔS * S
            V = ΔV * U
            if (det(U) < 0.0)
                error("Detected negative det(U)")
            end
            if (det(V) < 0.0)
                error("Detected negative det(V)")
            end
            s[i, :] = log.(diag(S))
            uold = RVofRT(U)
            vold = RVofRT(V)
            uprev = u[i, :]'
            vprev = v[i, :]'
            unew = RVcontin(uold, uprev)
            vnew = RVcontin(vold, vprev)
            if abs(ref[i, 1]) > 2.0 * 3.14
                print("n : ", n, "\n")
                print("prev: ", uprev, "\n")
                print("old : ", uold, "\n")
                print("new : ", unew, "\n")
            end
            u[i, :] = unew
            v[i, :] = vnew
        end
        Fprev = Fcurr
        tick = @sprintf("%04d", n)
        filename = string("helix_", tick, ".vtk")
        write_vtk(filename, coord, conn, u, s, v, theta)
    end
end

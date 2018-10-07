#
# Skew symmetric part of matrix
#
function skew(A)
    return 0.5*(A - A')
end

#
# Convert rotation vector to skew symmetric tensor
#
function check(w)
    W = [0 -w[3] w[2]; w[3] 0 -w[1]; -w[2] w[1] 0]
    return W
end

#
# Convert skew symmetric tensor to rotation vector
#
function uncheck(W)
    W = skew(W)
    w = [W[3, 2] W[1, 3] W[2, 1]]
    return w
end

#
# Symmetric part of matrix
#
function symm(A)
    return 0.5*(A + A')
end

#
# Baker-Campbell-Hausdorff formula, up to 8 terms.
#
# The Baker–Campbell–Hausdorff formula is the solution to the equation
#
# z = log[exp(x) exp(y)]
#
# for possibly noncommutative "x" and "y" in the Lie algebra of a Lie
# group. This formula tightly links Lie groups to Lie algebras by
# expressing the logarithm of the product of two Lie group elements as
# a Lie algebra element using only Lie algebraic operations. The
# solution on this form, whenever defined, means that multiplication
# in the group can be expressed entirely in Lie algebraic terms. The
# solution on commutative forms is obtained by substituting the power
# series for exp and log in the equation and rearranging. The point
# is to express the solution in Lie algebraic terms.
#
# The coefficients on the series were computed by using the
# Mathematica implementation of Goldberg's algorithm given in:
# Computing the Baker-Campbell-Hausdorff series and the Zassenhaus
# product, Weyrauch, Michael and Scholz, Daniel, COMPUTER PHYSICS
# COMMUNICATIONS, 2009, 180:9,1558-1565.
# 
function BCH(x, y)
    
    z1 = x+y
    
    z2 = 0.5*(x*y - y*x)
    
    z3 = x*x*y/12 - x*y*x/6 + x*y*y/12 + y*x*x/12 - y*x*y/6 + y*y*x/12
    
    z4 = x*x*y*y/24 - x*y*x*y/12 + y*x*y*x/12 - y*y*x*x/24
    
    z5 = -x*x*x*x*y/720 + x*x*x*y*x/180 + x*x*x*y*y/180 - x*x*y*x*x/120 -
    x*x*y*x*y/120 - x*x*y*y*x/120 + x*x*y*y*y/180 + x*y*x*x*x/180 -
    x*y*x*x*y/120 + x*y*x*y*x/30 - x*y*x*y*y/120 - x*y*y*x*x/120 - x*y*y*x*y/120
    + x*y*y*y*x/180 - x*y*y*y*y/720 - y*x*x*x*x/720 + y*x*x*x*y/180 -
    y*x*x*y*x/120 - y*x*x*y*y/120 - y*x*y*x*x/120 + y*x*y*x*y/30 - y*x*y*y*x/120
    + y*x*y*y*y/180 + y*y*x*x*x/180 - y*y*x*x*y/120 - y*y*x*y*x/120 -
    y*y*x*y*y/120 + y*y*y*x*x/180 + y*y*y*x*y/180 - y*y*y*y*x/720

    z6 = -x*x*x*x*y*y/1440 + x*x*x*y*x*y/360 + x*x*x*y*y*y/360 - x*x*y*x*x*y/240
    - x*x*y*x*y*y/240 - x*x*y*y*x*y/240 - x*x*y*y* y*y/1440 + x*y*x*x*x*y/360 -
    x*y*x*x*y*y/240 + x*y*x*y*x*y/60 + x*y*x*y*y*y/360 - x*y*y*x*x*y/240 -
    x*y*y*x*y*y/240 + x*y*y*y*x*y/360 - y*x*x*x*y*x/360 + y*x*x*y*x*x/240 +
    y*x*x*y*y*x/240 - y*x*y*x*x*x/360 - y*x*y*x*y*x/60 + y*x*y*y*x*x/240 -
    y*x*y*y*y*x/360 + y*y*x*x*x*x/1440 + y*y*x*x*y*x/240 + y*y*x*y*x*x/240 +
    y*y*x*y*y*x/240 - y*y*y*x*x*x/360 - y*y*y*x*y*x/360 + y*y*y*y*x*x/1440

    z7 = x*x*x*x*x*x*y/30240 - x*x*x*x*x*y*x/5040 - x*x*x*x*x*y*y/5040 +
    x*x*x*x*y*x*x/2016 + x*x*x*x*y*x*y/2016 + x*x*x*x*y*y*x/2016 +
    x*x*x*x*y*y*y/3780 - x*x*x*y*x*x*x/1512 - x*x*x*y*x*x*y/5040 -
    x*x*x*y*x*y*x/630 - x*x*x*y*x*y*y/5040 - x*x*x*y*y*x*x/5040 -
    x*x*x*y*y*x*y/5040 - x*x*x*y*y*y*x/1512 + x*x*x*y*y*y*y/3780 +
    x*x*y*x*x*x*x/2016 - x*x*y*x*x*x*y/5040 + x*x*y*x*x*y*x/840 -
    x*x*y*x*x*y*y/1120 + x*x*y*x*y*x*x/840 + x*x*y*x*y*x*y/840 +
    x*x*y*x*y*y*x/840 - x*x*y*x*y*y*y/5040 - x*x*y*y*x*x*x/5040 -
    x*x*y*y*x*x*y/1120 + x*x*y*y*x*y*x/840 - x*x*y*y*x*y*y/1120 -
    x*x*y*y*y*x*x/5040 - x*x*y*y*y*x*y/5040 + x*x*y*y*y*y*x/2016 -
    x*x*y*y*y*y*y/5040 - x*y*x*x*x*x*x/5040 + x*y*x*x*x*x*y/2016 -
    x*y*x*x*x*y*x/630 - x*y*x*x*x*y*y/5040 + x*y*x*x*y*x*x/840 +
    x*y*x*x*y*x*y/840 + x*y*x*x*y*y*x/840 - x*y*x*x*y*y*y/5040 -
    x*y*x*y*x*x*x/630 + x*y*x*y*x*x*y/840 - x*y*x*y*x*y*x/140 +
    x*y*x*y*x*y*y/840 + x*y*x*y*y*x*x/840 + x*y*x*y*y*x*y/840 -
    x*y*x*y*y*y*x/630 + x*y*x*y*y*y*y/2016 + x*y*y*x*x*x*x/2016 -
    x*y*y*x*x*x*y/5040 + x*y*y*x*x*y*x/840 - x*y*y*x*x*y*y/1120 +
    x*y*y*x*y*x*x/840 + x*y*y*x*y*x*y/840 + x*y*y*x*y*y*x/840 -
    x*y*y*x*y*y*y/5040 - x*y*y*y*x*x*x/1512 - x*y*y*y*x*x*y/5040 -
    x*y*y*y*x*y*x/630 - x*y*y*y*x*y*y/5040 + x*y*y*y*y*x*x/2016 +
    x*y*y*y*y*x*y/2016 - x*y*y*y*y*y*x/5040 + x*y*y*y*y*y*y/30240 +
    y*x*x*x*x*x*x/30240 - y*x*x*x*x*x*y/5040 + y*x*x*x*x*y*x/2016 +
    y*x*x*x*x*y*y/2016 - y*x*x*x*y*x*x/5040 - y*x*x*x*y*x*y/630 -
    y*x*x*x*y*y*x/5040 - y*x*x*x*y*y*y/1512 - y*x*x*y*x*x*x/5040 +
    y*x*x*y*x*x*y/840 + y*x*x*y*x*y*x/840 + y*x*x*y*x*y*y/840 -
    y*x*x*y*y*x*x/1120 + y*x*x*y*y*x*y/840 - y*x*x*y*y*y*x/5040 +
    y*x*x*y*y*y*y/2016 + y*x*y*x*x*x*x/2016 - y*x*y*x*x*x*y/630 +
    y*x*y*x*x*y*x/840 + y*x*y*x*x*y*y/840 + y*x*y*x*y*x*x/840 -
    y*x*y*x*y*x*y/140 + y*x*y*x*y*y*x/840 - y*x*y*x*y*y*y/630 -
    y*x*y*y*x*x*x/5040 + y*x*y*y*x*x*y/840 + y*x*y*y*x*y*x/840 +
    y*x*y*y*x*y*y/840 - y*x*y*y*y*x*x/5040 - y*x*y*y*y*x*y/630 +
    y*x*y*y*y*y*x/2016 - y*x*y*y*y*y*y/5040 - y*y*x*x*x*x*x/5040 +
    y*y*x*x*x*x*y/2016 - y*y*x*x*x*y*x/5040 - y*y*x*x*x*y*y/5040 -
    y*y*x*x*y*x*x/1120 + y*y*x*x*y*x*y/840 - y*y*x*x*y*y*x/1120 -
    y*y*x*x*y*y*y/5040 - y*y*x*y*x*x*x/5040 + y*y*x*y*x*x*y/840 +
    y*y*x*y*x*y*x/840 + y*y*x*y*x*y*y/840 - y*y*x*y*y*x*x/1120 +
    y*y*x*y*y*x*y/840 - y*y*x*y*y*y*x/5040 + y*y*x*y*y*y*y/2016 +
    y*y*y*x*x*x*x/3780 - y*y*y*x*x*x*y/1512 - y*y*y*x*x*y*x/5040 -
    y*y*y*x*x*y*y/5040 - y*y*y*x*y*x*x/5040 - y*y*y*x*y*x*y/630 -
    y*y*y*x*y*y*x/5040 - y*y*y*x*y*y*y/1512 + y*y*y*y*x*x*x/3780 +
    y*y*y*y*x*x*y/2016 + y*y*y*y*x*y*x/2016 + y*y*y*y*x*y*y/2016 -
    y*y*y*y*y*x*x/5040 - y*y*y*y*y*x*y/5040 + y*y*y*y*y*y*x/30240
    
    z8 = x*x*x*x*x*x*y*y/60480 - x*x*x*x*x*y*x*y/10080 - x*x*x*x*x*y*y*y/10080 +
    x*x*x*x*y*x*x*y/4032 + x*x*x*x*y*x*y*y/4032 + x*x*x*x*y*y*x*y/4032 +
    23*x*x*x*x*y*y*y*y/120960 - x*x*x*y*x*x*x*y/3024 - x*x*x*y*x*x*y*y/10080 -
    x*x*x*y*x*y*x*y/1260 - x*x*x*y*x*y*y*y/3024 - x*x*x*y*y*x*x*y/10080 -
    x*x*x*y*y*x*y*y/10080 - x*x*x*y*y*y*x*y/3024 - x*x*x*y*y*y*y*y/10080 +
    x*x*y*x*x*x*x*y/4032 - x*x*y*x*x*x*y*y/10080 + x*x*y*x*x*y*x*y/1680 -
    x*x*y*x*x*y*y*y/10080 + x*x*y*x*y*x*x*y/1680 + x*x*y*x*y*x*y*y/1680 +
    x*x*y*x*y*y*x*y/1680 + x*x*y*x*y*y*y*y/4032 - x*x*y*y*x*x*x*y/10080 -
    x*x*y*y*x*x*y*y/2240 + x*x*y*y*x*y*x*y/1680 - x*x*y*y*x*y*y*y/10080 -
    x*x*y*y*y*x*x*y/10080 - x*x*y*y*y*x*y*y/10080 + x*x*y*y*y*y*x*y/4032 +
    x*x*y*y*y*y*y*y/60480 - x*y*x*x*x*x*x*y/10080 + x*y*x*x*x*x*y*y/4032 -
    x*y*x*x*x*y*x*y/1260 - x*y*x*x*x*y*y*y/3024 + x*y*x*x*y*x*x*y/1680 +
    x*y*x*x*y*x*y*y/1680 + x*y*x*x*y*y*x*y/1680 + x*y*x*x*y*y*y*y/4032 -
    x*y*x*y*x*x*x*y/1260 + x*y*x*y*x*x*y*y/1680 - x*y*x*y*x*y*x*y/280 -
    x*y*x*y*x*y*y*y/1260 + x*y*x*y*y*x*x*y/1680 + x*y*x*y*y*x*y*y/1680 -
    x*y*x*y*y*y*x*y/1260 - x*y*x*y*y*y*y*y/10080 + x*y*y*x*x*x*x*y/4032 -
    x*y*y*x*x*x*y*y/10080 + x*y*y*x*x*y*x*y/1680 - x*y*y*x*x*y*y*y/10080 +
    x*y*y*x*y*x*x*y/1680 + x*y*y*x*y*x*y*y/1680 + x*y*y*x*y*y*x*y/1680 +
    x*y*y*x*y*y*y*y/4032 - x*y*y*y*x*x*x*y/3024 - x*y*y*y*x*x*y*y/10080 -
    x*y*y*y*x*y*x*y/1260 - x*y*y*y*x*y*y*y/3024 + x*y*y*y*y*x*x*y/4032 +
    x*y*y*y*y*x*y*y/4032 - x*y*y*y*y*y*x*y/10080 + y*x*x*x*x*x*y*x/10080 -
    y*x*x*x*x*y*x*x/4032 - y*x*x*x*x*y*y*x/4032 + y*x*x*x*y*x*x*x/3024 +
    y*x*x*x*y*x*y*x/1260 + y*x*x*x*y*y*x*x/10080 + y*x*x*x*y*y*y*x/3024 -
    y*x*x*y*x*x*x*x/4032 - y*x*x*y*x*x*y*x/1680 - y*x*x*y*x*y*x*x/1680 -
    y*x*x*y*x*y*y*x/1680 + y*x*x*y*y*x*x*x/10080 - y*x*x*y*y*x*y*x/1680 +
    y*x*x*y*y*y*x*x/10080 - y*x*x*y*y*y*y*x/4032 + y*x*y*x*x*x*x*x/10080 +
    y*x*y*x*x*x*y*x/1260 - y*x*y*x*x*y*x*x/1680 - y*x*y*x*x*y*y*x/1680 +
    y*x*y*x*y*x*x*x/1260 + y*x*y*x*y*x*y*x/280 - y*x*y*x*y*y*x*x/1680 +
    y*x*y*x*y*y*y*x/1260 - y*x*y*y*x*x*x*x/4032 - y*x*y*y*x*x*y*x/1680 -
    y*x*y*y*x*y*x*x/1680 - y*x*y*y*x*y*y*x/1680 + y*x*y*y*y*x*x*x/3024 +
    y*x*y*y*y*x*y*x/1260 - y*x*y*y*y*y*x*x/4032 + y*x*y*y*y*y*y*x/10080 -
    y*y*x*x*x*x*x*x/60480 - y*y*x*x*x*x*y*x/4032 + y*y*x*x*x*y*x*x/10080 +
    y*y*x*x*x*y*y*x/10080 + y*y*x*x*y*x*x*x/10080 - y*y*x*x*y*x*y*x/1680 +
    y*y*x*x*y*y*x*x/2240 + y*y*x*x*y*y*y*x/10080 - y*y*x*y*x*x*x*x/4032 -
    y*y*x*y*x*x*y*x/1680 - y*y*x*y*x*y*x*x/1680 - y*y*x*y*x*y*y*x/1680 +
    y*y*x*y*y*x*x*x/10080 - y*y*x*y*y*x*y*x/1680 + y*y*x*y*y*y*x*x/10080 -
    y*y*x*y*y*y*y*x/4032 + y*y*y*x*x*x*x*x/10080 + y*y*y*x*x*x*y*x/3024 +
    y*y*y*x*x*y*x*x/10080 + y*y*y*x*x*y*y*x/10080 + y*y*y*x*y*x*x*x/3024 +
    y*y*y*x*y*x*y*x/1260 + y*y*y*x*y*y*x*x/10080 + y*y*y*x*y*y*y*x/3024 -
    23*y*y*y*y*x*x*x*x/120960 - y*y*y*y*x*x*y*x/4032 - y*y*y*y*x*y*x*x/4032 -
    y*y*y*y*x*y*y*x/4032 + y*y*y*y*y*x*x*x/10080 + y*y*y*y*y*x*y*x/10080 -
    y*y*y*y*y*y*x*x/60480

    println("norm z1: ", norm(z1))
    println("norm z2: ", norm(z2))
    println("norm z3: ", norm(z3))
    println("norm z4: ", norm(z4))
    println("norm z5: ", norm(z5))
    println("norm z6: ", norm(z6))
    println("norm z7: ", norm(z7))
    println("norm z8: ", norm(z8))

    z=z1+z2+z3+z4+z5+z6+z7+z8

    return z

end

#
# Most of the functions that follow were adopted and adapted from the
# ONDAP FEM code.  See: Object-oriented finite-element dynamic
# simulation of geometrically nonlinear space structures, Victor
# Balopoulos, Ph.D. dissertation, Cornell University, 1997
#

#
# QofRT
#
# This function implements the singularity-free algorithm for
# obtaining from a rotation tensor, R, its corresponding quaternion,
# q.  This algorithm is due to Spurrier and is described below.
#
# First the scalar "maxm" is defined as the maximum among the diagonal
# terms and the trace of the rotation tensor.
#
# i)  If maxm is equal to the trace of R, then:
#       2 * qs = sqrt( 1 + maxm )
#     and
#       qv = axial_vec( skew(R) ) / (2 * qs)
#
# ii) If maxm is equal to R[j][j] (no summation), then:
#       2 * qv[j] = sqrt( 2 * maxm + 1 - tr(R) )
#       qs = axial_vec( skew(R) )[j] / (2 * qv[j])
#     and for i,j,k all different:
#       qv[k] = off_diag_vec( symm(R) )[i] / (2 * qv[j])
#
# The following paragraph is my (Victor's) attempt to explain why this
# algorithm works and is efficient and accurate.
#
# Since the rotation tensor is a quadratic function of the quaternion,
# at least one square root has to be evaluated to get back the
# quaternion.  After that, only divisions are needed and the divisor
# should be bounded as far from zero as possible.  This is attained by
# Spurrier's algorithm in a simple manner.  Looking at R as expressed
# at the end of the comments on RTofQ(), one can easily notice that
# the maximum among the four quantities ( R[1, 1], R[2, 2], R[3, 3],
# trace(R) ) corresponds to the component with the maximum absolute
# value in the quaternion ( qv[0], qv[1], qv[2], qs ).  Thus
# Spurrier's algorithm always computes the absolutely maximum
# component of the quaternion first and uses that in subsequent
# divisions.
#
# NOTICE that obtaining all components of the quaternion by square
# roots is both expensive and AMBIGUOUS, because four signs have to be
# chosen, only one of which is arbitrary.
#
function QofRT(R)
    trR = tr(R)
    maxm = trR
    maxi = 4
    q = zeros(4, 1)
    for i = 1 : 3
        if R[i, i] > maxm
            maxm = R[i, i]
            maxi = i
        end
    end
    if maxi == 4
        root = sqrt(maxm + 1.0)
        factor = 0.5 / root
        q[1] = factor * (R[3, 2] - R[2, 3])
        q[2] = factor * (R[1, 3] - R[3, 1])
        q[3] = factor * (R[2, 1] - R[1, 2])
        q[4] = 0.5 * root
    elseif maxi == 3
        root = sqrt(2.0 * maxm + 1.0 - trR)
        factor = 0.5 / root
        q[1] = factor * (R[1, 3] + R[3, 1])
        q[2] = factor * (R[2, 3] + R[3, 2])
        q[3] = 0.5 * root
        q[4] = factor * (R[2, 1] - R[1, 2])
    elseif maxi == 2
        root = sqrt(2.0 * maxm + 1.0 - trR)
        factor = 0.5 / root
        q[1] = factor * (R[1, 2] + R[2, 1])
        q[2] = 0.5 * root
        q[3] = factor * (R[2, 3] + R[3, 2])
        q[4] = factor * (R[1, 3] - R[3, 1])
    elseif maxi == 1
        root = sqrt(2.0 * maxm + 1.0 - trR)
        factor = 0.5 / root
        q[1] = 0.5 * root
        q[2] = factor * (R[1, 2] + R[2, 1])
        q[3] = factor * (R[1, 3] + R[3, 1])
        q[4] = factor * (R[3, 2] - R[2, 3])
    else
        error("QofRT failed!")
    end
    return q
end

#
# RVofQ
#
# This function maps a quaternion, qq = ( qv, qs ), to its
# corresponding "principal" rotation pseudo-vector, aa, where
# "principal" signifies that |aa| <= pi.  Both qq and -qq map into the
# same rotation matrix It is convenient to require that qs >= 0, for
# reasons explained below.  The sign inversion is applied to a local
# copy of qq to avoid side effects.
#
#    |qv| = | sin( |aa| / 2 ) |
#    qs  =   cos( |aa| / 2 )
#      <==>
#    |aa| / 2 = k * pi (+ or -) asin( |qv| )
#    |aa| / 2 = 2 * l * pi (+ or -) acos( qs )
#
# The smallest positive solution is:	|aa| = 2 * acos( qs )
# which satisfies the inequality:	0 <= |aa| <= pi
# because of the assumption  qs >= 0.  Given |aa|, aa
# is obtained as:
#
#    aa = ( |aa| / sin( acos(qs) ) ) qv
#       = ( |aa| / sqrt(1 - qs^2) ) qv
#
# The procedure described above is prone to numerical errors when qs
# is close to 1, i.e. when |aa| is close to 0.  Since this is the most
# common case, special care must be taken.  It is observed that the
# cosine function is insensitive to perturbations of its argument in
# the neighborhood of points for which the sine function is conversely
# at its most sensitive.  Thus the numerical difficulties are avoided
# by computing |aa| and aa as:
#
#    |aa| = 2 * asin( |qv| )
#    aa  = ( |aa| / |qv| ) qv
#
# whenever qs  is close to 1.
#
function RVofQ(qq)
    if qq[4] >= 0
        q = qq
    else
        q = -qq
    end
    qs = q[4]
    qv = [q[1] q[2] q[3]]
    qvnorm = norm(qv)
    aanorm = 2.0 * (qvnorm < sqrt(0.5) ? asin(qvnorm) : acos(qs))
    coef = qvnorm < sqrt(eps()) ? 2.0 : aanorm / qvnorm
    aa = coef * qv
    return aa
end

# Tricky mathematical functions
#
# In the algebra of rotations one often comes across functions that
# take undefined (0/0) values at some points.  Close to such points
# these functions must be evaluated using their asymptotic
# expansions; otherwise the computer may produce wildly erroneous
# results or a floating point exception.  To avoid unreadable code
# everywhere such functions are used, we introduce here functions to
# the same effect.
#
# NAME  FUNCTION FORM      X    ASSYMPTOTICS    FIRST RADIUS    SECOND RADIUS
# °°°°  °°°°°°°°°°°°°      °    °°°°°°°°°°°°    °°°°°°°°°°°°    °°°°°°°°°°°°°
# Ψ     sin(x)/x           0    1.0(-x^2/6)     (6*EPS)^.5      (120*EPS)^.25
#
# ΨΧΤ
function Ψ(x)
    y = abs(x)
    e2 = sqrt(eps())
    e4 = sqrt(e2)
    if (y > e4)
        return sin(y) / y
    elseif (y > e2)
        return 1.0 - y * y / 6.0
    else
        return 1.0
    end
end

#
# QofRV
#
# This function maps a rotation pseudo-vector, aa, to a quaternion, qq
# = ( qv, qs ), where qv is a vector and qs a scalar, defined as
# follows:
#
#   qv = sin( |aa| / 2 ) * aa / |aa|
#   qs = cos( |aa| / 2 )
#
function QofRV(aa)
    halfnorm = 0.5 * norm(aa)
    temp = 0.5 * Ψ(halfnorm)
    qq = zeros(4)
    qq[1 : 3] = temp * aa
    qq[4] = cos(halfnorm)
    return qq
end

#
# RTofQ
#
# For qq = ( qv, qs ), where qv is a vector and qs is a scalar, one
# can write:
#
#    R = 2 * qv \otimes qv
#      + 2 * qs * check(qv)
#      + (2 * qs^2 - 1) * t_iden
#
function RTofQ(qq)
    qs = qq[4]
    qv = [qq[1]; qq[2]; qq[3]]
    I = [1 0 0; 0 1 0; 0 0 1]
    R = 2.0 * qv * qv' + 2.0 * qs * check(qv) + (2.0 * qs * qs - 1.0) * I
    return R
end

#
# Rotation pseudo-vector of rotation tensor
#
function RVofRT(R)
    q = QofRT(R)
    w = RVofQ(q)
    return w
end

#
# Rotation tensor of rotation pseudo-vector
#
function RTofRV(w)
    q = QofRV(w)
    R = RTofQ(q)
    return R
end

#
# RVcontin
#
# This function takes as input two rotation pseudo- vectors, "old" and
# "last". The function returns a pseudo-vector which maps into the
# same rotation tensor as "old", while being as close to "last" as
# possible.  "last" is strictly a parameter (read-only).
#
# Vectors that map into the same rotation as "old" are parallel to it
# and differ in length by integer multiples of TWO_PI.  The one
# closest to "last" is also closest to the projection of "last" along
# "old", since the perpendicular component is common to all.
#
# NOTE: This routine has no way of knowing how much the rotation
# pseudo-vector should have changed from its last value.  The outside
# code has to enforce any necessary limits on the size of rotation
# increments.
#
# Compute the "unit" vector along "old" and its dot product with
# "last" ( "proj * unit" is the projection of "last" along "old).
# Round "proj" to the closest multiple of TWO_PI and add to it the
# "norm" of "old".  It is obvious that "proj * unit" (which overwrites
# "old") is the required continuation vector (rotation-equivalent to
# the original "old" and closest to "last").  If "old" is zero, the
# continuation vector should be the closest multiple of TWO_PI along
# "last"; hence "proj" and "unit" must satisfy "last == proj * unit".
#
function RVcontin(old, last)
    norm_old = norm(old)
    unit = normalize(vec(old))
    if (norm_old > 0.0)
        diff = last - old
        proj = dot(unit, diff)
    else
        unit = normalize(vec(last))
        proj = norm(last)
    end
    if (proj == 0.0)
        # No change required
        return old
    end
    kk = ceil(0.5 * proj / pi - 0.5)
    if (kk == 0.0)
        # No change required
        return old
    end
    proj = 2.0 * kk * pi + norm_old
    new = proj * unit
    return new
end

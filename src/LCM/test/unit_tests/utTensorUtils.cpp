//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include<ctime>

#include "Intrepid_FieldContainer.hpp"
#include "Sacado.hpp"
#include "VectorTensorBase.h"
#include "Teuchos_UnitTestHarness.hpp"

using namespace std;

typedef double ScalarT;

namespace {

  TEUCHOS_UNIT_TEST( TensorUtils, Initialization )
  {
    Intrepid::FieldContainer<ScalarT> FC(3, 3);
    FC(0, 0) = 1.0;
    FC(0, 1) = 2.0;
    FC(0, 2) = 3.0;
    FC(1, 0) = 4.0;
    FC(1, 1) = 5.0;
    FC(1, 2) = 6.0;
    FC(2, 0) = 7.0;
    FC(2, 1) = 8.0;
    FC(2, 2) = 9.0;

    ScalarT const * dataPtr0 = &FC(0, 0);

    LCM::Index const N = 3;
    LCM::Vector<ScalarT> u(N, dataPtr0);

    TEST_COMPARE( u(0), ==, 1.0);
    TEST_COMPARE( u(1), ==, 2.0);
    TEST_COMPARE( u(2), ==, 3.0);

    ScalarT const * dataPtr1 = &FC(1, 0);

    u = LCM::Vector<ScalarT>(N, dataPtr1);

    TEST_COMPARE( u(0), ==, 4.0);
    TEST_COMPARE( u(1), ==, 5.0);
    TEST_COMPARE( u(2), ==, 6.0);

    ScalarT const * dataPtr2 = &FC(2, 0);

    u = LCM::Vector<ScalarT>(N, dataPtr2);

    TEST_COMPARE( u(0), ==, 7.0);
    TEST_COMPARE( u(1), ==, 8.0);
    TEST_COMPARE( u(2), ==, 9.0);
  }

  TEUCHOS_UNIT_TEST( TensorUtils, VectorAddition )
  {
    LCM::Vector<ScalarT> const u(1.0, 0.0, 0.0);
    LCM::Vector<ScalarT> const v(0.0, 1.0, 0.0);
    LCM::Vector<ScalarT> const w(1.0, 1.0, 0.0);

    TEST_COMPARE( u + v == w, !=, 0);
  }

  TEUCHOS_UNIT_TEST( TensorUtils, VectorSubtraction )
  {
    LCM::Vector<ScalarT> u(3);
    LCM::Vector<ScalarT> v(3);
    u(0) = 1.0;
    u(1) = 2.0;
    u(2) = 3.0;

    v = u - u;

    TEST_COMPARE( LCM::norm(v), <=, LCM::machine_epsilon<ScalarT>());
  }

  TEUCHOS_UNIT_TEST( TensorUtils, VectorScalarMultipliaction )
  {
    LCM::Vector<ScalarT> u(3);
    LCM::Vector<ScalarT> v(3);
    LCM::Vector<ScalarT> w(3);
    u(0) = 1.0;
    u(1) = 2.0;
    u(2) = 3.0;

    v(0) = -2.0;
    v(1) = -4.0;
    v(2) = -6.0;

    w = 4.0 * u + 2.0 * v;

    TEST_COMPARE( LCM::norm(w), <=, LCM::machine_epsilon<ScalarT>());
  }

  TEUCHOS_UNIT_TEST( TensorUtils, TensorInstantiation )
  {
    Intrepid::FieldContainer<ScalarT> FC(2, 3, 3);
    FC(0, 0, 0) = 1.0;
    FC(0, 0, 1) = 2.0;
    FC(0, 0, 2) = 3.0;
    FC(0, 1, 0) = 4.0;
    FC(0, 1, 1) = 5.0;
    FC(0, 1, 2) = 6.0;
    FC(0, 2, 0) = 7.0;
    FC(0, 2, 1) = 8.0;
    FC(0, 2, 2) = 9.0;
    FC(1, 0, 0) = 10.0;
    FC(1, 0, 1) = 11.0;
    FC(1, 0, 2) = 12.0;
    FC(1, 1, 0) = 13.0;
    FC(1, 1, 1) = 14.0;
    FC(1, 1, 2) = 15.0;
    FC(1, 2, 0) = 16.0;
    FC(1, 2, 1) = 17.0;
    FC(1, 2, 2) = 18.0;

    ScalarT const * dataPtr0 = &FC(0, 0, 0);

    LCM::Tensor<ScalarT> const A(3, dataPtr0);

    TEST_COMPARE( A(0,0), ==, 1.0);
    TEST_COMPARE( A(0,1), ==, 2.0);
    TEST_COMPARE( A(0,2), ==, 3.0);
    TEST_COMPARE( A(1,0), ==, 4.0);
    TEST_COMPARE( A(1,1), ==, 5.0);
    TEST_COMPARE( A(1,2), ==, 6.0);
    TEST_COMPARE( A(2,0), ==, 7.0);
    TEST_COMPARE( A(2,1), ==, 8.0);
    TEST_COMPARE( A(2,2), ==, 9.0);

    ScalarT const * dataPtr1 = &FC(1, 0, 0);

    LCM::Tensor<ScalarT> const B(3, dataPtr1);

    TEST_COMPARE( B(0,0), ==, 10.0);
    TEST_COMPARE( B(0,1), ==, 11.0);
    TEST_COMPARE( B(0,2), ==, 12.0);
    TEST_COMPARE( B(1,0), ==, 13.0);
    TEST_COMPARE( B(1,1), ==, 14.0);
    TEST_COMPARE( B(1,2), ==, 15.0);
    TEST_COMPARE( B(2,0), ==, 16.0);
    TEST_COMPARE( B(2,1), ==, 17.0);
    TEST_COMPARE( B(2,2), ==, 18.0);
  }

  TEUCHOS_UNIT_TEST( TensorUtils, TensorAddition )
  {
    LCM::Tensor<ScalarT> const A(3, 1.0);
    LCM::Tensor<ScalarT> const B(3, 2.0);
    LCM::Tensor<ScalarT> const C(3, 3.0);

    TEST_COMPARE( C == A + B, !=, 0);
  }

  TEUCHOS_UNIT_TEST( TensorUtils, TensorInverse )
  {
    std::srand(std::time(NULL));
    LCM::Index const N = double(std::rand())/double(RAND_MAX) * 7.0 + 3.0;
    LCM::Tensor<ScalarT> A(N);
    LCM::Tensor<ScalarT> B(N);
    LCM::Tensor<ScalarT> C(N);

    for (LCM::Index i = 0; i < N; ++i) {
      for (LCM::Index j = 0; j < N; ++j) {
        A(i,j) = double(std::rand())/double(RAND_MAX) * 20.0 - 10.0;
      }
    }

    B = inverse(A);

    C = A * B;

    ScalarT const
    error = LCM::norm(C - LCM::eye<ScalarT>(N)) / norm(A);

    TEST_COMPARE( error, <=, 100.0 * LCM::machine_epsilon<ScalarT>());
  }

  TEUCHOS_UNIT_TEST( TensorUtils, TensorManipulation )
  {
    LCM::Tensor<ScalarT> A = LCM::eye<ScalarT>(3);
    LCM::Tensor<ScalarT> B(3);
    LCM::Tensor<ScalarT> C(3);
    LCM::Vector<ScalarT> u(3);

    A = 2.0 * A;
    A(1, 0) = A(0, 1) = 1.0;
    A(2, 1) = A(1, 2) = 1.0;

    B = LCM::inverse(A);

    C = A * B;

    TEST_COMPARE( LCM::norm(C - LCM::eye<ScalarT>(3)), <=,
        LCM::machine_epsilon<ScalarT>());

    ScalarT I1 = LCM::I1(A);
    ScalarT I2 = LCM::I2(A);
    ScalarT I3 = LCM::I3(A);

    u(0) = I1 - 6;
    u(1) = I2 - 10;
    u(2) = I3 - 4;

    ScalarT const
    error = LCM::norm(u);

    TEST_COMPARE( error, <=, LCM::machine_epsilon<ScalarT>());
  }

  TEUCHOS_UNIT_TEST( TensorUtils, TensorExp )
  {
    LCM::Tensor<ScalarT> const
    A(1, 2, 3, 4, 5, 6, 7, 8, 9);

    LCM::Tensor<ScalarT> const
    B = LCM::exp_pade(A);

    LCM::Tensor<ScalarT> const
    C = LCM::exp_taylor(A);

    LCM::Tensor<ScalarT> const
    D = B - C;

    ScalarT const
    error = norm(D) / norm(B);

    TEST_COMPARE( error, <=, 100.0 * LCM::machine_epsilon<ScalarT>());
  }

  TEUCHOS_UNIT_TEST( TensorUtils, TensorEig_Spd )
  {
    LCM::Tensor<ScalarT> A = LCM::eye<ScalarT>(3);
    A(0, 1) = 0.1;
    A(1, 0) = 0.1;
    LCM::Tensor<ScalarT> V(3);
    LCM::Tensor<ScalarT> D(3);
    boost::tie(V, D) = LCM::eig_sym(A);

    TEST_COMPARE( std::abs(D(0,0) - 1.1), <=, LCM::machine_epsilon<ScalarT>());
    TEST_COMPARE( std::abs(D(1,1) - 1.0), <=, LCM::machine_epsilon<ScalarT>());
    TEST_COMPARE( std::abs(D(2,2) - 0.9), <=, LCM::machine_epsilon<ScalarT>());
  }

  TEUCHOS_UNIT_TEST( TensorUtils, TensorLeftPolarDecomposition )
  {
    LCM::Tensor<ScalarT> V0(1.1, 0.2, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0, 1.2);
    LCM::Tensor<ScalarT> R0(sqrt(2) / 2, -sqrt(2) / 2, 0.0, sqrt(2) / 2,
        sqrt(2) / 2, 0.0, 0.0, 0.0, 1.0);

    LCM::Tensor<ScalarT> F = V0 * R0;
    LCM::Tensor<ScalarT> V(3);
    LCM::Tensor<ScalarT> R(3);
    boost::tie(V, R) = LCM::polar_left(F);

    TEST_COMPARE( LCM::norm(V-V0), <=, 10.0*LCM::machine_epsilon<ScalarT>());
    TEST_COMPARE( LCM::norm(R-R0), <=, LCM::machine_epsilon<ScalarT>());
  }

  TEUCHOS_UNIT_TEST( TensorUtils, TensorLogRotation )
  {
    LCM::Tensor<ScalarT> R = LCM::identity<ScalarT>(3);
    LCM::Tensor<ScalarT> R0(sqrt(2) / 2, -sqrt(2) / 2, 0.0, sqrt(2) / 2,
        sqrt(2) / 2, 0.0, 0.0, 0.0, 1.0);

    LCM::Tensor<ScalarT> r = log_rotation(R);
    LCM::Tensor<ScalarT> r0 = log_rotation(R0);

    TEST_COMPARE( LCM::norm(r), <=, LCM::machine_epsilon<ScalarT>());
    TEST_COMPARE( std::abs(r0(0,1) + 0.785398163397448), <=,
        10.0*LCM::machine_epsilon<ScalarT>());
    TEST_COMPARE( std::abs(r0(0,1) + r0(1,0)), <=,
        LCM::machine_epsilon<ScalarT>());

    ScalarT theta = std::acos(-1.0)
        + 10 * LCM::machine_epsilon<ScalarT>(); // rotation angle

    R(0, 0) = cos(theta);
    R(1, 1) = cos(theta);
    R(0, 1) = sin(theta);
    R(1, 0) = -sin(theta);
    R(2, 2) = 1.0;

    LCM::Tensor<ScalarT> logR = log_rotation(R);

    LCM::Tensor<ScalarT> Rref(3, 0.0);
    Rref(0, 1) = -theta;
    Rref(1, 0) = theta;

    TEST_COMPARE( LCM::norm(logR - Rref), <=,
        100*LCM::machine_epsilon<ScalarT>());
  }

  TEUCHOS_UNIT_TEST( TensorUtils, TensorBCHExp )
  {
    LCM::Tensor<ScalarT> F = 3.0 * LCM::identity<ScalarT>(3);
    LCM::Tensor<ScalarT> V(3), R(3), logV(3), logR(3);

    boost::tie(V, R, logV) = polar_left_logV(F);
    logR = log_rotation(R);

    LCM::Tensor<ScalarT> f = LCM::bch(logV, logR);

    TEST_COMPARE( std::abs(f(0,0) - std::log(3.0)), <=,
        LCM::machine_epsilon<ScalarT>());

    LCM::Vector<ScalarT> u(3);
    u(0) = std::acos(-1.0) / std::sqrt(2.0);
    u(1) = u(0);
    u(2) = 0.0;

    LCM::Tensor<ScalarT> R1(3, 0.0);
    LCM::Tensor<ScalarT> logR2(3, 0.0);
    logR2(0, 2) = u(1);
    logR2(1, 2) = -u(0);
    logR2(2, 0) = -u(1);
    logR2(2, 1) = u(0);
    logR2(0, 1) = -u(2);
    logR2(1, 0) = u(2);

    R1 = LCM::exp_skew_symmetric(logR2);
    LCM::Tensor<ScalarT> Rref = LCM::zero<ScalarT>(3);
    Rref(0, 1) = 1.0;
    Rref(1, 0) = 1.0;
    Rref(2, 2) = -1.0;

    TEST_COMPARE( norm(Rref-R1), <=,
        100.0*LCM::machine_epsilon<ScalarT>());
    TEST_COMPARE( norm(exp_skew_symmetric(logR) - R), <=,
        100.0*LCM::machine_epsilon<ScalarT>());
  }

  TEUCHOS_UNIT_TEST( TensorUtils, TensorPolarLeftLogV )
  {
    LCM::Tensor<ScalarT> const
    Finc(3.60070151614402, 0.00545892068653966, 0.144580850331452,
        -5.73345529510674, 0.176660910549112,   1.39627497290058,
         2.51510445213514, 0.453212159218359,  -1.44616077859513);

    LCM::Tensor<ScalarT> const
    solLogVinc(0.265620603957487, -1.066921781600734, -0.089540974250415,
        -1.066921781600734, 0.927394431410918, -0.942214085118614,
        -0.089540974250415, -0.942214085118613, 0.105672693695746);

    LCM::Tensor<ScalarT> Vinc(3), Rinc(3), logVinc(3), logRinc(3);

    boost::tie(Vinc, Rinc, logVinc) = polar_left_logV(Finc);

    TEST_COMPARE( std::abs(logVinc(0,0) - solLogVinc(0,0)), <=, 100*LCM::machine_epsilon<ScalarT>());
    TEST_COMPARE( std::abs(logVinc(0,1) - solLogVinc(0,1)), <=, 100*LCM::machine_epsilon<ScalarT>());
    TEST_COMPARE( std::abs(logVinc(0,2) - solLogVinc(0,2)), <=, 100*LCM::machine_epsilon<ScalarT>());
    TEST_COMPARE( std::abs(logVinc(1,0) - solLogVinc(1,0)), <=, 100*LCM::machine_epsilon<ScalarT>());
    TEST_COMPARE( std::abs(logVinc(1,1) - solLogVinc(1,1)), <=, 100*LCM::machine_epsilon<ScalarT>());
    TEST_COMPARE( std::abs(logVinc(1,2) - solLogVinc(1,2)), <=, 100*LCM::machine_epsilon<ScalarT>());
    TEST_COMPARE( std::abs(logVinc(2,0) - solLogVinc(2,0)), <=, 100*LCM::machine_epsilon<ScalarT>());
    TEST_COMPARE( std::abs(logVinc(2,1) - solLogVinc(2,1)), <=, 100*LCM::machine_epsilon<ScalarT>());
    TEST_COMPARE( std::abs(logVinc(2,2) - solLogVinc(2,2)), <=, 100*LCM::machine_epsilon<ScalarT>());
  }

  TEUCHOS_UNIT_TEST( TensorUtils, TensorVolDev )
  {
    LCM::Tensor<ScalarT>
    A = 3.0 * LCM::eye<ScalarT>(3);

    TEST_COMPARE( norm(A - vol(A)), <=, 100.0*LCM::machine_epsilon<ScalarT>());

    LCM::Tensor<ScalarT>
    B = dev(A);

    A(0, 0) = 0.0;
    A(1, 1) = 0.0;
    A(2, 2) = 0.0;

    TEST_COMPARE( norm(A - B), <=, 100.0*LCM::machine_epsilon<ScalarT>());
  }

  TEUCHOS_UNIT_TEST(TensorUtils, TensorSVD2x2)
  {
    ScalarT const
    phi = 1.0;

    ScalarT const
    psi = 2.0;

    ScalarT const
    s0 = sqrt(3.0);

    ScalarT const
    s1 = sqrt(2.0);

    ScalarT const
    cl = cos(phi);

    ScalarT const
    sl = sin(phi);

    ScalarT const
    cr = cos(psi);

    ScalarT const
    sr = sin(psi);

    LCM::Tensor<ScalarT> const
    X(cl, -sl, sl, cl);

    LCM::Tensor<ScalarT> const
    Y(cr, -sr, sr, cr);

    LCM::Tensor<ScalarT> const
    D(s0, 0.0, 0.0, s1);

    LCM::Tensor<ScalarT> const
    A = X * D * LCM::transpose(Y);

    LCM::Tensor<ScalarT>
    U(2), S(2), V(2);

    boost::tie(U, S, V) = LCM::svd(A);

    LCM::Tensor<ScalarT>
    B = U * S * LCM::transpose(V);

    ScalarT const
    error = norm(A - B) / norm(A);

    TEST_COMPARE(error, <=, 100.0*LCM::machine_epsilon<ScalarT>());
  }

  TEUCHOS_UNIT_TEST(TensorUtils, TensorSVD3x3)
  {
    LCM::Tensor<ScalarT> const
    A(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);

    LCM::Tensor<ScalarT>
    U(3), S(3), V(3);

    boost::tie(U, S, V) = LCM::svd(A);

    LCM::Tensor<ScalarT> const
    B = U * S * LCM::transpose(V);

    ScalarT const
    error = norm(A - B) / norm(A);

    TEST_COMPARE(error, <=, 100.0*LCM::machine_epsilon<ScalarT>());
  }

  TEUCHOS_UNIT_TEST(TensorUtils, TensorEigenSym2x2)
  {
    LCM::Tensor<ScalarT> const
    A(2.0, 1.0, 1.0, 2.0);

    LCM::Tensor<ScalarT>
    V(2), D(2);

    boost::tie(V, D) = LCM::eig_sym(A);

    LCM::Tensor<ScalarT> const
    B = V * D * LCM::transpose(V);

    ScalarT const
    error = norm(A - B) / norm(A);

    TEST_COMPARE(error, <=, 100.0*LCM::machine_epsilon<ScalarT>());
  }

  TEUCHOS_UNIT_TEST(TensorUtils, TensorEigenSym3x3)
  {
    LCM::Tensor<ScalarT> const
    A(2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0);

    LCM::Tensor<ScalarT>
    V(3), D(3);

    boost::tie(V, D) = LCM::eig_sym(A);

    LCM::Tensor<ScalarT> const
    B = V * D * LCM::transpose(V);

    ScalarT const
    error = norm(A - B) / norm(A);

    TEST_COMPARE(error, <=, 100.0*LCM::machine_epsilon<ScalarT>());
  }

  TEUCHOS_UNIT_TEST(TensorUtils, TensorInverse4x4)
  {
    LCM::Tensor<ScalarT>
    A = 2.0 * LCM::identity<ScalarT>(4);

    A(0,1) = 1.0;
    A(1,0) = 1.0;

    A(1,2) = 1.0;
    A(2,1) = 1.0;

    A(2,3) = 1.0;
    A(3,2) = 1.0;

    LCM::Tensor<ScalarT> const
    B = inverse(A);

    LCM::Tensor<ScalarT> const
    C = A * B;

    LCM::Tensor<ScalarT> const
    I = LCM::eye<ScalarT>(4);

    ScalarT const
    error = norm(C - I) / norm(A);

    TEST_COMPARE(error, <=, 100.0*LCM::machine_epsilon<ScalarT>());
  }

  TEUCHOS_UNIT_TEST(TensorUtils, TensorPolar3x3)
  {
    LCM::Tensor<ScalarT>
    A(2.0, 1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 2.0);

    LCM::Tensor<ScalarT>
    R(3), U(3);

    boost::tie(R, U) = LCM::polar_right(A);

    LCM::Tensor<ScalarT>
    X(3), D(3), Y(3);

    boost::tie(X, D, Y) = LCM::svd(A);


    LCM::Tensor<ScalarT>
    B = R - X * LCM::transpose(Y) + U - Y * D * LCM::transpose(Y);

    ScalarT const
    error = norm(B) / norm(A);

    TEST_COMPARE(error, <=, 100.0*LCM::machine_epsilon<ScalarT>());
  }

  TEUCHOS_UNIT_TEST(TensorUtils, TensorSVD3x3Fad)
  {
    LCM::Tensor < Sacado::Fad::DFad<double> >
    A(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);

    LCM::Tensor<Sacado::Fad::DFad<double> >
    U(3), S(3), V(3);

    boost::tie(U, S, V) = LCM::svd(A);

    LCM::Tensor < Sacado::Fad::DFad<double> >
    B = U * S * LCM::transpose(V);

    Sacado::Fad::DFad<double> const
    error = norm(B - A) / norm(A);

    TEST_COMPARE(error, <=, 100.0*LCM::machine_epsilon<ScalarT>());
  }

  TEUCHOS_UNIT_TEST(TensorUtils, TensorCholesky)
  {
    LCM::Tensor<ScalarT>
    A(1.0, 1.0, 1.0, 1.0, 5.0, 3.0, 1.0, 3.0, 3.0);

    LCM::Tensor<ScalarT>
    G(3);

    bool
    is_spd;

    boost::tie(G, is_spd) = LCM::cholesky(A);

    LCM::Tensor<ScalarT>
    B(1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 1.0, 1.0, 1.0);

    ScalarT const
    error = norm(G - B) / norm(A);

    TEST_COMPARE(error, <=, 100.0*LCM::machine_epsilon<ScalarT>());
  }

} // namespace

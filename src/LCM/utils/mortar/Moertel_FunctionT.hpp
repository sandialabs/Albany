//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef MOERTEL_FUNCTIONT_HPP
#define MOERTEL_FUNCTIONT_HPP

#include <ctime>
#include <iostream>

/*!
\brief MOERTEL: namespace of the Moertel package

The Moertel package depends on \ref Epetra, \ref EpetraExt, \ref Teuchos,
\ref Amesos, \ref ML and \ref AztecOO:<br>
Use at least the following lines in the configure of Trilinos:<br>
\code
--enable-moertel
--enable-epetra
--enable-epetraext
--enable-teuchos
--enable-ml
--enable-aztecoo --enable-aztecoo-teuchos
--enable-amesos
\endcode

*/
namespace MoertelT {

/*!
\class Function

\brief <b> A virtual class to form a shape function of some type </b>



\author Glen Hansen (gahanse@sandia.gov)

*/

// Helper class for segment functions
struct SegFunction_Traits
{
};

// MOERTEL_TEMPLATE_STATEMENT
template <class traits = SegFunction_Traits>
class FunctionT
{
 public:
  /*!
  \brief Type of function

  \param func_none : default value
  \param func_Constant1D : Constant function on a 1D Segment
  \param func_Linear1D : Linear function on a 1D Segment
  \param func_DualLinear1D : Dual linear function on a 1D Segment
  \param func_LinearTri : Linear function on a 3-noded triangle
  \param func_DualLinearTri : Dual linear function on a 3-noded triangle
  \param func_ConstantTri : Constant function on a 3-noded triangle
  */

  // @{ \name Constructors and destructors

  /*!
  \brief Constructor

  Constructs an instance of this base class.<br>

  \param outlevel : Level of output information written to stdout ( 0 - 10 )
  */

  FunctionT(int outlevel) : outputlevel_(outlevel){};

  /*!
  \brief Copy-Constructor

  Makes a deep copy<br>

  \param old : Instance to be copied
  */

  FunctionT(const FunctionT<traits>& old) : outputlevel_(old.outputlevel_) {}

  /*!
  \brief Destructor
  */
  virtual ~FunctionT() {}

  //@}
  // @{ \name Public members

  //! Type of traits class being used
  typedef traits traits_type;

  /*!
  \brief Return the level of output written to stdout ( 0 - 10 )

  */
  int
  OutLevel()
  {
    return outputlevel_;
  }

  /*!
  \brief Return the function type

  */
  //  MoertelT::FunctionT<traits>::traits_type Type() const { return type_;}

  /*!
  \brief Evaluate the function and derivatives at a given local coordinate

  \param xi (in) : Local coordinate where to evaluate the function
  \param val (out) : Function values at xi
  \param valdim (in) : Dimension of val
  \param deriv (out) : Derivative of functions at xi
  */
  void
  EvaluateFunction(
      const double* xi,
      double*       val,
      const int     valdim,
      double*       deriv)
  {
    TEUCHOS_ASSERT_EQUALITY(valdim, traits_type::valdim);
    TEUCHOS_ASSERT(!xi);

    traits_type::EvaluateFunction(xi, val, valdim, deriv);
  }

  //@}

 protected:
  int outputlevel_;

  // Note that this base class does not hold any data.
  // If Your derived class needs to hold data, make sure it's carefully
  // taken care of in the copy-ctor!
};

/*!
\brief Type of function

\param Constant1D : Constant function on a 1D Segment
\param Linear1D : Linear function on a 1D Segment
\param DualLinear1D : Dual linear function on a 1D Segment
\param LinearTri : Linear function on a 3-noded triangle
\param DualLinearTri : Dual linear function on a 3-noded triangle
\param ConstantTri : Constant function on a 3-noded triangle

*/

/*!
\class Function_Constant1D

\brief <b> A 1D constant shape function for a 2-noded 1D Segment</b>

phi_1 = 1 <br>
phi_2 = 1 <br>
phi_1,xi = 0 <br>
phi_2,xi = 0 <br>

\author Glen Hansen (gahanse@sandia.gov)

*/

struct Constant1D
{
  static const int valdim = 2;

  static void
  EvaluateFunction(
      const double* xi,
      double*       val,
      const int     valdim,
      double*       deriv)
  {
    // for this linear function, we get 2 values and 2 derivatives
    //

    if (val) {
      val[0] = 1.;
      val[1] = 1.;
    }
    if (deriv) {
      deriv[0] = 0.0;
      ;
      deriv[1] = 0.0;
    }
    return;
  }
};

/*!
\class Function_Linear1D

\brief <b> A 1D linear shape function </b>

phi_1 = 0.5*(1-xi) <br>
phi_2 = 0.5*(1+xi) <br>
phi_1,xi = -0.5 <br>
phi_2,xi = 0.5 <br>

\author Glen Hansen (gahanse@sandia.gov)

*/
struct Linear1D
{
  static const int valdim = 2;

  static void
  EvaluateFunction(
      const double* xi,
      double*       val,
      const int     valdim,
      double*       deriv)
  {
    // for this linear function, we get 2 values and 2 derivatives
    //

    if (val) {
      val[0] = 0.5 * (1.0 - xi[0]);
      val[1] = 0.5 * (1.0 + xi[0]);
    }
    if (deriv) {
      deriv[0] = -0.5;
      deriv[1] = 0.5;
    }
    return;
  }
};

/*!
\class Function_DualLinear1D

\brief <b> A 1D dual linear shape function </b>

phi_1 = -1.5*xi+0.5 <br>
phi_2 = 1.5*xi+0.5 <br>
phi_1,xi = -1.5 <br>
phi_2,xi = 1.5 <br>

\author Glen Hansen (gahanse@sandia.gov)

*/
struct DualLinear1D
{
  static const int valdim = 2;

  static void
  EvaluateFunction(
      const double* xi,
      double*       val,
      const int     valdim,
      double*       deriv)
  {
    // for this linear function, we get 2 values and 2 derivatives

    if (val) {
      val[0] = -1.5 * xi[0] + 0.5;
      val[1] = 1.5 * xi[0] + 0.5;
    }
    if (deriv) {
      deriv[0] = -1.5;
      deriv[1] = 1.5;
    }
    return;
  }
};

/*!
\class Function_LinearTri

\brief <b> A 2D linear shape function for a 3-noded triangle</b>

phi_1 = 1 - xi_1 - xi_2 <br>
phi_2 = xi_1 <br>
phi_3 = xi_2 <br>
phi_1,xi_1 = -1 <br>
phi_1,xi_2 = -1 <br>
phi_2,xi_1 = 1 <br>
phi_2,xi_2 = 0 <br>
phi_3,xi_1 = 0 <br>
phi_3,xi_2 = 1 <br>

\author Glen Hansen (gahanse@sandia.gov)

*/
struct LinearTri
{
  static const int valdim = 3;

  static void
  EvaluateFunction(
      const double* xi,
      double*       val,
      const int     valdim,
      double*       deriv)
  {
    // for this function, we get 3 values and six derivatives

    if (val) {
      val[0] = 1. - xi[0] - xi[1];
      val[1] = xi[0];
      val[2] = xi[1];
    }

    if (deriv) {
      deriv[0] = -1.;
      deriv[1] = -1.;
      deriv[2] = 1.;
      deriv[3] = 0.;
      deriv[4] = 0.;
      deriv[5] = 1.;
    }

    return;
  }
};

/*!
\class Function_DualLinearTri

\brief <b> A 2D dual linear shape function for a 3-noded triangle</b>

phi_1 = 3 - 2 * xi_1 - 2 * xi_2 <br>
phi_2 = 4 * xi_1 - 1<br>
phi_3 = 4 * xi_2 - 1 <br>
phi_1,xi_1 = -2 <br>
phi_1,xi_2 = -2 <br>
phi_2,xi_1 = 4 <br>
phi_2,xi_2 = 0 <br>
phi_3,xi_1 = 0 <br>
phi_3,xi_2 = 4 <br>

\author Glen Hansen (gahanse@sandia.gov)

*/
struct DualLinearTri
{
  static const int valdim = 3;

  static void
  EvaluateFunction(
      const double* xi,
      double*       val,
      const int     valdim,
      double*       deriv)
  {
    // for this function, we get 3 values and six derivatives

    if (val) {
      val[0] = 3.0 - 4.0 * xi[0] - 4.0 * xi[1];
      val[1] = 4.0 * xi[0] - 1.0;
      val[2] = 4.0 * xi[1] - 1.0;
    }

    if (deriv) {
      deriv[0] = -4.0;
      deriv[1] = -4.0;
      deriv[2] = 4.0;
      deriv[3] = 0.0;
      deriv[4] = 0.0;
      deriv[5] = 4.0;
    }

    return;
  }
};

/*!
\class Function_ConstantTri

\brief <b> A 2D constant shape function for a 3-noded triangle</b>

phi_1 = 1 <br>
phi_2 = 1<br>
phi_3 = 1 <br>
phi_1,xi_1 = 0 <br>
phi_1,xi_2 = 0 <br>
phi_2,xi_1 = 0 <br>
phi_2,xi_2 = 0 <br>
phi_3,xi_1 = 0 <br>
phi_3,xi_2 = 0 <br>

\author Glen Hansen (gahanse@sandia.gov)

*/
struct ConstantTri
{
  static const int valdim = 3;

  static void
  EvaluateFunction(
      const double* xi,
      double*       val,
      const int     valdim,
      double*       deriv)
  {
    // for this function, we get 3 values and six derivatives

    if (val) {
      val[0] = 1.;
      val[1] = 1.;
      val[2] = 1.;
    }

    if (deriv) {
      deriv[0] = 0.;
      deriv[1] = 0.;
      deriv[2] = 0.;
      deriv[3] = 0.;
      deriv[4] = 0.;
      deriv[5] = 0.;
    }

    return;
  }
};

struct BiLinearQuad
{
  static const int valdim = 4;

  static void
  EvaluateFunction(
      const double* xi,
      double*       val,
      const int     valdim,
      double*       deriv)
  {
    if (val) {
      val[0] = .25 * (1. - xi[0]) * (1. - xi[1]);
      val[1] = .25 * (1. + xi[0]) * (1. - xi[1]);
      val[2] = .25 * (1. + xi[0]) * (1. + xi[1]);
      val[3] = .25 * (1. - xi[0]) * (1. + xi[1]);
    }

    if (deriv) {
      deriv[0] = -.25 * (1. - xi[1]);
      deriv[1] = -.25 * (1. - xi[0]);
      deriv[2] = .25 * (1. - xi[1]);
      deriv[3] = -.25 * (1. + xi[0]);
      deriv[4] = .25 * (1. + xi[1]);
      deriv[5] = .25 * (1. + xi[0]);
      deriv[6] = -.25 * (1. + xi[1]);
      deriv[7] = .25 * (1. - xi[0]);
    }
    return;
  }
};

struct DualBiLinearQuad
{
  static const int valdim = 4;

  static void
  EvaluateFunction(
      const double* xi,
      double*       val,
      const int     valdim,
      double*       deriv)
  {
    const double onemxi  = 1.0 - xi[0];
    const double onepxi  = 1.0 + xi[0];
    const double onemeta = 1.0 - xi[1];
    const double onepeta = 1.0 + xi[1];

    if (val) {
      const double phi0 = .25 * onemxi * onemeta;
      const double phi1 = .25 * onepxi * onemeta;
      const double phi2 = .25 * onepxi * onepeta;
      const double phi3 = .25 * onemxi * onepeta;
      val[0]            = 4. * phi0 - 2. * phi1 - 2. * phi3 + phi2;
      val[1]            = 4. * phi1 - 2. * phi0 - 2. * phi2 + phi3;
      val[2]            = 4. * phi2 - 2. * phi1 - 2. * phi3 + phi0;
      val[3]            = 4. * phi3 - 2. * phi2 - 2. * phi0 + phi1;
    }

    if (deriv) {
      const double phi0xi  = -.25 * onemeta;
      const double phi0eta = -.25 * onemxi;
      const double phi1xi  = .25 * onemeta;
      const double phi1eta = -.25 * onepxi;
      const double phi2xi  = .25 * onepeta;
      const double phi2eta = .25 * onepxi;
      const double phi3xi  = -.25 * onepeta;
      const double phi3eta = .25 * onemxi;
      deriv[0]             = 4. * phi0xi - 2. * phi1xi - 2. * phi3xi + phi2xi;
      deriv[1] = 4. * phi0eta - 2. * phi1eta - 2. * phi3eta + phi2eta;
      deriv[2] = 4. * phi1xi - 2. * phi0xi - 2. * phi2xi + phi3xi;
      deriv[3] = 4. * phi1eta - 2. * phi0eta - 2. * phi2eta + phi3eta;
      deriv[4] = 4. * phi2xi - 2. * phi1xi - 2. * phi3xi + phi0xi;
      deriv[5] = 4. * phi2eta - 2. * phi1eta - 2. * phi3eta + phi0eta;
      deriv[6] = 4. * phi3xi - 2. * phi2xi - 2. * phi0xi + phi1xi;
      deriv[7] = 4. * phi3eta - 2. * phi2eta - 2. * phi0eta + phi1eta;
    }

    return;
  }
};

}  // namespace MoertelT

#endif  // MOERTEL_FUNCTION_H

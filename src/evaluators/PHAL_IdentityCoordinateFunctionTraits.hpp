//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#if !defined(PHAL_IDENTITYCOORDINATEFUNCTIONTRAITS_HPP)
#define PHAL_IDENTITYCOORDINATEFUNCTIONTRAITS_HPP

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_VerboseObject.hpp>

#include <Sacado_ScalarParameterLibrary.hpp>

#include "PHAL_AlbanyTraits.hpp"

namespace PHAL {

///
/// \brief Interface for representing a coordinate function to be applied as a Dirichlet boundary condition
///

template<typename EvalT>
class IdentityCoordFunctionTraits {

  public:

    typedef typename EvalT::ScalarT ScalarT;

    ///
    /// Only constructor
    ///
    IdentityCoordFunctionTraits(Teuchos::ParameterList& params_);

    ///
    /// Destructor
    ///
    ~IdentityCoordFunctionTraits() {};

    void computeBCs(double* coord, std::vector<ScalarT>& BCvals,
                    const RealType time);

    int getNumComponents() {
      return numEqn;
    }


  protected:

    // Number of equations
    int numEqn;

    // Equation offset into residual vector (Always treated as zero for now)
    int eqnOffset;

  private:

    //! Private to prohibit default or copy constructor
    IdentityCoordFunctionTraits();
    IdentityCoordFunctionTraits(const IdentityCoordFunctionTraits&);

    //! Private to prohibit copying
    IdentityCoordFunctionTraits& operator=(const IdentityCoordFunctionTraits&);

};
}

// Define macro for explicit template instantiation
#define COORD_FUNC_INSTANTIATE_TEMPLATE_CLASS_RESIDUAL(name) \
  template class name<PHAL::AlbanyTraits::Residual>;
#define COORD_FUNC_INSTANTIATE_TEMPLATE_CLASS_JACOBIAN(name) \
  template class name<PHAL::AlbanyTraits::Jacobian>;
#define COORD_FUNC_INSTANTIATE_TEMPLATE_CLASS_TANGENT(name) \
  template class name<PHAL::AlbanyTraits::Tangent>;
#define COORD_FUNC_INSTANTIATE_TEMPLATE_CLASS_DISTPARAMDERIV(name)   \
  template class name<PHAL::AlbanyTraits::DistParamDeriv>;

#ifdef ALBANY_SG_MP
#define COORD_FUNC_INSTANTIATE_TEMPLATE_CLASS_SGRESIDUAL(name) \
  template class name<PHAL::AlbanyTraits::SGResidual>;
#define COORD_FUNC_INSTANTIATE_TEMPLATE_CLASS_SGJACOBIAN(name) \
  template class name<PHAL::AlbanyTraits::SGJacobian>;
#define COORD_FUNC_INSTANTIATE_TEMPLATE_CLASS_SGTANGENT(name) \
  template class name<PHAL::AlbanyTraits::SGTangent>;
#define COORD_FUNC_INSTANTIATE_TEMPLATE_CLASS_MPRESIDUAL(name) \
  template class name<PHAL::AlbanyTraits::MPResidual>;
#define COORD_FUNC_INSTANTIATE_TEMPLATE_CLASS_MPJACOBIAN(name) \
  template class name<PHAL::AlbanyTraits::MPJacobian>;
#define COORD_FUNC_INSTANTIATE_TEMPLATE_CLASS_MPTANGENT(name) \
  template class name<PHAL::AlbanyTraits::MPTangent>;
#endif //ALBANY_SG_MP

#ifdef ALBANY_SG_MP
#define COORD_FUNC_INSTANTIATE_TEMPLATE_CLASS(name)             \
  COORD_FUNC_INSTANTIATE_TEMPLATE_CLASS_RESIDUAL(name)          \
  COORD_FUNC_INSTANTIATE_TEMPLATE_CLASS_JACOBIAN(name)          \
  COORD_FUNC_INSTANTIATE_TEMPLATE_CLASS_TANGENT(name)           \
  COORD_FUNC_INSTANTIATE_TEMPLATE_CLASS_DISTPARAMDERIV(name)    \
  COORD_FUNC_INSTANTIATE_TEMPLATE_CLASS_SGRESIDUAL(name)        \
  COORD_FUNC_INSTANTIATE_TEMPLATE_CLASS_SGJACOBIAN(name)        \
  COORD_FUNC_INSTANTIATE_TEMPLATE_CLASS_SGTANGENT(name)         \
  COORD_FUNC_INSTANTIATE_TEMPLATE_CLASS_MPRESIDUAL(name)        \
  COORD_FUNC_INSTANTIATE_TEMPLATE_CLASS_MPJACOBIAN(name)        \
  COORD_FUNC_INSTANTIATE_TEMPLATE_CLASS_MPTANGENT(name)
#else
#define COORD_FUNC_INSTANTIATE_TEMPLATE_CLASS(name)             \
  COORD_FUNC_INSTANTIATE_TEMPLATE_CLASS_RESIDUAL(name)          \
  COORD_FUNC_INSTANTIATE_TEMPLATE_CLASS_JACOBIAN(name)          \
  COORD_FUNC_INSTANTIATE_TEMPLATE_CLASS_TANGENT(name)           \
  COORD_FUNC_INSTANTIATE_TEMPLATE_CLASS_DISTPARAMDERIV(name)
#endif //ALBANY_SG_MP


#endif // PHAL_IDENTITYCOORDINATEFUNCTIONTRAITS_HPP

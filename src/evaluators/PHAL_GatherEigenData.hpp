//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/13/14: does not get compiled if ALBANY_EPETRA_EXE is off.  Has epetra.

#ifndef PHAL_GATHER_EIGENVECTORS2D_HPP
#define PHAL_GATHER_EIGENVECTORS2D_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Epetra_Vector.h"

namespace PHAL {
/** \brief Gathers solution values from the Newton solution vector into 
    the nodal fields of the field manager

    Currently makes an assumption that the stride is constant for dofs
    and that the nmber of dofs is equal to the size of the solution
    names vector.

*/
// **************************************************************
// Base Class with Generic Implementations: Specializations for
// Automatic Differentiation Below 
// **************************************************************

template<typename EvalT, typename Traits>
class GatherEigenDataBase
  : public PHX::EvaluatorWithBaseImpl<Traits>,
    public PHX::EvaluatorDerived<EvalT, Traits>  {
  
public:
  
  GatherEigenDataBase(const Teuchos::ParameterList& p,
                           const Teuchos::RCP<Albany::Layouts>& dl);
  
  void postRegistrationSetup(typename Traits::SetupData d,
                             PHX::FieldManager<Traits>& vm);
  
  // This function requires template specialization, in derived class below
  void evaluateFields(typename Traits::EvalData d);

protected:

  typedef typename EvalT::ScalarT ScalarT;
  std::vector< PHX::MDField<ScalarT,Cell,Node,Dim> > eigenvector_Re;
  std::vector< PHX::MDField<ScalarT,Cell,Node,Dim> > eigenvector_Im;
  std::vector< PHX::MDField<ScalarT> > eigenvalue_Re;
  std::vector< PHX::MDField<ScalarT> > eigenvalue_Im;
  std::size_t numNodes;
  std::size_t nEigenvectors;

};
  
template<typename EvalT, typename Traits> class GatherEigenData;

template<typename EvalT, typename Traits>
class GatherEigenData
   : public GatherEigenDataBase<EvalT, Traits>  {

  public:
    GatherEigenData(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl);

  //void evaluateFields(typename Traits::EvalData d);
  private:
    typedef typename EvalT::ScalarT ScalarT;
};

// **************************************************************
// **************************************************************
// * Specializations
// **************************************************************
// **************************************************************


// **************************************************************
// Residual
// **************************************************************
template<typename Traits>
class GatherEigenData<PHAL::AlbanyTraits::Residual,Traits>
   : public GatherEigenDataBase<PHAL::AlbanyTraits::Residual, Traits>  {

  public:
    GatherEigenData(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl);

    //void evaluateFields(typename Traits::EvalData d);
  private:
    typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
};

// **************************************************************
// Jacobian
// **************************************************************
template<typename Traits>
class GatherEigenData<PHAL::AlbanyTraits::Jacobian,Traits>
   : public GatherEigenDataBase<PHAL::AlbanyTraits::Jacobian, Traits>  {

  using GatherEigenDataBase<PHAL::AlbanyTraits::Jacobian,Traits>::nEigenvectors;

  public:
    GatherEigenData(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl);

    void evaluateFields(typename Traits::EvalData d);
  private:
    typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
};


// **************************************************************
}

#endif

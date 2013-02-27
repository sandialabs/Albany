//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(NeohookeanModel_hpp)
#define NeohookeanModel_hpp

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"
#include "LCM/models/ConstitutiveModel.hpp"

namespace LCM {

  //! \brief Constitutive Model Base Class
  template<typename EvalT, typename Traits>
  class NeohookeanModel : public LCM::ConstitutiveModel<EvalT, Traits>
  {
  public:

    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;

    ///
    /// Constructor
    ///
    NeohookeanModel(const Teuchos::ParameterList* p,
                    const Teuchos::RCP<Albany::Layouts>& dl);

    ///
    /// Method to compute the energy
    ///
    virtual 
    void 
    computeEnergy(typename Traits::EvalData workset,
                  std::vector<Teuchos::RCP<PHX::MDField<ScalarT> > > depFields,
                  std::vector<Teuchos::RCP<PHX::MDField<ScalarT> > > evalFields);

    ///
    /// Method to compute the state (e.g. stress)
    ///
    virtual 
    void 
    computeState(typename Traits::EvalData workset,
                 std::vector<Teuchos::RCP<PHX::MDField<ScalarT> > > depFields,
                 std::vector<Teuchos::RCP<PHX::MDField<ScalarT> > > evalFields);

    ///
    /// Method to compute the tangent
    ///
    virtual 
    void 
    computeTangent(typename Traits::EvalData workset,
                   std::vector<Teuchos::RCP<PHX::MDField<ScalarT> > > depFields,
                   std::vector<Teuchos::RCP<PHX::MDField<ScalarT> > > evalFields);

  private:

    ///
    /// Private to prohibit copying
    ///
    NeohookeanModel(const NeohookeanModel&);

    ///
    /// Private to prohibit copying
    ///
    NeohookeanModel& operator=(const NeohookeanModel&);

  };
}

#endif

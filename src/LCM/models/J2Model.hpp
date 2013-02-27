//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(J2Model_hpp)
#define J2Model_hpp

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"
#include "LCM/models/ConstitutiveModel.hpp"

namespace LCM {

  //! \brief Constitutive Model Base Class
  template<typename EvalT, typename Traits>
  class J2Model : public LCM::ConstitutiveModel<EvalT, Traits>
  {
  public:

    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;

    using ConstitutiveModel<EvalT,Traits>::num_dims_;
    using ConstitutiveModel<EvalT,Traits>::num_pts_;

    ///
    /// Constructor
    ///
    J2Model(const Teuchos::ParameterList* p,
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
    J2Model(const J2Model&);

    ///
    /// Private to prohibit copying
    ///
    J2Model& operator=(const J2Model&);

  };
}

#endif

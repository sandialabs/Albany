//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(GursonModel_hpp)
#define GursonModel_hpp

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"
#include "LCM/models/ConstitutiveModel.hpp"

namespace LCM {

  //! \brief Constitutive Model Base Class
  template<typename EvalT, typename Traits>
  class GursonModel : public LCM::ConstitutiveModel<EvalT, Traits>
  {
  public:

    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;

    using ConstitutiveModel<EvalT,Traits>::num_dims_;
    using ConstitutiveModel<EvalT,Traits>::num_pts_;
    using ConstitutiveModel<EvalT,Traits>::field_name_map_;
    
    ///
    /// Constructor
    ///
    GursonModel(Teuchos::ParameterList* p,
                const Teuchos::RCP<Albany::Layouts>& dl);

    ///
    /// Method to compute the energy
    ///
    virtual 
    void 
    computeEnergy(typename Traits::EvalData workset,
                  std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > dep_fields,
                  std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > eval_fields);

    ///
    /// Method to compute the state (e.g. stress)
    ///
    virtual 
    void 
    computeState(typename Traits::EvalData workset,
                 std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > dep_fields,
                 std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > eval_fields);

    ///
    /// Method to compute the tangent
    ///
    virtual 
    void 
    computeTangent(typename Traits::EvalData workset,
                   std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > dep_fields,
                   std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > eval_fields);

  private:

    ///
    /// Private to prohibit copying
    ///
    GursonModel(const GursonModel&);

    ///
    /// Private to prohibit copying
    ///
    GursonModel& operator=(const GursonModel&);

    ///
    /// Saturation hardening constants
    /// 
    RealType sat_mod_, sat_exp_;

    ///
    /// Initial Void Volume
    /// 
    RealType f0_;

    ///
    /// Shear Damage Parameter
    /// 
    RealType kw_;

    ///
    /// Void Nucleation Parameters
    /// 
    RealType eN_, sN_, fN_;

    ///
    /// Critical Void Parameters
    /// 
    RealType fc_, ff_;

    ///
    /// Yield Parameters
    /// 
    RealType q1_, q2_, q3_;


  };
}

#endif

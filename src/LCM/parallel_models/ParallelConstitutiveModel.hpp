//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_ParallelConstitutiveModel_hpp)
#define LCM_ParallelConstitutiveModel_hpp

#include "ConstitutiveModel.hpp"
#include <functional>

namespace LCM
{
 
template<typename EvalT, typename Traits, typename Kernel>
class ParallelConstitutiveModel: public LCM::ConstitutiveModel<EvalT, Traits>
{
public:

  using ScalarT = typename EvalT::ScalarT;
  using MeshScalarT = typename EvalT::MeshScalarT;
  using FieldMap = std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > >;
  using EvalKernel = Kernel;
  
  using ConstitutiveModel<EvalT, Traits>::num_pts_;

  ParallelConstitutiveModel(Teuchos::ParameterList* p,
      const Teuchos::RCP<Albany::Layouts>& dl);
  
  virtual
  ~ParallelConstitutiveModel() = default;
  
  virtual
  void
  computeState(
      typename Traits::EvalData workset,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > dep_fields,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > eval_fields) override;
  
  virtual
  void
  computeStateParallel(typename Traits::EvalData workset,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > dep_fields,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > eval_fields) override {
         TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Not implemented.");
  }
  
protected:
  
  virtual
  EvalKernel
  createEvalKernel( FieldMap &dep_fields,
                    FieldMap &eval_fields,
                    int numCells) = 0;
  
  
};

}

#endif

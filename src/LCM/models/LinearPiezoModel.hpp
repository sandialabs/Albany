//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_LinearPiezoModel_hpp)
#define LCM_LinearPiezoModel_hpp

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"
#include "LCM/models/ConstitutiveModel.hpp"
#include <Intrepid_MiniTensor.h>

#include "Sacado.hpp"

namespace LCM
{

//! \brief LinearPiezo model for electromechanics
template<typename EvalT, typename Traits>
class LinearPiezoModel: public LCM::ConstitutiveModel<EvalT, Traits>
{
public:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;
  typedef typename Sacado::mpl::apply<FadType,ScalarT>::type DFadType;

  using ConstitutiveModel<EvalT, Traits>::num_dims_;
  using ConstitutiveModel<EvalT, Traits>::num_pts_;
  using ConstitutiveModel<EvalT, Traits>::field_name_map_;
  using ConstitutiveModel<EvalT, Traits>::weights_;

  ///
  /// Constructor
  ///
  LinearPiezoModel(Teuchos::ParameterList* p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  ///
  /// Virtual Destructor
  ///
  virtual
  ~LinearPiezoModel()
  {};

  ///
  /// Method to compute the state (e.g. energy, stress, tangent)
  ///
  virtual
  void
  computeState(typename Traits::EvalData workset,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>> > dep_fields,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>> > eval_fields);

  //Kokkos
  virtual
  void
  computeStateParallel(typename Traits::EvalData workset,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>> > dep_fields,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>> > eval_fields);

private:
  
  ///
  /// Private methods
  ///
  void initializeConstants();

  ///
  /// Private to prohibit copying
  ///
  LinearPiezoModel(const LinearPiezoModel&);

  ///
  /// Private to prohibit copying
  ///
  LinearPiezoModel& operator=(const LinearPiezoModel&);

  ///
  /// material parameters
  ///
  RealType C11, C33, C12, C23, C44, C66;
  RealType e31, e33, e15, E11, E33;

  Intrepid::Tensor4<ScalarT> C;
  Intrepid::Tensor3<ScalarT> e;
  Intrepid::Tensor<ScalarT> eps;
  Intrepid::Tensor<ScalarT> R;

  bool test;

  ///
  /// INDEPENDENT FIELD NAMES
  ///
  std::string strainName;
  std::string efieldName;
 
  ///
  /// EVALUATED FIELD NAMES
  ///
  std::string stressName;
  std::string edispName;
  

};
}

#endif

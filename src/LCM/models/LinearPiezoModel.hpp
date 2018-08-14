//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_LinearPiezoModel_hpp)
#define LCM_LinearPiezoModel_hpp

#include <MiniTensor.h>
#include "Albany_Layouts.hpp"
#include "LCM/models/ConstitutiveModel.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

#include "Sacado.hpp"

namespace LCM {

//! \brief LinearPiezo model for electromechanics
template <typename EvalT, typename Traits>
class LinearPiezoModel : public LCM::ConstitutiveModel<EvalT, Traits>
{
 public:
  using Base        = LCM::ConstitutiveModel<EvalT, Traits>;
  using DepFieldMap = typename Base::DepFieldMap;
  using FieldMap    = typename Base::FieldMap;

  typedef typename EvalT::ScalarT                             ScalarT;
  typedef typename EvalT::MeshScalarT                         MeshScalarT;
  typedef typename Sacado::mpl::apply<FadType, ScalarT>::type DFadType;

  using ConstitutiveModel<EvalT, Traits>::num_dims_;
  using ConstitutiveModel<EvalT, Traits>::num_pts_;
  using ConstitutiveModel<EvalT, Traits>::field_name_map_;
  using ConstitutiveModel<EvalT, Traits>::weights_;

  ///
  /// Constructor
  ///
  LinearPiezoModel(
      Teuchos::ParameterList*              p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  ///
  /// Virtual Destructor
  ///
  virtual ~LinearPiezoModel(){};

  ///
  /// Method to compute the state (e.g. energy, stress, tangent)
  ///
  virtual void
  computeState(
      typename Traits::EvalData workset,
      DepFieldMap               dep_fields,
      FieldMap                  eval_fields);

  // Kokkos
  virtual void
  computeStateParallel(
      typename Traits::EvalData workset,
      DepFieldMap               dep_fields,
      FieldMap                  eval_fields);

 private:
  ///
  /// Private methods
  ///
  void
  initializeConstants();

  ///
  /// Private to prohibit copying
  ///
  LinearPiezoModel(const LinearPiezoModel&);

  ///
  /// Private to prohibit copying
  ///
  LinearPiezoModel&
  operator=(const LinearPiezoModel&);

  ///
  /// material parameters
  ///
  RealType C11, C33, C12, C23, C44, C66;
  RealType e31, e33, e15, E11, E33;

  minitensor::Tensor4<ScalarT> C;
  minitensor::Tensor3<ScalarT> e;
  minitensor::Tensor<ScalarT>  eps;
  minitensor::Tensor<ScalarT>  R;

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
}  // namespace LCM

#endif

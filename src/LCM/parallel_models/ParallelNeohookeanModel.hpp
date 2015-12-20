//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_ParallelNeohookeanModel_hpp)
#define LCM_ParallelNeohookeanModel_hpp

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"
#include "ParallelConstitutiveModel.hpp"

namespace LCM
{

namespace detail {

template<typename EvalT, typename Traits>
struct NeohookeanKernel
{
  using ScalarT = typename EvalT::ScalarT;
  using MeshScalarT = typename EvalT::MeshScalarT;
  using ScalarField = PHX::MDField<ScalarT>;
  
  // Dependent MDFields
  ScalarField def_grad;
  ScalarField J;
  ScalarField poissons_ratio;
  ScalarField elastic_modulus;
  
  // Evaluated MDFields
  ScalarField stress;
  ScalarField energy;
  ScalarField tangent;
  
  // Misc parameters
  int num_dims, num_pts;
  bool compute_energy;
  bool compute_tangent;
  
  KOKKOS_INLINE_FUNCTION
  void operator() (int cell) const;
};

}

//! \brief Parallel Neohookean Model
template<typename EvalT, typename Traits>
class ParallelNeohookeanModel: public LCM::ParallelConstitutiveModel<EvalT, Traits, detail::NeohookeanKernel<EvalT, Traits> >
{
public:

  using Parent = ParallelConstitutiveModel<EvalT, Traits, detail::NeohookeanKernel<EvalT, Traits> >;
  using ScalarT = typename EvalT::ScalarT;
  using MeshScalarT = typename EvalT::MeshScalarT;

  using FieldMap = typename Parent::FieldMap;
  using EvalKernel = typename Parent::EvalKernel;
  
  using Parent::num_dims_;
  using Parent::num_pts_;
  using Parent::field_name_map_;
  using Parent::compute_energy_;
  using Parent::compute_tangent_;

  // optional temperature support
  using Parent::have_temperature_;
  using Parent::expansion_coeff_;
  using Parent::ref_temperature_;
  using Parent::heat_capacity_;
  using Parent::density_;
  using Parent::temperature_;

  ///
  /// Constructor
  ///
  ParallelNeohookeanModel(Teuchos::ParameterList* p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  ParallelNeohookeanModel(const ParallelNeohookeanModel&) = delete;
  ParallelNeohookeanModel& operator=(const ParallelNeohookeanModel&) = delete;

  ///
  /// Virtual Destructor
  ///
  virtual
  ~ParallelNeohookeanModel() = default;
  
protected:
  
  virtual
  EvalKernel
  createEvalKernel(FieldMap &dep_fields,
                   FieldMap &eval_fields,
                   int numCells) override;
  
};
}

#endif

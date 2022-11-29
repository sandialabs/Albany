//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_SCATTER_SIDE_EQN_RESIDUAL_HPP
#define PHAL_SCATTER_SIDE_EQN_RESIDUAL_HPP

#include "Albany_AbstractDiscretization.hpp"
#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_Dimension.hpp"

#include "Albany_KokkosTypes.hpp"
#include "Albany_Layouts.hpp"
#include "Albany_DiscretizationUtils.hpp"

#include "Teuchos_ParameterList.hpp"

#include <map>
#include <set>

namespace PHAL {
/** \brief Scatters result from the residual fields of equations
    defined on one or more side sets into the global (epetra/tpetra)
    data structures.
    This includes the post-processing of the AD data type for all evaluation
    types besides Residual.
*/

// **************************************************************
// Base Class for code that is independent of evaluation type
// **************************************************************

template<typename EvalT, typename Traits>
class ScatterSideEqnResidualBase : public PHX::EvaluatorWithBaseImpl<Traits>,
                                   public PHX::EvaluatorDerived<EvalT, Traits>
{
public:
  ScatterSideEqnResidualBase (const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& vm);

  void evaluateFields (typename Traits::EvalData d);
protected:
  using ScalarT = typename EvalT::ScalarT;
  using ref_t = typename PHAL::Ref<const ScalarT>::type;

  ref_t get_ref (const int side_idx, const int node, const int eq) const {
    switch (tensorRank) {
      case 0:
      {  ref_t ref=val[eq](side_idx,node); return ref; }
      case 1:
      {  ref_t ref=valVec(side_idx,node,eq); return ref; }
      case 2:
      {  ref_t ref=valTensor(side_idx,node,eq/tensorDim,eq%tensorDim); return ref; }
    }
    Kokkos::abort("Unsupported tensor rank");
  }

  void gatherSideSetNodeGIDs (const Albany::AbstractDiscretization& disc);

  void doEvaluateFieldsResidual(typename Traits::EvalData d,
                                const std::vector<Albany::SideStruct>& sideSet);

  virtual void doEvaluateFields(typename Traits::EvalData d,
                                const std::vector<Albany::SideStruct>& sideSet) = 0;
  virtual void doPostEvaluate(typename Traits::EvalData d);

  Teuchos::RCP<PHX::FieldTag> scatter_operation;
  std::vector< PHX::MDField<const ScalarT> > val;
  PHX::MDField<const ScalarT>  valVec;
  PHX::MDField<const ScalarT>  valTensor;

  std::string             sideSetName;        // The side set where the equation(s) are defined
  std::vector<Albany::SideStruct> sideSet;

  Teuchos::Array<int>     numSideNodes;   // Number of nodes on each side of a cell
  Teuchos::Array<Teuchos::Array<int> >  sideNodes;
  int numFields;  // Number of fields gathered in this call
  int offset;     // Offset of first DOF being gathered when numFields<neq
  int tensorDim;     // Only for tensor residuals

  int cellDim;

  int tensorRank;

  // We store the ss_nodes for all worksets, so we can zero residual
  // and diagonalize jacobian OUTSIDE the side set
  std::set<GO> ss_nodes_gids;
  int numCellNodes;
  bool ss_nodes_gids_gathered = false;
};

template<typename EvalT, typename Traits> class ScatterSideEqnResidual;

// **************************************************************
// **************************************************************
// * Specializations
// **************************************************************
// **************************************************************


// **************************************************************
// Residual
// **************************************************************
template<typename Traits>
class ScatterSideEqnResidual<AlbanyTraits::Residual,Traits>
  : public ScatterSideEqnResidualBase<AlbanyTraits::Residual, Traits>  {
public:
  using base_type = ScatterSideEqnResidualBase<AlbanyTraits::Residual,Traits>;
  using ScalarT = typename base_type::ScalarT;

  ScatterSideEqnResidual (const Teuchos::ParameterList& p,
                          const Teuchos::RCP<Albany::Layouts>& dl);

protected:
  void doEvaluateFields(typename Traits::EvalData d,
                        const std::vector<Albany::SideStruct>& sideSet);
};

// **************************************************************
// Jacobian
// **************************************************************
template<typename Traits>
class ScatterSideEqnResidual<AlbanyTraits::Jacobian,Traits>
  : public ScatterSideEqnResidualBase<AlbanyTraits::Jacobian, Traits>  {
public:
  using base_type = ScatterSideEqnResidualBase<AlbanyTraits::Jacobian,Traits>;
  using ScalarT = typename base_type::ScalarT;

  ScatterSideEqnResidual (const Teuchos::ParameterList& p,
                          const Teuchos::RCP<Albany::Layouts>& dl);

protected:
  void doEvaluateFields(typename Traits::EvalData d,
                        const std::vector<Albany::SideStruct>& sideSet);
  void doPostEvaluate(typename Traits::EvalData d);
};

// **************************************************************
// Tangent
// **************************************************************
template<typename Traits>
class ScatterSideEqnResidual<AlbanyTraits::Tangent,Traits>
  : public ScatterSideEqnResidualBase<AlbanyTraits::Tangent, Traits>  {
public:
  using base_type = ScatterSideEqnResidualBase<AlbanyTraits::Tangent,Traits>;
  using ScalarT = typename base_type::ScalarT;

  ScatterSideEqnResidual (const Teuchos::ParameterList& p,
                          const Teuchos::RCP<Albany::Layouts>& dl);

protected:
  void doEvaluateFields(typename Traits::EvalData d,
                        const std::vector<Albany::SideStruct>& sideSet);
};

// **************************************************************
// Distributed parameter derivative
// **************************************************************
template<typename Traits>
class ScatterSideEqnResidual<AlbanyTraits::DistParamDeriv,Traits>
  : public ScatterSideEqnResidualBase<AlbanyTraits::DistParamDeriv, Traits>  {
public:
  using base_type = ScatterSideEqnResidualBase<AlbanyTraits::DistParamDeriv,Traits>;
  using ScalarT = typename base_type::ScalarT;

  ScatterSideEqnResidual (const Teuchos::ParameterList& p,
                          const Teuchos::RCP<Albany::Layouts>& dl);

protected:
  void doEvaluateFields(typename Traits::EvalData d,
                        const std::vector<Albany::SideStruct>& sideSet);
};

// **************************************************************
// Hessian vector products
// **************************************************************
template<typename Traits>
class ScatterSideEqnResidual<AlbanyTraits::HessianVec,Traits>
  : public ScatterSideEqnResidualBase<AlbanyTraits::HessianVec, Traits>  {
public:
  using base_type = ScatterSideEqnResidualBase<AlbanyTraits::HessianVec,Traits>;
  using ScalarT = typename base_type::ScalarT;

  ScatterSideEqnResidual (const Teuchos::ParameterList& p,
                          const Teuchos::RCP<Albany::Layouts>& dl);

protected:
  void doEvaluateFields(typename Traits::EvalData d,
                        const std::vector<Albany::SideStruct>& sideSet);
};

} // namespace PHAL

#endif // PHAL_SCATTER_SIDE_EQN_RESIDUAL_HPP

//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_SCATTER_RESIDUAL2D_HPP
#define LANDICE_SCATTER_RESIDUAL2D_HPP

#include "PHAL_ScatterResidual.hpp"

// Fwd decl
namespace Albany {
template<typename T>
struct LayeredMeshNumbering; 
}

namespace PHAL {

// Scatter a cell-based residual field of a 2D equation defined
// at an arbitrary level of an extruded mesh. Only Jacobian does
// something different from base class (full specialization of the
// evaluateField method is in the _Def.hpp file)
template<typename EvalT, typename Traits>
class ScatterResidual2D : public ScatterResidual<EvalT, Traits>
{
public:
  ScatterResidual2D (const Teuchos::ParameterList& p,
                     const Teuchos::RCP<Albany::Layouts>& dl);

  void evaluateFields (typename Traits::EvalData d);

protected:
  void evaluateFieldsImpl (typename Traits::EvalData d) {
    Base::evaluateFields(d);
  }

  void compute_offsets (const Albany::DOFManager& dof_mgr, const int bot_pos);

  using Base = ScatterResidual<EvalT,Traits>;
  using ScalarT = typename Base::ScalarT;

  // This stuff is really needed only by Jacobian, since the other
  // eval types rely on base class implementation
  using Base::numFields;
  using Base::numNodes;

  int m_bot_side_pos;
  int m_top_side_pos;
  int numSideNodes;
  int sideDim;
  int fieldLevel; // The node-layer where the 2d field is defined

  std::string meshPart;

  Albany::DualView<int**> m_bot_dofs_offsets;
  Albany::DualView<int**> m_top_dofs_offsets;
};

// Scatter a cell-based residual field of a 3D equation when
// one of the solution dofs is defined on an arbitrary level
// of an extruded mesh. Only Jacobian does something different
// from base class (full specialization of the evaluateField
// method is in the _Def.hpp file)
template<typename EvalT, typename Traits>
class ScatterResidualWithExtrudedField : public ScatterResidual<EvalT, Traits>
{
public:
  ScatterResidualWithExtrudedField(const Teuchos::ParameterList& p,
                                   const Teuchos::RCP<Albany::Layouts>& dl);

  void evaluateFields (typename Traits::EvalData d);

protected:
  void evaluateFieldsImpl (typename Traits::EvalData d) {
    Base::evaluateFields(d);
  }

  // This stuff is really needed only by Jacobian, since the other
  // eval types rely on base class implementation
  using Base = ScatterResidual<EvalT,Traits>;
  using ScalarT = typename Base::ScalarT;

  using Base::numFields;

  Teuchos::RCP<Albany::LayeredMeshNumbering<int>> m_cell_layers_data;

  Albany::DualView<int*> m_bot_dofs_offsets;
  Albany::DualView<int*> m_top_dofs_offsets;
  Albany::DualView<int*> m_bot_nodes_offsets;
  Albany::DualView<int*> m_top_nodes_offsets;

  int m_bot_side_pos;
  int m_top_side_pos;
  int sideDim;
  int numSideNodes;
  int offset2DField;
  int fieldLevel; // Node level where field is defined
  int m_field_layer; // Cell layer containing the field level
};

} // namespace PHAL

#endif // LANDICE_SCATTER_RESIDUAL2D_HPP

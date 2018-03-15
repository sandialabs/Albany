//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Teuchos_TestForException.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace LCM {

//
//
//
template <typename EvalT, typename Traits>
NodePointVecInterpolation<EvalT, Traits>::NodePointVecInterpolation(
    Teuchos::ParameterList const&        p,
    Teuchos::RCP<Albany::Layouts> const& dl)
    : nodal_value_(p.get<std::string>("Variable Name"), dl->node_vector),
      basis_fn_(p.get<std::string>("BF Name"), dl->cell_scalar),
      point_value_(p.get<std::string>("Variable Name"), dl->cell_vector)
{
  this->addDependentField(nodal_value_);
  this->addDependentField(basis_fn_);
  this->addEvaluatedField(point_value_);

  this->setName("NodePointVecInterpolation" + PHX::typeAsString<EvalT>());

  std::vector<PHX::DataLayout::size_type> dimensions;

  basis_fn_.fieldTag().dataLayout().dimensions(dimensions);
  number_nodes_ = dimensions[1];

  nodal_value_.fieldTag().dataLayout().dimensions(dimensions);
  dimension_ = dimensions[2];
}

//
//
//
template <typename EvalT, typename Traits>
void
NodePointVecInterpolation<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(nodal_value_, fm);
  this->utils.setFieldData(basis_fn_, fm);
  this->utils.setFieldData(point_value_, fm);
}

//
//
//
template <typename EvalT, typename Traits>
void
NodePointVecInterpolation<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int i = 0; i < dimension_; i++) {
      // Zero out for node==0; then += for node = 1 to number_nodes_
      // ScalarT &
      // vpt = point_value_(cell, i);
      for (int j = 0; j < point_value_.dimension(2); j++) {
        point_value_(cell, i, j) =
            nodal_value_(cell, 0, i) * basis_fn_(cell, 0, i);
        for (int node = 1; node < number_nodes_; ++node) {
          point_value_(cell, i, j) +=
              nodal_value_(cell, node, i) * basis_fn_(cell, node, i);
        }
      }
    }
  }
}

//
//
//
template <typename Traits>
NodePointVecInterpolation<PHAL::AlbanyTraits::Jacobian, Traits>::
    NodePointVecInterpolation(
        Teuchos::ParameterList const&        p,
        Teuchos::RCP<Albany::Layouts> const& dl)
    : nodal_value_(p.get<std::string>("Variable Name"), dl->node_vector),
      basis_fn_(p.get<std::string>("BF Name"), dl->cell_scalar),
      point_value_(p.get<std::string>("Variable Name"), dl->cell_vector)
{
  this->addDependentField(nodal_value_);
  this->addDependentField(basis_fn_);
  this->addEvaluatedField(point_value_);

  this->setName(
      "NodePointVecInterpolation" +
      PHX::typeAsString<PHAL::AlbanyTraits::Jacobian>());

  std::vector<PHX::DataLayout::size_type> dimensions;

  basis_fn_.fieldTag().dataLayout().dimensions(dimensions);
  number_nodes_ = dimensions[1];

  nodal_value_.fieldTag().dataLayout().dimensions(dimensions);
  dimension_ = dimensions[2];

  offset_ = p.get<int>("Offset of First DOF");
}

//
//
//
template <typename Traits>
void
NodePointVecInterpolation<PHAL::AlbanyTraits::Jacobian, Traits>::
    postRegistrationSetup(
        typename Traits::SetupData d,
        PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(nodal_value_, fm);
  this->utils.setFieldData(basis_fn_, fm);
  this->utils.setFieldData(point_value_, fm);
}

//
//
//
template <typename Traits>
void
NodePointVecInterpolation<PHAL::AlbanyTraits::Jacobian, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  int const num_dof = nodal_value_(0, 0, 0).size();

  int const neq = num_dof / number_nodes_;

  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int i = 0; i < dimension_; i++) {
      // Zero out for node==0; then += for node = 1 to number_nodes_
      // ScalarT &
      // vpt = point_value_(cell, i);

      point_value_(cell, i) =
          FadType(num_dof, nodal_value_(cell, 0, i).val() * basis_fn_(cell, 0));

      (point_value_(cell, i)).fastAccessDx(offset_ + i) =
          nodal_value_(cell, 0, i).fastAccessDx(offset_ + i) *
          basis_fn_(cell, 0);

      for (int node = 1; node < number_nodes_; ++node) {
        (point_value_(cell, i)).val() +=
            nodal_value_(cell, node, i).val() * basis_fn_(cell, node);

        (point_value_(cell, i)).fastAccessDx(neq * node + offset_ + i) +=
            nodal_value_(cell, node, i).fastAccessDx(neq * node + offset_ + i) *
            basis_fn_(cell, node);
      }
    }
  }
}

}  // namespace LCM

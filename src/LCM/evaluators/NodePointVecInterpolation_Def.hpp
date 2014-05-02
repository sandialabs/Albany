//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

namespace LCM {

//
//
//
template<typename EvalT, typename Traits>
NodePointVecInterpolation<EvalT, Traits>::
NodePointVecInterpolation(
    Teuchos::ParameterList const & p,
    Teuchos::RCP<Albany::Layouts> const & dl) :
  nodal_value_(p.get<std::string>("Variable Name"), dl->node_vector),
  basis_fn_(p.get<std::string>("BF Name"),  dl->cell_scalar),
  point_value_(p.get<std::string>("Variable Name"), dl->cell_vector)
{
  this->addDependentField(nodal_value_);
  this->addDependentField(basis_fn_);
  this->addEvaluatedField(point_value_);

  this->setName("NodePointVecInterpolation" + PHX::TypeString<EvalT>::value);

  std::vector<PHX::DataLayout::size_type>
  dimensions;

  basis_fn_.fieldTag().dataLayout().dimensions(dimensions);
  number_nodes_ = dimensions[1];

  nodal_value_.fieldTag().dataLayout().dimensions(dimensions);
  dimension_ = dimensions[2];
}

//
//
//
template<typename EvalT, typename Traits>
void NodePointVecInterpolation<EvalT, Traits>::
postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits> & fm)
{
  this->utils.setFieldData(nodal_value_,fm);
  this->utils.setFieldData(basis_fn_,fm);
  this->utils.setFieldData(point_value_,fm);
}

//
//
//
template<typename EvalT, typename Traits>
void NodePointVecInterpolation<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
    for (std::size_t i = 0; i < dimension_; i++) {
      // Zero out for node==0; then += for node = 1 to number_nodes_
      ScalarT &
      vpt = point_value_(cell, i);
      vpt = nodal_value_(cell, 0, i) * basis_fn_(cell, 0);
      for (std::size_t node = 1; node < number_nodes_; ++node) {
        vpt += nodal_value_(cell, node, i) * basis_fn_(cell, node);
      }
    }
  }
}

//
//
//
template<typename Traits>
NodePointVecInterpolation<PHAL::AlbanyTraits::Jacobian, Traits>::
NodePointVecInterpolation(
    Teuchos::ParameterList const & p,
    Teuchos::RCP<Albany::Layouts> const & dl) :
  nodal_value_(p.get<std::string>("Variable Name"), dl->node_vector),
  basis_fn_(p.get<std::string>("BF Name"), dl->cell_scalar),
  point_value_(p.get<std::string>("Variable Name"), dl->cell_vector)
{
  this->addDependentField(nodal_value_);
  this->addDependentField(basis_fn_);
  this->addEvaluatedField(point_value_);

  this->setName(
    "NodePointVecInterpolation" +
    PHX::TypeString<PHAL::AlbanyTraits::Jacobian>::value
  );

  std::vector<PHX::DataLayout::size_type>
  dimensions;

  basis_fn_.fieldTag().dataLayout().dimensions(dimensions);
  number_nodes_ = dimensions[1];

  nodal_value_.fieldTag().dataLayout().dimensions(dimensions);
  dimension_ = dimensions[2];

  offset_ = p.get<int>("Offset of First DOF");
}

//
//
//
template<typename Traits>
void NodePointVecInterpolation<PHAL::AlbanyTraits::Jacobian, Traits>::
postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits> & fm)
{
  this->utils.setFieldData(nodal_value_, fm);
  this->utils.setFieldData(basis_fn_, fm);
  this->utils.setFieldData(point_value_, fm);
}

//
//
//
template<typename Traits>
void NodePointVecInterpolation<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  int const
  num_dof = nodal_value_(0, 0, 0).size();

  int const
  neq = num_dof / number_nodes_;

  for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
    for (std::size_t i = 0; i < dimension_; i++) {
      // Zero out for node==0; then += for node = 1 to number_nodes_
      ScalarT &
      vpt = point_value_(cell, i);

      vpt = FadType(
          num_dof,
          nodal_value_(cell, 0, i).val() * basis_fn_(cell, 0)
      );

      vpt.fastAccessDx(offset_ + i) =
          nodal_value_(cell, 0, i).fastAccessDx(offset_ + i) *
          basis_fn_(cell, 0);

      for (std::size_t node = 1; node < number_nodes_; ++node) {

        vpt.val() +=
            nodal_value_(cell, node, i).val() * basis_fn_(cell, node);

        vpt.fastAccessDx(neq * node + offset_ + i) +=
            nodal_value_(cell, node, i).fastAccessDx(neq * node + offset_ + i)
            *
            basis_fn_(cell, node);
      }
    }
  }
}

} //namespace LCM

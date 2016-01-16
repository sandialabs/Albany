//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits, typename Type>
DOFVecInterpolationSideBase<EvalT, Traits, Type>::
DOFVecInterpolationSideBase(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
  sideSetName (p.get<std::string> ("Side Set Name")),
  val_node    (p.get<std::string> ("Variable Name"), dl->side_node_vector),
  BF          (p.get<std::string> ("BF Name"), dl->side_node_qp_scalar),
  val_qp      (p.get<std::string> ("Variable Name"), dl->side_qp_vector )
{
  this->addDependentField(val_node);
  this->addDependentField(BF);
  this->addEvaluatedField(val_qp);

  this->setName("DOFVecInterpolationSide" );

  std::vector<PHX::DataLayout::size_type> dims;
  BF.fieldTag().dataLayout().dimensions(dims);
  numSideNodes = dims[2];
  numSideQPs   = dims[3];

  dl->side_qp_vector->dimensions(dims);
  vecDim = dims[3];
}

template<typename EvalT, typename Traits>
DOFVecInterpolationSide<EvalT, Traits>::
DOFVecInterpolationSide(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) : 
  DOFVecInterpolationSideBase<EvalT, Traits, typename EvalT::ScalarT>(p, dl) {};

template<typename EvalT, typename Traits>
DOFVecInterpolationSideParam<EvalT, Traits>::
DOFVecInterpolationSideParam(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) : 
  DOFVecInterpolationSideBase<EvalT, Traits, typename EvalT::ParamScalarT>(p, dl) {};



//**********************************************************************
template<typename EvalT, typename Traits, typename Type>
void DOFVecInterpolationSideBase<EvalT, Traits, Type>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_node,fm);
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(val_qp,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits, typename Type>
void DOFVecInterpolationSideBase<EvalT, Traits, Type>::
evaluateFields(typename Traits::EvalData workset)
{
  if (workset.sideSets->find(sideSetName)==workset.sideSets->end())
    return;

  const std::vector<Albany::SideStruct>& sideSet = workset.sideSets->at(sideSetName);
  for (auto const& it_side : sideSet)
  {
    // Get the local data of side and cell
    const int cell = it_side.elem_LID;
    const int side = it_side.side_local_id;

    for (int dim=0; dim<vecDim; ++dim)
    {
      for (int qp=0; qp<numSideQPs; ++qp)
      {
        val_qp(cell,side,qp,dim) = val_node(cell,side,0,dim) * BF(cell,side,0,qp);
        for (int node=1; node<numSideNodes; ++node)
        {
          val_qp(cell,side,qp,dim) += val_node(cell,side,node,dim) * BF(cell,side,node,qp);
        }
      }
    }
  }
}

} // Namespace PHAL

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
template<typename EvalT, typename Traits>
DOFGradInterpolationSide<EvalT, Traits>::
DOFGradInterpolationSide(const Teuchos::ParameterList& p,
                         const Teuchos::RCP<Albany::Layouts>& dl) :
  sideSetName (p.get<std::string> ("Side Set Name")),
  val_node    (p.get<std::string> ("Variable Name"), dl->side_node_scalar),
  gradBF      (p.get<std::string> ("Gradient BF Name"), dl->side_node_qp_gradient),
  val_qp      (p.get<std::string> ("Gradient Variable Name"), dl->side_qp_gradient )
{
  this->addDependentField(val_node);
  this->addDependentField(gradBF);
  this->addEvaluatedField(val_qp);

  this->setName("DOFGradInterpolationSide" );

  std::vector<PHX::DataLayout::size_type> dims;
  gradBF.fieldTag().dataLayout().dimensions(dims);
  numSideNodes = dims[2];
  numSideQPs   = dims[3];
  numDims      = dims[4];
}

//**********************************************************************
template<typename EvalT, typename Traits>
void DOFGradInterpolationSide<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_node,fm);
  this->utils.setFieldData(gradBF,fm);
  this->utils.setFieldData(val_qp,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void DOFGradInterpolationSide<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  const Albany::SideSetList& ssList = *(workset.sideSets);
  Albany::SideSetList::const_iterator it_ss = ssList.find(sideSetName);

  if (it_ss==ssList.end())
    return;

  const std::vector<Albany::SideStruct>& sideSet = it_ss->second;
  std::vector<Albany::SideStruct>::const_iterator iter_s;
  for (iter_s=sideSet.begin(); iter_s!=sideSet.end(); ++iter_s)
  {
    // Get the local data of side and cell
    const int cell = iter_s->elem_LID;
    const int side = iter_s->side_local_id;

    for (int qp=0; qp<numSideQPs; ++qp)
    {
      for (int dim=0; dim<numDims; ++dim)
      {
        val_qp(cell,side,qp,dim) = 0.;
        for (int node=0; node<numSideNodes; ++node)
        {
          val_qp(cell,side,qp,dim) += val_node(cell,side,node) * gradBF(cell,side,node,qp,dim);
        }
      }
    }
  }
}

} // Namespace PHAL

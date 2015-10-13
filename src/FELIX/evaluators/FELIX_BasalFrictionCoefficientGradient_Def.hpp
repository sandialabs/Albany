//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Intrepid_FunctionSpaceTools.hpp"
#include "Albany_Layouts.hpp"

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN

namespace FELIX
{

template<typename EvalT, typename Traits>
BasalFrictionCoefficientGradient<EvalT, Traits>::
BasalFrictionCoefficientGradient (const Teuchos::ParameterList& p,
                                  const Teuchos::RCP<Albany::Layouts>& dl) :
  grad_beta (p.get<std::string> ("Basal Friction Coefficient Gradient Name"), dl->side_qp_gradient)
{
#ifdef OUTPUT_TO_SCREEN
  Teuchos::RCP<Teuchos::FancyOStream> output(Teuchos::VerboseObjectBase::getDefaultOStream());
#endif

  Teuchos::ParameterList& beta_list = *p.get<Teuchos::ParameterList*>("Parameter List");

  std::string betaType = (beta_list.isParameter("Type") ? beta_list.get<std::string>("Type") : "From File");

  basalSideName = p.get<std::string>("Side Set Name");
  if (betaType == "Constant")
  {
#ifdef OUTPUT_TO_SCREEN
    *output << "Constant and uniform beta, loaded from xml input file.\n";
#endif
    beta_type = CONSTANT;
  }
  else if (betaType == "From File")
  {
#ifdef OUTPUT_TO_SCREEN
    *output << "Constant beta, loaded from file.\n";
#endif
    beta_type = FROM_FILE;

    beta_given = PHX::MDField<ScalarT,Cell,Node>(p.get<std::string> ("Given Beta Variable Name"), dl->node_scalar);
    GradBF     = PHX::MDField<RealType,Cell,Side,Node,QuadPoint,Dim>(p.get<std::string> ("Gradient BF Side Variable Name"), dl->side_node_qp_gradient);

    this->addDependentField (beta_given);
    this->addDependentField (GradBF);

    std::vector<PHX::DataLayout::size_type> dims;
    dl->side_node_qp_gradient->dimensions(dims);
    numSideNodes = dims[2];

    // Index of the nodes on the sides in the numeration of the cell
    Teuchos::RCP<shards::CellTopology> cellType;
    cellType = p.get<Teuchos::RCP <shards::CellTopology> > ("Cell Type");
    int numSides = dims[1];
    sideDim  = dims[4];
    sideNodes.resize(numSides);
    for (int side=0; side<numSides; ++side)
    {
      sideNodes[side].resize(numSideNodes);
      for (int node=0; node<numSideNodes; ++node)
        sideNodes[side][node] = cellType->getNodeMap(sideDim,side,node);
    }
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
        std::endl << "Error in FELIX::BasalFrictionCoefficientGradient:  \"" << betaType << "\" does not allow for gradient calculation\n");
  }

  std::vector<PHX::DataLayout::size_type> dims;
  dl->side_qp_scalar->dimensions(dims);
  numSideQPs = dims[2];

  this->addEvaluatedField(grad_beta);

  this->setName("BasalFrictionCoefficientGradient"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void BasalFrictionCoefficientGradient<EvalT, Traits>::
postRegistrationSetup (typename Traits::SetupData d,
                       PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(grad_beta,fm);

  if (beta_type==FROM_FILE)
  {
    this->utils.setFieldData(GradBF,fm);
    this->utils.setFieldData(beta_given,fm);
  }
}

//**********************************************************************
template<typename EvalT, typename Traits>
void BasalFrictionCoefficientGradient<EvalT, Traits>::evaluateFields (typename Traits::EvalData workset)
{
  const Albany::SideSetList& ssList = *(workset.sideSets);
  Albany::SideSetList::const_iterator it_ss = ssList.find(basalSideName);

  if (it_ss==ssList.end())
    return;

  const std::vector<Albany::SideStruct>& sideSet = it_ss->second;
  std::vector<Albany::SideStruct>::const_iterator iter_s;
  for (iter_s=sideSet.begin(); iter_s!=sideSet.end(); ++iter_s)
  {
    // Get the local data of side and cell
    const int cell = iter_s->elem_LID;
    const int side = iter_s->side_local_id;

    switch (beta_type)
    {
      case CONSTANT:
        for (int qp=0; qp<numSideQPs; ++qp)
        {
          for (int dim=0; dim<sideDim; ++dim)
            grad_beta(cell,side,qp,dim) = 0.;
        }
        break;

      case FROM_FILE:
        for (int qp=0; qp<numSideQPs; ++qp)
        {
          for (int dim=0; dim<sideDim; ++dim)
          {
            grad_beta(cell,side,qp,dim) = 0.;
            for (int node=0; node<numSideNodes; ++node)
            {
              grad_beta(cell,side,qp,dim) += GradBF(cell,side,node,qp,dim)*beta_given(cell,sideNodes[side][node]);
            }
          }
        }
        break;
    }
  }
}

} // Namespace FELIX

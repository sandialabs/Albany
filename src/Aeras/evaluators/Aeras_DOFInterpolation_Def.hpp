//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

namespace Aeras {

//**********************************************************************
template<typename EvalT, typename Traits>
DOFInterpolation<EvalT, Traits>::
DOFInterpolation(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Aeras::Layouts>& dl) :
  val_node    (p.get<std::string>   ("Variable Name"), dl->node_scalar_level),
  BF          (p.get<std::string>   ("BF Name"),       dl->node_qp_scalar),
  val_qp      (p.get<std::string>   ("Variable Name"), dl->qp_scalar_level )
{
  this->addDependentField(val_node);
  this->addDependentField(BF);
  this->addEvaluatedField(val_qp);

  this->setName("DOFInterpolation"+PHX::TypeString<EvalT>::value);

  std::vector<PHX::DataLayout::size_type> dims;
  BF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];
  numLevels = p.get< int >("Number of Vertical Levels");
}

//**********************************************************************
template<typename EvalT, typename Traits>
void DOFInterpolation<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_node,fm);
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(val_qp,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void DOFInterpolation<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  //Intrepid version:
  // for (int i=0; i < val_qp.size() ; i++) val_qp[i] = 0.0;
  // Intrepid::FunctionSpaceTools:: evaluate<ScalarT>(val_qp, val_node, BF);

  for (int cell=0; cell < workset.numCells; ++cell) {
    for (int qp=0; qp < numQPs; ++qp) {
      for (int level=0; level < numLevels; ++level) {
        ScalarT& vqp = val_qp(cell,qp,level);
        vqp = val_node(cell, 0, level) * BF(cell, 0, qp);
        for (int node=1; node < numNodes; ++node) {
          vqp += val_node(cell, node, level) * BF(cell, node, qp);
        }
      }
    }
  }
}

//**********************************************************************
template<typename Traits>
DOFInterpolation<PHAL::AlbanyTraits::Jacobian, Traits>::
DOFInterpolation(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Aeras::Layouts>& dl) :
  val_node    (p.get<std::string>   ("Variable Name"), dl->node_scalar_level),
  BF          (p.get<std::string>   ("BF Name"), dl->node_qp_scalar),
  val_qp      (p.get<std::string>   ("Variable Name"), dl->qp_scalar_level )
{
  this->addDependentField(val_node);
  this->addDependentField(BF);
  this->addEvaluatedField(val_qp);

  this->setName("DOFInterpolation"+PHX::TypeString<PHAL::AlbanyTraits::Jacobian>::value);

  std::vector<PHX::DataLayout::size_type> dims;
  BF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];
  numLevels = p.get< int >("Number of Vertical Levels");
}

//**********************************************************************
template<typename Traits>
void DOFInterpolation<PHAL::AlbanyTraits::Jacobian, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_node,fm);
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(val_qp,fm);
}

//**********************************************************************
template<typename Traits>
void DOFInterpolation<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  //Intrepid version:
  // for (int i=0; i < val_qp.size() ; i++) val_qp[i] = 0.0;
  // Intrepid::FunctionSpaceTools:: evaluate<ScalarT>(val_qp, val_node, BF);

  int num_dof = val_node(0,0).size();
  int neq = num_dof / numNodes;

  for (int cell=0; cell < workset.numCells; ++cell) {
    for (int qp=0; qp < numQPs; ++qp) {
      for (int level=0; level < numLevels; ++level) {
        ScalarT& vqp = val_qp(cell,qp,level);

        vqp = FadType(num_dof, val_node(cell, 0, level).val() * BF(cell, 0, qp));
        for (int node=1; node < numNodes; ++node)  {
          vqp.val() += val_node(cell, node, level).val() * BF(cell, node, qp);
        }

        if (vqp.hasFastAccess() && num_dof) {
          vqp.fastAccessDx(level) = val_node(cell, 0, level).fastAccessDx(level) * BF(cell, 0, qp);
          for (int node=1; node < numNodes; ++node) {
            const int firstunk = neq * node + level;
            vqp.fastAccessDx(firstunk) += 
              val_node(cell, node, level).fastAccessDx(firstunk) * BF(cell, node, qp);
          }
        }
      }
    }
  }
}

//**********************************************************************

}


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
  DOFVecGradInterpolation<EvalT, Traits>::
  DOFVecGradInterpolation(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
    val_node    (p.get<std::string>  ("Variable Name"), dl->node_vector),
    GradBF      (p.get<std::string>  ("Gradient BF Name"), dl->node_qp_gradient ),
    grad_val_qp (p.get<std::string>  ("Gradient Variable Name"), dl->qp_vecgradient )
  {
    this->addDependentField(val_node);
    this->addDependentField(GradBF);
    this->addEvaluatedField(grad_val_qp);

    this->setName("DOFVecGradInterpolation"+PHX::TypeString<EvalT>::value);

    std::vector<PHX::DataLayout::size_type> dims;
    GradBF.fieldTag().dataLayout().dimensions(dims);
    numNodes = dims[1];
    numQPs   = dims[2];
    numDims  = dims[3];

    val_node.fieldTag().dataLayout().dimensions(dims);
    vecDim  = dims[2];
  }

  //**********************************************************************
  template<typename EvalT, typename Traits>
  void DOFVecGradInterpolation<EvalT, Traits>::
  postRegistrationSetup(typename Traits::SetupData d,
                        PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(val_node,fm);
    this->utils.setFieldData(GradBF,fm);
    this->utils.setFieldData(grad_val_qp,fm);
  }

  //**********************************************************************
  template<typename EvalT, typename Traits>
  void DOFVecGradInterpolation<EvalT, Traits>::
  evaluateFields(typename Traits::EvalData workset)
  {
    // This is needed, since evaluate currently sums into
    for (int i=0; i < grad_val_qp.size() ; i++) grad_val_qp[i] = 0.0;
  
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t node=0; node < numNodes; ++node) {
        for (std::size_t qp=0; qp < numQPs; ++qp) {
          for (std::size_t i=0; i<vecDim; i++) {
            for (std::size_t dim=0; dim<numDims; dim++) {
              grad_val_qp(cell,qp,i,dim) += val_node(cell, node, i) * GradBF(cell, node, qp, dim);
            } 
          } 
        } 
      } 
    }
    //  Intrepid::FunctionSpaceTools::evaluate<ScalarT>(grad_val_qp, val_node, GradBF);
  }
  
  //**********************************************************************
}

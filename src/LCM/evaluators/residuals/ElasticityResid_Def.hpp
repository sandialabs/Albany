//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace LCM {

//**********************************************************************
template<typename EvalT, typename Traits>
ElasticityResid<EvalT, Traits>::
ElasticityResid(Teuchos::ParameterList& p) :
  Stress       (p.get<std::string>                   ("Stress Name"),
	        p.get<Teuchos::RCP<PHX::DataLayout>>("QP Tensor Data Layout") ),
  wGradBF      (p.get<std::string>                   ("Weighted Gradient BF Name"),
	        p.get<Teuchos::RCP<PHX::DataLayout>>("Node QP Vector Data Layout") ),
  ExResidual   (p.get<std::string>                   ("Residual Name"),
		p.get<Teuchos::RCP<PHX::DataLayout>>("Node Vector Data Layout") )
{
  this->addDependentField(Stress);
  this->addDependentField(wGradBF);

  this->addEvaluatedField(ExResidual);

  if (p.isType<bool>("Disable Transient"))
    enableTransient = !p.get<bool>("Disable Transient");
  else enableTransient = true;

  hasDensity = false;

  if (enableTransient) {
    // Additional fields required for transient capability

    if(p.isParameter("Density Name")){
      hasDensity = true;
      Teuchos::RCP<PHX::DataLayout> cell_scalar_dl =
	p.get< Teuchos::RCP<PHX::DataLayout>>("Cell Scalar Data Layout");
      PHX::MDField<ScalarT,Cell> density_tmp
	(p.get<std::string>("Density Name"), cell_scalar_dl);
      density = density_tmp;
      this->addDependentField(density);
    }

    Teuchos::RCP<PHX::DataLayout> node_qp_scalar_dl =
      p.get< Teuchos::RCP<PHX::DataLayout>>("Node QP Scalar Data Layout");
    PHX::MDField<MeshScalarT,Cell,Node,QuadPoint> wBF_tmp
      (p.get<std::string>("Weighted BF Name"), node_qp_scalar_dl);
    wBF = wBF_tmp;
    this->addDependentField(wBF);

    Teuchos::RCP<PHX::DataLayout> vector_dl =
      p.get< Teuchos::RCP<PHX::DataLayout>>("QP Vector Data Layout");
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim> uDotDot_tmp
      (p.get<std::string>("Time Dependent Variable Name"), vector_dl);
    uDotDot = uDotDot_tmp;
    this->addDependentField(uDotDot);
  }

  this->setName("ElasticityResid"+PHX::typeAsString<EvalT>());

  std::vector<PHX::DataLayout::size_type> dims;
  wGradBF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];
  numDims  = dims[3];
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ElasticityResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(Stress,fm);
  this->utils.setFieldData(wGradBF,fm);

  this->utils.setFieldData(ExResidual,fm);

  if (enableTransient) this->utils.setFieldData(uDotDot,fm);
  if (enableTransient) this->utils.setFieldData(wBF,fm);

  if (hasDensity) this->utils.setFieldData(density,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ElasticityResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  typedef Intrepid2::FunctionSpaceTools FST;

  for (int cell=0; cell < workset.numCells; ++cell) {
    for (int node=0; node < numNodes; ++node) {
      for (int dim=0; dim<numDims; dim++) {
	ExResidual(cell,node,dim)=0.0;
      }
      for (int qp=0; qp < numQPs; ++qp) {
	for (int i=0; i<numDims; i++) {
	  for (int dim=0; dim<numDims; dim++) {
	    ExResidual(cell,node,i) += Stress(cell, qp, i, dim) * wGradBF(cell, node, qp, dim);
	  }
	}
      }
    }
  }

  if (workset.transientTerms && enableTransient) {
    for (int cell=0; cell < workset.numCells; ++cell) {
      for (int node=0; node < numNodes; ++node) {
	for (int qp=0; qp < numQPs; ++qp) {
	  for (int i=0; i<numDims; i++) {

	    if(hasDensity){
	      ExResidual(cell,node,i) += density(cell) * uDotDot(cell, qp, i) * wBF(cell, node, qp);
	    }
	    else{
	      ExResidual(cell,node,i) += uDotDot(cell, qp, i) * wBF(cell, node, qp);
	    }
	  }
	}
      }
    }
  }

//   FST::integrate<ScalarT>(ExResidual, Stress, wGradBF, Intrepid2::COMP_CPP, false); // "false" overwrites

}

//**********************************************************************
}


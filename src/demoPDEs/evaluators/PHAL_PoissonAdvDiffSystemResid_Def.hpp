//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace PHAL {


//**********************************************************************
template<typename EvalT, typename Traits>
PoissonAdvDiffSystemResid<EvalT, Traits>::
PoissonAdvDiffSystemResid(const Teuchos::ParameterList& p) :
  wBF     (p.get<std::string>                   ("Weighted BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Scalar Data Layout") ),
  wGradBF    (p.get<std::string>                   ("Weighted Gradient BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Gradient Data Layout") ),
  U       (p.get<std::string>                   ("QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  UGrad      (p.get<std::string>                   ("Gradient QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
  Residual   (p.get<std::string>                   ("Residual Name"),
              p.get<Teuchos::RCP<PHX::DataLayout> >("Node Vector Data Layout") ) 
{

  //U = [Phi; rhop], where rhop is the concentration of positive ions 
  
  if (p.isType<bool>("Supports Transient"))
    supportsTransient = p.get<bool>("Supports Transient");

  std::cout << "IKT supportTransient = " << supportsTransient << "\n";
  if (supportsTransient) { 
    UDot = decltype(UDot)(
      p.get<std::string>("QP Time Derivative Variable Name"),
      p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") );
  }

  this->addDependentField(U.fieldTag());
  this->addDependentField(UGrad.fieldTag());
  if (supportsTransient) this->addDependentField(UDot.fieldTag());
  this->addDependentField(wBF.fieldTag());
  this->addDependentField(wGradBF.fieldTag());

  this->addEvaluatedField(Residual);


  this->setName("PoissonAdvDiffSystemResid" );

  std::vector<PHX::DataLayout::size_type> dims;
  wGradBF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];
  numDims  = dims[3];

  Teuchos::ParameterList* bf_list =
  p.get<Teuchos::ParameterList*>("Parameter List");


  U.fieldTag().dataLayout().dimensions(dims);
  vecDim  = dims[2];

  kappa = bf_list->get("Diffusivity", 4.4e9); 
  rhom = bf_list->get("Density of negative ions", 1.0e23); 
  d    = bf_list->get("Kinematic diffusivity", 1.0e-10); 
  c    = bf_list->get("Advection coefficient", 3.89e-9); 
  
  Teuchos::RCP<Teuchos::FancyOStream> out = Teuchos::VerboseObjectBase::getDefaultOStream();
  *out << " vecDim = " << vecDim << "\n";
  *out << " numDims = " << numDims << "\n";
  

}

//**********************************************************************
template<typename EvalT, typename Traits>
void PoissonAdvDiffSystemResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData /* d */,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(U,fm);
  this->utils.setFieldData(UGrad,fm);
  if (supportsTransient) this->utils.setFieldData(UDot,fm);
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(wGradBF,fm);

  this->utils.setFieldData(Residual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void PoissonAdvDiffSystemResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t node=0; node < numNodes; ++node) {
      for (std::size_t i=0; i<vecDim; i++)  Residual(cell,node,i) = 0.0;
      for (std::size_t qp=0; qp < numQPs; ++qp) {
        //- div(kappa*grad(phi)) = rhop - rhom 
        Residual(cell,node,0) += kappa*UGrad(cell,qp,0,0)*wGradBF(cell,node,qp,0) + 
                                 kappa*UGrad(cell,qp,0,1)*wGradBF(cell,node,qp,1) -
				 U(cell,qp,1)*wBF(cell,node,qp) +
                                 rhom*wBF(cell,node,qp);
	//drhop/dt - div(d*grad(rhop) + c*rhop*grad(phi)) = 0 
	if (supportsTransient) {
          Residual(cell,node,1) += UDot(cell,qp,1)*wBF(cell,node,qp); 
	}
        Residual(cell,node,1) += d*UGrad(cell,qp,1,0)*wGradBF(cell,node,qp,0) + 
		                 d*UGrad(cell,qp,1,1)*wGradBF(cell,node,qp,1) + 
		                 c*U(cell,qp,1)*UGrad(cell,qp,0,0)*wGradBF(cell,node,qp,0) + 
		                 c*U(cell,qp,1)*UGrad(cell,qp,0,1)*wGradBF(cell,node,qp,1); 
				 
      }
    }        
  }
}

//**********************************************************************
}




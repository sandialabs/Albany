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
CoupledAdvDiffResid<EvalT, Traits>::
CoupledAdvDiffResid(const Teuchos::ParameterList& p) :
  wBF     (p.get<std::string>                   ("Weighted BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Scalar Data Layout") ),
  wGradBF    (p.get<std::string>                   ("Weighted Gradient BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Gradient Data Layout") ),
  rhop       (p.get<std::string>                   ("QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  rhopDot      (p.get<std::string>                   ("QP Time Derivative Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  rhopGrad      (p.get<std::string>                   ("Gradient QP rhop Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  PhiGrad      (p.get<std::string>                   ("Gradient QP Phi Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  Residual   (p.get<std::string>                   ("Residual Name"),
              p.get<Teuchos::RCP<PHX::DataLayout> >("Node Scalar Data Layout") ) 
{

  this->addDependentField(rhop.fieldTag());
  this->addDependentField(rhopGrad.fieldTag());
  this->addDependentField(rhopDot.fieldTag());
  this->addDependentField(PhiGrad.fieldTag());
  this->addDependentField(wBF.fieldTag());
  this->addDependentField(wGradBF.fieldTag());

  this->addEvaluatedField(Residual);


  this->setName("CoupledAdvDiffResid" );

  std::vector<PHX::DataLayout::size_type> dims;
  wGradBF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];
  numDims  = dims[3];

  Teuchos::ParameterList* bf_list =
  p.get<Teuchos::ParameterList*>("Parameter List");

  d    = bf_list->get("Kinematic diffusivity", 1.0e-10); 
  c    = bf_list->get("Advection coefficient", 3.89e-9); 
  
  //Teuchos::RCP<Teuchos::FancyOStream> out = Teuchos::VerboseObjectBase::getDefaultOStream();
  //*out << " numDims = " << numDims << "\n";
  

}

//**********************************************************************
template<typename EvalT, typename Traits>
void CoupledAdvDiffResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData /* d */,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(rhop,fm);
  this->utils.setFieldData(rhopGrad,fm);
  this->utils.setFieldData(rhopDot,fm);
  this->utils.setFieldData(PhiGrad,fm);
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(wGradBF,fm);

  this->utils.setFieldData(Residual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void CoupledAdvDiffResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t node=0; node < numNodes; ++node) {
      Residual(cell,node) = 0.0; 
      for (std::size_t qp=0; qp < numQPs; ++qp) {
	//drhop/dt - div(d*grad(rhop) + c*rhop*grad(phi)) = 0 
        Residual(cell,node) += rhopDot(cell,qp)*wBF(cell,node,qp) + 
		               d*rhopGrad(cell,qp,0)*wGradBF(cell,node,qp,0) + 
		               d*rhopGrad(cell,qp,1)*wGradBF(cell,node,qp,1) + 
		               c*rhop(cell,qp)*PhiGrad(cell,qp,0)*wGradBF(cell,node,qp,0) + 
		               c*rhop(cell,qp)*PhiGrad(cell,qp,1)*wGradBF(cell,node,qp,1);
	if (cell == 0) std::cout << "IKT PhiGrad = (" << PhiGrad(cell,qp,0) << ", " << PhiGrad(cell,qp,1) << ")\n"; 	
      }
    }        
  }
}

//**********************************************************************
}




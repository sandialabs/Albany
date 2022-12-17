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
ReactDiffSystemResid<EvalT, Traits>::
ReactDiffSystemResid(const Teuchos::ParameterList& p) :
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



  this->addDependentField(U.fieldTag());
  this->addDependentField(UGrad.fieldTag());
  this->addDependentField(wBF.fieldTag());
  this->addDependentField(wGradBF.fieldTag());

  this->addEvaluatedField(Residual);


  this->setName("ReactDiffSystemResid" );

  std::vector<PHX::DataLayout::size_type> dims;
  wGradBF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];
  numDims  = dims[3];


  Teuchos::ParameterList* bf_list =
  p.get<Teuchos::ParameterList*>("Parameter List");


  U.fieldTag().dataLayout().dimensions(dims);
  vecDim  = dims[2];

  mu0 = bf_list->get("Viscosity mu0", 0.1); 
  mu1 = bf_list->get("Viscosity mu1", 0.1); 
  mu2 = bf_list->get("Viscosity mu2", 0.1); 
  
  forces = arcpFromArray(bf_list->get<Teuchos::Array<double> >("Forces", Teuchos::Array<double>()));

  reactCoeff0 = arcpFromArray(bf_list->get<Teuchos::Array<double> >("Reaction Coefficients0", Teuchos::Array<double>()));
  reactCoeff1 = arcpFromArray(bf_list->get<Teuchos::Array<double> >("Reaction Coefficients1", Teuchos::Array<double>()));
  reactCoeff2 = arcpFromArray(bf_list->get<Teuchos::Array<double> >("Reaction Coefficients2", Teuchos::Array<double>()));

  if (forces.size() != 3) {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        std::logic_error,
        "Length of Forces array for ReactDiffSystem problem must be 3." << 
        "  You have provided an array of length " << forces.size() << ".\n");
  }

  if (reactCoeff0.size() != 3) {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        std::logic_error,
        "Length of Reaction Coefficients0 array for ReactDiffSystem problem must be 3." << 
        "  You have provided an array of length " << reactCoeff0.size() << ".\n");
  }

  if (reactCoeff1.size() != 3) {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        std::logic_error,
        "Length of Reaction Coefficients1 array for ReactDiffSystem problem must be 3." << 
        "  You have provided an array of length " << reactCoeff1.size() << ".\n");
  }
 
  if (reactCoeff2.size() != 3) {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        std::logic_error,
        "Length of Reaction Coefficients2 array for ReactDiffSystem problem must be 3." << 
        "  You have provided an array of length " << reactCoeff2.size() << ".\n");
  }

  Teuchos::RCP<Teuchos::FancyOStream> out = Teuchos::VerboseObjectBase::getDefaultOStream();
  *out << " vecDim = " << vecDim << "\n";
  *out << " numDims = " << numDims << "\n";


}

//**********************************************************************
template<typename EvalT, typename Traits>
void ReactDiffSystemResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(U,fm);
  this->utils.setFieldData(UGrad,fm);
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(wGradBF,fm);

  this->utils.setFieldData(Residual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ReactDiffSystemResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  typedef Intrepid2::FunctionSpaceTools<PHX::Device> FST;

  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t node=0; node < numNodes; ++node) {
      for (std::size_t i=0; i<vecDim; i++)  Residual(cell,node,i) = 0.0;
      for (std::size_t qp=0; qp < numQPs; ++qp) {
        //- mu0*delta(u0) + a0*u0 + a1*u1 + a2*u2 = f0
        Residual(cell,node,0) += mu0*UGrad(cell,qp,0,0)*wGradBF(cell,node,qp,0) + 
                                 mu0*UGrad(cell,qp,0,1)*wGradBF(cell,node,qp,1) +
                                 mu0*UGrad(cell,qp,0,2)*wGradBF(cell,node,qp,2) -
                                 reactCoeff0[0]*U(cell,qp,0)*wBF(cell,node,qp) -
                                 reactCoeff0[1]*U(cell,qp,1)*wBF(cell,node,qp) -
                                 reactCoeff0[2]*U(cell,qp,2)*wBF(cell,node,qp) -
                                 forces[0]*wBF(cell,node,qp); 
        //- mu1*delta(u1) + b0*u0 + b1*u1 + b2*u2 = f1
        Residual(cell,node,1) += mu1*UGrad(cell,qp,1,0)*wGradBF(cell,node,qp,0) + 
                                 mu1*UGrad(cell,qp,1,1)*wGradBF(cell,node,qp,1) +
                                 mu1*UGrad(cell,qp,1,2)*wGradBF(cell,node,qp,2) -
                                 reactCoeff1[0]*U(cell,qp,0)*wBF(cell,node,qp) -
                                 reactCoeff1[1]*U(cell,qp,1)*wBF(cell,node,qp) -
                                 reactCoeff1[2]*U(cell,qp,2)*wBF(cell,node,qp) -
                                 forces[1]*wBF(cell,node,qp); 
        //- mu2*delta(u2) + c0*u0 + c1*u1 + c2*u2 = f2
        Residual(cell,node,2) += mu2*UGrad(cell,qp,2,0)*wGradBF(cell,node,qp,0) + 
                                 mu2*UGrad(cell,qp,2,1)*wGradBF(cell,node,qp,1) +
                                 mu2*UGrad(cell,qp,2,2)*wGradBF(cell,node,qp,2) -
                                 reactCoeff2[0]*U(cell,qp,0)*wBF(cell,node,qp) -
                                 reactCoeff2[1]*U(cell,qp,1)*wBF(cell,node,qp) -
                                 reactCoeff2[2]*U(cell,qp,2)*wBF(cell,node,qp) -
                                 forces[2]*wBF(cell,node,qp); 
      }
    }        
  }
}

//**********************************************************************
}




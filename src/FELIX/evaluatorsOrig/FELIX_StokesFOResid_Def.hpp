/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

namespace FELIX {

//**********************************************************************
template<typename EvalT, typename Traits>
StokesFOResid<EvalT, Traits>::
StokesFOResid(const Teuchos::ParameterList& p) :
  wBF     (p.get<std::string>                   ("Weighted BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Scalar Data Layout") ),
  wGradBF    (p.get<std::string>                   ("Weighted Gradient BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Gradient Data Layout") ),
  C          (p.get<std::string>                   ("QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  Cgrad      (p.get<std::string>                   ("Gradient QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Concentration Tensor Data Layout") ),
  CDot       (p.get<std::string>                   ("QP Time Derivative Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  force       (p.get<std::string>              ("Body Force Name"),
 	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  Residual   (p.get<std::string>                   ("Residual Name"),
              p.get<Teuchos::RCP<PHX::DataLayout> >("Node Vector Data Layout") )
{
  this->addDependentField(C);
  this->addDependentField(Cgrad);
  this->addDependentField(force);
  //this->addDependentField(CDot);
  this->addDependentField(wBF);
  this->addDependentField(wGradBF);

  this->addEvaluatedField(Residual);


  this->setName("StokesFOResid"+PHX::TypeString<EvalT>::value);

  std::vector<PHX::DataLayout::size_type> dims;
  wGradBF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];
  numDims  = dims[3];

  C.fieldTag().dataLayout().dimensions(dims);
  vecDim  = dims[2];

cout << " vecDim = " << vecDim << endl;
cout << " numDims = " << numDims << endl;
cout << " in FELIX Stokes FO residual! " << numDims << endl;
cout << " numQPs = " << numQPs << endl; 
cout << " numNodes = " << numNodes << endl; 


if (vecDim != 2)  {TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
				  std::endl << "Error in FELIX::StokesFOResid constructor:  " <<
				  "Invalid Parameter vecDim.  Problem implemented for 2 dofs per node only (u and v). " << std::endl);}

}

//**********************************************************************
template<typename EvalT, typename Traits>
void StokesFOResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(C,fm);
  this->utils.setFieldData(Cgrad,fm);
  this->utils.setFieldData(force,fm);
  //this->utils.setFieldData(CDot,fm);
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(wGradBF,fm);

  this->utils.setFieldData(Residual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void StokesFOResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  MeshScalarT muqp = 1.0; //mu is constant for now
  typedef Intrepid::FunctionSpaceTools FST;
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t node=0; node < numNodes; ++node) {
              for (std::size_t i=0; i<vecDim; i++)  Residual(cell,node,i)=0.0;
          for (std::size_t qp=0; qp < numQPs; ++qp) {
             Residual(cell,node,0) += 2.0*muqp*((2.0*Cgrad(cell,qp,0,0) + Cgrad(cell,qp,1,1))*wGradBF(cell,node,qp,0) + 
                                      0.5*(Cgrad(cell,qp,0,1) + Cgrad(cell,qp,1,0))*wGradBF(cell,node,qp,1)) + 
                                      force(cell,qp,0)*wBF(cell,node,qp);
             Residual(cell,node,1) += 2.0*muqp*(0.5*(Cgrad(cell,qp,0,1) + Cgrad(cell,qp,1,0))*wGradBF(cell,node,qp,0) +
                                      (Cgrad(cell,qp,0,0) + 2.0*Cgrad(cell,qp,1,1))*wGradBF(cell,node,qp,1)) + force(cell,qp,1)*wBF(cell,node,qp); 
              }
           
    } } 
}

//**********************************************************************
}


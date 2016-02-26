//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"

namespace FELIX {

//**********************************************************************
template<typename EvalT, typename Traits>
StokesL1L2Resid<EvalT, Traits>::
StokesL1L2Resid(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl) :
  wBF        (p.get<std::string> ("Weighted BF Name"), dl->node_qp_scalar),
  wGradBF    (p.get<std::string> ("Weighted Gradient BF Name"), dl->node_qp_gradient),
  U          (p.get<std::string> ("QP Variable Name"), dl->qp_vector),
  Ugrad      (p.get<std::string> ("Gradient QP Variable Name"), dl->qp_vecgradient),
  UDot       (p.get<std::string> ("QP Time Derivative Variable Name"), dl->qp_vector),
  force      (p.get<std::string> ("Body Force Name"), dl->qp_vector),
  muFELIX    (p.get<std::string> ("FELIX Viscosity QP Variable Name"), dl->qp_scalar),
  epsilonXX  (p.get<std::string> ("FELIX EpsilonXX QP Variable Name"), dl->qp_scalar), 
  epsilonYY  (p.get<std::string> ("FELIX EpsilonYY QP Variable Name"), dl->qp_scalar), 
  epsilonXY  (p.get<std::string> ("FELIX EpsilonXY QP Variable Name"), dl->qp_scalar), 
  Residual   (p.get<std::string> ("Residual Name"), dl->node_vector)
{
  this->addDependentField(U);
  this->addDependentField(Ugrad);
  this->addDependentField(force);
  //this->addDependentField(UDot);
  this->addDependentField(wBF);
  this->addDependentField(wGradBF);
  this->addDependentField(muFELIX);
  this->addDependentField(epsilonXX);
  this->addDependentField(epsilonYY);
  this->addDependentField(epsilonXY);

  this->addEvaluatedField(Residual);


  this->setName("StokesL1L2Resid"+PHX::typeAsString<EvalT>());

  std::vector<PHX::DataLayout::size_type> dims;
  wGradBF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];
  numDims  = dims[3];

  U.fieldTag().dataLayout().dimensions(dims);
  vecDim  = dims[2];

//  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());
//*out << " in FELIX Stokes L1L2 residual! " << endl;
//*out << " vecDim = " << vecDim << endl;
//*out << " numDims = " << numDims << endl;
//*out << " numQPs = " << numQPs << endl; 
//*out << " numNodes = " << numNodes << endl; 


if (vecDim != 2)  {TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
				  std::endl << "Error in FELIX::StokesL1L2Resid constructor:  " <<
				  "Invalid Parameter vecDim.  Problem implemented for 2 dofs per node only (u and v). " << std::endl);}

}

//**********************************************************************
template<typename EvalT, typename Traits>
void StokesL1L2Resid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(U,fm);
  this->utils.setFieldData(Ugrad,fm);
  this->utils.setFieldData(force,fm);
  //this->utils.setFieldData(UDot,fm);
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(wGradBF,fm);
  this->utils.setFieldData(muFELIX,fm);
  this->utils.setFieldData(epsilonXX,fm);
  this->utils.setFieldData(epsilonYY,fm);
  this->utils.setFieldData(epsilonXY,fm);

  this->utils.setFieldData(Residual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void StokesL1L2Resid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  typedef Intrepid2::FunctionSpaceTools FST;
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t node=0; node < numNodes; ++node) {
            for (std::size_t i=0; i<vecDim; i++)  Residual(cell,node,i)=0.0;
        for (std::size_t qp=0; qp < numQPs; ++qp) {
           Residual(cell,node,0) += 2.0*muFELIX(cell,qp)*((2.0*epsilonXX(cell,qp) + epsilonYY(cell,qp))*wGradBF(cell,node,qp,0) + 
                                    epsilonXY(cell,qp)*wGradBF(cell,node,qp,1)) + 
                                    force(cell,qp,0)*wBF(cell,node,qp);
           Residual(cell,node,1) += 2.0*muFELIX(cell,qp)*(epsilonXY(cell,qp)*wGradBF(cell,node,qp,0) +
                                      (epsilonXX(cell,qp) + 2.0*epsilonYY(cell,qp))*wGradBF(cell,node,qp,1)) 
                                  + force(cell,qp,1)*wBF(cell,node,qp); 
              }
      }     
    }  
}

//**********************************************************************
}


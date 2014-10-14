//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_RCP.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"
#include "Aeras_Layouts.hpp"

namespace Aeras {

//**********************************************************************
template<typename EvalT, typename Traits>
XScalarAdvectionResid<EvalT, Traits>::
XScalarAdvectionResid(Teuchos::ParameterList& p,
                      const Teuchos::RCP<Aeras::Layouts>& dl) :
  wBF      (p.get<std::string> ("Weighted BF Name"),                 dl->node_qp_scalar),
  wGradBF  (p.get<std::string> ("Weighted Gradient BF Name"),        dl->node_qp_gradient),
  coordVec (p.get<std::string> ("QP Coordinate Vector Name"),        dl->qp_gradient),
  X        (p.get<std::string> ("QP Variable Name"),              
            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Variable Layout",                  dl->qp_scalar_level)),        
  XGrad    (p.get<std::string> ("Gradient QP Variable Name"),     
            p.get<Teuchos::RCP<PHX::DataLayout> >("Gradient QP Variable Layout",         dl->qp_gradient_level)),        
  XDot     (p.get<std::string> ("QP Time Derivative Variable Name"), 
            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Time Derivative Variable Layout",  dl->qp_scalar_level)),        
  Residual (p.get<std::string> ("Residual Name"),          
            p.get<Teuchos::RCP<PHX::DataLayout> >("Residual Layout",                     dl->node_scalar_level)),        
  numNodes   (dl->node_scalar             ->dimension(1)),
  numQPs     (dl->node_qp_scalar          ->dimension(2)),
  numDims    (dl->node_qp_gradient        ->dimension(3)),
  numLevels  (dl->node_scalar_level       ->dimension(2)),
  numRank    (X.fieldTag().dataLayout().rank())
{

  Teuchos::ParameterList* xsa_params = p.get<Teuchos::ParameterList*>("XScalarAdvection Problem");
  Re = xsa_params->get<double>("Reynolds Number", 1.0); //Default: Re=1
  std::cout << "XScalarAdvectionResid: Re= " << Re << std::endl;

  this->addDependentField(X);
  this->addDependentField(XGrad);
  this->addDependentField(XDot);
  this->addDependentField(wBF);
  this->addDependentField(wGradBF);
  this->addDependentField(coordVec);

  this->addEvaluatedField(Residual);

  this->setName("Aeras::XScalarAdvectionResid"+PHX::TypeString<EvalT>::value);

  // Register Reynolds number as Sacado-ized Parameter
  Teuchos::RCP<ParamLib> paramLib = p.get<Teuchos::RCP<ParamLib> >("Parameter Library");
  new Sacado::ParameterRegistration<EvalT, SPL_Traits>("Reynolds Number", this, paramLib);

  TEUCHOS_TEST_FOR_EXCEPTION( (numRank!=2 && numRank!=3) ,
     std::logic_error,"Aeras::XScalarAdvectionResid supports scalar or vector only");
}

//**********************************************************************
template<typename EvalT, typename Traits>
void XScalarAdvectionResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(X,       fm);
  this->utils.setFieldData(XGrad,   fm);
  this->utils.setFieldData(XDot,    fm);
  this->utils.setFieldData(wBF,     fm);
  this->utils.setFieldData(wGradBF, fm);
  this->utils.setFieldData(coordVec,fm);

  this->utils.setFieldData(Residual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void XScalarAdvectionResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  std::vector<ScalarT> vel(numLevels);
  for (int level=0; level < numLevels; ++level) {
    vel[level] = (level+1)*Re;
  }

  for (int i=0; i < Residual.size(); ++i) Residual(i)=0.0;

  for (int cell=0; cell < workset.numCells; ++cell) {
    for (int qp=0; qp < numQPs; ++qp) {
      for (int node=0; node < numNodes; ++node) {
        if (2==numRank) {
          Residual(cell,node) += XDot(cell,qp)*wBF(cell,node,qp);
          for (int j=0; j < numDims; ++j) 
            Residual(cell,node) += vel[0] * XGrad(cell,qp,j)*wBF(cell,node,qp);
        } else {
          for (int level=0; level < numLevels; ++level) {
            Residual(cell,node,level) += XDot(cell,qp,level)*wBF(cell,node,qp);
            for (int j=0; j < numDims; ++j) 
              Residual(cell,node,level) += vel[level] * XGrad(cell,qp,level,j)*wBF(cell,node,qp);
          }
        }
      }
    }
  }
}

//**********************************************************************
// Provide Access to Parameter for sensitivity/optimization/UQ
template<typename EvalT,typename Traits>
typename XScalarAdvectionResid<EvalT,Traits>::ScalarT&
XScalarAdvectionResid<EvalT,Traits>::getValue(const std::string &n)
{
  return Re;
}

}

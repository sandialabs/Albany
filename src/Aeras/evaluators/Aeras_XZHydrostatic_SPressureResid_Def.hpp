//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_RCP.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"
#include "Aeras_Layouts.hpp"

namespace Aeras {

//**********************************************************************
template<typename EvalT, typename Traits>
XZHydrostatic_SPressureResid<EvalT, Traits>::
XZHydrostatic_SPressureResid(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Aeras::Layouts>& dl) :
  wBF      (p.get<std::string> ("Weighted BF Name"), dl->node_qp_scalar),
  wGradBF  (p.get<std::string> ("Weighted Gradient BF Name"),dl->node_qp_gradient),
  sp       (p.get<std::string> ("QP Variable Name"), dl->qp_scalar),
  spDot    (p.get<std::string> ("QP Time Derivative Variable Name"), dl->qp_scalar),
  Residual (p.get<std::string> ("Residual Name"), dl->node_scalar)
{

  this->addDependentField(spDot);
  this->addDependentField(wBF);

  this->addEvaluatedField(Residual);


  this->setName("Aeras::XZHydrostatic_SPressureResid"+PHX::TypeString<EvalT>::value);

  std::vector<PHX::DataLayout::size_type> dims;
  wGradBF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];
  numDims  = dims[3];

  sp.fieldTag().dataLayout().dimensions(dims);
  numLevels =  p.get< int >("Number of Vertical Levels");
  std::cout << "XZHydrostatic_SPressureResid: numLevels= " << numLevels << std::endl;

  sp0 = 0.0;
}

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_SPressureResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(spDot,fm);
  this->utils.setFieldData(wBF,fm);

  this->utils.setFieldData(Residual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_SPressureResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  std::vector<ScalarT> vel(numLevels);
  for (std::size_t level=0; level < numLevels; ++level) {
    vel[level] = (level+1)*Re;
  }

  for (std::size_t i=0; i < Residual.size(); ++i) Residual(i)=0.0;

  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {

      for (std::size_t node=0; node < numNodes; ++node) {
        for (std::size_t level=0; level < numLevels; ++level) {
          // Transient Term
          Residual(cell,node,level) += spDot(cell,qp,level)*wBF(cell,node,qp);
          // Advection Term
          for (std::size_t j=0; j < numDims; ++j) {
              Residual(cell,node,level) += 0.0;
          }
        }
      }
    }
  }
}

//**********************************************************************
template<typename EvalT,typename Traits>
typename XZHydrostatic_SPressureResid<EvalT,Traits>::ScalarT& 
XZHydrostatic_SPressureResid<EvalT,Traits>::getValue(const std::string &n)
{
  if (n=="SPressure") return sp0;
}

}

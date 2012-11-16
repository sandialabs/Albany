//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"
#include "Intrepid_RealSpaceTools.hpp"
#include "Sacado_ParameterRegistration.hpp"

#include "Tensor.h"

namespace LCM {

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  MechanicsResidual<EvalT, Traits>::
  MechanicsResidual(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl) :
    stress (p.get<std::string>("Stress Name"),dl->qp_tensor),
    J      (p.get<std::string>("DetDefGrad Name"),dl->qp_scalar),
    defgrad(p.get<std::string>("DefGrad Name"),dl->qp_tensor),
    wGradBF(p.get<std::string>("Weighted Gradient BF Name"),dl->node_qp_vector),
    wBF    (p.get<std::string>("Weighted BF Name"),dl->node_qp_scalar),
    Residual(p.get<std::string>("Residual Name"),dl->node_vector)
  {
    this->addDependentField(stress);
    this->addDependentField(J);
    this->addDependentField(defgrad);
    this->addDependentField(wGradBF);
    this->addDependentField(wBF);

    this->addEvaluatedField(Residual);

    this->setName("MechanicsResidual"+PHX::TypeString<EvalT>::value);

    std::vector<PHX::DataLayout::size_type> dims;
    wGradBF.fieldTag().dataLayout().dimensions(dims);
    numNodes = dims[1];
    numQPs   = dims[2];
    numDims  = dims[3];
    int worksetSize = dims[0];

    Teuchos::RCP<ParamLib> paramLib = 
      p.get< Teuchos::RCP<ParamLib> >("Parameter Library");

    zGrav=0.0;
    new Sacado::ParameterRegistration<EvalT, SPL_Traits>("zGrav", 
                                                         this, 
                                                         paramLib);

  }

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void MechanicsResidual<EvalT, Traits>::
  postRegistrationSetup(typename Traits::SetupData d,
                        PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(stress,fm);
    this->utils.setFieldData(J,fm);
    this->utils.setFieldData(defgrad,fm);
    this->utils.setFieldData(wGradBF,fm);
    this->utils.setFieldData(wBF,fm);
    this->utils.setFieldData(Residual,fm);
  }

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void MechanicsResidual<EvalT, Traits>::
  evaluateFields(typename Traits::EvalData workset)
  {
    cout.precision(15);
    typedef Intrepid::FunctionSpaceTools FST;
    typedef Intrepid::RealSpaceTools<ScalarT> RST;

    LCM::Tensor<ScalarT> F, P, sig;
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t node=0; node < numNodes; ++node) {
        for (std::size_t dim=0; dim<numDims; dim++)  
          Residual(cell,node,dim)=0.0;
        for (std::size_t qp=0; qp < numQPs; ++qp) {
          F = LCM::Tensor<ScalarT>( numDims, &defgrad(cell,qp,0,0) );
          sig = LCM::Tensor<ScalarT>( numDims, &stress(cell,qp,0,0) );
          P = J(cell,qp)*sig*LCM::inverse(LCM::transpose(F));
          for (std::size_t i=0; i<numDims; i++) {
            for (std::size_t j=0; j<numDims; j++) {
              Residual(cell,node,i) += P(i, j) * wGradBF(cell, node, qp, j);
            } 
          } 
        } 
      } 
    }
    /** // Gravity term used for load stepping 
        for (std::size_t cell=0; cell < workset.numCells; ++cell) {
        for (std::size_t node=0; node < numNodes; ++node) {
        for (std::size_t qp=0; qp < numQPs; ++qp) {
        Residual(cell,node,2) +=  zGrav * wBF(cell, node, qp);
        } 
        } 
        }
    **/
  }
  //----------------------------------------------------------------------------
  template<typename EvalT,typename Traits>
  typename MechanicsResidual<EvalT,Traits>::ScalarT&
  MechanicsResidual<EvalT,Traits>::getValue(const std::string &n)
  {
    return zGrav;
  }

  //----------------------------------------------------------------------------
}


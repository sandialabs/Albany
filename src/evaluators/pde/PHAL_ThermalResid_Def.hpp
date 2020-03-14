//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"
#include "PHAL_Utilities.hpp"

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits>
ThermalResid<EvalT, Traits>::
ThermalResid(const Teuchos::ParameterList& p) :
  wBF         (p.get<std::string>                   ("Weighted BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Scalar Data Layout") ),
  Tdot        (p.get<std::string>                   ("QP Time Derivative Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  wGradBF     (p.get<std::string>                   ("Weighted Gradient BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Vector Data Layout") ),
  TGrad       (p.get<std::string>                   ("Gradient QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  TResidual   (p.get<std::string>                   ("Residual Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node Scalar Data Layout") ),
  kappa(p.get<Teuchos::Array<double>>("Thermal Conductivity")),
  rho(p.get<double>("Density")),
  C(p.get<double>("Heat Capacity"))
{

  this->addDependentField(wBF);
  this->addDependentField(Tdot);
  this->addDependentField(TGrad);
  this->addDependentField(wGradBF);
  this->addEvaluatedField(TResidual);

  Teuchos::RCP<PHX::DataLayout> vector_dl =
      p.get<Teuchos::RCP<PHX::DataLayout>>("Node QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  worksetSize = dims[0];
  numNodes    = dims[1];
  numQPs      = dims[2];
  numDims     = dims[3];
  this->setName("ThermalResid" );
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ThermalResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(wBF, fm);
  this->utils.setFieldData(TGrad, fm);
  this->utils.setFieldData(wGradBF, fm);
  this->utils.setFieldData(Tdot, fm);
  this->utils.setFieldData(TResidual, fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ThermalResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

  // We are solving the following PDE:
  // rho*CdT/dt - kappa_1*dT/dx - kappa_2*dT/dy - kappa_3*dT/dz = 0 in 3D
  for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
    for (std::size_t node = 0; node < numNodes; ++node) {
      TResidual(cell, node) = 0.0;
      for (std::size_t qp = 0; qp < numQPs; ++qp) {
        // Time-derivative contribution to residual
        TResidual(cell, node) += rho * C * Tdot(cell, qp) * wBF(cell, node, qp);
        // Diffusion part of residual
        for (std::size_t ndim = 0; ndim < numDims; ++ndim) {
          TResidual(cell, node) += kappa[ndim] * TGrad(cell, qp, ndim) *
                                   wGradBF(cell, node, qp, ndim);
        }
      }
    }
  }

}

//**********************************************************************
}


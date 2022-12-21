//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_Print.hpp"

namespace LandIce {


//**********************************************************************
template<typename EvalT, typename Traits>
StokesTauM<EvalT, Traits>::
StokesTauM(const Teuchos::ParameterList& p,
           const Teuchos::RCP<Albany::Layouts>& dl) :
  V       (p.get<std::string> ("Velocity QP Variable Name"), dl->qp_vector),
  Gc      (p.get<std::string> ("Contravarient Metric Tensor Name"), dl->qp_tensor),
  muLandIce (p.get<std::string> ("LandIce Viscosity QP Variable Name"), dl->qp_scalar),
  jacobian_det (p.get<std::string>  ("Jacobian Det Name"), dl->qp_scalar ),
  TauM    (p.get<std::string> ("Tau M Name"), dl->qp_scalar)
{
   Teuchos::ParameterList* tauM_list =
   p.get<Teuchos::ParameterList*>("Parameter List");

  delta = tauM_list->get("Delta", 1.0);

  this->addDependentField(V);
  this->addDependentField(Gc);
  this->addDependentField(muLandIce);
  this->addDependentField(jacobian_det);

  this->addEvaluatedField(TauM);

  std::vector<PHX::DataLayout::size_type> dims;
  dl->qp_gradient->dimensions(dims);
  numCells= dims[0];
  numQPs  = dims[1];
  numDims = dims[2];

  this->setName("StokesTauM"+PHX::print<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void StokesTauM<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(V,fm);
  this->utils.setFieldData(Gc,fm);
  this->utils.setFieldData(muLandIce,fm);
  this->utils.setFieldData(jacobian_det,fm);

  this->utils.setFieldData(TauM,fm);

  // Allocate workspace
  normGc = Kokkos::createDynRankView(Gc.get_view(), "XXX", numCells, numQPs);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void StokesTauM<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  //tau = h^2*delta - stabilization from Bochev et. al. "taxonomy" paper
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {
        meshSize = 2.0*pow(jacobian_det(cell,qp), 1.0/numDims);
        TauM(cell, qp) = delta*meshSize*meshSize;
    }
  }
}

//**********************************************************************
}


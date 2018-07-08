//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"
#include "Sacado.hpp"


namespace Tsunami {


template<typename EvalT, typename Traits>
NavierStokesBodyForce<EvalT, Traits>::
NavierStokesBodyForce(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl) :
  force              (p.get<std::string>("Body Force Name"),dl->qp_vector)
{

  Teuchos::ParameterList* bf_list =
    p.get<Teuchos::ParameterList*>("Parameter List");

  std::string type = bf_list->get("Type", "None");
  
  if (type == "None") {
    bf_type = NONE;
  }
  else if (type == "Poly Steady Stokes 2D") {
    bf_type = POLY;
    coordVec = decltype(coordVec)(
            p.get<std::string>("Coordinate Vector Name"),dl->qp_gradient);
    viscosityQP = decltype(viscosityQP)(
            p.get<std::string>("Fluid Viscosity QP Name"),dl->qp_scalar);
    this->addDependentField(coordVec);
    this->addDependentField(viscosityQP);
  }
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true, std::logic_error,
        "Error in Tsunami::NavierStokes: Invalid Body Force Type = "
            << type << "!  Valid types are 'None' and 'Poly'.");
  }

  this->addEvaluatedField(force);

  std::vector<PHX::DataLayout::size_type> dims;
  dl->qp_gradient->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  this->setName("NavierStokesBodyForce"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void NavierStokesBodyForce<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  if (bf_type == POLY) {
    this->utils.setFieldData(coordVec,fm);
    this->utils.setFieldData(viscosityQP,fm);
  }
  this->utils.setFieldData(force,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void NavierStokesBodyForce<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  if (bf_type == NONE) {
    for (std::size_t cell=0; cell < workset.numCells; ++cell)
      for (std::size_t qp=0; qp < numQPs; ++qp)
        for (std::size_t i=0; i < numDims; ++i)
          force(cell,qp,i) = 0.0;
  }
  //The following is hard-coded for a 2D Stokes problem with manufactured solution
  else if (bf_type == POLY) {
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
        //ScalarT* f = &force(cell,qp,0);
        MeshScalarT X0 = coordVec(cell,qp,0);
        MeshScalarT X1 = coordVec(cell,qp,1);
        force(cell,qp,0) =  40.0*viscosityQP(cell,qp)*(2.0*X1*X1 - 3.0*X1+1.0)*X1*(6.0*X0*X0 -6.0*X0 + 1.0)
                         + 120*viscosityQP(cell,qp)*(X0-1.0)*(X0-1.0)*X0*X0*(2.0*X1-1.0)
                         + 10.0*viscosityQP(cell,qp);
        force(cell,qp,1) = - 120.0*viscosityQP(cell,qp)*(1.0-X1)*(1.0-X1)*X1*X1*(2.0*X0-1.0)
                           - 40.0*viscosityQP(cell,qp)*(2.0*X0*X0 - 3.0*X0 + 1.0)*X0*(6.0*X1*X1 - 6.0*X1 + 1.0)
                           - 5*viscosityQP(cell,qp)*X1;
      }
    }
  }
}

}

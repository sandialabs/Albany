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
NSTauT<EvalT, Traits>::
NSTauT(const Teuchos::ParameterList& p) :
  V           (p.get<std::string>                   ("Velocity QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  ThermalCond (p.get<std::string>                   ("Thermal Conductivity Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  Gc            (p.get<std::string>                   ("Contravarient Metric Tensor Name"),  
                 p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
  rho       (p.get<std::string>                   ("Density QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  Cp       (p.get<std::string>                  ("Specific Heat QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  TauT            (p.get<std::string>                 ("Tau T Name"),
                 p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") )
  
{
  this->addDependentField(V);
  this->addDependentField(ThermalCond);
  this->addDependentField(Gc);
  this->addDependentField(rho);
  this->addDependentField(Cp);
 
  this->addEvaluatedField(TauT);

  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numCells = dims[0];
  numQPs  = dims[1];
  numDims = dims[2];

  this->setName("NSTauT" );
}

//**********************************************************************
template<typename EvalT, typename Traits>
void NSTauT<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(V,fm);
  this->utils.setFieldData(ThermalCond,fm);
  this->utils.setFieldData(Gc,fm);
  this->utils.setFieldData(rho,fm);
  this->utils.setFieldData(Cp,fm);
  
  this->utils.setFieldData(TauT,fm);

  // Allocate workspace
  normGc = Kokkos::createDynRankView(Gc.get_view(), "YYY", numCells, numQPs);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void NSTauT<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{ 
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {       
        TauT(cell,qp) = 0.0;
        normGc(cell,qp) = 0.0;
        for (std::size_t i=0; i < numDims; ++i) {
          for (std::size_t j=0; j < numDims; ++j) {
            TauT(cell,qp) += rho(cell,qp) * Cp(cell,qp) * rho(cell,qp) * Cp(cell,qp)* V(cell,qp,i)*Gc(cell,qp,i,j)*V(cell,qp,j);
            normGc(cell,qp) += Gc(cell,qp,i,j)*Gc(cell,qp,i,j);          
          }
        }
        TauT(cell,qp) += 12.*ThermalCond(cell,qp)*ThermalCond(cell,qp)*std::sqrt(normGc(cell,qp));
        TauT(cell,qp) = 1./std::sqrt(TauT(cell,qp));
      }
    }
  

}

//**********************************************************************
}


//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

namespace LCM {

//**********************************************************************
template<typename EvalT, typename Traits>
TrappedConcentration<EvalT, Traits>::
TrappedConcentration(const Teuchos::ParameterList& p) :
  Vmolar       (p.get<std::string>                   ("Molar Volume Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  Clattice       (p.get<std::string>                   ("Lattice Concentration Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  Keq       (p.get<std::string>                   ("Equilibrium Constant Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  Ntrap       (p.get<std::string>                   ("Trapped Solvent Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  Ctrapped      (p.get<std::string>      ("Trapped Concentration Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") )
{
  this->addDependentField(Vmolar);
  this->addDependentField(Keq);
  this->addDependentField(Ntrap);
  this->addDependentField(Clattice);

  this->addEvaluatedField(Ctrapped);

  this->setName("Trapped Concentration"+PHX::TypeString<EvalT>::value);

  Teuchos::RCP<PHX::DataLayout> scalar_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  scalar_dl->dimensions(dims);
  numQPs  = dims[1];
}

//**********************************************************************
template<typename EvalT, typename Traits>
void TrappedConcentration<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(Ctrapped,fm);
  this->utils.setFieldData(Vmolar,fm);
  this->utils.setFieldData(Keq,fm);
  this->utils.setFieldData(Ntrap,fm);
  this->utils.setFieldData(Clattice,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void TrappedConcentration<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Compute Strain tensor from displacement gradient
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {

    	// Nlattice = avogadroNUM(cell,qp)/Vmolar(cell,qp);
    	Nlattice = 1.0/Vmolar(cell,qp);

    	Ctrapped(cell,qp) = Ntrap(cell,qp)*Keq(cell,qp)*Clattice(cell,qp)/
    			            ( Keq(cell,qp)*Clattice(cell,qp) + Nlattice );



    }
  }

}

//**********************************************************************
}


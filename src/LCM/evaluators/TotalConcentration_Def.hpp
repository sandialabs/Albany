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
TotalConcentration<EvalT, Traits>::
TotalConcentration(const Teuchos::ParameterList& p) :
  Clattice       (p.get<std::string>     ("Lattice Concentration Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  Ctrapped      (p.get<std::string>      ("Trapped Concentration Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  Ctotal      (p.get<std::string>      ("Total Concentration Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") )
{

  this->addDependentField(Clattice);
  this->addDependentField(Ctrapped);
  this->addEvaluatedField(Ctotal);

  this->setName("Total Concentration"+PHX::TypeString<EvalT>::value);

  Teuchos::RCP<PHX::DataLayout> scalar_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  scalar_dl->dimensions(dims);
  numQPs  = dims[1];
}

//**********************************************************************
template<typename EvalT, typename Traits>
void TotalConcentration<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(Ctrapped,fm);
  this->utils.setFieldData(Clattice,fm);
  this->utils.setFieldData(Ctotal,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void TotalConcentration<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Compute Strain tensor from displacement gradient
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {

    	Ctotal(cell,qp) = Clattice(cell,qp) + Ctrapped(cell,qp) ;

    }
  }

}

//**********************************************************************
}


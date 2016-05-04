//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Albany_Layouts.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace ATO {

//**********************************************************************
template<typename EvalT, typename Traits>
BodyForce<EvalT, Traits>::
BodyForce(const Teuchos::ParameterList& p) :
  density          (p.get<std::string>                   ("Density Field Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("Density Field Data Layout") ),
  outVector        (p.get<std::string>                   ("Body Force Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("Body Force Data Layout") )
{
  // Pull out numQPs and numDims from a Layout
  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("Body Force Data Layout");
  std::vector<PHX::Device::size_type> dims;
  vector_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  bodyForceVector = p.get<Teuchos::Array<double> >("Body Force Direction");
  double mag = p.get<double>("Body Force Magnitude");
  for(int i=0; i<numDims; i++) bodyForceVector[i] *= mag;

  this->addDependentField(density);
  this->addEvaluatedField(outVector);


  this->setName("BodyForce"+PHX::typeAsString<EvalT>());

}

//**********************************************************************
template<typename EvalT, typename Traits>
void BodyForce<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(density,fm);
  this->utils.setFieldData(outVector,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void BodyForce<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  for (std::size_t cell=0; cell < workset.numCells; ++cell)
    for (std::size_t qp=0; qp < numQPs; ++qp)
      for (std::size_t i=0; i < numDims; ++i)
        outVector(cell,qp,i) = bodyForceVector[i]*density(cell,qp);
}

//**********************************************************************
}


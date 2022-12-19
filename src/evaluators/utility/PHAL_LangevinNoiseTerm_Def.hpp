//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "PHAL_LangevinNoiseTerm.hpp"
#include "Albany_SacadoTypes.hpp"

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace PHAL {


//**********************************************************************
template<typename EvalT, typename Traits>
LangevinNoiseTerm<EvalT, Traits>::
LangevinNoiseTerm(const Teuchos::ParameterList& p) :
  rho        (p.get<std::string>                   ("Rho QP Variable Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  noiseTerm   (p.get<std::string>                ("Langevin Noise Term"),
 	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  duration(2),
  engine(std::random_device()()) // seed the rng
{
  sd = p.get<double>("SD Value");
  duration = p.get<Teuchos::Array<int> >("Langevin Noise Time Period");

  auto vector_dl = p.get< Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  normal_pdf = std::normal_distribution<double>(0.0, Albany::ADValue(sd));

  this->addDependentField(rho.fieldTag());

  this->addEvaluatedField(noiseTerm);

  this->setName("LangevinNoiseTerm" );
}

//**********************************************************************
template<typename EvalT, typename Traits>
void LangevinNoiseTerm<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData /* d */,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(rho,fm);
  this->utils.setFieldData(noiseTerm,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void LangevinNoiseTerm<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  if(duration[0] < 0 || (workset.current_time >= duration[0] && workset.current_time < duration[1])) {

    // Standard deviation as sd about a mean of zero. Perturb the rho solution by this.
    for (std::size_t cell=0; cell < workset.numCells; ++cell)
      for (std::size_t qp=0; qp < numQPs; ++qp)
        noiseTerm(cell, qp) = rho(cell, qp) + normal_pdf(engine);
  }
}

template<typename EvalT, typename Traits>
typename LangevinNoiseTerm<EvalT, Traits>::ScalarT&
LangevinNoiseTerm<EvalT, Traits>::getValue(const std::string &n)
{
  TEUCHOS_TEST_FOR_EXCEPTION(n!="sd", Teuchos::Exceptions::InvalidParameter,
      "Error! [LangevinNoiseTerm::getValue] Invalid parameter name.\n"
      "  - input name: " << n << "\n"
      "  - valid name: sd\n");

  return sd;
}

} // namespace PHAL

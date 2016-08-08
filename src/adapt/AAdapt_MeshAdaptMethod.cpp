//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "AAdapt_MeshAdaptMethod.hpp"

namespace AAdapt {

MeshAdaptMethod::MeshAdaptMethod(
    const Teuchos::RCP<Albany::APFDiscretization>& disc):
    apf_disc(disc),
    mesh_struct(disc->getAPFMeshStruct()),
    commT(disc->getComm())
{
}

void MeshAdaptMethod::setCommonMeshAdaptOptions(
    const Teuchos::RCP<Teuchos::ParameterList>& adapt_params_,
    ma::Input *in) {
  Teuchos::Array<std::string> defaultStArgs =
    Teuchos::tuple<std::string>("zoltan", "parma", "parma");

  double lbMaxImbalance = adapt_params_->get<double>("Maximum LB Imbalance", 1.30);
  in->maximumImbalance = lbMaxImbalance;

  Teuchos::Array<std::string> loadBalancing =
    adapt_params_->get<Teuchos::Array<std::string> >(
        "Load Balancing", defaultStArgs);
  TEUCHOS_TEST_FOR_EXCEPTION(loadBalancing.size() != 3, std::logic_error,
      "parameter \"Load Balancing\" needs to be three strings");

  if (loadBalancing[0] == "zoltan") {
    in->shouldRunPreZoltan = true;
  } else if (loadBalancing[0] == "parma") {
    in->shouldRunPreParma = true;
  } else if (loadBalancing[0] == "none") {
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
        "Unknown \"Load Balancing\" option " << loadBalancing[0] << std::endl);
  }

  if(loadBalancing[1] == "zoltan") {
    in->shouldRunMidZoltan = true;
  } else if(loadBalancing[1] == "parma") {
    in->shouldRunMidParma = true;
  } else if(loadBalancing[1] == "none") {
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
        "Unknown \"Load Balancing\" option " << loadBalancing[1] << std::endl);
  }

  /* we don't use MeshAdapts' "Post" options here, instead this is
   * run manually in AAdapt::MeshAdapt::afterAdapt() for better control
   * of Albany's needs regarding the partitioning.
   */

  in->shouldCoarsen = adapt_params_->get<bool>("Should Coarsen", true);

}

}

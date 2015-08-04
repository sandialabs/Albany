//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "AAdapt_MeshSizeField.hpp"
// Shall we do a case insensitive compare for convenience?
#include <boost/algorithm/string/predicate.hpp>

namespace AAdapt {

MeshSizeField::MeshSizeField(
    const Teuchos::RCP<Albany::APFDiscretization>& disc): 
    mesh_struct(disc->getAPFMeshStruct()),
    commT(disc->getComm())
{
}

void MeshSizeField::setMAInputParams(const Teuchos::RCP<Teuchos::ParameterList>& adapt_params_, ma::Input *in) {

     // Set everything here that should apply to all the size field types

//     Teuchos::Array<std::string> defaultStArgs("Zoltan", "Parma", "Parma");
     Teuchos::Array<std::string> defaultStArgs = 
       Teuchos::tuple<std::string>("zoltan", "parma", "parma");

     double lbMaxImbalance = adapt_params_->get<double>("Maximum LB Imbalance", 1.30);
     in->maximumImbalance = lbMaxImbalance;

     Teuchos::Array<std::string> loadBalancing = adapt_params_->get<Teuchos::Array<std::string> >("Load Balancing", defaultStArgs);
     if (loadBalancing.size() == 3) { // check for error

       if(boost::iequals(loadBalancing[0], "zoltan"))
          in->shouldRunPreZoltan = true;
       else if(boost::iequals(loadBalancing[0], "parma"))
          in->shouldRunPreParma = true;
       else if(boost::iequals(loadBalancing[0], "none"))
          ;
       else
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
         "Error in input for \"Load Balancing\": cannot determine load balancing strategy for pre-adaptation. Found token: " 
         << loadBalancing[0] << std::endl);

       if(boost::iequals(loadBalancing[1], "zoltan"))
          in->shouldRunMidZoltan = true;
       else if(boost::iequals(loadBalancing[1], "parma"))
          in->shouldRunMidParma = true;
       else if(boost::iequals(loadBalancing[1], "none"))
          ;
       else
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
         "Error in input for \"Load Balancing\": cannot determine load balancing strategy for mid-adaptation. Found token: " 
         << loadBalancing[1] << std::endl);

     }
     else
       TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                                   "Error in input for \"Load Balancing\": cannot determine load balancing strategy." << std::endl);

     in->shouldCoarsen = adapt_params_->get<bool>("Should Coarsen", true);

}

}

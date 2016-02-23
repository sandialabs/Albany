//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <fstream>
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "Albany_Utils.hpp"

#include <iostream>
#include <boost/tuple/tuple.hpp>

namespace LCM {


template<typename EvalT, typename Traits>
StabParameter<EvalT, Traits>::
StabParameter(Teuchos::ParameterList& p) :
   stabParameter            (p.get<std::string>("Stabilization Parameter Name"),
		 p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout"))
{
  Teuchos::ParameterList* elmd_list = 
    p.get<Teuchos::ParameterList*>("Parameter List");


  Teuchos::RCP<PHX::DataLayout> vector_dl =
  p.get< Teuchos::RCP<PHX::DataLayout>>("Node QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);

  worksetSize = dims[0];
  numNodes = dims[1];
  numQPs  = dims[2];
  numDims = dims[3];

  Teuchos::RCP<ParamLib> paramLib = 
    p.get< Teuchos::RCP<ParamLib>>("Parameter Library", Teuchos::null);

  std::string type = elmd_list->get("Stabilization Parameter Type", "Constant");
  if (type == "Constant") {
    is_constant = true;
    constant_value = elmd_list->get("Value", 1.0);

    // Add StabParameter as a Sacado-ized parameter
    this->registerSacadoParameter("Stabilization Parameter", paramLib);
  }
  else if (type == "Gradient Dependent") {
    is_constant = false;
    constant_value = elmd_list->get("Value", 1.0);

  }
  else {
	  TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
		       "Invalid stabilization parameter type " << type);
  } 


  // Get additional input to construct adaptive stabilization

  if ( p.isType<std::string>("Gradient QP Variable Name") ) {
	//  is_constant = false;

 //   Teuchos::RCP<PHX::DataLayout> scalar_dl =
 //     p.get< Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout");
 //   PHX::MDField<ScalarT,Cell,QuadPoint>
 //     tmp(p.get<std::string>("QP Pore Pressure Name"), scalar_dl);
 //   porePressure = tmp;
  //  this->addDependentField(porePressure);

	  Teuchos::RCP<PHX::DataLayout> vector_dl =
	        p.get< Teuchos::RCP<PHX::DataLayout>>("QP Vector Data Layout");
	  PHX::MDField<ScalarT,Cell,QuadPoint,Dim>
	        ts(p.get<std::string>("Gradient QP Variable Name"), vector_dl);
	       TGrad = ts;
	  this->addDependentField(TGrad);


  }

  if ( p.isType<std::string>("Gradient BF Name") ) {
	//  is_constant = false;

   //   Teuchos::RCP<PHX::DataLayout> scalar_dl =
   //     p.get< Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout");
   //   PHX::MDField<ScalarT,Cell,QuadPoint>
   //     tmp(p.get<std::string>("QP Pore Pressure Name"), scalar_dl);
   //   porePressure = tmp;
    //  this->addDependentField(porePressure);

  	  Teuchos::RCP<PHX::DataLayout> node_vector_dl =
  	        p.get< Teuchos::RCP<PHX::DataLayout>>("Node QP Vector Data Layout");
  	  PHX::MDField<MeshScalarT,Cell,Node, QuadPoint,Dim>
  	        ts(p.get<std::string>("Gradient BF Name"), node_vector_dl);
  	       GradBF = ts;
  	  this->addDependentField(GradBF);


    }

  if ( p.isType<std::string>("Diffusive Parameter Name") ) {
	  is_constant = false;
       Teuchos::RCP<PHX::DataLayout> scalar_dl =
         p.get< Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout");
       PHX::MDField<ScalarT,Cell,QuadPoint>
         btp(p.get<std::string>("Diffusive Parameter Name"), scalar_dl);
       diffusionParameter = btp;
       this->addDependentField(diffusionParameter);
    }



  this->addEvaluatedField(stabParameter);
  this->setName("Stabilization Parameter"+PHX::typeAsString<EvalT>());
}

// **********************************************************************
template<typename EvalT, typename Traits>
void StabParameter<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(stabParameter,fm);
  if (!is_constant) {
	  this->utils.setFieldData(TGrad,fm);
	  this->utils.setFieldData(GradBF,fm);
	  this->utils.setFieldData(diffusionParameter,fm);
  }


}

// **********************************************************************
template<typename EvalT, typename Traits>
void StabParameter<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  int numCells = workset.numCells;

  ScalarT L2GradT;
  ScalarT UGNparameter;

//  std::cout <<  "Constant? " << is_constant << endl;

  if (is_constant) {
    for (int cell=0; cell < numCells; ++cell) {
      for (int qp=0; qp < numQPs; ++qp) {
    	  stabParameter(cell,qp) = constant_value;
      }
    }
  } else {




    for (int cell=0; cell < numCells; ++cell) {
      for (int qp=0; qp < numQPs; ++qp) {

    	  L2GradT = 0.0;
    	  UGNparameter = 0.0;

    	  		// calculate L2 norm of gradient T
    	  	        for (int dim=0; dim <numDims; ++dim){
    	  		    		 L2GradT += TGrad(cell,qp,dim)*TGrad(cell,qp,dim);
    	  		}


    	  		if (L2GradT > 0.0){

                    L2GradT = std::sqrt(L2GradT);

    	  			for (int node=0; node < numNodes; ++node) {
    	  				for (int dim=0; dim <numDims; ++dim){
    	  					UGNparameter += std::abs(GradBF(cell, node, qp,dim)*TGrad(cell,qp,dim)/L2GradT);

    	  				}
    	  			}
    	  		}
    	  		if ((UGNparameter !=0.0) && (diffusionParameter(cell,qp) != 0.0)){
    	  			UGNparameter = 1.0/UGNparameter;
    	  		}
    	  		//stabParameter(cell,qp) = constant_value*UGNparameter*UGNparameter*diffusionParameter(cell,qp);
    	  		stabParameter(cell,qp) = constant_value;
    	  		// std::cout <<  "stabilization parameter value " << UGNparameter << endl;




    	  				 /* biotCoefficient(cell,qp)*strain(cell,qp,i,i)
    	  				             + porePressure(cell,qp)
    	  				             *(biotCoefficient(cell,qp)-initialPorosity_value)/GrainBulkModulus; */


//    	  	// for debug
//    	  	std::cout << "initial Porosity: " << initialPorosity_value << endl;
//    	  	std::cout << "Pore Pressure: " << porePressure << endl;
//    	  	std::cout << "Biot Coefficient: " << biotCoefficient << endl;
//    	  	std::cout << "Grain Bulk Modulus " << GrainBulkModulus << endl;

//			porosity(cell,qp) += (1.0 - initialPorosity_value)
//								  /GrainBulkModulus*porePressure(cell,qp);
    	  // for large deformation, \phi = J \dot \phi_{o}
      }
    }

  }
}




// **********************************************************************
template<typename EvalT,typename Traits>
typename StabParameter<EvalT,Traits>::ScalarT&
StabParameter<EvalT,Traits>::getValue(const std::string &n)
{
  if (n == "Stabilization Parameter")
    return constant_value;


  TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
		     std::endl <<
		     "Error! Logic error in getting parameter " << n
		     << " in Porosity::getValue()" << std::endl);
  return constant_value;
}

// **********************************************************************
// **********************************************************************
}


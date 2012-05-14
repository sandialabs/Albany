/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


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
		 p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout"))
{
  Teuchos::ParameterList* elmd_list = 
    p.get<Teuchos::ParameterList*>("Parameter List");


  Teuchos::RCP<PHX::DataLayout> vector_dl =
  p.get< Teuchos::RCP<PHX::DataLayout> >("Node QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);

  worksetSize = dims[0];
  numNodes = dims[1];
  numQPs  = dims[2];
  numDims = dims[3];

  Teuchos::RCP<ParamLib> paramLib = 
    p.get< Teuchos::RCP<ParamLib> >("Parameter Library", Teuchos::null);

  std::string type = elmd_list->get("Stabilization Parameter Type", "Constant");
  if (type == "Constant") {
    is_constant = true;
    constant_value = elmd_list->get("Value", 1.0);

    // Add StabParameter as a Sacado-ized parameter
    new Sacado::ParameterRegistration<EvalT, SPL_Traits>(
	"Stabilization Parameter", this, paramLib);
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

  if ( p.isType<string>("Gradient QP Variable Name") ) {
	//  is_constant = false;

 //   Teuchos::RCP<PHX::DataLayout> scalar_dl =
 //     p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
 //   PHX::MDField<ScalarT,Cell,QuadPoint>
 //     tmp(p.get<string>("QP Pore Pressure Name"), scalar_dl);
 //   porePressure = tmp;
  //  this->addDependentField(porePressure);

	  Teuchos::RCP<PHX::DataLayout> vector_dl =
	        p.get< Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout");
	  PHX::MDField<ScalarT,Cell,QuadPoint,Dim>
	        ts(p.get<string>("Gradient QP Variable Name"), vector_dl);
	       TGrad = ts;
	  this->addDependentField(TGrad);


  }

  if ( p.isType<string>("Gradient BF Name") ) {
	//  is_constant = false;

   //   Teuchos::RCP<PHX::DataLayout> scalar_dl =
   //     p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
   //   PHX::MDField<ScalarT,Cell,QuadPoint>
   //     tmp(p.get<string>("QP Pore Pressure Name"), scalar_dl);
   //   porePressure = tmp;
    //  this->addDependentField(porePressure);

  	  Teuchos::RCP<PHX::DataLayout> node_vector_dl =
  	        p.get< Teuchos::RCP<PHX::DataLayout> >("Node QP Vector Data Layout");
  	  PHX::MDField<MeshScalarT,Cell,Node, QuadPoint,Dim>
  	        ts(p.get<string>("Gradient BF Name"), node_vector_dl);
  	       GradBF = ts;
  	  this->addDependentField(GradBF);


    }

  if ( p.isType<string>("Diffusive Parameter Name") ) {
	  is_constant = false;
       Teuchos::RCP<PHX::DataLayout> scalar_dl =
         p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
       PHX::MDField<ScalarT,Cell,QuadPoint>
         btp(p.get<string>("Diffusive Parameter Name"), scalar_dl);
       diffusionParameter = btp;
       this->addDependentField(diffusionParameter);
    }



  this->addEvaluatedField(stabParameter);
  this->setName("Stabilization Parameter"+PHX::TypeString<EvalT>::value);
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
  std::size_t numCells = workset.numCells;

  ScalarT L2GradT;
  ScalarT UGNparameter;

//  std::cout <<  "Constant? " << is_constant << endl;

  if (is_constant) {
    for (std::size_t cell=0; cell < numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
    	  stabParameter(cell,qp) = constant_value;
      }
    }
  } else {




    for (std::size_t cell=0; cell < numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {

    	  L2GradT = 0.0;
    	  UGNparameter = 0.0;



    	  		 // calculate L2 norm of gradient T
    	  		 for (std::size_t dim=0; dim <numDims; ++dim){
    	  		    		  L2GradT += TGrad(cell,qp,dim)*TGrad(cell,qp,dim);
    	  		 }
    	  		 L2GradT = std::sqrt(L2GradT);

    	  		if (L2GradT != 0.0){

    	  			for (std::size_t node=0; node < numNodes; ++node) {
    	  				for (std::size_t dim=0; dim <numDims; ++dim){
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


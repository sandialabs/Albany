//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <fstream>
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"

template<typename EvalT, typename Traits>
QCAD::Permittivity<EvalT, Traits>::
Permittivity(Teuchos::ParameterList& p,
             const Teuchos::RCP<Albany::Layouts>& dl) :
  permittivity(p.get<std::string>("QP Variable Name"), dl->qp_scalar),
  temp_dependent(false), position_dependent(false)
{
	Teuchos::ParameterList* perm_list = 
    	p.get<Teuchos::ParameterList*>("Parameter List");

  Teuchos::RCP<const Teuchos::ParameterList> reflist = 
  		this->getValidPermittivityParameters();
  perm_list->validateParameters(*reflist,0);

  std::vector<PHX::DataLayout::size_type> dims;
  dl->qp_vector->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  // Material database
  materialDB = p.get< Teuchos::RCP<QCAD::MaterialDatabase> >("MaterialDB");

  // Permittivity type
  typ = perm_list->get("Permittivity Type", "Constant");
  
  // Permittivity (relative) value is constant
  if (typ == "Constant") 
  {
    position_dependent = false;
    temp_dependent = false;

    constant_value = perm_list->get("Value", 1.0);

    // Add Permittivity as a Sacado-ized parameter
    Teuchos::RCP<ParamLib> paramLib = 
      	p.get< Teuchos::RCP<ParamLib> >("Parameter Library", Teuchos::null);
    this->registerSacadoParameter("Permittivity", paramLib);

  }
  
  // Permittivity (relative) has temperature dependence
  else if (typ == "Temperature Dependent") 
  {
    position_dependent = false;
    temp_dependent = true;

    constant_value = perm_list->get("Value", 1.0);
    factor = perm_list->get("Factor", 1.0);

    // Add Permittivity as a Sacado-ized parameter
    Teuchos::RCP<ParamLib> paramLib = 
      	p.get< Teuchos::RCP<ParamLib> >("Parameter Library", Teuchos::null);
    this->registerSacadoParameter("Permittivity", paramLib);
    this->registerSacadoParameter("Permittivity Factor", paramLib);

  }

  else if (typ == "Block Dependent") 
  {
    // So far *not* position dependent, since we don't need the coordinate vector (just the block name)
    // However, this would be set to true if there is a spatial distribution of permittivity within a single block
    position_dependent = false; 
    temp_dependent = false; 
  }
  
  // for testing 1D MOSCapacitor
  else if (typ == "Position Dependent")
  {
    position_dependent = true; 
    temp_dependent = false; 
    oxideWidth = perm_list->get("Oxide Width", 0.);
    siliconWidth = perm_list->get("Silicon Width", 0.);
  }

  // Parse for other functional form for permittivity variation here
  // This effectively validates the 2nd argument in perm_list->get (...);
  else 
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
		       "Invalid Permittivity Type: " << typ);
  } 


  // Add coordinate dependence to permittivity evaluator
  if(position_dependent) {
    PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>
      	tmp(p.get<std::string>("Coordinate Vector Name"), dl->qp_vector);
    coordVec = tmp;
    this->addDependentField(coordVec);
  }

  // Add temperature dependence to permittivity evaluator
  if(temp_dependent) {
    PHX::MDField<ScalarT,Cell,QuadPoint>
      	tmp(p.get<std::string>("Temperature Variable Name"), dl->qp_scalar);
    Temp = tmp;
    this->addDependentField(Temp);
  }

  this->addEvaluatedField(permittivity);
  this->setName( "Permittivity" + PHX::typeAsString<EvalT>() );
}

// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::Permittivity<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(permittivity,fm);
  if (position_dependent) this->utils.setFieldData(coordVec,fm);
  if (temp_dependent) this->utils.setFieldData(Temp,fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::Permittivity<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // assign constant_value to permittivity defined in .hpp file
  if (typ == "Constant") 
  {	
    for (std::size_t cell=0; cell < workset.numCells; ++cell) 
    {	
      for (std::size_t qp=0; qp < numQPs; ++qp) 
      {
        permittivity(cell,qp) = constant_value;
      }
    }
  }
  
  // calculate temp-dep value and fill in the permittivity field
  else if (typ == "Temperature Dependent") 
  {
    for (std::size_t cell=0; cell < workset.numCells; ++cell) 
    {
      for (std::size_t qp=0; qp < numQPs; ++qp) 
      {
        ScalarT denom = 1.0 + factor * Temp(cell,qp);
        permittivity(cell,qp) = constant_value / denom;;
      }
    }
  }

  else if (typ == "Block Dependent") 
  {	
    const ScalarT& value = materialDB->getElementBlockParam<double>(workset.EBName,"Permittivity");

    // loop through all elements in one workset
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {	
      // loop through the QPs for each element
      for (std::size_t qp=0; qp < numQPs; ++qp)
        permittivity(cell,qp) = value;
    }
  }
  
  // for testing 1D MOSCapacitor
  else if (typ == "Position Dependent")
  {
    for (std::size_t cell=0; cell < workset.numCells; ++cell) 
    {	
      for (std::size_t qp=0; qp < numQPs; ++qp)
      {
        MeshScalarT coord0 = coordVec(cell,qp,0);
        
        // Silicon region
        if ( (coord0 > oxideWidth) && (coord0 <= (oxideWidth + siliconWidth)) )
        {
          const std::string matName = "Silicon";
          const ScalarT& value = materialDB->getMaterialParam<double>(matName,"Permittivity");
          permittivity(cell,qp) = value;
        }
        
        // Oxide region
        else if ((coord0 >= 0) && (coord0 <= oxideWidth))
        {
          const std::string matName = "SiliconDioxide";
          const ScalarT& value = materialDB->getMaterialParam<double>(matName,"Permittivity");
          permittivity(cell,qp) = value;
        }
        
        else
          TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
	          std::endl << "Error!  x-coord:" << coord0 << "is outside the oxideWidth" << 
	          " + siliconWidth range: " << oxideWidth + siliconWidth << "!"<< std::endl);
        
      }  // end of loop over QPs
    }  // end of loop over cells
  }
  
  // otherwise, throw out error message and exit the program
  else 
  {
    std::cout << "Error: permittivity has to be either constant, " <<  
    		"block dependent, or temperature dependent !" << std::endl;
    exit(1);
  } 
  
}

// **********************************************************************
template<typename EvalT,typename Traits>
typename QCAD::Permittivity<EvalT,Traits>::ScalarT& 
QCAD::Permittivity<EvalT,Traits>::getValue(const std::string &n)
{
	// constant permittivity, n must match the value string in the "Parameters"
	// section of the input.xml file as parameters for response analysis 
  if (n == "Permittivity")	
    return constant_value;
  
  // temperature factor used in the temp-dep permittivity calculation
  else if (n == "Permittivity Factor")
    return factor;
  
  // otherwise, throw out error message and continue the program
  else 
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
				std::endl <<
				"Error! Logic error in getting paramter " << n <<
				" in Permittivity::getValue()" << std::endl);
    return constant_value;
  }
}

// **********************************************************************
template<typename EvalT,typename Traits>
Teuchos::RCP<const Teuchos::ParameterList>
QCAD::Permittivity<EvalT,Traits>::getValidPermittivityParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
     	rcp(new Teuchos::ParameterList("Valid Permittivty Params"));;

  validPL->set<std::string>("Permittivity Type", "Constant", 
  		"Constant permittivity in the entire device");
  validPL->set<double>("Value", 1.0, "Constant permittivity value");
  validPL->set<double>("Factor", 1.0, "Permittivity temperature factor");

  validPL->set<double>("Oxide Width", 0., "Oxide width for 1D MOSCapacitor device");
  validPL->set<double>("Silicon Width", 0., "Silicon width for 1D MOSCapacitor device");

  return validPL;
}

// **********************************************************************
// **********************************************************************


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

template<typename EvalT, typename Traits>
QCAD::Permittivity<EvalT, Traits>::
Permittivity(Teuchos::ParameterList& p) :
  permittivity(p.get<std::string>("QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")),
  temp_dependent(false), position_dependent(false)
{
	Teuchos::ParameterList* perm_list = 
    	p.get<Teuchos::ParameterList*>("Parameter List");

  Teuchos::RCP<const Teuchos::ParameterList> reflist = 
  		this->getValidPermittivityParameters();
  perm_list->validateParameters(*reflist,0);

  Teuchos::RCP<PHX::DataLayout> scalar_dl =
    	p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
  Teuchos::RCP<PHX::DataLayout> vector_dl =
    	p.get< Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  typ = perm_list->get("Permittivity Type", "Constant");
  
  // Permittivity (relative) value is constant
  if (typ == "Constant") 
  {
    constant_value = perm_list->get("Value", 1.0);

    // Add Permittivity as a Sacado-ized parameter
    Teuchos::RCP<ParamLib> paramLib = 
      	p.get< Teuchos::RCP<ParamLib> >("Parameter Library", Teuchos::null);
    new Sacado::ParameterRegistration<EvalT, SPL_Traits>(
				"Permittivity", this, paramLib);
  }
  
  // Permittivity (relative) has position dependence 
  else if (typ == "Position Dependent") 
  {
  	position_dependent = true;
    silicon_value = perm_list->get("Silicon Value", 1.0);
    oxide_value = perm_list->get("Oxide Value", 1.0);
  	
  	// Add coordinate dependence to permittivity evaluator
    PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>
      	tmp(p.get<string>("Coordinate Vector Name"), vector_dl);
    coordVec = tmp;
    this->addDependentField(coordVec);

    // Add Silicon and Oxide Permittivity as Sacado-ized parameters
    Teuchos::RCP<ParamLib> paramLib = 
      	p.get< Teuchos::RCP<ParamLib> >("Parameter Library", Teuchos::null);
    new Sacado::ParameterRegistration<EvalT, SPL_Traits>(
				"Silicon Permittivity", this, paramLib);
    new Sacado::ParameterRegistration<EvalT, SPL_Traits>(
				"Oxide Permittivity", this, paramLib);
  }
  
  // Permittivity (relative) has temperature dependence
  else if (typ == "Temperature Dependent") 
  {
    temp_dependent = true;
    constant_value = perm_list->get("Value", 1.0);
    factor = perm_list->get("Factor", 1.0);

		// Add temperature dependence to permittivity evaluator
    PHX::MDField<ScalarT,Cell,QuadPoint>
      	tmp(p.get<string>("Temperature Variable Name"), scalar_dl);
    Temp = tmp;
    this->addDependentField(Temp);

    // Add Permittivity as a Sacado-ized parameter
    Teuchos::RCP<ParamLib> paramLib = 
      	p.get< Teuchos::RCP<ParamLib> >("Parameter Library", Teuchos::null);
    new Sacado::ParameterRegistration<EvalT, SPL_Traits>(
				"Permittivity", this, paramLib);
    new Sacado::ParameterRegistration<EvalT, SPL_Traits>(
				"Permittivity Factor", this, paramLib);
  }

  else if (typ == "Block Dependent") 
  {
    constant_value = perm_list->get("Value", 1.0);
    silicon_value = perm_list->get("Silicon Value", 1.0);
    oxide_value = perm_list->get("Oxide Value", 1.0);
    poly_value = perm_list->get("Poly Value", 1.0);

    // Add Material Permittivities as Sacado-ized parameters
    Teuchos::RCP<ParamLib> paramLib = 
      	p.get< Teuchos::RCP<ParamLib> >("Parameter Library", Teuchos::null);
    new Sacado::ParameterRegistration<EvalT, SPL_Traits>(
				"Silicon Permittivity", this, paramLib);
    new Sacado::ParameterRegistration<EvalT, SPL_Traits>(
				"Oxide Permittivity", this, paramLib);
    new Sacado::ParameterRegistration<EvalT, SPL_Traits>(
				"Poly Permittivity", this, paramLib);
  }

  
  // Parse for other functional form for permittivity variation here
  // This effectively validates the 2nd argument in perm_list->get (...);
  else 
  {
    TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
		       "Invalid Permittivity Type: " << typ);
  } 

  this->addEvaluatedField(permittivity);
  this->setName("Permittivity"+PHX::TypeString<EvalT>::value);
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
  // assign silicon_value for y>0 and oxide_value for y<=0 to permittivity
  else if (typ == "Position Dependent") 
  {	
  	// loop through all elements in one workset
    for (std::size_t cell=0; cell < workset.numCells; ++cell) 
    {	
    	// loop through the QPs for each element
      for (std::size_t qp=0; qp < numQPs; ++qp) 
      {
	if (coordVec(cell,qp,1) > 0.0) //3rd argument: 0 for x, 1 for y, 2 for z
	  permittivity(cell,qp) = silicon_value;
	else
	  permittivity(cell,qp) = oxide_value;
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
    ScalarT value;

    if(workset.EBName == "silicon" || workset.EBName == "nsilicon" || workset.EBName == "psilicon")
      value = silicon_value;
    else if(workset.EBName == "sio2") value = oxide_value;
    else if(workset.EBName == "poly") value = poly_value;
    else {
      TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
			 std::endl << "Error!  Unknown element block name "
			 << workset.EBName << "!" << std::endl);
    }

    // loop through all elements in one workset
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {	
      // loop through the QPs for each element
      for (std::size_t qp=0; qp < numQPs; ++qp)
	permittivity(cell,qp) = value;
    }
  }
  
  // otherwise, throw out error message and exit the program
  else 
  {
    std::cout << "Error: permittivity has to be either constant, " <<  
    		"position dependent, or temperature dependent !" << endl;
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
  
  // register the Silicon and Oxide permittivity for position-dep. case
  else if (n == "Silicon Permittivity")
  	return silicon_value;
  	
  else if (n == "Oxide Permittivity")
  	return oxide_value;

  else if (n == "Poly Permittivity")
        return poly_value;

  // temperature factor used in the temp-dep permittivity calculation
  else if (n == "Permittivity Factor")
    return factor;
  
  // otherwise, throw out error message and continue the program
  else 
  {
    TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
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

  validPL->set<string>("Permittivity Type", "Constant", 
  		"Constant permittivity in the entire device");
  validPL->set<double>("Value", 1.0, "Constant permittivity value");
  validPL->set<double>("Factor", 1.0, "Permittivity temperature factor");
  validPL->set<double>("Silicon Value", 1.0, "Silicon permittivity value");
  validPL->set<double>("Oxide Value", 1.0, "SiO2 permittivity value");
  validPL->set<double>("Poly Value", 1.0, "Poly-silicon permittivity value");

  return validPL;
}

// **********************************************************************
// **********************************************************************


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
QCAD::SchrodingerPotential<EvalT, Traits>::
SchrodingerPotential(Teuchos::ParameterList& p) :
  coordVec(p.get<std::string>("QP Coordinate Vector Name"),
     p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout")),
  psi(p.get<std::string>("QP Variable Name"),
      p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")),
  V(p.get<std::string>("QP Potential Name"),
      p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout"))
{
  Teuchos::ParameterList* psList = p.get<Teuchos::ParameterList*>("Parameter List");

  Teuchos::RCP<const Teuchos::ParameterList> reflist = 
      this->getValidSchrodingerPotentialParameters();
  psList->validateParameters(*reflist,0);

  Teuchos::RCP<PHX::DataLayout> vector_dl =
      p.get< Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  energy_unit_in_eV = p.get<double>("Energy unit in eV");
  length_unit_in_m = p.get<double>("Length unit in m");
  potentialType = psList->get("Type", "defaultType");
  E0 = psList->get("E0", 1.0);
  scalingFactor = psList->get("Scaling Factor", 1.0);

  // Add E0 as a Sacado-ized parameter
  Teuchos::RCP<ParamLib> paramLib =
      p.get< Teuchos::RCP<ParamLib> >("Parameter Library", Teuchos::null);
  new Sacado::ParameterRegistration<EvalT, SPL_Traits>(
      "Schrodinger Potential E0", this, paramLib);
  new Sacado::ParameterRegistration<EvalT, SPL_Traits> (
      "Schrodinger Potential Scaling Factor", this, paramLib);

  this->addDependentField(psi);
  this->addDependentField(coordVec);

  this->addEvaluatedField(V);
  this->setName("Schrodinger Potential"+PHX::TypeString<EvalT>::value);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::SchrodingerPotential<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(V,fm);
  this->utils.setFieldData(psi,fm);
  this->utils.setFieldData(coordVec,fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::SchrodingerPotential<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  for (std::size_t cell=0; cell < workset.numCells; ++cell) 
  {
    for (std::size_t qp=0; qp < numQPs; ++qp) 
    {
      V(cell, qp) = potentialValue(numDims, &coordVec(cell,qp,0));
    }
  }
}

// **********************************************************************
template<typename EvalT,typename Traits>
typename QCAD::SchrodingerPotential<EvalT,Traits>::ScalarT& 
QCAD::SchrodingerPotential<EvalT,Traits>::getValue(const std::string &n)
{
  if(n == "Schrodinger Potential Scaling Factor") return scalingFactor;
  else if(n == "Schrodinger Potential E0") return E0;
  else TEST_FOR_EXCEPT(true); return E0; //dummy so all control paths return
}

// **********************************************************************
template<typename EvalT,typename Traits>
Teuchos::RCP<const Teuchos::ParameterList>
QCAD::SchrodingerPotential<EvalT,Traits>::getValidSchrodingerPotentialParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
       rcp(new Teuchos::ParameterList("Valid Schrodinger Potential Params"));;

  validPL->set<string>("Type", "defaultType", "Switch between different potential types");
  validPL->set<double>("E0", 1.0, "Energy scale - dependent on type");
  validPL->set<double>("Scaling Factor", 1.0, "Constant scaling factor");

  return validPL;
}

// **********************************************************************

//Return potential in energy_unit_in_eV * eV units
template<typename EvalT,typename Traits>
typename QCAD::SchrodingerPotential<EvalT,Traits>::ScalarT
QCAD::SchrodingerPotential<EvalT,Traits>::potentialValue(
    const int numDim, const MeshScalarT* coord)
{
  ScalarT val, r2;

  //std::cout << "x = " << coord[0] << endl; //in 1D, x-coords run from zero to 1

  /***** define universal constants as double constants *****/
  const double hbar = 1.0546e-34;  // Planck constant [J s]
  const double emass = 9.1094e-31; // Electron mass [kg]
  const double evPerJ = 6.2415e18; // eV per Joule (J/eV)

  const double parabolicFctr = 0.5 * (emass / (evPerJ * hbar*hbar) );

  ScalarT prefactor;  // prefactor from constant, including scaling due to units
  
  /***** implement RHS of the scaled Poisson equation in pn diode *****/
  if (potentialType == "Parabolic") 
  {
    prefactor = parabolicFctr * E0*E0 * (energy_unit_in_eV * pow(length_unit_in_m,2));  
    switch (numDim) 
    {
      case 1: 
	r2 = (coord[0]-0.5)*(coord[0]-0.5);
	val = prefactor * r2;
	break;
      case 2:
	r2 = (coord[0]-0.5)*(coord[0]-0.5) + (coord[1]-0.5)*(coord[1]-0.5);
	val = prefactor * r2;
        break;
      case 3:
	r2 = (coord[0]-0.5)*(coord[0]-0.5) + (coord[1]-0.5)*(coord[1]-0.5) 
	   + (coord[2]-0.5)*(coord[2]-0.5);
	val = prefactor * r2;
	break;
      default:
        TEST_FOR_EXCEPT(true);
    }  // end of switch(numDim)
  }
  
  /***** otherwise error? ******/
  else 
  {
    TEST_FOR_EXCEPT(true);
  }

  return scalingFactor * val;
}


// *****************************************************************************

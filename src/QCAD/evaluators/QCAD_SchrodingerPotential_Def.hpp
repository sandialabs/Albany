//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <fstream>
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "QCAD_StringFormulaEvaluator.hpp"

template<typename EvalT, typename Traits>
QCAD::SchrodingerPotential<EvalT, Traits>::
SchrodingerPotential(Teuchos::ParameterList& p,
                 const Teuchos::RCP<Albany::Layouts>& dl) :
  coordVec(p.get<std::string>("QP Coordinate Vector Name"), dl->qp_gradient),
  psi(p.get<std::string>("QP Variable Name"), dl->qp_scalar),
  V(p.get<std::string>("QP Potential Name"), dl->qp_scalar)
{
  // psList == "Potential" sublist of main input file
  Teuchos::ParameterList* psList = p.get<Teuchos::ParameterList*>("Parameter List");

  Teuchos::RCP<const Teuchos::ParameterList> reflist = 
      this->getValidSchrodingerPotentialParameters();
  psList->validateParameters(*reflist,0);

  std::vector<PHX::DataLayout::size_type> dims;
  dl->qp_gradient->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  energy_unit_in_eV = p.get<double>("Energy unit in eV");
  length_unit_in_m = p.get<double>("Length unit in m");
  
  potentialType = psList->get("Type", "defaultType");
  E0 = psList->get("E0", 1.0);
  scalingFactor = psList->get("Scaling Factor", 1.0);
  
  // Parameters for String Formula
  stringFormula = psList->get("Formula", "0");

  // Parameters for Finite Wall 
  barrEffMass = psList->get<double>("Barrier Effective Mass", 0.0);
  barrWidth = psList->get<double>("Barrier Width", 0.0);
  wellEffMass = psList->get<double>("Well Effective Mass", 0.0);
  wellWidth = psList->get<double>("Well Width", 0.0);
  
  potentialStateName = psList->get<std::string>("State Name","Not Specified");

  // Add E0 as a Sacado-ized parameter
  Teuchos::RCP<ParamLib> paramLib =
      p.get< Teuchos::RCP<ParamLib> >("Parameter Library", Teuchos::null);

  this->registerSacadoParameter("Schrodinger Potential E0", paramLib);
  this->registerSacadoParameter("Schrodinger Potential Scaling Factor", paramLib);

  this->addDependentField(psi);
  this->addDependentField(coordVec);

  this->addEvaluatedField(V);
  this->setName( "Schrodinger Potential" + PHX::typeAsString<EvalT>() );
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

  // Parabolic potential (test case)
  if (potentialType == "Parabolic") 
  {
    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t qp = 0; qp < numQPs; ++qp) {
        V(cell, qp) = parabolicPotentialValue(numDims,coordVec,cell,qp);
      }
    }
  }
  
  // Infinite barrier wall
  else if (potentialType == "Infinite Wall")
  {
    for (std::size_t cell = 0; cell < workset.numCells; ++cell)
      for (std::size_t qp = 0; qp < numQPs; ++qp)
        V(cell, qp) = 0.0; 
  } 

  // Finite barrier wall
  else if (potentialType == "Finite Wall")
  {
    for (std::size_t cell = 0; cell < workset.numCells; ++cell)
    {
      for (std::size_t qp = 0; qp < numQPs; ++qp)
        V(cell, qp) = finiteWallPotential(numDims,coordVec,cell,qp);
    }    
  }

  // Specifed by formula string
  else if (potentialType == "String Formula")
  {
    for (std::size_t cell = 0; cell < workset.numCells; ++cell)
    {
      for (std::size_t qp = 0; qp < numQPs; ++qp)
        V(cell, qp) = stringFormulaPotential(numDims,coordVec,cell,qp);
    }    
  }

  
  // Potential energy taken from a StateManager state
  else if (potentialType == "From State") 
  {
    Albany::StateArray& states = *workset.stateArrayPtr;
    Albany::MDArray& potentialState = states[potentialStateName];

    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
        double d =  potentialState(cell, qp);
        V(cell, qp) = d; //scalingFactor * d;

	//ANDY: behavior I don't understand - scalingFactor gets set to 1000?
	//if( scalingFactor != 1.0 ) 
	//  std::cout << "DEBUG: scaling factor = " << scalingFactor << std::endl;
	//HACK to help anasazi solve
	//if(workset.EBName == "silicon" || scalingFactor < 0) {
	//  V(cell, qp) =  d;
	//}
	//else V(cell, qp) = scalingFactor;
      }
    }
  }    

  /***** otherwise error? ******/
  else 
  {
    TEUCHOS_TEST_FOR_EXCEPT(true);
  }
}

// **********************************************************************
template<typename EvalT,typename Traits>
typename QCAD::SchrodingerPotential<EvalT,Traits>::ScalarT& 
QCAD::SchrodingerPotential<EvalT,Traits>::getValue(const std::string &n)
{
  if(n == "Schrodinger Potential Scaling Factor") return scalingFactor;
  else if(n == "Schrodinger Potential E0") return E0;
  else TEUCHOS_TEST_FOR_EXCEPT(true); return E0; //dummy so all control paths return
}

// **********************************************************************
template<typename EvalT,typename Traits>
Teuchos::RCP<const Teuchos::ParameterList>
QCAD::SchrodingerPotential<EvalT,Traits>::getValidSchrodingerPotentialParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
       rcp(new Teuchos::ParameterList("Valid Schrodinger Potential Params"));;

  validPL->set<std::string>("Type", "defaultType", "Switch between different potential types");
  validPL->set<double>("E0", 1.0, "Energy scale - dependent on type");
  validPL->set<double>("Scaling Factor", 1.0, "Constant scaling factor");

  // For string formula potential
  validPL->set<std::string>("Formula", "0", "Mathematical expression containing x,y,z that specifies the potential at coordinates (x,y,z).  In 1D and 2D use just x and x,y respectively.");
  
  // For Finite Wall potential 
  validPL->set<double>("Barrier Effective Mass", 0.0, "Barrier effective mass in [m0]");
  validPL->set<double>("Barrier Width", 0.0, "Barrier width in length_unit_in_m");
  validPL->set<double>("Well Effective Mass", 0.0, "Well effective mass in [m0]");
  validPL->set<double>("Well Width", 0.0, "Well width in length_unit_in_m");

  // For 1D MOSCapacitor to test the 1D P-S iteration
  /* specific parameters for 1D MOSCapacitor to set correct effective mass 
     for oxide and silicon regions (could be given in the Poisson Coupling
     section, putting here for less perturbation to the main code). */

  validPL->set<double>("Oxide Width", 0.0, "Oxide width in length_unit_in_m");
  validPL->set<double>("Silicon Width", 0.0, "Silicon width in length_unit_in_m");

  // For potential imported from a state
  validPL->set<std::string>("State Name", "", "Name of state to import potential from");

  return validPL;
}

// **********************************************************************

//Return potential in energy_unit_in_eV * eV units
template<typename EvalT,typename Traits>
typename QCAD::SchrodingerPotential<EvalT,Traits>::ScalarT
QCAD::SchrodingerPotential<EvalT,Traits>::parabolicPotentialValue(
   const int numDim, const PHX::MDField<const MeshScalarT,Cell,QuadPoint,Dim> & coord,
   const int cell, const int qp )
{
  ScalarT val;
  MeshScalarT r2;
  int i;

  //std::cout << "x = " << coord[0] << endl; //in 1D, x-coords run from zero to 1

  /***** define universal constants as double constants *****/
  const double hbar = 1.0546e-34;  // Planck constant [J s]
  const double emass = 9.1094e-31; // Electron mass [kg]
  const double evPerJ = 6.2415e18; // eV per Joule (J/eV)

  const double parabolicFctr = 0.5 * (emass / (evPerJ * hbar*hbar) );

  // prefactor from constant, including scaling due to units
  ScalarT prefactor;  
  
  prefactor = parabolicFctr * E0*E0 * (energy_unit_in_eV * pow(length_unit_in_m,2));  
  for(i=0, r2=0.0; i<numDim; i++)
    r2 += (coord(cell,qp,i)-0.5)*(coord(cell,qp,i)-0.5);
  val = prefactor * r2;  
  
  return scalingFactor * val;
}

// **********************************************************************

//Return potential in energy_unit_in_eV * eV units
template<typename EvalT,typename Traits>
typename QCAD::SchrodingerPotential<EvalT,Traits>::ScalarT
QCAD::SchrodingerPotential<EvalT,Traits>::finiteWallPotential(
   const int numDim, const PHX::MDField<const MeshScalarT,Cell,QuadPoint,Dim> & coord,
   const int cell, const int qp )
{
  ScalarT val;
  
  switch (numDim)
  {
    case 1: // 1D: total width = 2*barrWidth + wellWidth
    {
      if ( (coord(cell,qp,0) >= 0) && (coord(cell,qp,0) < barrWidth) )
        val = E0;  // barrier
      else if ( (coord(cell,qp,0) >= barrWidth) && (coord(cell,qp,0) <= (barrWidth+wellWidth)) )
        val = 0;   // well
      else if ( (coord(cell,qp,0) > (barrWidth+wellWidth)) && (coord(cell,qp,0) <= (2*barrWidth+wellWidth)) )
        val = E0;  // barrier
      else 
        TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, std::endl 
			    << "Error! x coordinate is outside [0, 2*barrWidth+wellWidth] range,"
			    << " make sure 1D Scale in Discretization equal to 2*barrWidth+wellWidth !" << std::endl);
      break;
    }
    
    case 2: // 2D
    {
      if ( (coord(cell,qp,0) >= barrWidth) && (coord(cell,qp,0) <= (barrWidth+wellWidth)) &&
           (coord(cell,qp,1) >= barrWidth) && (coord(cell,qp,1) <= (barrWidth+wellWidth)) )
        val = 0.0;  // well
      else
        val = E0;   // barrier
      break;
    }

    case 3: // 3D
    {
      if ( (coord(cell,qp,0) >= barrWidth) && (coord(cell,qp,0) <= (barrWidth+wellWidth)) &&
           (coord(cell,qp,1) >= barrWidth) && (coord(cell,qp,1) <= (barrWidth+wellWidth)) && 
           (coord(cell,qp,2) >= barrWidth) && (coord(cell,qp,2) <= (barrWidth+wellWidth)) )
        val = 0.0;  // well
      else
        val = E0;   // barrier
      break;
    }
    
    default: 
    {
      TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, std::endl 
			  << "Error! Invalid numDim = " << numDim << ", must be 1 or 2 or 3 !" << std::endl);
			break;  
    }
    
  }  // end of switch (numDim)
  
  return scalingFactor * val;
}


// **********************************************************************

//Return potential in energy_unit_in_eV * eV units
template<typename EvalT,typename Traits>
typename QCAD::SchrodingerPotential<EvalT,Traits>::ScalarT
QCAD::SchrodingerPotential<EvalT,Traits>::stringFormulaPotential(
   const int numDim, const PHX::MDField<const MeshScalarT,Cell,QuadPoint,Dim> & coord,
   const int cell, const int qp )
{
  ScalarT val;
  MeshScalarT x, y, z, result;
  
  switch (numDim)
  {
    case 1: // 1D: total width = 2*barrWidth + wellWidth
    {
      x = coord(cell,qp,0);
      y = 0; z=0;
      break;
    }
    
    case 2: // 2D
    {
      x = coord(cell,qp,0);
      y = coord(cell,qp,1);
      z = 0;
      break;
    }

    case 3: // 3D
    {
      x = coord(cell,qp,0);
      y = coord(cell,qp,1);
      z = coord(cell,qp,2);
      break;
    }
    
    default: 
    {
      TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, std::endl 
			  << "Error! Invalid numDim = " << numDim << ", must be 1 or 2 or 3 !" << std::endl);
			break;  
    }    
  }  // end of switch (numDim)
  
  result = Evaluate(stringFormula,x,y,z); //Parse and evaluate formula string 
  val = result;                                // (could speed this up in future by saving intermediate parse info)
  return scalingFactor * val;
}


// *****************************************************************************


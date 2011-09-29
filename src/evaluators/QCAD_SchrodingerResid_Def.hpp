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


#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"


//**********************************************************************
template<typename EvalT, typename Traits>
QCAD::SchrodingerResid<EvalT, Traits>::
SchrodingerResid(const Teuchos::ParameterList& p) :
  wBF         (p.get<std::string>                   ("Weighted BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Scalar Data Layout") ),
  psi         (p.get<std::string>                   ("QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  psiDot      (p.get<std::string>                   ("QP Time Derivative Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  wGradBF     (p.get<std::string>                   ("Weighted Gradient BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Vector Data Layout") ),
  psiGrad     (p.get<std::string>                   ("Gradient QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  V           (p.get<std::string>                   ("Potential Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  coordVec    (p.get<std::string>                   ("QP Coordinate Vector Name"),
               p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout")),
  invEffMass  (                                      "Intermediate Eff Mass",
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  havePotential (p.get<bool>("Have Potential")),
  haveMaterial  (p.get<bool>("Have Material")),
  psiResidual (p.get<std::string>                   ("Residual Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node Scalar Data Layout") )
{
  // obtain Finite Wall potential parameters
  Teuchos::ParameterList* psList = p.get<Teuchos::ParameterList*>("Parameter List");
  potentialType = psList->get<std::string>("Type", "defaultType");
  barrEffMass = psList->get<double>("Barrier Effective Mass", 0.0);
  barrWidth = psList->get<double>("Barrier Width", 0.0);
  wellEffMass = psList->get<double>("Well Effective Mass", 0.0);
  wellWidth = psList->get<double>("Well Width", 0.0);

  // obtain Oxide and Silicon Width for 1D MOSCapacitor
  oxideWidth = psList->get<double>("Oxide Width", 0.0);
  siliconWidth = psList->get<double>("Silicon Width", 0.0); 
  
  enableTransient = true; //always true - problem doesn't make sense otherwise
  energy_unit_in_eV = p.get<double>("Energy unit in eV");
  length_unit_in_m = p.get<double>("Length unit in m");
  bOnlyInQuantumBlocks = p.get<bool>("Only solve in quantum blocks");

  // Material database
  materialDB = p.get< Teuchos::RCP<QCAD::MaterialDatabase> >("MaterialDB");

  Teuchos::RCP<PHX::DataLayout> vector_dl =
      p.get< Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  // Allocate workspace
  psiGradWithMass.resize(dims[0], numQPs, numDims);
  psiV.resize(dims[0], numQPs);

  this->addDependentField(wBF);
  this->addDependentField(psi);
  this->addDependentField(psiDot);
  this->addDependentField(psiGrad);
  this->addDependentField(wGradBF);
  this->addDependentField(coordVec);
  if (havePotential) this->addDependentField(V);

  this->addEvaluatedField(psiResidual);
  this->addEvaluatedField(invEffMass);
  
  this->setName("SchrodingerResid"+PHX::TypeString<EvalT>::value);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void QCAD::SchrodingerResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(psi,fm);
  this->utils.setFieldData(psiDot,fm);
  this->utils.setFieldData(psiGrad,fm);
  this->utils.setFieldData(wGradBF,fm);
  this->utils.setFieldData(coordVec,fm);
  if (havePotential)  this->utils.setFieldData(V,fm);

  this->utils.setFieldData(psiResidual,fm);
  this->utils.setFieldData(invEffMass,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void QCAD::SchrodingerResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  typedef Intrepid::FunctionSpaceTools FST;

  /***** define universal constants as double constants *****/
  const double hbar = 1.0546e-34;  // Planck constant [J s]
  const double evPerJ = 6.2415e18; // eV per Joule (eV/J)
  bool bValidRegion = true;

  if(bOnlyInQuantumBlocks)
    bValidRegion = materialDB->getElementBlockParam<bool>(workset.EBName,"quantum",false);

  if(bValidRegion)
  {
    //compute inverse effective mass here (no separate evaluator)
    for (std::size_t cell=0; cell < workset.numCells; ++cell) 
    {
      for (std::size_t qp=0; qp < numQPs; ++qp) 
      {
        //scaled hbar^2/2m so kinetic energy has specified units (EnergyUnitInEV)
        invEffMass(cell, qp) =  0.5*pow(hbar,2)*evPerJ / 
            (energy_unit_in_eV * pow(length_unit_in_m,2)) *
            getInvEffMass(workset.EBName, numDims, &coordVec(cell,qp,0)); 
        
        //std::cout << "hbar^2/2m = " << invEffMass(cell, qp) << std::endl;     
      }
    }

    //compute hbar^2/2m * Grad(psi)
    FST::scalarMultiplyDataData<ScalarT> (psiGradWithMass, invEffMass, psiGrad);

    //Kinetic term: add integral( hbar^2/2m * Grad(psi) * Grad(BF)dV ) to residual
    FST::integrate<ScalarT>(psiResidual, psiGradWithMass, wGradBF, Intrepid::COMP_CPP, false); // "false" overwrites
  
    //Potential term: add integral( psi * V * BF dV ) to residual
    if (havePotential) {
      FST::scalarMultiplyDataData<ScalarT> (psiV, V, psi);
      FST::integrate<ScalarT>(psiResidual, psiV, wBF, Intrepid::COMP_CPP, true); // "true" sums into
    }

    //**Note: I think this should always be used with enableTransient = True
    //psiDot term (to use loca): add integral( psi_dot * BF dV ) to residual
    if (workset.transientTerms && enableTransient) 
      FST::integrate<ScalarT>(psiResidual, psiDot, wBF, Intrepid::COMP_CPP, true); // "true" sums into
      
  }
  
  else 
  { 
    // Invalid region (don't perform calc here - evectors should all be zero)
    // So, set psiDot term to zero and set psi term (H) = Identity

    //This doesn't work - results in NaN error
    /*for (std::size_t cell=0; cell < workset.numCells; ++cell) 
    {
      for (std::size_t qp=0; qp < numQPs; ++qp)
        psiResidual(cell, qp) = 1.0*psi(cell,qp);
    }*/

    //Potential term: add integral( psi * V * BF dV ) to residual
    if (havePotential) {
      FST::scalarMultiplyDataData<ScalarT> (psiV, V, psi);
      FST::integrate<ScalarT>(psiResidual, psiV, wBF, Intrepid::COMP_CPP, false); // "false" overwrites
    }

  }
}
//**********************************************************************

template<typename EvalT,typename Traits>
Teuchos::RCP<const Teuchos::ParameterList>
QCAD::SchrodingerResid<EvalT,Traits>::getValidMaterialParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
       rcp(new Teuchos::ParameterList("Valid Material Params"));;

  validPL->set<string>("Name", "defaultName", "Switch between different materials, e.g. GaAs");

  return validPL;
}

// **********************************************************************

//Inverse effective mass in kg^-1
template<typename EvalT, typename Traits>
typename QCAD::SchrodingerResid<EvalT,Traits>::ScalarT
QCAD::SchrodingerResid<EvalT, Traits>::getInvEffMass(const std::string& EBName, const int numDim, 
						     const MeshScalarT* coord)
{
  ScalarT effMass;
  const double emass = 9.1094e-31; // Electron mass [kg]

  //TODO - return tensor instead of scalar - now just return effective mass in X-direction
  // effMass = materialDB->getElementBlockParam<double>(EBName,"Electron Effective Mass X",1.0) * emass;
  
  if (potentialType == "Finite Wall")
  {
    if (numDim == 1)  // 1D
    {
      if ( (coord[0] >= barrWidth) && (coord[0] <= (barrWidth+wellWidth)) )
        effMass = wellEffMass * emass;  // well
      else
        effMass = barrEffMass * emass;  // barrier
    }
    
    else if (numDim == 2)  // 2D
    {
      if ( (coord[0] >= barrWidth) && (coord[0] <= (barrWidth+wellWidth)) &&
           (coord[1] >= barrWidth) && (coord[1] <= (barrWidth+wellWidth)) )
        effMass = wellEffMass * emass;  
      else
        effMass = barrEffMass * emass;   
    }
    
    else if (numDim == 3)  // 3D
    {
      if ( (coord[0] >= barrWidth) && (coord[0] <= (barrWidth+wellWidth)) &&
           (coord[1] >= barrWidth) && (coord[1] <= (barrWidth+wellWidth)) && 
           (coord[2] >= barrWidth) && (coord[2] <= (barrWidth+wellWidth)) )
        effMass = wellEffMass * emass;   
      else
        effMass = barrEffMass * emass;      
    }
    
    else
      TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, std::endl 
			  << "Error! Invalid numDim = " << numDim << ", must be 1 or 2 or 3 !" << std::endl);
    
  }  // end of if (potentialType == "Finite Wall")

  
  // For 1D MOSCapacitor 
  else if ( (numDim == 1) && (oxideWidth > 0.0) ) 
  {
    // Oxide region
    if ((coord[0] >= 0) && (coord[0] <= oxideWidth))  // Oxide region
      effMass = materialDB->getMaterialParam<double>("SiliconDioxide","Longitudinal Electron Effective Mass",1.0) * emass;
    
    // Silicon region
    else if ((coord[0] > oxideWidth) && (coord[0] <= oxideWidth+siliconWidth))  
      effMass = materialDB->getMaterialParam<double>("Silicon","Longitudinal Electron Effective Mass",1.0) * emass;
    
    else
      TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
	       std::endl << "Error!  x-coord:" << coord[0] << "is outside the oxideWidth" << 
	       " + siliconWidth range: " << oxideWidth + siliconWidth << "!"<< std::endl);
  }

  
  /* Effective mass depends on the wafer orientation and growth direction.
  For SiO2/Si interface parallel to the [100] plane (growth direction along [001]),
  the confinement is in [001] direction, and the mass for Delta2-band (along [001])
  is ml (longitudinal).Consider only the Delta2-band for the time being 
  (need to include Delta4-band later)
  */
  else
    effMass = materialDB->getElementBlockParam<double>(EBName,"Longitudinal Electron Effective Mass",1.0) * emass;
  
  return 1.0/effMass;
}



// **********************************************************************

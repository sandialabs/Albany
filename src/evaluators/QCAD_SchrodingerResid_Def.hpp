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
  enableTransient = true; //always true - problem doesn't make sense otherwise
  energy_unit_in_eV = p.get<double>("Energy unit in eV");
  length_unit_in_m = p.get<double>("Length unit in m");
  bOnlyInQuantumBlocks = p.get<bool>("Only solve in quantum blocks");

  if(haveMaterial) {
     Teuchos::ParameterList* pMatList = p.get<Teuchos::ParameterList*>("Material Parameter List");
     Teuchos::RCP<const Teuchos::ParameterList> reflist = 
       this->getValidMaterialParameters();
     pMatList->validateParameters(*reflist,0);
     materialName = pMatList->get("Name", "defaultName");
  }

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
  const double evPerJ = 6.2415e18; // eV per Joule (J/eV)
  bool bValidRegion = true;

  if(bOnlyInQuantumBlocks)
    bValidRegion = materialDB->getElementBlockParam<bool>(workset.EBName,
							  "quantum");

  if(bValidRegion)
  {
  
    //compute inverse effective mass here (no separate evaluator)
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
	//scaled hbar^2/2m so kinetic energy has specified units (EnergyUnitInEV)
	invEffMass(cell, qp) =  0.5*pow(hbar,2)*evPerJ / 
	  (energy_unit_in_eV * pow(length_unit_in_m,2)) *
	  getInvEffMass(numDims, &coordVec(cell,qp,0)); 
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
  else { 
    // Invalid region (don't perform calc here - evectors should all be zero)
    // So, set psiDot term to zero and set psi term (H) = Identity
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp)
	psiResidual(cell, qp) = 1.0*psi(cell,qp);
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
QCAD::SchrodingerResid<EvalT, Traits>::getInvEffMass(const int numDim, 
						     const MeshScalarT* coord)
{
  ScalarT effMass;
  const double emass = 9.1094e-31; // Electron mass [kg]

  if(haveMaterial)
  {
    if (materialName == "GaAs") {
      effMass = 0.067 * emass;
    }
    else if (materialName == "Vacuum") {
      effMass = emass;
    }
    else {
      TEST_FOR_EXCEPT(true);
    }
  }
  else
  {
    effMass = emass;
  }
  return 1.0/effMass;
}



// **********************************************************************

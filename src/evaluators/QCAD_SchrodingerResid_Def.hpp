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
  //invEffMass  (                                      "Intermediate Eff Mass",
	//       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
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

  // calculate hbar^2/2m0 so kinetic energy has specified units (EnergyUnitInEV)
  const double hbar = 1.0546e-34;   // Planck constant [J s]
  const double evPerJ = 6.2415e18;  // eV per Joule (eV/J)
  const double emass = 9.1094e-31;  // Electron mass [kg]
  hbar2_over_2m0 = 0.5*pow(hbar,2)*evPerJ /(emass *energy_unit_in_eV *pow(length_unit_in_m,2));

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
  V_barrier.resize(dims[0], numQPs);

  this->addDependentField(wBF);
  this->addDependentField(psi);
  this->addDependentField(psiDot);
  this->addDependentField(psiGrad);
  this->addDependentField(wGradBF);
  this->addDependentField(coordVec);
  if (havePotential) this->addDependentField(V);

  this->addEvaluatedField(psiResidual);
  // this->addEvaluatedField(invEffMass);
  
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
  // this->utils.setFieldData(invEffMass,fm);
}


//**********************************************************************
template<typename EvalT, typename Traits>
void QCAD::SchrodingerResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  typedef Intrepid::FunctionSpaceTools FST;
  bool bValidRegion = true;
  double invEffMass = 1.0; 

  if(bOnlyInQuantumBlocks)
    bValidRegion = materialDB->getElementBlockParam<bool>(workset.EBName,"quantum",false);

  if(bValidRegion)
  {
    // For potentialType == FromState
    if ( potentialType == "FromState")
    {
      if ( (numDims == 1) && (oxideWidth > 0.0) )   // 1D MOSCapacitor 
      {
        for (std::size_t cell = 0; cell < workset.numCells; ++cell) 
        { 
          for (std::size_t qp = 0; qp < numQPs; ++qp) 
          {
            invEffMass = hbar2_over_2m0 * getInvEffMass1DMosCap(&coordVec(cell,qp,0));
            for (std::size_t dim = 0; dim < numDims; ++dim)
              psiGradWithMass(cell,qp,dim) = invEffMass * psiGrad(cell,qp,dim);
          }  
        }    
      }
      
      // Effective mass depends on the wafer orientation and growth direction.
      // Assume SiO2/Si interface is parallel to the [100] plane and consider only the 
      // Delta2 Valleys whose principal axis is perpendicular to the SiO2/Si interface. 
      // (need to include Delta4-band for high temperatures > 50 K).

      else  // General case
      {
        const string& matrlCategory = materialDB->getElementBlockParam<string>(workset.EBName,"Category","");
        double ml = 1.0; 
        double mt = 1.0; 
        
        // obtain ml and mt
        if (matrlCategory == "Semiconductor") 
        {
          const string& condBandMinVal = materialDB->getElementBlockParam<string>(workset.EBName,"Conduction Band Minimum");
          ml = materialDB->getElementBlockParam<double>(workset.EBName,"Longitudinal Electron Effective Mass");
          mt = materialDB->getElementBlockParam<double>(workset.EBName,"Transverse Electron Effective Mass");
    
          if ((condBandMinVal == "Gamma Valley") && (fabs(ml-mt) > 1e-10))
          {
            TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Gamma Valley's longitudinal and "
              << "transverse electron effective mass must be equal ! "
              << "Please check the values in materials.xml" << std::endl);
          }
        }      

        else if (matrlCategory == "Insulator")
        {
          ml = materialDB->getElementBlockParam<double>(workset.EBName,"Longitudinal Electron Effective Mass");
          mt = materialDB->getElementBlockParam<double>(workset.EBName,"Transverse Electron Effective Mass");
          if (fabs(ml-mt) > 1e-10) 
          {
            TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Insulator's longitudinal and "
              << "transverse electron effective mass must be equal ! "
              << "Please check the values in materials.xml" << std::endl);
	        }
	      }
	      
        // calculate psiGradWithMass (good for diagonal effective mass tensor !)
        for (std::size_t cell = 0; cell < workset.numCells; ++cell) 
        { 
          for (std::size_t qp = 0; qp < numQPs; ++qp) 
          {
            for (std::size_t dim = 0; dim < numDims; ++dim)
            {
              if (dim == numDims-1)
                invEffMass = hbar2_over_2m0 / ml;
              else
                invEffMass = hbar2_over_2m0 / mt;
              psiGradWithMass(cell,qp,dim) = invEffMass * psiGrad(cell,qp,dim);
            }  
          }  
        }    
	             
      }  // end of General case
      
    }  // end of if ( potentialType == "FromState")


    // For potentialType == Finite Wall 
    else if ( potentialType == "Finite Wall" ) 
    {
      for (std::size_t cell = 0; cell < workset.numCells; ++cell) 
      { 
        for (std::size_t qp = 0; qp < numQPs; ++qp) 
        {
          invEffMass = hbar2_over_2m0 * getInvEffMassFiniteWall(&coordVec(cell,qp,0));
          for (std::size_t dim = 0; dim < numDims; ++dim)
            psiGradWithMass(cell,qp,dim) = invEffMass * psiGrad(cell,qp,dim);
        }  
      }    
    }  
    
    // For other potentialType
    else
    {
      for (std::size_t cell = 0; cell < workset.numCells; ++cell) 
        for (std::size_t qp = 0; qp < numQPs; ++qp) 
          for (std::size_t dim = 0; dim < numDims; ++dim)
            psiGradWithMass(cell,qp,dim) = hbar2_over_2m0 * psiGrad(cell,qp,dim);
    }    


/*
    // Define universal constants as double constants
    const double hbar = 1.0546e-34;  // Planck constant [J s]
    const double evPerJ = 6.2415e18; // eV per Joule (eV/J)
  
    // prefactor for scaled hbar^2/2m so kinetic energy has specified units (EnergyUnitInEV)
    ScalarT prefactor = 0.5*pow(hbar,2)*evPerJ /(energy_unit_in_eV * pow(length_unit_in_m,2));

    // Reset effective mass tensor to 0.0
    for (std::size_t cell = 0; cell < workset.numCells; ++cell) 
      for (std::size_t qp = 0; qp < numQPs; ++qp) 
        for (std::size_t i = 0; i < numDims; ++i) 
          for (std::size_t j = 0; j < numDims; ++j)
            invEffMass(cell,qp,i,j) = 0.0;

    // Set diagonal elements of effective mass tensor
    for (std::size_t cell = 0; cell < workset.numCells; ++cell) 
      for (std::size_t qp = 0; qp < numQPs; ++qp) 
        for (std::size_t i = 0; i < numDims; ++i) 
          invEffMass(cell,qp,i,i) = prefactor * getInvEffMass(workset.EBName, i, &coordVec(cell,qp,0));

    //compute hbar^2/2m * Grad(psi)
    FST::tensorMultiplyDataData<ScalarT> (psiGradWithMass, invEffMass, psiGrad);
*/

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
      
  }  // end of if(bValidRegion)
  
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
    if (havePotential) 
    {
      for (std::size_t cell = 0; cell < workset.numCells; ++cell)
        for (std::size_t qp = 0; qp < numQPs; ++qp)
          V_barrier(cell,qp) = 100.0;
          
      FST::scalarMultiplyDataData<ScalarT> (psiV, V_barrier, psi);
      // FST::scalarMultiplyDataData<ScalarT> (psiV, V, psi);
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
template<typename EvalT, typename Traits> double
QCAD::SchrodingerResid<EvalT, Traits>::getInvEffMassFiniteWall(const MeshScalarT* coord)
{
  double effMass; 
  switch (numDims) 
  {
    case 1:  // 1D
    {
      if ( (coord[0] >= barrWidth) && (coord[0] <= (barrWidth+wellWidth)) )
        effMass = wellEffMass;  // well
      else
        effMass = barrEffMass;  // barrier
      break;  
    }
    case 2:  // 2D
    {
      if ( (coord[0] >= barrWidth) && (coord[0] <= (barrWidth+wellWidth)) &&
         (coord[1] >= barrWidth) && (coord[1] <= (barrWidth+wellWidth)) )
        effMass = wellEffMass;  
      else
        effMass = barrEffMass;
      break;   
    }
    case 3:  // 3D
    {
      if ( (coord[0] >= barrWidth) && (coord[0] <= (barrWidth+wellWidth)) &&
         (coord[1] >= barrWidth) && (coord[1] <= (barrWidth+wellWidth)) && 
         (coord[2] >= barrWidth) && (coord[2] <= (barrWidth+wellWidth)) )
         
        effMass = wellEffMass;   
      else
        effMass = barrEffMass;  
      break;    
    }
    default:
    {
      effMass = 0.0; // should never get here (suppresses uninitialized warning)
      TEUCHOS_TEST_FOR_EXCEPT( effMass == 0 );
    }
  }  
    
  return 1.0/effMass;
}


// **********************************************************************
template<typename EvalT, typename Traits> double
QCAD::SchrodingerResid<EvalT, Traits>::getInvEffMass1DMosCap(const MeshScalarT* coord)
{
  double effMass; 
  
  // Oxide region
  if ((coord[0] >= 0) && (coord[0] <= oxideWidth))  
    effMass = materialDB->getMaterialParam<double>("SiliconDioxide","Longitudinal Electron Effective Mass",1.0);
    
  // Silicon region
  else if ((coord[0] > oxideWidth) && (coord[0] <= oxideWidth+siliconWidth))  
    effMass = materialDB->getMaterialParam<double>("Silicon","Longitudinal Electron Effective Mass",1.0);
   
  else
  {
   TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
	     std::endl << "Error!  x-coord:" << coord[0] << "is outside the oxideWidth" << 
	     " + siliconWidth range: " << oxideWidth + siliconWidth << "!"<< std::endl);
  }
  
  return 1.0/effMass;
}


// **********************************************************************
/* temporarily keep it

template<typename EvalT, typename Traits>
typename QCAD::SchrodingerResid<EvalT,Traits>::ScalarT
QCAD::SchrodingerResid<EvalT, Traits>::getInvEffMass(const std::string& EBName, 
    const std::size_t dim, const MeshScalarT* coord)
{
  ScalarT effMass; 
  const double emass = 9.1094e-31; // Electron mass [kg]
  
  // For Finite Wall potential
  if (potentialType == "Finite Wall")
  {
    switch (numDims) 
    {
      case 1:  // 1D
      {
        if ( (coord[0] >= barrWidth) && (coord[0] <= (barrWidth+wellWidth)) )
          effMass = wellEffMass * emass;  // well
        else
          effMass = barrEffMass * emass;  // barrier
        break;  
      }
      case 2:  // 2D
      {
        if ( (coord[0] >= barrWidth) && (coord[0] <= (barrWidth+wellWidth)) &&
           (coord[1] >= barrWidth) && (coord[1] <= (barrWidth+wellWidth)) )
          effMass = wellEffMass * emass;  
        else
          effMass = barrEffMass * emass;
        break;   
      }
      case 3:  // 3D
      {
        if ( (coord[0] >= barrWidth) && (coord[0] <= (barrWidth+wellWidth)) &&
           (coord[1] >= barrWidth) && (coord[1] <= (barrWidth+wellWidth)) && 
           (coord[2] >= barrWidth) && (coord[2] <= (barrWidth+wellWidth)) )
          effMass = wellEffMass * emass;   
        else
          effMass = barrEffMass * emass;  
        break;    
      }
      default:
      {
        effMass = 0; // should never get here (suppresses uninitialized warning)
        TEUCHOS_TEST_FOR_EXCEPT( effMass == 0 );
      }
    }  // end of switch (numDims) 
    
    return 1.0/effMass;
    
  }  // end of if (potentialType == "Finite Wall")


  // For 1D MOSCapacitor 
  if ( (numDims == 1) && (oxideWidth > 0.0) ) 
  {
    // Oxide region
    if ((coord[0] >= 0) && (coord[0] <= oxideWidth))  // Oxide region
      effMass = materialDB->getMaterialParam<double>("SiliconDioxide","Longitudinal Electron Effective Mass",1.0) * emass;
    
    // Silicon region
    else if ((coord[0] > oxideWidth) && (coord[0] <= oxideWidth+siliconWidth))  
      effMass = materialDB->getMaterialParam<double>("Silicon","Longitudinal Electron Effective Mass",1.0) * emass;
    
    else
      TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
	       std::endl << "Error!  x-coord:" << coord[0] << "is outside the oxideWidth" << 
	       " + siliconWidth range: " << oxideWidth + siliconWidth << "!"<< std::endl);
  
    return 1.0/effMass;
    
  }  // end of 1D MOSCapacitor 


  // Effective mass depends on the wafer orientation and growth direction.
  // Assume SiO2/Si interface is parallel to the [100] plane and consider only the 
  // Delta2 Valleys whose principal axis is perpendicular to the SiO2/Si interface. 
  // (need to include Delta4-band for high temperatures).

  // For General Case
  const string& matrlCategory = materialDB->getElementBlockParam<string>(EBName,"Category","");
    
  if (matrlCategory == "Semiconductor") 
  {
    const string& condBandMinVal = materialDB->getElementBlockParam<string>(EBName,"Conduction Band Minimum");
    double ml = materialDB->getElementBlockParam<double>(EBName,"Longitudinal Electron Effective Mass");
    double mt = materialDB->getElementBlockParam<double>(EBName,"Transverse Electron Effective Mass");
    
    if ((condBandMinVal == "Gamma Valley") && (abs(ml-mt) > 1e-10))
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Gamma Valley's longitudinal and "
        << "transverse electron effective mass must be equal ! "
        << "Please check the values in materials.xml" << std::endl);
      
    if (condBandMinVal == "Delta2 Valley")
    {
      if (dim == numDims-1) 
        effMass = ml*emass;
      else
        effMass = mt*emass;
    }  
    else if (condBandMinVal == "Gamma Valley")
      effMass = ml*emass;
    else
      TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, std::endl
        << "Invalid Conduction Band Minimum ! Must be Delta2 or Gamma Valley !" << std::endl);

  }  // end of if (matrlCategory == "Semiconductor")
    
  else if (matrlCategory == "Insulator")
  {
    double ml = materialDB->getElementBlockParam<double>(EBName,"Longitudinal Electron Effective Mass");
    double mt = materialDB->getElementBlockParam<double>(EBName,"Transverse Electron Effective Mass");
    if (abs(ml-mt) > 1e-10) 
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Insulator's longitudinal and "
	       << "transverse electron effective mass must be equal ! "
	       << "Please check the values in materials.xml" << std::endl);
    effMass = ml*emass;
  }
  
  else
    effMass = emass;
   
  return 1.0/effMass;  
}
*/

// **********************************************************************

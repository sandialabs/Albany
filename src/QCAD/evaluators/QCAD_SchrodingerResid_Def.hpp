//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"


//**********************************************************************
template<typename EvalT, typename Traits>
QCAD::SchrodingerResid<EvalT, Traits>::
SchrodingerResid(const Teuchos::ParameterList& p,
                 const Teuchos::RCP<Albany::Layouts>& dl) :
  wBF         (p.get<std::string>  ("Weighted BF Name"), dl->node_qp_scalar),
  psi         (p.get<std::string>  ("QP Variable Name"), dl->qp_scalar),
  psiDot      (p.get<std::string>  ("QP Time Derivative Variable Name"), dl->qp_scalar),
  wGradBF     (p.get<std::string>  ("Weighted Gradient BF Name"), dl->node_qp_gradient),
  psiGrad     (p.get<std::string>  ("Gradient QP Variable Name"), dl->qp_gradient),
  V           (p.get<std::string>  ("Potential Name"), dl->qp_scalar),
  coordVec    (p.get<std::string>  ("QP Coordinate Vector Name"), dl->qp_gradient),
  havePotential (p.get<bool>("Have Potential")),
  psiResidual (p.get<std::string>  ("Residual Name"), dl->node_scalar)
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
  materialDB = p.get< Teuchos::RCP<Albany::MaterialDatabase> >("MaterialDB");

  std::vector<PHX::DataLayout::size_type> dims;
  dl->qp_gradient->dimensions(dims);
  numCells = dims[0];
  numQPs  = dims[1];
  numDims = dims[2];

  this->addDependentField(wBF);
  this->addDependentField(psi);
  this->addDependentField(psiDot);
  this->addDependentField(psiGrad);
  this->addDependentField(wGradBF);
  this->addDependentField(coordVec);
  if (havePotential) this->addDependentField(V);

  this->addEvaluatedField(psiResidual);
  
  this->setName( "SchrodingerResid" + PHX::typeAsString<EvalT>() );
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

  // Allocate workspace
  psiGradWithMass = Kokkos::createDynRankView(V.get_view(), "XXX", numCells, numQPs, numDims);
  psiV = Kokkos::createDynRankView(V.get_view(), "XXX", numCells, numQPs);
  V_barrier = Kokkos::createDynRankView(V.get_view(), "XXX", numCells, numQPs);
}


//**********************************************************************
template<typename EvalT, typename Traits>
void QCAD::SchrodingerResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  typedef Intrepid2::FunctionSpaceTools<PHX::Device> FST;
  bool bValidRegion = true;
  double invEffMass = 1.0; 

  if(bOnlyInQuantumBlocks)
    bValidRegion = materialDB->getElementBlockParam<bool>(workset.EBName,"quantum",false);
  
  // Put loops inside the if branches for speed
  if(bValidRegion)
  {
    if ( potentialType == "From State" || potentialType == "String Formula" || potentialType == "From Aux Data Vector")
    {
      if ( (numDims == 1) && (oxideWidth > 0.0) )   // 1D MOSCapacitor 
      {
        for (std::size_t cell = 0; cell < workset.numCells; ++cell) 
        { 
          for (std::size_t qp = 0; qp < numQPs; ++qp) 
          {
            invEffMass = hbar2_over_2m0 * getInvEffMass1DMosCap(coordVec(cell,qp,0));
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
        double ml, mt;
        
        const std::string& matrlCategory = materialDB->getElementBlockParam<std::string>(workset.EBName,"Category","");

        // obtain ml and mt
        if (matrlCategory == "Semiconductor") 
        {
          const std::string& condBandMinVal = materialDB->getElementBlockParam<std::string>(workset.EBName,"Conduction Band Minimum");
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

	else {
	  // Default releative effective masses == 1.0 if matrl category is not recognized.
	  // Perhaps we should throw an error here instead?
	  ml = mt = 1.0;
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
      
    }  // end of if ( potentialType == "From State" || ... )


    // For potentialType == Finite Wall 
    else if ( potentialType == "Finite Wall" ) 
    {
      for (std::size_t cell = 0; cell < workset.numCells; ++cell) 
      { 
        for (std::size_t qp = 0; qp < numQPs; ++qp) 
        {
          invEffMass = hbar2_over_2m0 * getInvEffMassFiniteWall(coordVec,cell,qp);
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

    //Kinetic term: add integral( hbar^2/2m * Grad(psi) * Grad(BF)dV ) to residual
    FST::integrate(psiResidual.get_view(), psiGradWithMass, wGradBF.get_view(), false); // "false" overwrites
  
    //Potential term: add integral( psi * V * BF dV ) to residual
    if (havePotential) {
      FST::scalarMultiplyDataData (psiV, V.get_view(), psi.get_view());
      FST::integrate(psiResidual.get_view(), psiV, wBF.get_view(), true); // "true" sums into
    }

    //**Note: I think this should always be used with enableTransient = True
    //psiDot term (to use loca): add integral( psi_dot * BF dV ) to residual
    if (workset.transientTerms && enableTransient) 
      FST::integrate(psiResidual.get_view(), psiDot.get_view(), wBF.get_view(), true); // "true" sums into
      
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
          
      FST::scalarMultiplyDataData(psiV, V_barrier, psi.get_view());
      // FST::scalarMultiplyDataData(psiV, V, psi);
      FST::integrate(psiResidual.get_view(), psiV, wBF.get_view(), false); // "false" overwrites
    }


    //POSSIBLE NEW - same as valid region but use a very large potential...
    /*for (std::size_t cell = 0; cell < workset.numCells; ++cell)  {
      for (std::size_t qp = 0; qp < numQPs; ++qp) {

	V_barrier(cell,qp) = 1e+10;

	for (std::size_t dim = 0; dim < numDims; ++dim)
	  psiGradWithMass(cell,qp,dim) = hbar2_over_2m0 * psiGrad(cell,qp,dim);
      }
    }

    //Kinetic term: add integral( hbar^2/2m * Grad(psi) * Grad(BF)dV ) to residual
    FST::integrate(psiResidual, psiGradWithMass, wGradBF, false); // "false" overwrites
  
    //Potential term: add integral( psi * V * BF dV ) to residual
    FST::scalarMultiplyDataData (psiV, V_barrier, psi);
    //FST::scalarMultiplyDataData (psiV, V, psi);
    FST::integrate(psiResidual, psiV, wBF, true); // "true" sums into

    // **Note: I think this should always be used with enableTransient = True
    //psiDot term (to use loca): add integral( psi_dot * BF dV ) to residual
    if (workset.transientTerms && enableTransient) 
      FST::integrate(psiResidual, psiDot, wBF, true); // "true" sums into
    */
  }
}


// **********************************************************************
template<typename EvalT, typename Traits> double
QCAD::SchrodingerResid<EvalT, Traits>::getInvEffMassFiniteWall(
  const PHX::MDField<const MeshScalarT,Cell,QuadPoint,Dim> & coord,
  const int cell, const int qp )
{
  double effMass; 
  switch (numDims) 
  {
    case 1:  // 1D
    {
      if ( (coord(cell,qp,0) >= barrWidth) && (coord(cell,qp,0) <= (barrWidth+wellWidth)) )
        effMass = wellEffMass;  // well
      else
        effMass = barrEffMass;  // barrier
      break;  
    }
    case 2:  // 2D
    {
      if ( (coord(cell,qp,0) >= barrWidth) && (coord(cell,qp,0) <= (barrWidth+wellWidth)) &&
         (coord(cell,qp,1) >= barrWidth) && (coord(cell,qp,1) <= (barrWidth+wellWidth)) )
        effMass = wellEffMass;  
      else
        effMass = barrEffMass;
      break;   
    }
    case 3:  // 3D
    {
      if ( (coord(cell,qp,0) >= barrWidth) && (coord(cell,qp,0) <= (barrWidth+wellWidth)) &&
         (coord(cell,qp,1) >= barrWidth) && (coord(cell,qp,1) <= (barrWidth+wellWidth)) && 
         (coord(cell,qp,2) >= barrWidth) && (coord(cell,qp,2) <= (barrWidth+wellWidth)) )
         
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
QCAD::SchrodingerResid<EvalT, Traits>::getInvEffMass1DMosCap(const MeshScalarT coord0)
{
  double effMass; 
  
  // Oxide region
  if ((coord0 >= 0) && (coord0 <= oxideWidth))  
    effMass = materialDB->getMaterialParam<double>("SiliconDioxide","Longitudinal Electron Effective Mass",1.0);
    
  // Silicon region
  else if ((coord0 > oxideWidth) && (coord0 <= oxideWidth+siliconWidth))  
    effMass = materialDB->getMaterialParam<double>("Silicon","Longitudinal Electron Effective Mass",1.0);
   
  else
  {
   TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
	     std::endl << "Error!  x-coord:" << coord0 << "is outside the oxideWidth" << 
	     " + siliconWidth range: " << oxideWidth + siliconWidth << "!"<< std::endl);
  }
  
  return 1.0/effMass;
}


// **********************************************************************

//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Aeras_ShallowWaterProblem.hpp"

#include "Intrepid2_FieldContainer.hpp"
#include "Shards_CellTopology.hpp"
#include "PHAL_FactoryTraits.hpp"
#include "Albany_Utils.hpp"
#include "Albany_BCUtils.hpp"
#include "Albany_ProblemUtils.hpp"
#include <string>


Aeras::ShallowWaterProblem::
ShallowWaterProblem( const Teuchos::RCP<Teuchos::ParameterList>& params_,
             const Teuchos::RCP<ParamLib>& paramLib_,
             const int spatialDim_) :
  Albany::AbstractProblem(params_, paramLib_),
  spatialDim(spatialDim_)
{
  TEUCHOS_TEST_FOR_EXCEPTION(spatialDim!=2 && spatialDim!=3,std::logic_error,"Shallow water problem is only written for 2 or 3D.");
  std::string eqnSet = params_->sublist("Equation Set").get<std::string>("Type", "Shallow Water"); 
  // Set number of scalar equation per node, neq,  based on spatialDim
  if      (spatialDim==2) { modelDim=2; neq=3; } // Planar 2D problem
  else if (spatialDim ==3 ) { //2D shells embedded in 3D
    if (eqnSet == "Scalar") { modelDim=2; neq=1; } 
    else { modelDim=2; neq=3; } 
  }

  bool useExplHyperviscosity = params_->sublist("Shallow Water Problem").get<bool>("Use Explicit Hyperviscosity", false);
  bool useImplHyperviscosity = params_->sublist("Shallow Water Problem").get<bool>("Use Implicit Hyperviscosity", false);
  bool usePrescribedVelocity = params_->sublist("Shallow Water Problem").get<bool>("Use Prescribed Velocity", false); 
  bool plotVorticity = params_->sublist("Shallow Water Problem").get<bool>("Plot Vorticity", false); 

  TEUCHOS_TEST_FOR_EXCEPTION( useExplHyperviscosity && useImplHyperviscosity ,std::logic_error,"Use only explicit or implicit hyperviscosity, not both.");


  if (useImplHyperviscosity) {
    if (usePrescribedVelocity) //TC1 case: only 1 extra hyperviscosity dof 
      neq = 4; 
    //If we're using hyperviscosity for Shallow water equations, we have double the # of dofs. 
    else  
      neq = 2*neq; 
  }

//No need to plot vorticity when prescrVel == 1.
  if (plotVorticity) {
     if (!usePrescribedVelocity) {
       //one extra stationary equation for vorticity
       neq++;
     }
     else {
       std::cout << "Prescribed Velocity is ON, in this case option PlotVorticity=true is ignored." << std::endl; 
     }
  }


  std::cout << "eqnSet, modelDim, neq: " << eqnSet << ", " << modelDim << ", " << neq << std::endl; 
  // Set the num PDEs for the null space object to pass to ML
  this->rigidBodyModes->setNumPDEs(neq);
}

Aeras::ShallowWaterProblem::
~ShallowWaterProblem()
{
}

void
Aeras::ShallowWaterProblem::
buildProblem(
  Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
  Albany::StateManager& stateMgr)
{
  using Teuchos::rcp;

 /* Construct All Phalanx Evaluators */
  TEUCHOS_TEST_FOR_EXCEPTION(meshSpecs.size()!=1,std::logic_error,"Problem supports one Material Block");
  fm.resize(1);
  fm[0]  = rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
  buildEvaluators(*fm[0], *meshSpecs[0], stateMgr, Albany::BUILD_RESID_FM, 
		  Teuchos::null);
}

Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
Aeras::ShallowWaterProblem::
buildEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fmchoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  Albany::ConstructEvaluatorsOp<ShallowWaterProblem> op(
    *this, fm0, meshSpecs, stateMgr, fmchoice, responseList);
  Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes> fe(op);
  return *op.tags;
}


Teuchos::RCP<const Teuchos::ParameterList>
Aeras::ShallowWaterProblem::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidShallowWaterProblemParams");

  validPL->sublist("Shallow Water Problem", false, "");
  validPL->sublist("Aeras Surface Height", false, "");
  validPL->sublist("Aeras Shallow Water Source", false, "");
  validPL->sublist("Equation Set", false, "");
  return validPL;
}


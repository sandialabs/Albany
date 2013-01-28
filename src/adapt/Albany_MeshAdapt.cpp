//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_MeshAdapt.hpp"

#include "Teuchos_TimeMonitor.hpp"

Albany::MeshAdapt::
MeshAdapt(const Teuchos::RCP<Teuchos::ParameterList>& params_,
                     const Teuchos::RCP<ParamLib>& paramLib_,
                     Albany::StateManager& StateMgr_,
                     const Teuchos::RCP<const Epetra_Comm>& comm_) :
    Albany::AbstractAdapter(params_, paramLib_, StateMgr_, comm_),
    remeshFileIndex(1)
{


}

Albany::MeshAdapt::
~MeshAdapt()
{
}

bool
Albany::MeshAdapt::queryAdaptationCriteria(){


// FIXME Dumb criteria

   if(iter == 5 || iter == 10 || iter == 15){ // fracture at iter = 5, 10, 15

    // First, check and see if the mesh fracture criteria is met anywhere before messing with things.

    return true;

  }

  return false; 
 
}

bool
Albany::MeshAdapt::adaptMesh(){

  std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
  std::cout << "Adapting mesh using Albany::MeshAdapt method        " << std::endl;
  std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;

  return true;

}

//! Transfer solution between meshes.
void
Albany::MeshAdapt::
solutionTransfer(const Epetra_Vector& oldSolution,
        Epetra_Vector& newSolution){

}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::MeshAdapt::getValidAdapterParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericAdapterParams("ValidMeshAdaptParams");

/*
  if (numDim==1)
    validPL->set<bool>("Periodic BC", false, "Flag to indicate periodic BC for 1D problems");
  validPL->sublist("Thermal Conductivity", false, "");
  validPL->set("Convection Velocity", "{0,0,0}", "");
  validPL->set<bool>("Have Rho Cp", false, "Flag to indicate if rhoCp is used");
  validPL->set<string>("MaterialDB Filename","materials.xml","Filename of material database xml file");
*/

  return validPL;
}


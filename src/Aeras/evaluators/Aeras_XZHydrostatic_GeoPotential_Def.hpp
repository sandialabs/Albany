//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_RCP.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "Albany_Utils.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"
#include "Aeras_Layouts.hpp"

#include "Aeras_Eta.hpp"
namespace Aeras {

//**********************************************************************
template<typename EvalT, typename Traits>
XZHydrostatic_GeoPotential<EvalT, Traits>::
XZHydrostatic_GeoPotential(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Aeras::Layouts>& dl) :
  density   (p.get<std::string> ("Density")     , dl->node_scalar_level),
  Pi        (p.get<std::string> ("Pi")          , dl->node_scalar_level),
  Phi       (p.get<std::string> ("GeoPotential"), dl->node_scalar_level),
  PhiSurf   (p.get<std::string> ("SurfaceGeopotential")  , dl->node_scalar),
  
  numNodes ( dl->node_scalar          ->dimension(1)),
  numLevels( dl->node_scalar_level    ->dimension(2)),
  Phi0(0.0),
  E (Eta<EvalT>::self())
{

  Teuchos::ParameterList* xzhydrostatic_params =
    p.isParameter("XZHydrostatic Problem") ? 
      p.get<Teuchos::ParameterList*>("XZHydrostatic Problem"):
      p.get<Teuchos::ParameterList*>("Hydrostatic Problem");
  Phi0 = xzhydrostatic_params->get<double>("Phi0", 0.0); //Default: Phi0=0.0
  //std::cout << "XZHydrostatic_GeoPotential: Phi0 = " << Phi0 << std::endl;

  this->addDependentField(density);
  this->addDependentField(Pi);

  this->addDependentField(PhiSurf);
  
  this->addEvaluatedField(Phi);

  this->setName("Aeras::XZHydrostatic_GeoPotential" + PHX::typeAsString<EvalT>());

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  delta = E.delta_kokkos;
#endif
}

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_GeoPotential<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(density  , fm);
  this->utils.setFieldData(Pi       , fm);
  this->utils.setFieldData(Phi      , fm);
  
  this->utils.setFieldData(PhiSurf  , fm);
}

//**********************************************************************
// Kokkos kernels
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void XZHydrostatic_GeoPotential<EvalT, Traits>::
operator() (const int cell, const int node, const int level) const{
      ScalarT sum =
      PhiSurf(cell,node) +
      0.5 * Pi(cell,node,level) * delta(level) / density(cell,node,level);
      for (int j=level+1; j < numLevels; ++j) sum += Pi(cell,node,j)     * delta(j)     / density(cell,node,j);

      Phi(cell,node,level) = sum;
}

#endif

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_GeoPotential<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  for (int cell=0; cell < workset.numCells; ++cell) {
    for (int node=0; node < numNodes; ++node) {
      for (int level=0; level < numLevels; ++level) {
        ScalarT sum =
        PhiSurf(cell,node) +
        0.5 * Pi(cell,node,level) * E.delta(level) / density(cell,node,level);
        for (int j=level+1; j < numLevels; ++j) sum += Pi(cell,node,j)     * E.delta(j)     / density(cell,node,j);

        Phi(cell,node,level) = sum;
        
        //std::cout <<"Inside GeoP, cell, node, PhiSurf(cell,node)="<<cell<<
        //", "<<node<<", "<<PhiSurf(cell,node) <<std::endl;
      }
    }
  }

  /* OG Debugging statements
  std::cout << "Printing PHI at level 0 ----------------------------------------- \n";
  //for(int level=0; level < numLevels; ++level){
  for (int node=0; node < numNodes; ++node) {
	 //std::cout << "lev= " << level <<  ", phi = " << Phi(23,0,level) <<"\n";
	std::cout << "node = " << node <<  ", phi = " << Phi(23,node,0) <<"\n";
  }
  //}*/

#else
  XZHydrostatic_GeoPotential_Policy range(
      {0,0,0}, {(int)workset.numCells,(int)numNodes,(int)numLevels});
  Kokkos::Experimental::md_parallel_for(range,*this);

  cudaCheckError();

#endif
}
}

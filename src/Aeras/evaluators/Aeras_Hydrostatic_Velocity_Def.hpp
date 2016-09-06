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
Hydrostatic_Velocity<EvalT, Traits>::
Hydrostatic_Velocity(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Aeras::Layouts>& dl) :
  Velx         (p.get<std::string> ("Velx Name"),            dl->node_vector_level),
  sphere_coord (p.get<std::string> ("Spherical Coord Name"), dl->qp_gradient ),
  pressure     (p.get<std::string> ("Pressure"),             dl->node_scalar_level),
  Velocity     (p.get<std::string> ("Velocity"),             dl->node_vector_level),
  numNodes ( dl->node_scalar             ->dimension(1)),
  numDims  ( dl->node_qp_gradient        ->dimension(3)),
  numLevels( dl->node_scalar_level       ->dimension(2)),
  E (Eta<EvalT>::self()),
  out(Teuchos::VerboseObjectBase::getDefaultOStream())
{
  Teuchos::ParameterList* hs_list = p.get<Teuchos::ParameterList*>("Hydrostatic Problem");
  
  std::string advType = hs_list->get("Advection Type", "Unknown");
  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());

  *out << "Advection Type = " << advType << std::endl; 
  
  if (advType == "Unknown") {
    adv_type = UNKNOWN;
  }

  else if (advType == "Prescribed 1-1") {
    //Eqns. 19-23, DCMIP 2012, pp. 16,17 3D Deformational Flow
    adv_type = PRESCRIBED_1_1; 
    PI          = 3.141592653589793;
    earthRadius = 6.3712e6;
    ptop        = 25494.4;
    p0          = 100000.0;
    tau         = 1036800.0;
    omega0      = 23000.0*PI/tau; 
    k           = 10.0*earthRadius/tau;
  }

  else if (advType == "Prescribed 1-2") {
    adv_type = PRESCRIBED_1_2;
  }

  else {
    TEUCHOS_TEST_FOR_EXCEPTION(true,
                   Teuchos::Exceptions::InvalidParameter,"Aeras::Hydrostatic_Velocity: " 
                               << "Advection Type = " << advType << " is invalid!"); 
  }

  this->addDependentField(Velx);
  this->addDependentField(sphere_coord);
  this->addDependentField(pressure);
  this->addEvaluatedField(Velocity);

  this->setName("Aeras::Hydrostatic_Velocity" + PHX::typeAsString<EvalT>());

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  B = E.B_kokkos;
#endif
}

//**********************************************************************
template<typename EvalT, typename Traits>
void Hydrostatic_Velocity<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(Velx,         fm);
  this->utils.setFieldData(sphere_coord, fm);
  this->utils.setFieldData(pressure,     fm);
  this->utils.setFieldData(Velocity,     fm);
}

//**********************************************************************
// Kokkos kernels
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void Hydrostatic_Velocity<EvalT, Traits>::
operator() (const Hydrostatic_Velocity_Tag& tag, const int& cell) const{
  for (int node=0; node < numNodes; ++node) 
    for (int level=0; level < numLevels; ++level) 
      for (int dim=0; dim < numDims; ++dim)  
        Velocity(cell,node,level,dim) = Velx(cell,node,level,dim); 
}

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void Hydrostatic_Velocity<EvalT, Traits>::
operator() (const Hydrostatic_Velocity_PRESCRIBED_1_1_Tag& tag, const int& cell) const{
  for (int node=0; node < numNodes; ++node) {
    const MeshScalarT lambda = sphere_coord(cell, node, 0);
    const MeshScalarT theta = sphere_coord(cell, node, 1);
    ScalarT lambdap = lambda - 2.0*PI*time/tau;

    ScalarT Ua = k*sin(lambdap)*sin(lambdap)*sin(2.0*theta)*cos(PI*time/tau)
               + (2.0*PI*earthRadius/tau)*cos(theta);

    ScalarT Va = k*sin(2.0*lambdap)*cos(theta)*cos(PI*time/tau);

    for (int level=0; level < numLevels; ++level) {
      ScalarT B = this->B(level);
      ScalarT p = pressure(cell,node,level);

      ScalarT taper = - exp( (p    - p0)/(B*ptop) )
                      + exp( (ptop - p )/(B*ptop) );

      ScalarT Ud = (omega0*earthRadius)/(B*ptop)
                   *cos(lambdap)*cos(theta)*cos(theta)*cos(2.0*PI*time/tau)*taper;

      Velocity(cell,node,level,0) = Ua + Ud; 
      Velocity(cell,node,level,1) = Va; 
    }
  }
}

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void Hydrostatic_Velocity<EvalT, Traits>::
operator() (const Hydrostatic_Velocity_PRESCRIBED_1_2_Tag& tag, const int& cell) const{
  //FIXME: Pete, Tom - please fill in
  for (int node=0; node < numNodes; ++node) {
    const MeshScalarT lambda = sphere_coord(cell, node, 0);
    const MeshScalarT theta = sphere_coord(cell, node, 1);
    for (int level=0; level < numLevels; ++level) {
      for (int dim=0; dim < numDims; ++dim) {
        Velocity(cell,node,level,dim) = 0.0; //FIXME  
      }
    }
  }
}

#endif

//**********************************************************************
template<typename EvalT, typename Traits>
void Hydrostatic_Velocity<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  time = workset.current_time; 

#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  //*out << "Aeras::Hydrostatic_Velocity time = " << time << std::endl; 
  switch (adv_type) {
    case UNKNOWN: //velocity is an unknown that we solve for (not prescribed)
    {
      for (int cell=0; cell < workset.numCells; ++cell) 
        for (int node=0; node < numNodes; ++node) 
          for (int level=0; level < numLevels; ++level) 
            for (int dim=0; dim < numDims; ++dim)  
              Velocity(cell,node,level,dim) = Velx(cell,node,level,dim); 
    } 
    break; 

    case PRESCRIBED_1_1: //velocity is prescribed to that of 1-1 test
    {
      for (int cell=0; cell < workset.numCells; ++cell) { 
        for (int node=0; node < numNodes; ++node) {

          const MeshScalarT lambda = sphere_coord(cell, node, 0);
          const MeshScalarT theta = sphere_coord(cell, node, 1);
          ScalarT lambdap = lambda - 2.0*PI*time/tau;

          for (int level=0; level < numLevels; ++level) {

              ScalarT Ua = k*sin(lambdap)*sin(lambdap)*sin(2.0*theta)*cos(PI*time/tau)
                         + (2.0*PI*earthRadius/tau)*cos(theta);
    
              ScalarT Va = k*sin(2.0*lambdap)*cos(theta)*cos(PI*time/tau);

              ScalarT B = E.B(level);
              ScalarT p = pressure(cell,node,level);

              ScalarT taper = - exp( (p    - p0)/(B*ptop) )
                              + exp( (ptop - p )/(B*ptop) );

              ScalarT Ud = (omega0*earthRadius)/(B*ptop)
                           *cos(lambdap)*cos(theta)*cos(theta)*cos(2.0*PI*time/tau)*taper;

              Velocity(cell,node,level,0) = Ua + Ud; 
              Velocity(cell,node,level,1) = Va; 
          }
        }
      }
    }
    break; 

    case PRESCRIBED_1_2: //velocity is prescribed to that of 1-2 test
    {
      //FIXME: Pete, Tom - please fill in
      for (int cell=0; cell < workset.numCells; ++cell) { 
        for (int node=0; node < numNodes; ++node) {
          const MeshScalarT lambda = sphere_coord(cell, node, 0);
          const MeshScalarT theta = sphere_coord(cell, node, 1);
          for (int level=0; level < numLevels; ++level) {
            for (int dim=0; dim < numDims; ++dim) {
              Velocity(cell,node,level,dim) = 0.0; //FIXME  
            }
          }
        }
      }
    }
    break; 
  }

#else
  switch (adv_type) {
    case UNKNOWN: //velocity is an unknown that we solve for (not prescribed)
    {
      Kokkos::parallel_for(Hydrostatic_Velocity_Policy(0,workset.numCells),*this);
      cudaCheckError();
      break; 
    } 

    case PRESCRIBED_1_1: //velocity is prescribed to that of 1-1 test
    {
      Kokkos::parallel_for(Hydrostatic_Velocity_PRESCRIBED_1_1_Policy(0,workset.numCells),*this);
      cudaCheckError();
      break; 
    }

    case PRESCRIBED_1_2: //velocity is prescribed to that of 1-2 test
    {
      Kokkos::parallel_for(Hydrostatic_Velocity_PRESCRIBED_1_2_Policy(0,workset.numCells),*this);
      cudaCheckError();
      break; 
    }
  }

#endif
}
}

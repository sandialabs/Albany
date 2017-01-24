//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AERAS_HYDROSTATIC_SURFACEGEOPOTENTIAL_HPP
#define AERAS_HYDROSTATIC_SURFACEGEOPOTENTIAL_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Aeras_Layouts.hpp"
#include "Aeras_Dimension.hpp"

namespace Aeras {
/** \brief Surface geopotential (phi_s) for Hydrostatic atmospheric model

    This evaluator computes the surface geopotential for the Hydrostatic model
    of atmospheric dynamics.

*/
template<typename EvalT, typename Traits>
class Hydrostatic_SurfaceGeopotential : public PHX::EvaluatorWithBaseImpl<Traits>,
                   public PHX::EvaluatorDerived<EvalT, Traits> {

public:
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  Hydrostatic_SurfaceGeopotential(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Aeras::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:
  // Input
  PHX::MDField<MeshScalarT,Cell,Node,Dim> coordVec;

  // Output:
  PHX::MDField<ScalarT,Cell,Node> PhiSurf;

  const int numNodes;
                     
  enum TOPOGRAPHYTYPE {NONE, SPHERE_MOUNTAIN1, ASP_BAROCLINIC};
  TOPOGRAPHYTYPE topoType;
  
  int numParam;
  
  Teuchos::Array<double> topoData;

  // SPHERE_MOUNTAIN1 parameters:
  double cntrLat, cntrLon, mtnHeight, mtnWidth, mtnHalfWidth, PI, G;

  // ASP_BAROCLINIC parameters:
  double a, omega, eta0, etas, u0, pi;

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
public:
  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;

  struct Hydrostatic_SurfaceGeopotential_SPHERE_MOUNTAIN1_Tag{};
  struct Hydrostatic_SurfaceGeopotential_ASP_BAROCLINIC_Tag{};

  typedef Kokkos::RangePolicy<ExecutionSpace, Hydrostatic_SurfaceGeopotential_SPHERE_MOUNTAIN1_Tag> Hydrostatic_SurfaceGeopotential_SPHERE_MOUNTAIN1_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, Hydrostatic_SurfaceGeopotential_ASP_BAROCLINIC_Tag> Hydrostatic_SurfaceGeopotential_ASP_BAROCLINIC_Policy;

  KOKKOS_INLINE_FUNCTION
  void operator() (const Hydrostatic_SurfaceGeopotential_SPHERE_MOUNTAIN1_Tag& tag, const int& i) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (const Hydrostatic_SurfaceGeopotential_ASP_BAROCLINIC_Tag& tag, const int& i) const;

#endif
};
}

#endif

//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AERAS_XZHYDROSTATIC_SURFACEGEOPOTENTIAL_HPP
#define AERAS_XZHYDROSTATIC_SURFACEGEOPOTENTIAL_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Aeras_Layouts.hpp"
#include "Aeras_Dimension.hpp"

namespace Aeras {
/** \brief Surface geopotential (phi_s) for XZHydrostatic atmospheric model

    This evaluator computes the surface geopotential for the XZHydrostatic model
    of atmospheric dynamics.

*/
template<typename EvalT, typename Traits>
class XZHydrostatic_SurfaceGeopotential : public PHX::EvaluatorWithBaseImpl<Traits>,
                   public PHX::EvaluatorDerived<EvalT, Traits> {

public:
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  XZHydrostatic_SurfaceGeopotential(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Aeras::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:
  // Input
  PHX::MDField<MeshScalarT,Cell,Vertex,Dim> coordVec;

  // Output:
  PHX::MDField<ScalarT,Cell,Node> PhiSurf;

  const int numNodes;
                     
  enum TOPOGRAPHYTYPE {NONE, MOUNTAIN1};
  TOPOGRAPHYTYPE topoType;
  
  int numParam;
  
  Teuchos::Array<double> topoData;

  // MOUNTAIN1 parameters:
  double local_pi, local_gravity, center, width, height;

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
public:
  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;

  struct XZHydrostatic_SurfaceGeopotential_MOUNTAIN1_Tag{};

  typedef Kokkos::RangePolicy<ExecutionSpace, XZHydrostatic_SurfaceGeopotential_MOUNTAIN1_Tag> XZHydrostatic_SurfaceGeopotential_MOUNTAIN1_Policy;

  KOKKOS_INLINE_FUNCTION
  void operator() (const XZHydrostatic_SurfaceGeopotential_MOUNTAIN1_Tag& tag, const int& i) const;

private:
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > wsCoords;

#endif
};
}

#endif

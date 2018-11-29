//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_UPDATEZCOORDINATE_HPP
#define LANDICE_UPDATEZCOORDINATE_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Sacado_ParameterAccessor.hpp" 
#include "Albany_Layouts.hpp"

namespace LandIce {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits>
class UpdateZCoordinateMovingTop : public PHX::EvaluatorWithBaseImpl<Traits>,
		    public PHX::EvaluatorDerived<EvalT, Traits> {

public:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  UpdateZCoordinateMovingTop(const Teuchos::ParameterList& p,
            const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  // Input:
  PHX::MDField<const MeshScalarT, Cell, Node,Dim> coordVecIn;
  PHX::MDField<const MeshScalarT, Cell, Node> bedTopo;
  PHX::MDField<const MeshScalarT, Cell, Node> H;
  PHX::MDField<const MeshScalarT, Cell, Node> H0;
  PHX::MDField<const MeshScalarT, Cell, Node> dH;
  PHX::MDField<MeshScalarT, Cell, Node> topSurface;

  // Output:
  PHX::MDField<MeshScalarT, Cell, Node, Dim> coordVecOut;

  bool haveThickness;
  double minH, rho_i, rho_w;
  unsigned int numDims, numNodes;
};


template<typename EvalT, typename Traits>
class UpdateZCoordinateMovingBed : public PHX::EvaluatorWithBaseImpl<Traits>,
        public PHX::EvaluatorDerived<EvalT, Traits> {

public:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  UpdateZCoordinateMovingBed(const Teuchos::ParameterList& p,
            const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  // Input:
  PHX::MDField<const MeshScalarT, Cell, Node,Dim> coordVecIn;
  PHX::MDField<const MeshScalarT, Cell, Node> dH;
  PHX::MDField<const MeshScalarT, Cell, Node> bedTopo;
  PHX::MDField<const MeshScalarT, Cell, Node> topSurface;
  PHX::MDField<const MeshScalarT, Cell, Node> H;

  // Output:
  PHX::MDField<MeshScalarT, Cell, Node, Dim> coordVecOut;
  PHX::MDField<MeshScalarT, Cell, Node> topSurfaceOut;
  PHX::MDField<MeshScalarT, Cell, Node> bedTopoOut;

  double minH, rho_i, rho_w;
  unsigned int numDims, numNodes;
};
}

#endif

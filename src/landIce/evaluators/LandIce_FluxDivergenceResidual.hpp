//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_FLUX_DIVERGENCE_RESIDUAL_HPP
#define LANDICE_FLUX_DIVERGENCE_RESIDUAL_HPP

#include "PHAL_SeparableScatterScalarResponse.hpp"
#include "Shards_CellTopology.hpp"

namespace LandIce {
/**
 * \brief Computes the residual for the Layered Flux Divergence.
 *        The divergence is obtained in a Finite Volume fashion, computing the flux through the dual Voronoi cell and divided by the cell area.
 *        Let L be the number of layers. The residual at the node x_{i,l}, where l refers to the node level and i to the node 2d id,  is computed as
 *        resid_{i,L} = A_i fluxDiv_{i,L}
 *        resid_{i,l} = A_i fluxDiv_{i,l} - r_l \sum_{e_i} H  0.5(v_l+v_{l+1}) n_{e_i}, for l=0, 1, L-1
 *         *
 *        where H_l and v_l are the thickness and velocity at layer l; A_i is the area of the Voronoi cell at point i and e_i its edges;
 *        r_l is the thickness layer ratio of layer l.
 *
 *        Once fluxDiv_{i,L} has been computed, the total flux divergence div(H v) can be computed by \sum_l fluxDiv_{i,L},
 *        which is implemented in the Gather Vertically Contracted Solution evaluator.
 *
 *        This evaluator works only for prismatic elements (otherwise error is thrown),
 *        and it is correct only when the triangulation does not have obtuse angles.
 */



  template<typename EvalT, typename Traits, typename ThicknessScalarT>
  class LayeredFluxDivergenceResidual : public PHX::EvaluatorWithBaseImpl<Traits>,
        public PHX::EvaluatorDerived<EvalT, Traits>
  {
  public:
    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;

    LayeredFluxDivergenceResidual(Teuchos::ParameterList& p,
       const Teuchos::RCP<Albany::Layouts>& dl);

    void postRegistrationSetup(typename Traits::SetupData d,
             PHX::FieldManager<Traits>& vm);

    void evaluateFields(typename Traits::EvalData d);

  private:

    unsigned int numCells;
    unsigned int numNodes;
    bool upwindStabilization;

    PHX::MDField<const MeshScalarT, Cell, Node, Dim> coords;
    PHX::MDField<const ThicknessScalarT> H;
    PHX::MDField<const ScalarT> vel;
    PHX::MDField<const ScalarT> flux_div;

    PHX::MDField<ScalarT> residual;
  };




  /* Original nonlayered implementation. The issue with this implementation is that,
   * in order to correctly compute the derivatives it would require the Jacobian pattern to include links between any node
   * and the node at the base of the column. This is because the flux div would depend on the vertically averaged velocity.

  template<typename EvalT, typename Traits>
  class FluxDivergenceResidual : public PHX::EvaluatorWithBaseImpl<Traits>,
        public PHX::EvaluatorDerived<EvalT, Traits>
  {
  public:
    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;
    typedef typename EvalT::ParamScalarT ParamScalarT;

    FluxDivergenceResidual(Teuchos::ParameterList& p,
       const Teuchos::RCP<Albany::Layouts>& dl);

    void postRegistrationSetup(typename Traits::SetupData d,
             PHX::FieldManager<Traits>& vm);

    void evaluateFields(typename Traits::EvalData d);

  private:

    Teuchos::RCP<shards::CellTopology> cellType;
    std::vector<std::vector<int> >  sideNodes;
    std::string sideName;

    int numCells;
    int numNodes;
    int sideDim;
    int numSideNodes;
    bool upwindStabilization;

    PHX::MDField<const MeshScalarT, Cell, Node, Dim> coords;
    PHX::MDField<const ParamScalarT> H;
    PHX::MDField<const ScalarT> vel;
    PHX::MDField<const ScalarT> flux_div;

    PHX::MDField<ScalarT> residual;
  };
  */

} // Namespace LandIce

#endif // LANDICE_FLUX_DIVERGENCE_RESIDUAL_HPP

//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_L2_PROJECTED_BOUNDARY_LAPLACIAN_RESIDUAL_HPP
#define FELIX_L2_PROJECTED_BOUNDARY_LAPLACIAN_RESIDUAL_HPP

//#include "FELIX_MeshRegion.hpp"
#include "PHAL_SeparableScatterScalarResponse.hpp"

namespace FELIX {
/**
 * \brief Response Description
 */
  template<typename EvalT, typename Traits>
  class L2ProjectedBoundaryLaplacianResidual : public PHX::EvaluatorWithBaseImpl<Traits>,
        public PHX::EvaluatorDerived<EvalT, Traits>
  {
  public:
    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;
    typedef typename EvalT::ParamScalarT ParamScalarT;

    L2ProjectedBoundaryLaplacianResidual(Teuchos::ParameterList& p,
       const Teuchos::RCP<Albany::Layouts>& dl);

    void postRegistrationSetup(typename Traits::SetupData d,
             PHX::FieldManager<Traits>& vm);

    void evaluateFields(typename Traits::EvalData d);


  private:

    std::string sideName;
    std::vector<std::vector<int> >  sideNodes;
    
    int numCells;
    int numNodes;

    int numSideNodes;
    int numBasalQPs;
    int numSideDims;
    int sideDim;

    PHX::MDField<const ScalarT,Cell,Node>             solution;
    PHX::MDField<const ParamScalarT,Cell,Side,Node>                 field;
    PHX::MDField<const ScalarT,Cell,Side,QuadPoint,Dim>        gradField;
    PHX::MDField<const MeshScalarT,Cell,Side,Node,QuadPoint,Dim>   gradBF;
    PHX::MDField<const MeshScalarT,Cell,Side,QuadPoint>        w_side_measure;

    PHX::MDField<ScalarT,Cell,Node> bdLaplacian_L2Projection_res;

    ScalarT p_reg, reg;
    double laplacian_coeff, mass_coeff;
    int offset;
  };

} // Namespace FELIX

#endif // FELIX_RESPONSE_SURFACE_VELOCITY_MISMATCH_HPP

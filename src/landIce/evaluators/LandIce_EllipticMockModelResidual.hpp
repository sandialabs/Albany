//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef MOCK_ELLIPTIC_EQUATION_RESIDUAL_HPP
#define MOCK_ELLIPTIC_EQUATION_RESIDUAL_HPP

#include "PHAL_SeparableScatterScalarResponse.hpp"
#include "Shards_CellTopology.hpp"

namespace LandIce {
/**
 * \brief Response Description
 */
  template<typename EvalT, typename Traits>
  class EllipticMockModelResidual : public PHX::EvaluatorWithBaseImpl<Traits>,
        public PHX::EvaluatorDerived<EvalT, Traits>
  {
  public:
    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;
    typedef typename EvalT::ParamScalarT ParamScalarT;

    EllipticMockModelResidual(Teuchos::ParameterList& p,
       const Teuchos::RCP<Albany::Layouts>& dl);

    void postRegistrationSetup(typename Traits::SetupData d,
             PHX::FieldManager<Traits>& vm);

    void evaluateFields(typename Traits::EvalData d);


  private:

    Albany::LocalSideSetInfo sideSet;

    std::string sideName;
    Kokkos::DualView<int**, PHX::Device> sideNodes;
    Teuchos::RCP<shards::CellTopology> cellType;
    Teuchos::RCP<shards::CellTopology> sideType;
    
    unsigned int numCells;
    unsigned int numNodes;
    unsigned int numQPs;
    unsigned int cellDim;

    unsigned int numSideNodes;
    unsigned int numSideQPs;
    unsigned int sideDim;

    PHX::MDField<const ScalarT>       field;
    PHX::MDField<const ScalarT>       side_field;
    PHX::MDField<const ScalarT>       gradField;
    PHX::MDField<const ParamScalarT>  forcing;
    PHX::MDField<const ParamScalarT>  param0; // nominal parameter for linearized model
    PHX::MDField<const ParamScalarT>  param; // parameter 
    PHX::MDField<const ParamScalarT>  field0; // nominal field for linearized model
    PHX::MDField<const ParamScalarT>  gradField0; // nominal field grad for linearized model
    PHX::MDField<const MeshScalarT>   gradBF;
    PHX::MDField<const RealType>      BF;
    PHX::MDField<const RealType>      side_BF;
    PHX::MDField<const MeshScalarT>   w_measure;
    PHX::MDField<const MeshScalarT>   w_side_measure;

    PHX::MDField<ScalarT> residual;

    RealType eps; //regularization for viscosity   [yr^-1]
    RealType n; //Glen's law exponent
    double laplacian_coeff, robin_coeff;
    bool linearized_model;

  public:

    typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;

    struct EllipticMockModel_Cell_Tag{};
    struct EllipticMockModel_Side_Tag{};

    typedef Kokkos::RangePolicy<ExecutionSpace,EllipticMockModel_Cell_Tag> EllipticMockModel_Cell_Policy;
    typedef Kokkos::RangePolicy<ExecutionSpace,EllipticMockModel_Side_Tag> EllipticMockModel_Side_Policy;

    KOKKOS_INLINE_FUNCTION
    void operator() (const EllipticMockModel_Cell_Tag& tag, const int& i) const;
    KOKKOS_INLINE_FUNCTION
    void operator() (const EllipticMockModel_Side_Tag& tag, const int& i) const;
  };

} // Namespace LandIce

#endif // LANDICE_RESPONSE_SURFACE_VELOCITY_MISMATCH_HPP

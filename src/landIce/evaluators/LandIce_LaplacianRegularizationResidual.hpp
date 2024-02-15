//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_LAPLACIAN_REGULAIZATION_RESIDUAL_HPP
#define LANDICE_LAPLACIAN_REGULAIZATION_RESIDUAL_HPP

#include "PHAL_SeparableScatterScalarResponse.hpp"
#include "Shards_CellTopology.hpp"

namespace LandIce {
/**
 * \brief Response Description
 */
  template<typename EvalT, typename Traits>
  class LaplacianRegularizationResidual : public PHX::EvaluatorWithBaseImpl<Traits>,
        public PHX::EvaluatorDerived<EvalT, Traits>
  {
  public:
    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;
    typedef typename EvalT::ParamScalarT ParamScalarT;

    LaplacianRegularizationResidual(Teuchos::ParameterList& p,
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
    PHX::MDField<const MeshScalarT>   gradBF;
    PHX::MDField<const RealType>      BF;
    PHX::MDField<const RealType>      side_BF;
    PHX::MDField<const MeshScalarT>   w_measure;
    PHX::MDField<const MeshScalarT>   w_side_measure;

    PHX::MDField<ScalarT> residual;

    ScalarT p_reg, reg;
    double laplacian_coeff, mass_coeff, robin_coeff;
    Teuchos::Array<double> advection_vect;

    bool lumpedMassMatrix;

  public:

    typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;

    struct LaplacianRegularization_Cell_Tag{};
    struct LaplacianRegularization_Side_Tag{};

    typedef Kokkos::RangePolicy<ExecutionSpace,LaplacianRegularization_Cell_Tag> LaplacianRegularization_Cell_Policy;
    typedef Kokkos::RangePolicy<ExecutionSpace,LaplacianRegularization_Side_Tag> LaplacianRegularization_Side_Policy;

    KOKKOS_INLINE_FUNCTION
    void operator() (const LaplacianRegularization_Cell_Tag& tag, const int& i) const;
    KOKKOS_INLINE_FUNCTION
    void operator() (const LaplacianRegularization_Side_Tag& tag, const int& i) const;
  };

} // Namespace LandIce

#endif // LANDICE_RESPONSE_SURFACE_VELOCITY_MISMATCH_HPP

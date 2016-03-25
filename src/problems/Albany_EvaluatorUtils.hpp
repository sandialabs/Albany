//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_EVALUATORUTILS_HPP
#define ALBANY_EVALUATORUTILS_HPP

#include <vector>
#include <string>

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Phalanx.hpp"
#include "Albany_DataTypes.hpp"
#include "PHAL_AlbanyTraits.hpp"

#include "Teuchos_VerboseObject.hpp"

#include "Albany_ProblemUtils.hpp"

#include "Intrepid2_Basis.hpp"
#include "Intrepid2_FieldContainer.hpp"
#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"


namespace Albany {
  /*!
   * \brief Generic Functions to construct evaluators more succinctly
   */
  template<typename EvalT, typename Traits>
  class EvaluatorUtils {

   public:

    EvaluatorUtils(Teuchos::RCP<Albany::Layouts> dl);

    //! Function to create parameter list for construction of GatherSolution
    //! evaluator with standard Field names
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructGatherSolutionEvaluator(
       bool isVectorField,
       Teuchos::ArrayRCP<std::string> dof_names,
       Teuchos::ArrayRCP<std::string> dof_names_dot,
       int offsetToFirstDOF=0);

    //! Function to create parameter list for construction of GatherSolution
    //! evaluator with standard Field names.
    //! Tensor rank of solution variable is 0, 1, or 2
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructGatherSolutionEvaluator(
       int tensorRank,
       Teuchos::ArrayRCP<std::string> dof_names,
       Teuchos::ArrayRCP<std::string> dof_names_dot,
       int offsetToFirstDOF=0);


    //! Function to create parameter list for construction of GatherSolution
    //! evaluator with acceleration terms
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructGatherSolutionEvaluator_withAcceleration(
       bool isVectorField,
       Teuchos::ArrayRCP<std::string> dof_names,
       Teuchos::ArrayRCP<std::string> dof_names_dot, // can be Teuchos::null
       Teuchos::ArrayRCP<std::string> dof_names_dotdot,
       int offsetToFirstDOF=0);

    //! Function to create parameter list for construction of GatherSolution
    //! evaluator with acceleration terms.
    //! Tensor rank of solution variable is 0, 1, or 2
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructGatherSolutionEvaluator_withAcceleration(
       int tensorRank,
       Teuchos::ArrayRCP<std::string> dof_names,
       Teuchos::ArrayRCP<std::string> dof_names_dot, // can be Teuchos::null
       Teuchos::ArrayRCP<std::string> dof_names_dotdot,
       int offsetToFirstDOF=0);


    //! Same as above, but no ability to gather time dependent x_dot field
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructGatherSolutionEvaluator_noTransient(
       bool isVectorField,
       Teuchos::ArrayRCP<std::string> dof_names,
       int offsetToFirstDOF=0);

    //! Same as above, but no ability to gather time dependent x_dot field
    //! Tensor rank of solution variable is 0, 1, or 2
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructGatherSolutionEvaluator_noTransient(
       int tensorRank,
       Teuchos::ArrayRCP<std::string> dof_names,
       int offsetToFirstDOF=0);

    //! Function to create parameter list for construction of ScatterResidual
    //! evaluator with standard Field names
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructScatterResidualEvaluator(
       bool isVectorField,
       Teuchos::ArrayRCP<std::string> resid_names,
       int offsetToFirstDOF=0, std::string scatterName="Scatter");

    //! Function to create parameter list for construction of GatherScalarNodalParameter
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructGatherScalarNodalParameter(
        const std::string& param_name,
        const std::string& field_name="");

    //! Function to create parameter list for construction of ScatterResidual
    //! evaluator with standard Field names
    //! Tensor rank of solution variable is 0, 1, or 2
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructScatterResidualEvaluator(
       int tensorRank,
       Teuchos::ArrayRCP<std::string> resid_names,
       int offsetToFirstDOF=0, std::string scatterName="Scatter");

    //! Function to create parameter list for construction of DOFInterpolation
    //! evaluator with standard field names
    //! AGS Note 10/13: oddsetToFirstDOF is added to DOF evaluators
    //!  for template specialization of Jacobian evaluation for
    //   performance. Otherwise it was not needed. With this info,
    //   the location of the nonzero partial derivatives can be
    //   computed, and the chain rule is coded with that known sparsity.
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructDOFInterpolationEvaluator(
       const std::string& dof_names, int offsetToFirstDOF=0);

    Teuchos::RCP< PHX::Evaluator<Traits> >
      constructDOFInterpolationEvaluator_noDeriv(
         const std::string& dof_names);

    //! Same as above, for Interpolating the Gradient
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructDOFGradInterpolationEvaluator(
       const std::string& dof_names, int offsetToFirstDOF=0);

    //! Interpolating the Gradient of quantity with no derivs
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructDOFGradInterpolationEvaluator_noDeriv(
       const std::string& dof_names);

    //! Interpolation functions for vector quantities
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructDOFVecInterpolationEvaluator(
       const std::string& dof_names, int offsetToFirstDOF=0);
    //! Same as above, for Interpolating the Gradient for Vector quantities
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructDOFVecGradInterpolationEvaluator(
       const std::string& dof_names, int offsetToFirstDOF=0);

    //! Interpolation functions for Tensor quantities
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructDOFTensorInterpolationEvaluator(
       const std::string& dof_names, int offsetToFirstDOF=0);
    //! Same as above, for Interpolating the Gradient for Tensor quantities
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructDOFTensorGradInterpolationEvaluator(
       const std::string& dof_names, int offsetToFirstDOF=0);

    //! Interpolation functions for scalar quantities defined on a side set
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructDOFInterpolationSideEvaluator(
       const std::string& dof_names,
       const std::string& sideSetName);

    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructDOFInterpolationSideEvaluator_noDeriv(
       const std::string& dof_names,
       const std::string& sideSetName);

    //! Interpolation functions for vector ScalarT defined on a side set
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructDOFVecInterpolationSideEvaluator(
       const std::string& dof_names,
       const std::string& sideSetName);

    //! Interpolation functions for vector ParamScalarT quantities defined on a side set
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructDOFVecInterpolationSideEvaluator_noDeriv(
       const std::string& dof_names,
       const std::string& sideSetName);

    //! Interpolation functions for gradient of quantities defined on a side set
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructDOFGradInterpolationSideEvaluator(
      const std::string& dof_names,
      const std::string& sideSetName);

    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructDOFGradInterpolationSideEvaluator_noDeriv(
      const std::string& dof_names,
      const std::string& sideSetName);

    //! Interpolation functions for gradient of vector quantities defined on a side set
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructDOFVecGradInterpolationSideEvaluator(
      const std::string& dof_names,
      const std::string& sideSetName);

    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructDOFVecGradInterpolationSideEvaluator_noDeriv(
      const std::string& dof_names,
      const std::string& sideSetName);

    //! Function to create parameter list for construction of GatherCoordinateVector
    //! evaluator with standard Field names
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructGatherCoordinateVectorEvaluator(std::string strCurrentDisp="");

    //! Function to create parameter list for construction of MapToPhysicalFrame
    //! evaluator with standard Field names
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructMapToPhysicalFrameEvaluator(
      const Teuchos::RCP<shards::CellTopology>& cellType,
      const Teuchos::RCP<Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,PHX::Device> > > cubature,
      const Teuchos::RCP<Intrepid2::Basis<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > > intrepidBasis = Teuchos::null);

    //! Function to create parameter list for construction of MapToPhysicalFrameSide
    //! evaluator with standard Field names
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructMapToPhysicalFrameSideEvaluator(
      const Teuchos::RCP<shards::CellTopology>& cellType,
      const Teuchos::RCP<Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,PHX::Device> > > cubature,
      const std::string& sideSetName);

    //! Function to create evaluator for restriction to side set
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructDOFCellToSideEvaluator(
       const std::string& cell_dof_name,
       const std::string& sideSetName,
       const Teuchos::RCP<shards::CellTopology>& cellType,
       const std::string& side_dof_name = "");

    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructDOFCellToSideEvaluator_noDeriv(
       const std::string& cell_dof_name,
       const std::string& sideSetName,
       const Teuchos::RCP<shards::CellTopology>& cellType,
       const std::string& side_dof_name = "");

    //! Same as above, for vector quantities
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructDOFVecCellToSideEvaluator(
       const std::string& cell_dof_name,
       const std::string& sideSetName,
       const Teuchos::RCP<shards::CellTopology>& cellType,
       const std::string& side_dof_name = "");

    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructDOFVecCellToSideEvaluator_noDeriv(
       const std::string& cell_dof_name,
       const std::string& sideSetName,
       const Teuchos::RCP<shards::CellTopology>& cellType,
       const std::string& side_dof_name = "");

    //! Function to create evaluator NodesToCellInterpolation (=DOFInterpolation+QuadPointsToCellInterpolation)
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructNodesToCellInterpolationEvaluator(
      const std::string& dof_name,
      bool isVectorField = false);

    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructNodesToCellInterpolationEvaluator_noDeriv(
      const std::string& dof_name,
      bool isVectorField = false);

    //! Function to create evaluator QuadPointsToCellInterpolation
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructQuadPointsToCellInterpolationEvaluator(
      const std::string& dof_name,
      bool isVectorField = false);

    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructQuadPointsToCellInterpolationEvaluator_noDeriv(
      const std::string& dof_name,
      bool isVectorField = false);

    //! Function to create evaluator QuadPointsToCellInterpolation
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructSideQuadPointsToSideInterpolationEvaluator(
      const std::string& dof_name,
      const std::string& sideSetName,
      bool isVectorField = false);

    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructSideQuadPointsToSideInterpolationEvaluator_noDeriv(
      const std::string& dof_name,
      const std::string& sideSetName,
      bool isVectorField = false);

    //! Function to create parameter list for construction of ComputeBasisFunctions
    //! evaluator with standard Field names
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructComputeBasisFunctionsEvaluator(
      const Teuchos::RCP<shards::CellTopology>& cellType,
      const Teuchos::RCP<Intrepid2::Basis<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > > intrepidBasis,
      const Teuchos::RCP<Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,PHX::Device> > > cubature);

    //! Function to create parameter list for construction of ComputeBasisFunctionsSide
    //! evaluator with standard Field names
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructComputeBasisFunctionsSideEvaluator(
      const Teuchos::RCP<shards::CellTopology>& cellType,
      const Teuchos::RCP<Intrepid2::Basis<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > > intrepidBasisSide,
      const Teuchos::RCP<Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,PHX::Device> > > cubatureSide,
      const std::string& sideSetName);

  private:

    //! Struct of PHX::DataLayout objects defined all together.
    Teuchos::RCP<Albany::Layouts> dl;

  };
}

#endif

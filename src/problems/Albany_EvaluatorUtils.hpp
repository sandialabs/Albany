//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
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
  template<typename EvalT, typename Traits, typename ScalarT>
  class EvaluatorUtilsBase {

   public:

    typedef typename EvalT::MeshScalarT   MeshScalarT;
    typedef typename EvalT::ParamScalarT  ParamScalarT;

    EvaluatorUtilsBase(Teuchos::RCP<Albany::Layouts> dl);

    const EvaluatorUtilsBase<EvalT,Traits,MeshScalarT>&
    getMSTUtils()
    {
      if (utils_MST==Teuchos::null)
        utils_MST = Teuchos::rcp(new EvaluatorUtilsBase<EvalT,Traits,MeshScalarT>(dl));
      return *utils_MST;
    }

    const EvaluatorUtilsBase<EvalT,Traits,ParamScalarT>&
    getPSTUtils()
    {
      if (utils_PST==Teuchos::null)
        utils_PST = Teuchos::rcp(new EvaluatorUtilsBase<EvalT,Traits,ParamScalarT>(dl));
      return *utils_PST;
    }

    const EvaluatorUtilsBase<EvalT,Traits,RealType>&
    getRTUtils()
    {
      if (utils_RT==Teuchos::null)
        utils_RT = Teuchos::rcp(new EvaluatorUtilsBase<EvalT,Traits,RealType>(dl));
      return *utils_RT;
    }

    //! Function to create parameter list for construction of GatherSolution
    //! evaluator with standard Field names
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructGatherSolutionEvaluator(
       bool isVectorField,
       Teuchos::ArrayRCP<std::string> dof_names,
       Teuchos::ArrayRCP<std::string> dof_names_dot,
       int offsetToFirstDOF=0) const;

    //! Function to create parameter list for construction of GatherSolution
    //! evaluator with standard Field names.
    //! Tensor rank of solution variable is 0, 1, or 2
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructGatherSolutionEvaluator(
       int tensorRank,
       Teuchos::ArrayRCP<std::string> dof_names,
       Teuchos::ArrayRCP<std::string> dof_names_dot,
       int offsetToFirstDOF=0) const;


    //! Function to create parameter list for construction of GatherSolution
    //! evaluator with acceleration terms
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructGatherSolutionEvaluator_withAcceleration(
       bool isVectorField,
       Teuchos::ArrayRCP<std::string> dof_names,
       Teuchos::ArrayRCP<std::string> dof_names_dot, // can be Teuchos::null
       Teuchos::ArrayRCP<std::string> dof_names_dotdot,
       int offsetToFirstDOF=0) const;

    //! Function to create parameter list for construction of GatherSolution
    //! evaluator with acceleration terms.
    //! Tensor rank of solution variable is 0, 1, or 2
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructGatherSolutionEvaluator_withAcceleration(
       int tensorRank,
       Teuchos::ArrayRCP<std::string> dof_names,
       Teuchos::ArrayRCP<std::string> dof_names_dot, // can be Teuchos::null
       Teuchos::ArrayRCP<std::string> dof_names_dotdot,
       int offsetToFirstDOF=0) const;


    //! Same as above, but no ability to gather time dependent x_dot field
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructGatherSolutionEvaluator_noTransient(
       bool isVectorField,
       Teuchos::ArrayRCP<std::string> dof_names,
       int offsetToFirstDOF=0) const;

    //! Same as above, but no ability to gather time dependent x_dot field
    //! Tensor rank of solution variable is 0, 1, or 2
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructGatherSolutionEvaluator_noTransient(
       int tensorRank,
       Teuchos::ArrayRCP<std::string> dof_names,
       int offsetToFirstDOF=0) const;

    //! Function to create parameter list for construction of ScatterResidual
    //! evaluator with standard Field names
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructScatterResidualEvaluator(
       bool isVectorField,
       Teuchos::ArrayRCP<std::string> resid_names,
       int offsetToFirstDOF=0, std::string scatterName="Scatter") const;

    //! Function to create parameter list for construction of ScatterResidual
    //! evaluator with standard Field names
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructScatterResidualEvaluatorWithExtrudedParams(
       bool isVectorField,
       Teuchos::ArrayRCP<std::string> resid_names,
       Teuchos::RCP<std::map<std::string, int> > extruded_params_levels,
       int offsetToFirstDOF=0, std::string scatterName="Scatter") const;

    //! Function to create parameter list for construction of GatherScalarNodalParameter
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructGatherScalarNodalParameter(
        const std::string& param_name,
        const std::string& field_name="") const;

    //! Function to create parameter list for construction of GatherScalarExtruded2DNodalParameter
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructGatherScalarExtruded2DNodalParameter(
        const std::string& param_name,
        const std::string& field_name="") const;

    //! Function to create parameter list for construction of ScatterResidual
    //! evaluator with standard Field names
    //! Tensor rank of solution variable is 0, 1, or 2
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructScatterResidualEvaluator(
       int tensorRank,
       Teuchos::ArrayRCP<std::string> resid_names,
       int offsetToFirstDOF=0, std::string scatterName="Scatter") const;

    //! Function to create parameter list for construction of DOFInterpolation
    //! evaluator with standard field names
    //! AGS Note 10/13: oddsetToFirstDOF is added to DOF evaluators
    //!  for template specialization of Jacobian evaluation for
    //   performance. Otherwise it was not needed. With this info,
    //   the location of the nonzero partial derivatives can be
    //   computed, and the chain rule is coded with that known sparsity.
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructDOFInterpolationEvaluator(
       const std::string& dof_names, int offsetToFirstDOF=0) const;

    Teuchos::RCP< PHX::Evaluator<Traits> >
      constructDOFInterpolationEvaluator_noDeriv(
         const std::string& dof_names) const;

    //! Same as above, for Interpolating the Gradient
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructDOFGradInterpolationEvaluator(
       const std::string& dof_names, int offsetToFirstDOF=0) const;

    //! Interpolating the Gradient of quantity with no derivs
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructDOFGradInterpolationEvaluator_noDeriv(
       const std::string& dof_names) const;

    //! Interpolation functions for vector quantities
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructDOFVecInterpolationEvaluator(
       const std::string& dof_names, int offsetToFirstDOF=0) const;

    //! Same as above, for Interpolating the Gradient for Vector quantities
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructDOFVecGradInterpolationEvaluator(
       const std::string& dof_names, int offsetToFirstDOF=0) const;

    //! Interpolation functions for Tensor quantities
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructDOFTensorInterpolationEvaluator(
       const std::string& dof_names, int offsetToFirstDOF=0) const;
    //! Same as above, for Interpolating the Gradient for Tensor quantities
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructDOFTensorGradInterpolationEvaluator(
       const std::string& dof_names, int offsetToFirstDOF=0) const;

    //! Interpolation functions for scalar quantities defined on a side set
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructDOFInterpolationSideEvaluator(
       const std::string& dof_names,
       const std::string& sideSetName) const;

    //! Interpolation functions for vector defined on a side set
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructDOFVecInterpolationSideEvaluator(
       const std::string& dof_names,
       const std::string& sideSetName) const;

    //! Interpolation functions for gradient of quantities defined on a side set
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructDOFGradInterpolationSideEvaluator(
      const std::string& dof_names,
      const std::string& sideSetName) const;

    //! Interpolation functions for gradient of vector quantities defined on a side set
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructDOFVecGradInterpolationSideEvaluator(
      const std::string& dof_names,
      const std::string& sideSetName) const;

    //! Interpolation functions for divergence of quantities defined on a side set
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructDOFDivInterpolationSideEvaluator(
      const std::string& dof_names,
      const std::string& sideSetName) const;

    //! Function to create parameter list for construction of GatherCoordinateVector
    //! evaluator with standard Field names
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructGatherCoordinateVectorEvaluator(std::string strCurrentDisp="") const;

    //! Function to create parameter list for construction of MapToPhysicalFrame
    //! evaluator with standard Field names
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructMapToPhysicalFrameEvaluator(
      const Teuchos::RCP<shards::CellTopology>& cellType,
      const Teuchos::RCP<Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,PHX::Device> > > cubature,
      const Teuchos::RCP<Intrepid2::Basis<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > > intrepidBasis = Teuchos::null) const;

    //! Function to create parameter list for construction of MapToPhysicalFrameSide
    //! evaluator with standard Field names
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructMapToPhysicalFrameSideEvaluator(
      const Teuchos::RCP<shards::CellTopology>& cellType,
      const Teuchos::RCP<Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,PHX::Device> > > cubature,
      const std::string& sideSetName) const;

    //! Function to create evaluator for restriction to side set
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructDOFCellToSideEvaluator(
       const std::string& cell_dof_name,
       const std::string& sideSetName,
       const std::string& layout,
       const Teuchos::RCP<shards::CellTopology>& cellType = Teuchos::null,
       const std::string& side_dof_name = "") const;

    //! Function to create evaluator for prolongation to cell
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructDOFSideToCellEvaluator(
       const std::string& side_dof_name,
       const std::string& sideSetName,
       const std::string& layout,
       const Teuchos::RCP<shards::CellTopology>& cellType = Teuchos::null,
       const std::string& cell_dof_name = "") const;

    //! Function to create evaluator NodesToCellInterpolation (=DOFInterpolation+QuadPointsToCellInterpolation)
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructNodesToCellInterpolationEvaluator(
      const std::string& dof_name,
      bool isVectorField = false) const;

    //! Function to create evaluator QuadPointsToCellInterpolation
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructQuadPointsToCellInterpolationEvaluator(
      const std::string& dof_name,
      bool isVectorField = false) const;

    //! Function to create evaluator QuadPointsToCellInterpolation
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructSideQuadPointsToSideInterpolationEvaluator(
      const std::string& dof_name,
      const std::string& sideSetName,
      bool isVectorField = false) const;

    //! Function to create parameter list for construction of ComputeBasisFunctions
    //! evaluator with standard Field names
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructComputeBasisFunctionsEvaluator(
      const Teuchos::RCP<shards::CellTopology>& cellType,
      const Teuchos::RCP<Intrepid2::Basis<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > > intrepidBasis,
      const Teuchos::RCP<Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,PHX::Device> > > cubature) const;

    //! Function to create parameter list for construction of ComputeBasisFunctionsSide
    //! evaluator with standard Field names
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructComputeBasisFunctionsSideEvaluator(
      const Teuchos::RCP<shards::CellTopology>& cellType,
      const Teuchos::RCP<Intrepid2::Basis<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > > intrepidBasisSide,
      const Teuchos::RCP<Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,PHX::Device> > > cubatureSide,
      const std::string& sideSetName) const;

  private:

    //! Evaluator Utils with different ScalarType
    Teuchos::RCP<EvaluatorUtilsBase<EvalT,Traits,MeshScalarT>>    utils_MST;
    Teuchos::RCP<EvaluatorUtilsBase<EvalT,Traits,ParamScalarT>>   utils_PST;
    Teuchos::RCP<EvaluatorUtilsBase<EvalT,Traits,RealType>>   utils_RT;

    //! Struct of PHX::DataLayout objects defined all together.
    Teuchos::RCP<Albany::Layouts> dl;

  };

template<typename EvalT, typename Traits>
using EvaluatorUtils = EvaluatorUtilsBase<EvalT,Traits,typename EvalT::ScalarT>;

}

#endif

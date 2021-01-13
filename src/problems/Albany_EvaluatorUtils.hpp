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
#include "Teuchos_VerboseObject.hpp"

#include "Intrepid2_Basis.hpp"
#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"

#include <Phalanx_Evaluator.hpp>

#include "Albany_ScalarOrdinalTypes.hpp"
#include "Albany_Layouts.hpp"

namespace Albany {

  /*!
   * \brief Generic Functions to construct evaluators more succinctly
   */
  template<typename Traits>
  class EvaluatorUtilsBase {
    public:

    using IntrepidBasis    = Intrepid2::Basis<PHX::Device, RealType, RealType>;
    using IntrepidCubature = Intrepid2::Cubature<PHX::Device>;

    virtual ~EvaluatorUtilsBase() = default;

    //! Function to create parameter list for construction of GatherSolution
    //! evaluator with standard Field names
    Teuchos::RCP< PHX::Evaluator<Traits> >
    virtual constructGatherSolutionEvaluator(
       bool isVectorField,
       Teuchos::ArrayRCP<std::string> dof_names,
       Teuchos::ArrayRCP<std::string> dof_names_dot,
       int offsetToFirstDOF=0) const = 0;

    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructGatherSolutionEvaluator(
       bool isVectorField,
       const std::string& dof_name,
       const std::string& dof_name_dot,
       int offsetToFirstDOF=0) const {
      return constructGatherSolutionEvaluator(isVectorField,arcp_str(dof_name),arcp_str(dof_name_dot),offsetToFirstDOF);
    }

    //! Function to create parameter list for construction of GatherSolution
    //! evaluator with standard Field names.
    //! Tensor rank of solution variable is 0, 1, or 2
    Teuchos::RCP< PHX::Evaluator<Traits> >
    virtual constructGatherSolutionEvaluator(
       int tensorRank,
       Teuchos::ArrayRCP<std::string> dof_names,
       Teuchos::ArrayRCP<std::string> dof_names_dot,
       int offsetToFirstDOF=0) const = 0;


    //! Function to create parameter list for construction of GatherSolution
    //! evaluator with acceleration terms
    Teuchos::RCP< PHX::Evaluator<Traits> >
    virtual constructGatherSolutionEvaluator_withAcceleration(
       bool isVectorField,
       Teuchos::ArrayRCP<std::string> dof_names,
       Teuchos::ArrayRCP<std::string> dof_names_dot, // can be Teuchos::null
       Teuchos::ArrayRCP<std::string> dof_names_dotdot,
       int offsetToFirstDOF=0) const = 0;

    //! Function to create parameter list for construction of GatherSolution
    //! evaluator with acceleration terms.
    //! Tensor rank of solution variable is 0, 1, or 2
    Teuchos::RCP< PHX::Evaluator<Traits> >
    virtual constructGatherSolutionEvaluator_withAcceleration(
       int tensorRank,
       Teuchos::ArrayRCP<std::string> dof_names,
       Teuchos::ArrayRCP<std::string> dof_names_dot, // can be Teuchos::null
       Teuchos::ArrayRCP<std::string> dof_names_dotdot,
       int offsetToFirstDOF=0) const = 0;


    //! Same as above, but no ability to gather time dependent x_dot field
    Teuchos::RCP< PHX::Evaluator<Traits> >
    virtual constructGatherSolutionEvaluator_noTransient(
       bool isVectorField,
       Teuchos::ArrayRCP<std::string> dof_names,
       int offsetToFirstDOF=0) const = 0;

    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructGatherSolutionEvaluator_noTransient(
       bool isVectorField,
       const std::string& dof_name,
       int offsetToFirstDOF=0) const {
      return constructGatherSolutionEvaluator_noTransient(isVectorField,arcp_str(dof_name),offsetToFirstDOF);
    }

    //! Same as above, but no ability to gather time dependent x_dot field
    //! Tensor rank of solution variable is 0, 1, or 2
    Teuchos::RCP< PHX::Evaluator<Traits> >
    virtual constructGatherSolutionEvaluator_noTransient(
       int tensorRank,
       Teuchos::ArrayRCP<std::string> dof_names,
       int offsetToFirstDOF=0) const = 0;

    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructGatherSolutionEvaluator_noTransient(
       int tensorRank,
       const std::string& dof_name,
       int offsetToFirstDOF=0) const {
      return constructGatherSolutionEvaluator_noTransient(tensorRank,Teuchos::ArrayRCP<std::string>(1,dof_name),offsetToFirstDOF);
    }

    //! Function to create parameter list for construction of ScatterResidual
    //! evaluator with standard Field names
    Teuchos::RCP< PHX::Evaluator<Traits> >
    virtual constructScatterResidualEvaluator(
       bool isVectorField,
       Teuchos::ArrayRCP<std::string> resid_names,
       int offsetToFirstDOF=0, std::string scatterName="Scatter") const = 0;

    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructScatterResidualEvaluator(
       bool isVectorField,
       const std::string& resid_name,
       int offsetToFirstDOF=0, std::string scatterName="Scatter") const {
      return constructScatterResidualEvaluator (isVectorField,arcp_str(resid_name),offsetToFirstDOF,scatterName);
    }

    Teuchos::RCP< PHX::Evaluator<Traits> >
    virtual constructScatterSideEqnResidualEvaluator(
       const Teuchos::RCP<shards::CellTopology>& cellType,
       const std::string& sideSetName,
       bool residualsAreVolumeFields,
       bool isVectorField,
       Teuchos::ArrayRCP<std::string> resid_names,
       int offsetToFirstDOF=0, std::string scatterName="Scatter") const = 0;

    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructScatterSideEqnResidualEvaluator(
       const Teuchos::RCP<shards::CellTopology>& cellType,
       const std::string& sideSetName,
       bool residualsAreVolumeFields,
       bool isVectorField,
       const std::string& resid_name,
       int offsetToFirstDOF=0, std::string scatterName="Scatter") const {
      return constructScatterSideEqnResidualEvaluator (cellType, sideSetName,residualsAreVolumeFields,isVectorField,arcp_str(resid_name),offsetToFirstDOF,scatterName);
    }

    //! Function to create parameter list for construction of ScatterResidual
    //! evaluator with standard Field names
    Teuchos::RCP< PHX::Evaluator<Traits> >
    virtual constructScatterResidualEvaluatorWithExtrudedParams(
       bool isVectorField,
       Teuchos::ArrayRCP<std::string> resid_names,
       Teuchos::RCP<std::map<std::string, int> > extruded_params_levels,
       int offsetToFirstDOF=0, std::string scatterName="Scatter") const = 0;

    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructScatterResidualEvaluatorWithExtrudedParams(
       bool isVectorField,
       const std::string& resid_name,
       Teuchos::RCP<std::map<std::string, int> > extruded_params_levels,
       int offsetToFirstDOF=0, std::string scatterName="Scatter") const {
      return constructScatterResidualEvaluatorWithExtrudedParams (isVectorField,arcp_str(resid_name),extruded_params_levels,offsetToFirstDOF,scatterName);
    }

    //! Function to create parameter list for construction of ScatterResidual
    //! evaluator with standard Field names
    //! Tensor rank of solution variable is 0, 1, or 2
    Teuchos::RCP< PHX::Evaluator<Traits> >
    virtual constructScatterResidualEvaluator(
       int tensorRank,
       Teuchos::ArrayRCP<std::string> resid_names,
       int offsetToFirstDOF=0, std::string scatterName="Scatter") const = 0;

    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructScatterResidualEvaluator(
       int tensorRank,
       const std::string& resid_name,
       int offsetToFirstDOF=0, std::string scatterName="Scatter") const {
      return constructScatterResidualEvaluator(tensorRank,arcp_str(resid_name),offsetToFirstDOF,scatterName);
    }
    Teuchos::RCP< PHX::Evaluator<Traits> >
    virtual constructScatterSideEqnResidualEvaluator(
       const Teuchos::RCP<shards::CellTopology>& cellType,
       const std::string& sideSetName,
       bool residualsAreVolumeFields,
       int tensorRank,
       Teuchos::ArrayRCP<std::string> resid_names,
       int offsetToFirstDOF=0, std::string scatterName="Scatter") const = 0;

    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructScatterSideEqnResidualEvaluator(
       const Teuchos::RCP<shards::CellTopology>& cellType,
       const std::string& sideSetName,
       bool residualsAreVolumeFields,
       int tensorRank,
       const std::string& resid_name,
       int offsetToFirstDOF=0, std::string scatterName="Scatter") const {
      return constructScatterSideEqnResidualEvaluator(cellType,sideSetName,residualsAreVolumeFields,tensorRank,arcp_str(resid_name),offsetToFirstDOF,scatterName);
    }

    //! Function to create parameter list for construction of GatherScalarNodalParameter
    Teuchos::RCP< PHX::Evaluator<Traits> >
    virtual constructGatherScalarNodalParameter(
        const std::string& param_name,
        const std::string& field_name="") const = 0;

    //! Function to create parameter list for construction of ScatterScalarNodalParameter
    Teuchos::RCP< PHX::Evaluator<Traits> >
    virtual constructScatterScalarNodalParameter(
        const std::string& param_name,
        const std::string& field_name="") const = 0;

    //! Function to create parameter list for construction of GatherScalarExtruded2DNodalParameter
    Teuchos::RCP< PHX::Evaluator<Traits> >
    virtual constructGatherScalarExtruded2DNodalParameter(
        const std::string& param_name,
        const std::string& field_name="") const = 0;

    //! Function to create parameter list for construction of ScatterScalarExtruded2DNodalParameter
    Teuchos::RCP< PHX::Evaluator<Traits> >
    virtual constructScatterScalarExtruded2DNodalParameter(
        const std::string& param_name,
        const std::string& field_name="") const = 0;

    //! Function to create parameter list for construction of DOFInterpolation
    //! evaluator with standard field names
    //! AGS Note 10/13: oddsetToFirstDOF is added to DOF evaluators
    //!  for template specialization of Jacobian evaluation for
    //   performance. Otherwise it was not needed. With this info,
    //   the location of the nonzero partial derivatives can be
    //   computed, and the chain rule is coded with that known sparsity.
    Teuchos::RCP< PHX::Evaluator<Traits> >
    virtual constructDOFInterpolationEvaluator(
        const std::string& dof_names,
        int offsetToFirstDOF = -1) const = 0;

    //! Same as above, for Interpolating the Gradient
    Teuchos::RCP< PHX::Evaluator<Traits> >
    virtual constructDOFGradInterpolationEvaluator(
        const std::string& dof_names,
        int offsetToFirstDOF = -1) const = 0;

    //! Interpolation functions for vector quantities
    Teuchos::RCP< PHX::Evaluator<Traits> >
    virtual constructDOFVecInterpolationEvaluator(
       const std::string& dof_names, int offsetToFirstDOF=-1) const = 0;

    //! Same as above, for Interpolating the Gradient for Vector quantities
    Teuchos::RCP< PHX::Evaluator<Traits> >
    virtual constructDOFVecGradInterpolationEvaluator(
       const std::string& dof_names, int offsetToFirstDOF=-1) const = 0;

    //! Interpolation functions for Tensor quantities
    Teuchos::RCP< PHX::Evaluator<Traits> >
    virtual constructDOFTensorInterpolationEvaluator(
       const std::string& dof_names, int offsetToFirstDOF=-1) const = 0;
    //! Same as above, for Interpolating the Gradient for Tensor quantities
    Teuchos::RCP< PHX::Evaluator<Traits> >
    virtual constructDOFTensorGradInterpolationEvaluator(
       const std::string& dof_names, int offsetToFirstDOF=-1) const = 0;

    //! Interpolation functions for scalar quantities defined on a side set
    Teuchos::RCP< PHX::Evaluator<Traits> >
    virtual constructDOFInterpolationSideEvaluator(
        const std::string& dof_names,
        const std::string& sideSetName) const = 0;

    //! Interpolation functions for vector defined on a side set
    Teuchos::RCP< PHX::Evaluator<Traits> >
    virtual constructDOFVecInterpolationSideEvaluator(
       const std::string& dof_names,
       const std::string& sideSetName) const = 0;

    //! Interpolation functions for gradient of quantities defined on a side set
    Teuchos::RCP< PHX::Evaluator<Traits> >
    virtual constructDOFGradInterpolationSideEvaluator(
      const std::string& dof_names,
      const std::string& sideSetName,
      const bool planar = false) const = 0;

    //! Interpolation functions for gradient of vector quantities defined on a side set
    Teuchos::RCP< PHX::Evaluator<Traits> >
    virtual constructDOFVecGradInterpolationSideEvaluator(
      const std::string& dof_names,
      const std::string& sideSetName) const = 0;

    //! Function to create parameter list for construction of GatherCoordinateVector
    //! evaluator with standard Field names
    Teuchos::RCP< PHX::Evaluator<Traits> >
    virtual constructGatherCoordinateVectorEvaluator(
        std::string strCurrentDisp="") const = 0;

    //! Function to create parameter list for construction of MapToPhysicalFrame
    //! evaluator with standard Field names
    Teuchos::RCP< PHX::Evaluator<Traits> >
    virtual constructMapToPhysicalFrameEvaluator(
        const Teuchos::RCP<shards::CellTopology>& cellType,
        const Teuchos::RCP<IntrepidCubature> cubature,
        const Teuchos::RCP<IntrepidBasis> intrepidBasis = Teuchos::null) const = 0;

    //! Function to create parameter list for construction of MapToPhysicalFrameSide
    //! evaluator with standard Field names
    Teuchos::RCP< PHX::Evaluator<Traits> >
    virtual constructMapToPhysicalFrameSideEvaluator(
      const Teuchos::RCP<shards::CellTopology>& cellType,
      const Teuchos::RCP<IntrepidCubature> cubature,
      const std::string& sideSetName) const = 0;

    //! Function to create evaluator for restriction to side set
    Teuchos::RCP< PHX::Evaluator<Traits> >
    virtual constructDOFCellToSideEvaluator(
        const std::string& cell_dof_name,
        const std::string& sideSetName,
        const std::string& layout,
        const Teuchos::RCP<shards::CellTopology>& cellType = Teuchos::null,
        const std::string& side_dof_name = "") const = 0;

    //! Combo: restriction to side plus interpolation
    Teuchos::RCP< PHX::Evaluator<Traits> >
    virtual constructDOFCellToSideQPEvaluator(
       const std::string& cell_dof_name,
       const std::string& sideSetName,
       const std::string& layout,
       const Teuchos::RCP<shards::CellTopology>& cellType = Teuchos::null,
       const std::string& side_dof_name = "") const = 0;

    //! Function to create evaluator for prolongation to cell
    Teuchos::RCP< PHX::Evaluator<Traits> >
    virtual constructDOFSideToCellEvaluator(
       const std::string& side_dof_name,
       const std::string& sideSetName,
       const std::string& layout,
       const Teuchos::RCP<shards::CellTopology>& cellType = Teuchos::null,
       const std::string& cell_dof_name = "") const = 0;

    //! Function to create P0 interpolation evaluator
    //! Note: interpolationType can be 'Cell Average' or 'Value At Cell Barycenter',
    //!       with the latter only available for nodal fields.
    Teuchos::RCP< PHX::Evaluator<Traits> >
    virtual constructP0InterpolationEvaluator(
        const std::string& dof_name,
        const std::string& interpolationType = "Cell Average",
        const FieldLocation loc = FieldLocation::Node,
        const FieldRankType rank = FieldRankType::Scalar,
        const Teuchos::RCP<IntrepidBasis>& basis = Teuchos::null) const = 0;

    Teuchos::RCP< PHX::Evaluator<Traits> >
    virtual constructP0InterpolationSideEvaluator(
        const std::string& sideSetName,
        const std::string& dof_name,
        const std::string& interpolationType = "Cell Average",
        const FieldLocation loc = FieldLocation::Node,
        const FieldRankType rank = FieldRankType::Scalar,
        const Teuchos::RCP<IntrepidBasis>& basis = Teuchos::null) const = 0;

    // Convenience shortcuts for special cases of P0 interpolation
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructBarycenterEvaluator(
        const std::string& dof_name,
        const Teuchos::RCP<IntrepidBasis>& basis,
        const FieldRankType rank = FieldRankType::Scalar) const {
      return constructP0InterpolationEvaluator(
          dof_name,"Value At Cell Barycenter", FieldLocation::Node, rank, basis);
    }
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructBarycenterSideEvaluator(
        const std::string& ss_name,
        const std::string& dof_name,
        const Teuchos::RCP<IntrepidBasis>& basis,
        const FieldRankType rank = FieldRankType::Scalar) const {
      return constructP0InterpolationSideEvaluator(
          ss_name, dof_name,"Value At Cell Barycenter", FieldLocation::Node, rank, basis);
    }
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructCellAverageEvaluator(
        const std::string& dof_name,
        const FieldLocation loc = FieldLocation::Node,
        const FieldRankType rank = FieldRankType::Scalar) const {
      return constructP0InterpolationEvaluator(
          dof_name,"Cell Average", loc, rank);
    }
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructCellAverageSideEvaluator(
        const std::string& ss_name,
        const std::string& dof_name,
        const FieldLocation loc = FieldLocation::Node,
        const FieldRankType rank = FieldRankType::Scalar) const {
      return constructP0InterpolationSideEvaluator(
          ss_name, dof_name,"Cell Average", loc, rank);
    }

    //! Function to create parameter list for construction of ComputeBasisFunctions
    //! evaluator with standard Field names
    Teuchos::RCP< PHX::Evaluator<Traits> >
    virtual constructComputeBasisFunctionsEvaluator(
        const Teuchos::RCP<shards::CellTopology>& cellType,
        const Teuchos::RCP<IntrepidBasis> intrepidBasis,
        const Teuchos::RCP<IntrepidCubature> cubature) const = 0;

    //! Function to create parameter list for construction of ComputeBasisFunctionsSide
    //! evaluator with standard Field names
    Teuchos::RCP< PHX::Evaluator<Traits> >
    virtual constructComputeBasisFunctionsSideEvaluator(
        const Teuchos::RCP<shards::CellTopology>& cellType,
        const Teuchos::RCP<IntrepidBasis> intrepidBasisSide,
        const Teuchos::RCP<IntrepidCubature> cubatureSide,
        const std::string& sideSetName,
        const bool buildNormals = false,
        const bool palanar = false) const = 0;

    protected:
    Teuchos::ArrayRCP<std::string> arcp_str(const std::string& s) const {
      return Teuchos::ArrayRCP<std::string>(1,s);
    }

  };

  template<typename EvalT, typename Traits, typename ScalarType>
  class EvaluatorUtilsImpl : public EvaluatorUtilsBase<Traits> {

  public:

    using IntrepidBasis    = typename EvaluatorUtilsBase<Traits>::IntrepidBasis;
    using IntrepidCubature = typename EvaluatorUtilsBase<Traits>::IntrepidCubature;

    typedef typename EvalT::ScalarT       ScalarT;
    typedef typename EvalT::MeshScalarT   MeshScalarT;
    typedef typename EvalT::ParamScalarT  ParamScalarT;

    EvaluatorUtilsImpl(Teuchos::RCP<Albany::Layouts> dl);

    const EvaluatorUtilsBase<Traits>&
    getSTUtils() const
    {
      if (std::is_same<ScalarType,ScalarT>::value) {
        return *this;
      } else if (utils_ST==Teuchos::null)
        utils_ST = Teuchos::rcp(new EvaluatorUtilsImpl<EvalT,Traits,ScalarT>(dl));
      return *utils_ST;
    }

    const EvaluatorUtilsBase<Traits>&
    getMSTUtils() const
    {
      if (std::is_same<ScalarType,MeshScalarT>::value) {
        return *this;
      } else if (utils_MST==Teuchos::null)
        utils_MST = Teuchos::rcp(new EvaluatorUtilsImpl<EvalT,Traits,MeshScalarT>(dl));
      return *utils_MST;
    }

    const EvaluatorUtilsBase<Traits>&
    getPSTUtils() const
    {
      if (std::is_same<ScalarType,ParamScalarT>::value) {
        return *this;
      } else if (utils_PST==Teuchos::null)
        utils_PST = Teuchos::rcp(new EvaluatorUtilsImpl<EvalT,Traits,ParamScalarT>(dl));
      return *utils_PST;
    }

    const EvaluatorUtilsBase<Traits>&
    getRTUtils() const
    {
      if (utils_RT==Teuchos::null)
        utils_RT = Teuchos::rcp(new EvaluatorUtilsImpl<EvalT,Traits,RealType>(dl));
      return *utils_RT;
    }

    // Do not hide base class inlined methods
    using EvaluatorUtilsBase<Traits>::constructGatherSolutionEvaluator;
    using EvaluatorUtilsBase<Traits>::constructGatherSolutionEvaluator_noTransient;
    using EvaluatorUtilsBase<Traits>::constructScatterResidualEvaluator;
    using EvaluatorUtilsBase<Traits>::constructScatterSideEqnResidualEvaluator;
    using EvaluatorUtilsBase<Traits>::constructScatterResidualEvaluatorWithExtrudedParams;

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

    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructScatterSideEqnResidualEvaluator(
       const Teuchos::RCP<shards::CellTopology>& cellType,
       const std::string& sideSetName,
       bool residualsAreVolumeFields,
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

    //! Function to create parameter list for construction of ScatterResidual
    //! evaluator with standard Field names
    //! Tensor rank of solution variable is 0, 1, or 2
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructScatterResidualEvaluator(
       int tensorRank,
       Teuchos::ArrayRCP<std::string> resid_names,
       int offsetToFirstDOF=0, std::string scatterName="Scatter") const;
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructScatterSideEqnResidualEvaluator(
       const Teuchos::RCP<shards::CellTopology>& cellType,
       const std::string& sideSetName,
       bool residualsAreVolumeFields,
       int tensorRank,
       Teuchos::ArrayRCP<std::string> resid_names,
       int offsetToFirstDOF=0, std::string scatterName="Scatter") const;

    //! Function to create parameter list for construction of GatherScalarNodalParameter
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructGatherScalarNodalParameter(
        const std::string& param_name,
        const std::string& field_name="") const;

    //! Function to create parameter list for construction of ScatterScalarNodalParameter
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructScatterScalarNodalParameter(
        const std::string& param_name,
        const std::string& field_name="") const;

    //! Function to create parameter list for construction of GatherScalarExtruded2DNodalParameter
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructGatherScalarExtruded2DNodalParameter(
        const std::string& param_name,
        const std::string& field_name="") const;

    //! Function to create parameter list for construction of ScatterScalarExtruded2DNodalParameter
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructScatterScalarExtruded2DNodalParameter(
        const std::string& param_name,
        const std::string& field_name="") const;

    //! Function to create parameter list for construction of DOFInterpolation
    //! evaluator with standard field names
    //! AGS Note 10/13: oddsetToFirstDOF is added to DOF evaluators
    //!  for template specialization of Jacobian evaluation for
    //   performance. Otherwise it was not needed. With this info,
    //   the location of the nonzero partial derivatives can be
    //   computed, and the chain rule is coded with that known sparsity.
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructDOFInterpolationEvaluator(
        const std::string& dof_names,
        int offsetToFirstDOF = -1) const;

    //! Same as above, for Interpolating the Gradient
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructDOFGradInterpolationEvaluator(
        const std::string& dof_names,
        int offsetToFirstDOF = -1) const;

    //! Interpolation functions for vector quantities
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructDOFVecInterpolationEvaluator(
       const std::string& dof_names, int offsetToFirstDOF=-1) const;

    //! Same as above, for Interpolating the Gradient for Vector quantities
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructDOFVecGradInterpolationEvaluator(
       const std::string& dof_names, int offsetToFirstDOF=-1) const;

    //! Interpolation functions for Tensor quantities
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructDOFTensorInterpolationEvaluator(
       const std::string& dof_names, int offsetToFirstDOF=-1) const;
    //! Same as above, for Interpolating the Gradient for Tensor quantities
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructDOFTensorGradInterpolationEvaluator(
       const std::string& dof_names, int offsetToFirstDOF=-1) const;

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
      const std::string& sideSetName,
      const bool planar = false) const;

    //! Interpolation functions for gradient of vector quantities defined on a side set
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructDOFVecGradInterpolationSideEvaluator(
      const std::string& dof_names,
      const std::string& sideSetName) const;

    //! Function to create parameter list for construction of GatherCoordinateVector
    //! evaluator with standard Field names
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructGatherCoordinateVectorEvaluator(
        std::string strCurrentDisp="") const;

    //! Function to create parameter list for construction of MapToPhysicalFrame
    //! evaluator with standard Field names
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructMapToPhysicalFrameEvaluator(
        const Teuchos::RCP<shards::CellTopology>& cellType,
        const Teuchos::RCP<IntrepidCubature> cubature,
        const Teuchos::RCP<IntrepidBasis> intrepidBasis = Teuchos::null) const;

    //! Function to create parameter list for construction of MapToPhysicalFrameSide
    //! evaluator with standard Field names
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructMapToPhysicalFrameSideEvaluator(
      const Teuchos::RCP<shards::CellTopology>& cellType,
      const Teuchos::RCP<IntrepidCubature> cubature,
      const std::string& sideSetName) const;

    //! Function to create evaluator for restriction to side set
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructDOFCellToSideEvaluator(
        const std::string& cell_dof_name,
        const std::string& sideSetName,
        const std::string& layout,
        const Teuchos::RCP<shards::CellTopology>& cellType = Teuchos::null,
        const std::string& side_dof_name = "") const;

    //! Combo: restriction to side plus interpolation
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructDOFCellToSideQPEvaluator(
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

    //! Function to create P0 interpolation evaluator
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructP0InterpolationEvaluator(
        const std::string& dof_name,
        const std::string& interpolationType = "Cell Average",
        const FieldLocation loc = FieldLocation::Node,
        const FieldRankType rank = FieldRankType::Scalar,
        const Teuchos::RCP<IntrepidBasis>& basis = Teuchos::null) const;

    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructP0InterpolationSideEvaluator(
        const std::string& sideSetName,
        const std::string& dof_name,
        const std::string& interpolationType = "Cell Average",
        const FieldLocation loc = FieldLocation::Node,
        const FieldRankType rank = FieldRankType::Scalar,
        const Teuchos::RCP<IntrepidBasis>& basis = Teuchos::null) const;

    //! Function to create parameter list for construction of ComputeBasisFunctions
    //! evaluator with standard Field names
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructComputeBasisFunctionsEvaluator(
        const Teuchos::RCP<shards::CellTopology>& cellType,
        const Teuchos::RCP<IntrepidBasis> intrepidBasis,
        const Teuchos::RCP<IntrepidCubature> cubature) const;

    //! Function to create parameter list for construction of ComputeBasisFunctionsSide
    //! evaluator with standard Field names
    Teuchos::RCP< PHX::Evaluator<Traits> >
    constructComputeBasisFunctionsSideEvaluator(
        const Teuchos::RCP<shards::CellTopology>& cellType,
        const Teuchos::RCP<IntrepidBasis> intrepidBasisSide,
        const Teuchos::RCP<IntrepidCubature> cubatureSide,
        const std::string& sideSetName,
        const bool buildNormals = false,
        const bool planar = false) const;

  private:

    //! Evaluator Utils with different ScalarType. Mutable, so we can have getters with JIT build.
    //! NOTE: we CAN'T create them in the constructor, since we would have a never-ending construction.
    mutable Teuchos::RCP<EvaluatorUtilsBase<Traits>>   utils_ST;
    mutable Teuchos::RCP<EvaluatorUtilsBase<Traits>>   utils_MST;
    mutable Teuchos::RCP<EvaluatorUtilsBase<Traits>>   utils_PST;
    mutable Teuchos::RCP<EvaluatorUtilsBase<Traits>>   utils_RT;

    //! Struct of PHX::DataLayout objects defined all together.
    Teuchos::RCP<Albany::Layouts> dl;

  };

template<typename EvalT, typename Traits>
using EvaluatorUtils = EvaluatorUtilsImpl<EvalT,Traits,typename EvalT::ScalarT>;

} // Namespace Albany

#endif // ALBANY_EVALUATORUTILS_HPP

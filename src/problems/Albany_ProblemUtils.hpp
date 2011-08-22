/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


#ifndef ALBANY_PROBLEMUTILS_HPP
#define ALBANY_PROBLEMUTILS_HPP

#include <vector>
#include <string>

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Phalanx.hpp"
#include "Albany_DataTypes.hpp"
#include "PHAL_AlbanyTraits.hpp"

#include "Teuchos_VerboseObject.hpp"

#include "Intrepid_Basis.hpp"
#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"


namespace Albany {
  /*!
   * \brief Struct to construct and hold DataLayouts
   */
  struct Layouts {
    Layouts(int worksetSize, int  numVertices, int numNodes, int numQPts, int numDim);
    //! Data Layout for scalar quantity that lives at nodes
    Teuchos::RCP<PHX::DataLayout> node_scalar;
    //! Data Layout for scalar quantity that lives at quad points
    Teuchos::RCP<PHX::DataLayout> qp_scalar;
    //! Data Layout for scalar quantity that lives on a cell
    Teuchos::RCP<PHX::DataLayout> cell_scalar;
    //! Data Layout for vector quantity that lives at nodes
    Teuchos::RCP<PHX::DataLayout> node_vector;
    //! Data Layout for vector quantity that lives at quad points
    Teuchos::RCP<PHX::DataLayout> qp_vector;
    //! Data Layout for tensor quantity that lives at nodes
    Teuchos::RCP<PHX::DataLayout> node_tensor;
    //! Data Layout for tensor quantity that lives at quad points
    Teuchos::RCP<PHX::DataLayout> qp_tensor;
    //! Data Layout for vector quantity that lives at vertices (coordinates)
    Teuchos::RCP<PHX::DataLayout> vertices_vector;
    //! Data Layout for scalar basis functions
    Teuchos::RCP<PHX::DataLayout> node_qp_scalar;
    //! Data Layout for vector basis functions
    Teuchos::RCP<PHX::DataLayout> node_qp_vector;
    /*!
     * \brief Dummy Data Layout where one is needed but not accessed
     * For instance, the action of scattering residual data from a
     * Field into the residual vector in the workset struct needs an
     * evaluator, but the evaluator has no natural Field that it computes.
     * So, it computes the Scatter field with this (empty) Dummy layout.
     * Requesting this Dummy Field then activates this evaluator so
     * the action is performed.
     */
    Teuchos::RCP<PHX::DataLayout> dummy;
  };

  //! Helper Factory function to construct Intrepid Basis from Shards CellTopologyData
  Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > >
    getIntrepidBasis(const CellTopologyData& ctd, bool compositeTet=false);
}

namespace Albany {
  /*!
   * \brief Generic Functions to help define evaluators and problems more succinctly
   */
  class ProblemUtils {

   public:

    ProblemUtils(Teuchos::RCP<Albany::Layouts> dl, std::string facTraits="PHAL");

    //! Function to create parameter list for construction of GatherSolution
    //! evaluator with standard Field names
    Teuchos::RCP<Teuchos::ParameterList> 
    constructGatherSolutionEvaluator(
       bool isVectorField,
       Teuchos::ArrayRCP<std::string> dof_names,
       Teuchos::ArrayRCP<std::string> dof_names_dot,
       int offsetToFirstDOF=0);

    //! Same as above, but no ability to gather time dependent x_dot field
    Teuchos::RCP<Teuchos::ParameterList> 
    constructGatherSolutionEvaluator_noTransient(
       bool isVectorField,
       Teuchos::ArrayRCP<std::string> dof_names,
       int offsetToFirstDOF=0);

    //! Function to create parameter list for construction of ScatterResidual
    //! evaluator with standard Field names
    Teuchos::RCP<Teuchos::ParameterList> 
    constructScatterResidualEvaluator(
       bool isVectorField,
       Teuchos::ArrayRCP<std::string> resid_names,
       int offsetToFirstDOF=0, std::string scatterName="Scatter");

    //! Function to create parameter list for construction of DOFInterpolation 
    //! evaluator with standard field names
    Teuchos::RCP<Teuchos::ParameterList> 
    constructDOFInterpolationEvaluator(
       std::string& dof_names);
    //! Same as above, for Interpolating the Gradient
    Teuchos::RCP<Teuchos::ParameterList> 
    constructDOFGradInterpolationEvaluator(
       std::string& dof_names);

    //! Interpolation functions for vector quantities
    Teuchos::RCP<Teuchos::ParameterList> 
    constructDOFVecInterpolationEvaluator(
       std::string& dof_names);
    //! Same as above, for Interpolating the Gradient for Vector quantities
    Teuchos::RCP<Teuchos::ParameterList> 
    constructDOFVecGradInterpolationEvaluator(
       std::string& dof_names);

    //! Function to create parameter list for construction of GatherCoordinateVector
    //! evaluator with standard Field names
    Teuchos::RCP<Teuchos::ParameterList> 
    constructGatherCoordinateVectorEvaluator();

    //! Function to create parameter list for construction of MapToPhysicalFrame
    //! evaluator with standard Field names
    Teuchos::RCP<Teuchos::ParameterList> 
    constructMapToPhysicalFrameEvaluator(
      const Teuchos::RCP<shards::CellTopology>& cellType,
      const Teuchos::RCP<Intrepid::Cubature<RealType> > cubature);

    //! Function to create parameter list for construction of ComputeBasisFunctions
    //! evaluator with standard Field names
    Teuchos::RCP<Teuchos::ParameterList> 
    constructComputeBasisFunctionsEvaluator(
      const Teuchos::RCP<shards::CellTopology>& cellType,
      const Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > intrepidBasis,
      const Teuchos::RCP<Intrepid::Cubature<RealType> > cubature);

  private:

    //! Struct of PHX::DataLayout objects defined all together.
    Teuchos::RCP<Albany::Layouts> dl;

    //! Temporary variable inside most methods, defined just once for convenience.
    mutable int type;

    //! String flag to switch between FactorTraits structs, currently PHAL and LCM
    std::string facTraits;
  };
}

#endif 

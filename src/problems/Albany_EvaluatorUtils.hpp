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

#include "Intrepid_Basis.hpp"
#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
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

    //! Same as above, but no ability to gather time dependent x_dot field
    Teuchos::RCP< PHX::Evaluator<Traits> > 
    constructGatherSolutionEvaluator_noTransient(
       bool isVectorField,
       Teuchos::ArrayRCP<std::string> dof_names,
       int offsetToFirstDOF=0);

    //! Function to create parameter list for construction of ScatterResidual
    //! evaluator with standard Field names
    Teuchos::RCP< PHX::Evaluator<Traits> > 
    constructScatterResidualEvaluator(
       bool isVectorField,
       Teuchos::ArrayRCP<std::string> resid_names,
       int offsetToFirstDOF=0, std::string scatterName="Scatter");

    //! Function to create parameter list for construction of DOFInterpolation 
    //! evaluator with standard field names
    Teuchos::RCP< PHX::Evaluator<Traits> > 
    constructDOFInterpolationEvaluator(
       std::string& dof_names);
    //! Same as above, for Interpolating the Gradient
    Teuchos::RCP< PHX::Evaluator<Traits> > 
    constructDOFGradInterpolationEvaluator(
       std::string& dof_names);

    //! Interpolation functions for vector quantities
    Teuchos::RCP< PHX::Evaluator<Traits> > 
    constructDOFVecInterpolationEvaluator(
       std::string& dof_names);
    //! Same as above, for Interpolating the Gradient for Vector quantities
    Teuchos::RCP< PHX::Evaluator<Traits> > 
    constructDOFVecGradInterpolationEvaluator(
       std::string& dof_names);

    //! Function to create parameter list for construction of GatherCoordinateVector
    //! evaluator with standard Field names
    Teuchos::RCP< PHX::Evaluator<Traits> > 
    constructGatherCoordinateVectorEvaluator();

    //! Function to create parameter list for construction of MapToPhysicalFrame
    //! evaluator with standard Field names
    Teuchos::RCP< PHX::Evaluator<Traits> > 
    constructMapToPhysicalFrameEvaluator(
      const Teuchos::RCP<shards::CellTopology>& cellType,
      const Teuchos::RCP<Intrepid::Cubature<RealType> > cubature);

    //! Function to create parameter list for construction of ComputeBasisFunctions
    //! evaluator with standard Field names
    Teuchos::RCP< PHX::Evaluator<Traits> > 
    constructComputeBasisFunctionsEvaluator(
      const Teuchos::RCP<shards::CellTopology>& cellType,
      const Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > intrepidBasis,
      const Teuchos::RCP<Intrepid::Cubature<RealType> > cubature);

  private:

    //! Struct of PHX::DataLayout objects defined all together.
    Teuchos::RCP<Albany::Layouts> dl;

  };
}

#endif 

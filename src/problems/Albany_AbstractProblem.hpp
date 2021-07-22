//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

// IK, 9/12/14: no Epetra!

#ifndef ALBANY_ABSTRACTPROBLEM_HPP
#define ALBANY_ABSTRACTPROBLEM_HPP

#include <string>
#include <vector>

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"

#include "Albany_AbstractFieldContainer.hpp"  //has typedef needed to list the field requirements of the problem
#include "Albany_NullSpaceUtils.hpp"  // has defn of struct that holds null space info for ML
#include "Albany_StateInfoStruct.hpp"  // contains MeshSpecsStuct
#include "Albany_StateManager.hpp"

#include "Albany_DataTypes.hpp"

#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_Dimension.hpp"
#include "PHAL_Workset.hpp"

#include "Teuchos_VerboseObject.hpp"

#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Intrepid2_HGRAD_HEX_C1_FEM.hpp"
#include "Intrepid2_HGRAD_HEX_C2_FEM.hpp"
#include "Intrepid2_HGRAD_LINE_C1_FEM.hpp"
#include "Intrepid2_HGRAD_LINE_Cn_FEM.hpp"
#include "Intrepid2_HGRAD_QUAD_C1_FEM.hpp"
#include "Intrepid2_HGRAD_QUAD_C2_FEM.hpp"
#include "Intrepid2_HGRAD_TET_C1_FEM.hpp"
#include "Intrepid2_HGRAD_TET_C2_FEM.hpp"
#include "Intrepid2_HGRAD_TET_COMP12_FEM.hpp"
#include "Intrepid2_HGRAD_TRI_C1_FEM.hpp"
#include "Intrepid2_HGRAD_TRI_C2_FEM.hpp"
#include "PHAL_FactoryTraits.hpp"
#include "Shards_CellTopology.hpp"

namespace Albany {

// Define enumerator to store solution method name
enum SolutionMethodType {
  Steady,
  Continuation,
  Transient,
  Eigensolve,
  Unknown
};

enum FieldManagerChoice { BUILD_RESID_FM, BUILD_RESPONSE_FM, BUILD_STATE_FM };

/*!
 * \brief Abstract interface for representing a 1-D finite element
 * problem.
 */
class AbstractProblem {
 public:
  //! Only constructor
  AbstractProblem(
      const Teuchos::RCP<Teuchos::ParameterList>& params_,
      const Teuchos::RCP<ParamLib>& paramLib_,
      // const Teuchos::RCP<DistributedParameterLibrary>& distParamLib_,
      const int neq_ = 0);

  //! Destructor
  virtual ~AbstractProblem(){};

  //! Get the number of equations
  unsigned int
  numEquations() const;
  const std::map<int, std::vector<std::string>>&
  getSideSetEquations() const;
  void
  setNumEquations(const int neq_);
  unsigned int
  numStates() const;

  // Get the solution method type name
  SolutionMethodType
  getSolutionMethod();

  //! Get spatial dimension
  virtual int
  spatialDimension() const = 0;
 
  //! Get boolean telling code if SDBCs are utilized  
  virtual bool
  useSDBCs() const = 0;

  //! Build the PDE instantiations, boundary conditions, and initial solution
  //! And construct the evaluators and field managers
  virtual void
  buildProblem(
      Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct>> meshSpecs,
      StateManager& stateMgr) = 0;

  // Build evaluators
  virtual Teuchos::Array<Teuchos::RCP<const PHX::FieldTag>>
  buildEvaluators(
      PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
      const Albany::MeshSpecsStruct& meshSpecs, Albany::StateManager& stateMgr,
      Albany::FieldManagerChoice fmchoice,
      const Teuchos::RCP<Teuchos::ParameterList>& responseList) = 0;

  Teuchos::ArrayRCP<Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>>>
  getFieldManager();
  Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>>
  getDirichletFieldManager();
  Teuchos::ArrayRCP<Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>>>
  getNeumannFieldManager();

  Teuchos::Array<Teuchos::Array<int>>
  getOffsets() {
    return offsets_;
  }
  
  std::vector<std::string>
  getNodeSetIDs() {
    return nodeSetIDs_;
  }


  //! Return the Null space object used to communicate with MP
  const Teuchos::RCP<Albany::RigidBodyModes>&
  getNullSpace() {
    return rigidBodyModes;
  }

  //! Each problem must generate it's list of valide parameters
  virtual Teuchos::RCP<const Teuchos::ParameterList>
  getValidProblemParameters() const {
    return getGenericProblemParams("Generic Problem List");
  };

  virtual void
  getAllocatedStates(
      Teuchos::ArrayRCP<Teuchos::ArrayRCP<
          Teuchos::RCP<Kokkos::DynRankView<RealType, PHX::Device>>>>
          /* oldState_ */,
      Teuchos::ArrayRCP<Teuchos::ArrayRCP<
          Teuchos::RCP<Kokkos::DynRankView<RealType, PHX::Device>>>>
          /* newState_ */) const {};

  //! Get a list of the Special fields needed to implement the problem
  const AbstractFieldContainer::FieldContainerRequirements
  getFieldRequirements() {
    return requirements;
  }

  const std::map<
      std::string, AbstractFieldContainer::FieldContainerRequirements>&
  getSideSetFieldRequirements() const {
    return ss_requirements;
  }

  //! Allow the Problem to modify the solver settings, for example by adding a
  //! custom status test.
  virtual void
  applyProblemSpecificSolverSettings(
      Teuchos::RCP<Teuchos::ParameterList> ) {}

 protected:
  Teuchos::Array<Teuchos::Array<int>> offsets_;
  std::vector<std::string> nodeSetIDs_;
  //! List of valid problem params common to all problems, as
  //! a starting point for the specific  getValidProblemParameters
  Teuchos::RCP<Teuchos::ParameterList>
  getGenericProblemParams(std::string listname = "ProblemList") const;

  //! Configurable output stream, defaults to printing on proc=0
  Teuchos::RCP<Teuchos::FancyOStream> out;

  //! Number of equations per node being solved
  unsigned int neq;

  //! Number of time derivatives used in the problem
  //! 0 = x, 1 = xdot, 2 = xdotdot
  int number_of_time_deriv;

  //! Equations that are defined ONLY on some part of the mesh
  std::map<int, std::vector<std::string>> sideSetEquations;

  // Variable use to store the solution method name.
  SolutionMethodType SolutionMethodName;

  //! Problem parameters
  Teuchos::RCP<Teuchos::ParameterList> params;

  //! Parameter library
  Teuchos::RCP<ParamLib> paramLib;

  //! Distributed parameter library
  // Teuchos::RCP<DistributedParameterLibrary> distParamLib;

  //! Field manager for Volumetric Fill
  Teuchos::ArrayRCP<Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>>> fm;

  //! Field manager for Dirchlet Conditions Fill
  Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>> dfm;

  //! Field manager for Neumann Conditions Fill
  Teuchos::ArrayRCP<Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>>> nfm;

  //! Special fields needed to implement the problem
  AbstractFieldContainer::FieldContainerRequirements requirements;
  //! Special fields defined on the side sets needed to implement the problem
  std::map<std::string, AbstractFieldContainer::FieldContainerRequirements>
      ss_requirements;

  //! Null space object used to communicate with MP
  Teuchos::RCP<Albany::RigidBodyModes> rigidBodyModes;

 private:
  //! Private to prohibit default or copy constructor
  AbstractProblem();
  AbstractProblem(const AbstractProblem&);

  //! Private to prohibit copying
  AbstractProblem&
  operator=(const AbstractProblem&);
};

template <typename ProblemT>
struct ConstructEvaluatorsOp {
public:
  ConstructEvaluatorsOp(
      ProblemT& prob_,
      PHX::FieldManager<PHAL::AlbanyTraits>& fm_,
      const Albany::MeshSpecsStruct& meshSpecs_,
      Albany::StateManager& stateMgr_,
      Albany::FieldManagerChoice fmchoice_ = BUILD_RESID_FM,
      const Teuchos::RCP<Teuchos::ParameterList>& responseList_ = Teuchos::null)
      : prob(prob_),
        fm(fm_),
        meshSpecs(meshSpecs_),
        stateMgr(stateMgr_),
        fmchoice(fmchoice_),
        responseList(responseList_)
  {
    tags = Teuchos::rcp(new Teuchos::Array<Teuchos::RCP<const PHX::FieldTag>>);
  }

  template <typename EvalT>
  void
  operator()(EvalT /* e */) const
  {
    tags->push_back(prob.template constructEvaluators<EvalT>(
        fm, meshSpecs, stateMgr, fmchoice, responseList));
  }

private:
  ProblemT& prob;
  PHX::FieldManager<PHAL::AlbanyTraits>& fm;
  const Albany::MeshSpecsStruct& meshSpecs;
  Albany::StateManager& stateMgr;
  Albany::FieldManagerChoice fmchoice;
  Teuchos::RCP<Teuchos::ParameterList> responseList;

public:
  Teuchos::RCP<Teuchos::Array<Teuchos::RCP<const PHX::FieldTag>>> tags;
};

template <typename ProblemT>
struct ConstructFieldsOp
{
public:
  ConstructFieldsOp(
      ProblemT& prob_,
      PHX::FieldManager<PHAL::AlbanyTraits>& fm_)
      : prob(prob_),
        fm(fm_)
  {
  }

  template <typename EvalT>
  void operator() (EvalT /* e */) const
  {
    prob.template constructFields<EvalT>(fm);
  }

private:
  ProblemT& prob;
  PHX::FieldManager<PHAL::AlbanyTraits>& fm;
};

} // namespace Albany

#endif  // ALBANY_ABSTRACTPROBLEM_HPP

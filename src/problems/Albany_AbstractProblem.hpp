//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

// IK, 9/12/14: no Epetra!

#ifndef ALBANY_ABSTRACT_PROBLEM_HPP
#define ALBANY_ABSTRACT_PROBLEM_HPP

#include "Albany_NullSpaceUtils.hpp"  // has defn of struct that holds null space info for ML
#include "Albany_StateInfoStruct.hpp"  // contains MeshSpecsStuct
#include "Albany_StateManager.hpp"
#include "Albany_DataTypes.hpp"
#include "Albany_ScalarParameterAccessors.hpp"

#include "PHAL_AlbanyTraits.hpp"

#include "Phalanx_FieldManager.hpp"

#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"

#include <string>
#include <vector>

namespace Albany {

// Define enumerator to store solution method name
enum SolutionMethodType {
  Steady,
  Continuation,
  Transient,
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
  buildProblem (Teuchos::ArrayRCP<Teuchos::RCP<MeshSpecs>> meshSpecs,
                StateManager& stateMgr) = 0;

  // Build evaluators
  virtual Teuchos::Array<Teuchos::RCP<const PHX::FieldTag>>
  buildEvaluators(
      PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
      const MeshSpecs& meshSpecs, StateManager& stateMgr,
      FieldManagerChoice fmchoice,
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

  Teuchos::RCP<ScalarParameterAccessorsMap>
  getAccessors() {
    if (accessors_all.is_null()) {
      accessors_all =
        Teuchos::rcp(new ScalarParameterAccessorsMap());
    }
    if (accessors_all->at<PHAL::AlbanyTraits::Residual>().is_null()) {
      accessors_all->at<PHAL::AlbanyTraits::Residual>() =
        Teuchos::rcp(new ScalarParameterAccessors<PHAL::AlbanyTraits::Residual>());
    }
    if (accessors_all->at<PHAL::AlbanyTraits::Jacobian>().is_null()) {
      accessors_all->at<PHAL::AlbanyTraits::Jacobian>() =
        Teuchos::rcp(new ScalarParameterAccessors<PHAL::AlbanyTraits::Jacobian>());
    }
    if (accessors_all->at<PHAL::AlbanyTraits::Tangent>().is_null()) {
      accessors_all->at<PHAL::AlbanyTraits::Tangent>() =
        Teuchos::rcp(new ScalarParameterAccessors<PHAL::AlbanyTraits::Tangent>());
    }
    if (accessors_all->at<PHAL::AlbanyTraits::DistParamDeriv>().is_null()) {
      accessors_all->at<PHAL::AlbanyTraits::DistParamDeriv>() =
        Teuchos::rcp(new ScalarParameterAccessors<PHAL::AlbanyTraits::DistParamDeriv>());
    }
    if (accessors_all->at<PHAL::AlbanyTraits::HessianVec>().is_null()) {
      accessors_all->at<PHAL::AlbanyTraits::HessianVec>() =
        Teuchos::rcp(new ScalarParameterAccessors<PHAL::AlbanyTraits::HessianVec>());
    }
    return accessors_all;
  }


  //! Return the Null space object used to communicate with MP
  const Teuchos::RCP<RigidBodyModes>&
  getNullSpace() {
    return rigidBodyModes;
  }

  //! Each problem must generate it's list of valid parameters
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

  //! Parameter accessors
  Teuchos::RCP<ScalarParameterAccessorsMap> accessors_all;

  //! Field manager for Volumetric Fill
  Teuchos::ArrayRCP<Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>>> fm;

  //! Field manager for Dirchlet Conditions Fill
  Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>> dfm;

  //! Field manager for Neumann Conditions Fill
  Teuchos::ArrayRCP<Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>>> nfm;

  //! Null space object used to communicate with MP
  Teuchos::RCP<RigidBodyModes> rigidBodyModes;

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
      const MeshSpecs& meshSpecs_,
      StateManager& stateMgr_,
      FieldManagerChoice fmchoice_ = BUILD_RESID_FM,
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
  const MeshSpecs& meshSpecs;
  StateManager& stateMgr;
  FieldManagerChoice fmchoice;
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

#endif  // ALBANY_ABSTRACT_PROBLEM_HPP

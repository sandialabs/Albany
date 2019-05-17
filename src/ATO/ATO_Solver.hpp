//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ATO_SOLVER_HPP
#define ATO_SOLVER_HPP

// ATO includes
#include "ATO_OptInterface.hpp"
#include "ATO_Types.hpp"

// Albany includes
#include "Albany_ThyraTypes.hpp"
#include "Albany_CommTypes.hpp"

// Trilinos includes
#include "Thyra_ResponseOnlyModelEvaluatorBase.hpp"

// System includes
#include <vector>

// Albany forward declarations
namespace Albany {
class Application;
class CombineAndScatterManager;
class StateManager;
} // namespace Albany

namespace ATO {

// ATO forward declarations
class Aggregator;
class OptimizationProblem;
class Optimizer;
class SpatialFilter;

// Subsolver structs
struct SolverSubSolver {
  Teuchos::RCP<Albany::Application>   app;
  Teuchos::RCP<Thyra_ModelEvaluator>  model;
  Teuchos::RCP<Thyra_InArgs>          params_in;
  Teuchos::RCP<Thyra_OutArgs>         responses_out;

  void freeUp() {
    app   = Teuchos::null;
    model = Teuchos::null;
  }
};

struct SolverSubSolverData {
  int Np;
  int Ng;

  std::vector<int> pLength;
  std::vector<int> gLength;

  Teuchos::RCP<const Thyra_Vector>  p_init;
  Thyra_DerivativeSupport           deriv_support;
};

// Solver class
class Solver : public Thyra::ResponseOnlyModelEvaluatorBase<ST>,
               public OptInterface
{
public:
   Solver(const Teuchos::RCP<Teuchos::ParameterList>& appParams,
          const Teuchos::RCP<const Teuchos_Comm>&     comm,
          const Teuchos::RCP<const Thyra_Vector>&     initial_guess);

  ~Solver() = default;

  // Override methods from Thyra_ModelEvaluator
  Teuchos::RCP<const Thyra_VectorSpace> get_x_space() const override { return Teuchos::null; } 
  Teuchos::RCP<const Thyra_VectorSpace> get_f_space() const override { return Teuchos::null; }  
  Teuchos::RCP<const Thyra_VectorSpace> get_g_space(const int ig) const override;
  Teuchos::RCP<const Thyra_VectorSpace> get_p_space(const int /* ip */) const override { return m_x_vs; }
  Thyra_InArgs  createInArgs()      const override;

  // Override method from Thyra::ResponseOnlyModelEvaluatorBase<ST>
  Thyra_OutArgs createOutArgsImpl() const override;
  void evalModelImpl (const Thyra_InArgs& inArgs, const Thyra_OutArgs& outArgs) const override;

  // Override methods from OptInterface
  void Compute (      double* p, double& f, double* dfdp, double& g, double* dgdp) override;
  void Compute (const double* p, double& f, double* dfdp, double& g, double* dgdp) override;

  void ComputeConstraint (double* /* p */, double& /* c */, double* /* dcdp */) override {}

  void ComputeObjective (      double* p, double& g, double* dgdp) override;
  void ComputeObjective (const double* p, double& g, double* dgdp) override;

  void writeCurrentDesign ();
  void InitializeOptDofs(double* p) override;
  void getOptDofsLowerBound (Teuchos::Array<double>& b) const override;
  void getOptDofsUpperBound (Teuchos::Array<double>& b) const override;
  void ComputeVolume(double* p, const double* dfdp, double& v, double threshhold, double minP) override;
  void ComputeMeasure(const std::string& measureType, const double* p,
                      double& measure, double* dmdp, const std::string& integrationMethod) override;
  void ComputeMeasure(const std::string& measureType, double& measure) override;
  int GetNumOptDofs() const override;

private:

  // data
  int m_iteration;
  int m_writeDesignFrequency;
  int m_numDims;
  int m_num_parameters; // for sensitivity analysis
  int m_num_responses;

  Teuchos::RCP<const Thyra_VectorSpace> m_x_vs;

  int m_numPhysics; // number of sub problems

  std::vector<int> m_wsOffset;  //index offsets to map to/from workset to/from 1D array.

  bool m_is_verbose;    // verbose or not for topological optimization solver
  bool m_is_restart;

  Teuchos::RCP<Aggregator> m_objAggregator;
  Teuchos::RCP<Aggregator> m_conAggregator;
  Teuchos::RCP<Optimizer>  m_optimizer;

  struct TopologyInfoStruct {
    Teuchos::RCP<Topology>      topology;
    Teuchos::RCP<SpatialFilter> filter;
    Teuchos::RCP<SpatialFilter> postFilter;
    Teuchos::RCP<Thyra_Vector>  filteredOverlapVector;
    Teuchos::RCP<Thyra_Vector>  filteredVector;
    Teuchos::RCP<Thyra_Vector>  overlapVector;
    Teuchos::RCP<Thyra_Vector>  localVector;
    bool                        filterIsRecursive;
  };

  std::vector<Teuchos::RCP<TopologyInfoStruct> >  m_topologyInfoStructs;
  Teuchos::RCP<TopologyArray>                     m_topologyArray;

  // currently all topologies must have the same entity type
  std::string m_entityType;

  std::vector<Teuchos::RCP<SpatialFilter> > m_filters;
  Teuchos::RCP<SpatialFilter>               m_derivativeFilter;

  struct HomogenizationSet {
    std::string name;
    std::string type;
    int responseIndex;
    int homogDim;
    std::vector<Teuchos::RCP<Teuchos::ParameterList> > homogenizationAppParams;
    std::vector<SolverSubSolver> homogenizationProblems;
  };

  std::vector<HomogenizationSet> m_homogenizationSets;

  std::vector<Teuchos::RCP<Teuchos::ParameterList> >  m_subProblemAppParams;
  std::vector<SolverSubSolver>                        m_subProblems;

  OptimizationProblem* m_atoProblem;

  Teuchos::RCP<const Teuchos_Comm>      m_solverComm;
  Teuchos::RCP<Teuchos::ParameterList>  m_mainAppParams;

  Teuchos::RCP<const Thyra_SpmdVectorSpace> m_overlapNodeVS;
  Teuchos::RCP<const Thyra_SpmdVectorSpace> m_localNodeVS;

  Teuchos::Array< Teuchos::RCP<Thyra_Vector> > m_overlapObjectiveGradientVec;
  Teuchos::Array< Teuchos::RCP<Thyra_Vector> > m_ObjectiveGradientVec;

  Teuchos::Array< Teuchos::RCP<Thyra_Vector> > m_overlapConstraintGradientVec;
  Teuchos::Array< Teuchos::RCP<Thyra_Vector> > m_ConstraintGradientVec;

  Teuchos::RCP<double> m_objectiveValue;
  Teuchos::RCP<double> m_constraintValue;

  Teuchos::RCP<Albany::CombineAndScatterManager> m_cas_manager;

  std::map<std::string, std::vector<Teuchos::RCP<const Thyra_Vector>>>  m_responseMap;
  std::map<std::string, std::vector<Teuchos::RCP<Thyra_MultiVector>>>   m_responseDerivMap;

  // === Private methods === //
  void copyTopologyIntoStateMgr(const double* p, Albany::StateManager& stateMgr );
  void smoothTopology(double* p);
  void smoothTopology(Teuchos::RCP<TopologyInfoStruct> topoStruct);
  void copyTopologyFromStateMgr(double* p, Albany::StateManager& stateMgr );
  void copyTopologyIntoParameter(const double* p, SolverSubSolver& sub);
  void copyObjectiveFromStateMgr( double& g, double* dgdp );
  void copyConstraintFromStateMgr( double& c, double* dcdp );
  Teuchos::RCP<const Teuchos::ParameterList> getValidProblemParameters() const;

  Teuchos::RCP<Teuchos::ParameterList>
    createInputFile( const Teuchos::RCP<Teuchos::ParameterList>& appParams, int physIndex) const;

  SolverSubSolver CreateSubSolver(const Teuchos::RCP<Teuchos::ParameterList>  appParams,
                                  const Teuchos::RCP<const Teuchos_Comm>&     comm,
                                  const Teuchos::RCP<const Thyra_Vector>&     initial_guess  = Teuchos::null);

  Teuchos::RCP<Teuchos::ParameterList>
    createHomogenizationInputFile (const Teuchos::RCP<Teuchos::ParameterList>& appParams,
                                   const Teuchos::ParameterList& homog_subList,
                                   int homogProblemIndex,
                                   int homogSubIndex,
                                   int homogDim) const;

  SolverSubSolverData CreateSubSolverData(const SolverSubSolver& sub) const;
};

} // namespace ATO

#endif // ATO_SOLVER_HPP

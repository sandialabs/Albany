//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ATO_SOLVER_H
#define ATO_SOLVER_H

#include <iostream>

#include "LOCA.H"
#include "LOCA_Epetra.H"
#include "Epetra_Vector.h"
#include "Epetra_LocalMap.h"
#include "Epetra_CrsMatrix.h"
#include "LOCA_Epetra_ModelEvaluatorInterface.H"
#include <NOX_Epetra_MultiVector.H>

#include "Albany_ModelEvaluator.hpp"
#include "Albany_StateManager.hpp"
#include "Albany_Utils.hpp"
#include "Piro_Epetra_StokhosNOXObserver.hpp"
#include "ATO_Types.hpp"
#include "ATO_Aggregator.hpp"
#include "ATO_Optimizer.hpp"

namespace ATO {

  class SolverSubSolver;
  class SolverSubSolverData;
  class OptimizationProblem;
  class Topology;

  typedef struct GlobalPoint{ 
    GlobalPoint();
    int    gid; 
    double coords[3]; 
  } GlobalPoint;
  bool operator< (GlobalPoint const & a, GlobalPoint const & b);

  // eventually make this a base class and derive from it to make
  // various kernels.  Also add a factory.
  class SpatialFilter{
    public:
      SpatialFilter(Teuchos::ParameterList& params);
      void buildOperator(
             Teuchos::RCP<Albany::Application> app,
             Teuchos::RCP<Epetra_Map>          overlapNodeMap,
             Teuchos::RCP<Epetra_Map>          localNodeMap,
             Teuchos::RCP<Epetra_Import>       importer,
             Teuchos::RCP<Epetra_Export>       exporter);
      Teuchos::RCP<Epetra_CrsMatrix> FilterOperator(){return filterOperator;}
    protected:
      void importNeighbors(
             std::map< GlobalPoint, std::set<GlobalPoint> >& neighbors,
             Teuchos::RCP<Epetra_Import>       importer,
             Teuchos::RCP<Epetra_Export>       exporter);

      Teuchos::RCP<Epetra_CrsMatrix> filterOperator;
      double filterRadius;
      Teuchos::Array<std::string> blocks;
  };

  class OptInterface {
  public:
    virtual void Compute(double* p, double& g, double* dgdp, double& c, double* dcdp=NULL)=0;
    virtual void Compute(const double* p, double& g, double* dgdp, double& c, double* dcdp=NULL)=0;

    virtual void ComputeConstraint(double* p, double& c, double* dcdp=NULL)=0;

    virtual void ComputeObjective(double* p, double& g, double* dgdp=NULL)=0;
    virtual void ComputeObjective(const double* p, double& g, double* dgdp=NULL)=0;
    virtual void InitializeOptDofs(double* p)=0;
    virtual void getOptDofsLowerBound( Teuchos::Array<double>& b )=0;
    virtual void getOptDofsUpperBound( Teuchos::Array<double>& b )=0;
    virtual void ComputeVolume(double* p, const double* dfdp, double& v, double threshhold, double minP)=0;
    virtual void ComputeMeasure(std::string measureType, const double* p, double& measure, double* dmdp=0)=0;
    virtual void ComputeMeasure(std::string measureType, double& measure)=0;
    virtual int GetNumOptDofs()=0;
  };

  class Solver : public EpetraExt::ModelEvaluator , public OptInterface {
  public:

     Solver(const Teuchos::RCP<Teuchos::ParameterList>& appParams,
            const Teuchos::RCP<const Epetra_Comm>& comm,
            const Teuchos::RCP<const Epetra_Vector>& initial_guess);

    ~Solver();

    //pure virtual from EpetraExt::ModelEvaluator
    virtual Teuchos::RCP<const Epetra_Map> get_x_map() const;
    //pure virtual from EpetraExt::ModelEvaluator
    virtual Teuchos::RCP<const Epetra_Map> get_f_map() const;
    //pure virtual from EpetraExt::ModelEvaluator
    virtual EpetraExt::ModelEvaluator::InArgs createInArgs() const;
    //pure virtual from EpetraExt::ModelEvaluator
    virtual EpetraExt::ModelEvaluator::OutArgs createOutArgs() const;
    //pure virtual from EpetraExt::ModelEvaluator
    void evalModel( const InArgs& inArgs, const OutArgs& outArgs ) const;

    void Compute(double* p, double& f, double* dfdp, double& g, double* dgdp=NULL);
    void Compute(const double* p, double& f, double* dfdp, double& g, double* dgdp=NULL);

    void ComputeConstraint(double* p, double& c, double* dcdp=NULL);

    void ComputeObjective(double* p, double& g, double* dgdp=NULL);
    void ComputeObjective(const double* p, double& g, double* dgdp=NULL);
    void writeCurrentDesign();
    void InitializeOptDofs(double* p);
    void getOptDofsLowerBound( Teuchos::Array<double>& b );
    void getOptDofsUpperBound( Teuchos::Array<double>& b );
    void ComputeVolume(double* p, const double* dfdp, double& v, double threshhold, double minP);
    void ComputeMeasure(std::string measureType, const double* p, double& measure, double* dmdp=0);
    void ComputeMeasure(std::string measureType, double& measure);
    int GetNumOptDofs();

  private:

    // data
    int _iteration;
    int _writeDesignFrequency;
    int  numDims;
    int _num_parameters; // for sensitiviy analysis(?)
    int _num_responses;  //  ditto
    Teuchos::RCP<Epetra_LocalMap> _epetra_param_map;
    Teuchos::RCP<Epetra_LocalMap> _epetra_response_map;
    Teuchos::RCP<Epetra_Map>      _epetra_x_map;

    int _numPhysics; // number of sub problems

    std::vector<int> _wsOffset;  //index offsets to map to/from workset to/from 1D array.

    bool _is_verbose;    // verbose or not for topological optimization solver
    bool _is_restart;

    Teuchos::RCP<Aggregator> _objAggregator;
    Teuchos::RCP<Aggregator> _conAggregator;
    Teuchos::RCP<Optimizer> _optimizer;

    typedef struct TopologyInfoStruct {
      Teuchos::RCP<Topology>      topology;
      Teuchos::RCP<SpatialFilter> filter;
      Teuchos::RCP<SpatialFilter> postFilter;
      Teuchos::RCP<Epetra_Vector> filteredOverlapVector;
      Teuchos::RCP<Epetra_Vector> filteredVector;
      Teuchos::RCP<Epetra_Vector> overlapVector;
      Teuchos::RCP<Epetra_Vector> localVector;
      bool                        filterIsRecursive;
    } TopologyInfoStruct;

    std::vector<Teuchos::RCP<TopologyInfoStruct> > _topologyInfoStructs;
    std::vector<Teuchos::RCP<TopologyStruct> > _topologyStructs;
    Teuchos::RCP<TopologyArray> _topologyArray;

    // currently all topologies must have the same entity type
    std::string entityType;

    std::vector<Teuchos::RCP<SpatialFilter> > filters;
    Teuchos::RCP<SpatialFilter> _derivativeFilter;


    typedef struct HomogenizationSet { 
      std::string name;
      std::string type;
      int responseIndex;
      int homogDim; 
      std::vector<Teuchos::RCP<Teuchos::ParameterList> > homogenizationAppParams;
      std::vector<SolverSubSolver> homogenizationProblems;
    } HomogenizationSet;

    std::vector<HomogenizationSet> _homogenizationSets;

    std::vector<Teuchos::RCP<Teuchos::ParameterList> > _subProblemAppParams;
    std::vector<SolverSubSolver> _subProblems;

    OptimizationProblem* _atoProblem;

    Teuchos::RCP<const Epetra_Comm> _solverComm;
    Teuchos::RCP<Teuchos::ParameterList> _mainAppParams;

    Teuchos::RCP<Epetra_Map> overlapNodeMap;
    Teuchos::RCP<Epetra_Map> localNodeMap;


    Teuchos::Array< Teuchos::RCP<Epetra_Vector> > overlapObjectiveGradientVec;
    Teuchos::Array< Teuchos::RCP<Epetra_Vector> > ObjectiveGradientVec;

    Teuchos::Array< Teuchos::RCP<Epetra_Vector> > overlapConstraintGradientVec;
    Teuchos::Array< Teuchos::RCP<Epetra_Vector> > ConstraintGradientVec;

    Teuchos::RCP<double> objectiveValue;
    Teuchos::RCP<double> constraintValue;

    Teuchos::RCP<Epetra_Import> importer;
    Teuchos::RCP<Epetra_Export> exporter;

    std::map<std::string, Teuchos::RCP<const Epetra_Vector> > responseMap;
    std::map<std::string, Teuchos::RCP<Epetra_MultiVector> > responseDerivMap;


    // methods
    void copyTopologyIntoStateMgr(const double* p, Albany::StateManager& stateMgr );
    void smoothTopology(double* p);
    void smoothTopology(Teuchos::RCP<TopologyInfoStruct> topoStruct);
    void copyTopologyFromStateMgr(double* p, Albany::StateManager& stateMgr );
    void copyTopologyIntoParameter(const double* p, SolverSubSolver& sub);
    void copyObjectiveFromStateMgr( double& g, double* dgdp );
    void copyConstraintFromStateMgr( double& c, double* dcdp );
    void zeroSet();
    Teuchos::RCP<const Teuchos::ParameterList> getValidProblemParameters() const;

    Teuchos::RCP<const Epetra_Map> get_g_map(int j) const;

    Teuchos::RCP<Teuchos::ParameterList> 
      createInputFile( const Teuchos::RCP<Teuchos::ParameterList>& appParams, int physIndex) const;

    SolverSubSolver CreateSubSolver(const Teuchos::RCP<Teuchos::ParameterList> appParams, 
                                    const Epetra_Comm& comm,
				    const Teuchos::RCP<const Epetra_Vector>& initial_guess  = Teuchos::null);

    Teuchos::RCP<Teuchos::ParameterList> 
      createHomogenizationInputFile( 
            const Teuchos::RCP<Teuchos::ParameterList>& appParams, 
            const Teuchos::ParameterList& homog_subList, 
            int homogProblemIndex, 
            int homogSubIndex, 
            int homogDim) const;

    SolverSubSolverData CreateSubSolverData(const ATO::SolverSubSolver& sub) const;

  };

  class SolverSubSolver {
  public:
    Teuchos::RCP<Albany::Application> app;
    Teuchos::RCP<EpetraExt::ModelEvaluator> model;
    Teuchos::RCP<EpetraExt::ModelEvaluator::InArgs> params_in;
    Teuchos::RCP<EpetraExt::ModelEvaluator::OutArgs> responses_out;
    void freeUp() { app = Teuchos::null; model = Teuchos::null; }
  };

  class SolverSubSolverData {
  public:
    int Np;
    int Ng;
    std::vector<int> pLength;
    std::vector<int> gLength;
    Teuchos::RCP<const Epetra_Vector> p_init;
    EpetraExt::ModelEvaluator::DerivativeSupport deriv_support;
  };

}
#endif

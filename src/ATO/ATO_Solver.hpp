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

  class OptInterface {
  public:
    virtual void ComputeObjective(const double* p, double& f, double* dfdp=NULL)=0;
    virtual void ComputeConstraint(double* p, double& c, double* dcdp=NULL)=0;
    virtual void ComputeVolume(const double* p, double& v, double* dvdp=NULL)=0;
    virtual void ComputeVolume(double& v)=0;
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

    void ComputeObjective(const double* p, double& f, double* dfdp=NULL);
    void ComputeConstraint(double* p, double& c, double* dcdp=NULL);
    void ComputeVolume(const double* p, double& v, double* dvdp=NULL);
    void ComputeVolume(double& v);
    int GetNumOptDofs();

  private:

    // data
    int  numDims;
    int _num_parameters; // for sensitiviy analysis(?)
    int _num_responses;  //  ditto
    Teuchos::RCP<Epetra_LocalMap> _epetra_param_map;
    Teuchos::RCP<Epetra_LocalMap> _epetra_response_map;
    Teuchos::RCP<Epetra_Map>      _epetra_x_map;

    int _numPhysics; // number of sub problems

    std::vector<int> _wsOffset;  //index offsets to map to/from workset to/from 1D array.

    bool _is_verbose;    // verbose or not for topological optimization solver

    Teuchos::RCP<Aggregator> _aggregator;
    Teuchos::RCP<Optimizer> _optimizer;
    Teuchos::RCP<Topology> _topology;

    double  _filterRadius; // not sure if this is the best place but for now...
    bool _filterDerivative;
    bool _filterTopology;
    bool _postFilterTopology;

    std::vector<Teuchos::RCP<Teuchos::ParameterList> > _subProblemAppParams;
    std::vector<SolverSubSolver> _subProblems;
    OptimizationProblem* _atoProblem;

    Teuchos::RCP<const Epetra_Comm> _solverComm;
    Teuchos::RCP<Teuchos::ParameterList> _mainAppParams;

    Teuchos::RCP<Epetra_Map> overlapNodeMap;
    Teuchos::RCP<Epetra_Map> localNodeMap;

    Teuchos::RCP<Epetra_Vector> filteredOTopoVec;
    Teuchos::RCP<Epetra_Vector> filteredTopoVec;

    Teuchos::RCP<Epetra_Vector> overlapTopoVec;
    Teuchos::RCP<Epetra_Vector> topoVec;

    Teuchos::RCP<Epetra_Vector> overlapdfdpVec;
    Teuchos::RCP<Epetra_Vector> dfdpVec;

    Teuchos::RCP<Epetra_Import> importer;
    Teuchos::RCP<Epetra_Export> exporter;

    Teuchos::RCP<Epetra_CrsMatrix> filterOperator;

    // methods
    void copyTopologyIntoStateMgr(const double* p, Albany::StateManager& stateMgr );
    void copyObjectiveFromStateMgr( double& f, double* dfdp );
    void zeroSet();
    void buildFilterOperator(const Teuchos::RCP<Albany::Application> app);
    void importNeighbors(std::map< GlobalPoint, std::set<GlobalPoint> >& neighbors);
    Teuchos::RCP<const Teuchos::ParameterList> getValidProblemParameters() const;
    Teuchos::RCP<Teuchos::ParameterList> 
      createInputFile( const Teuchos::RCP<Teuchos::ParameterList>& appParams, int physIndex) const;

    Teuchos::RCP<const Epetra_Map> get_g_map(int j) const;

    SolverSubSolver CreateSubSolver(const Teuchos::RCP<Teuchos::ParameterList> appParams, 
                                    const Epetra_Comm& comm,
				    const Teuchos::RCP<const Epetra_Vector>& initial_guess  = Teuchos::null) const;

    SolverSubSolverData CreateSubSolverData(const ATO::SolverSubSolver& sub) const;

    Teuchos::RCP<Teuchos::ParameterList>
    createElasticityInputFile( const Teuchos::RCP<Teuchos::ParameterList>& appParams,
                               int numDims,
                               const std::string& exoOutputFile ) const;
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

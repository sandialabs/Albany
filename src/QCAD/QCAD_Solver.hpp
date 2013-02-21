//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef QCAD_SOLVER_H
#define QCAD_SOLVER_H

#include <iostream>

#include "LOCA.H"
#include "LOCA_Epetra.H"
#include "Epetra_Vector.h"
#include "Epetra_LocalMap.h"
#include "LOCA_Epetra_ModelEvaluatorInterface.H"
#include <NOX_Epetra_MultiVector.H>

#include "Albany_ModelEvaluator.hpp"
#include "Albany_Utils.hpp"
#include "Piro_Epetra_StokhosNOXObserver.hpp"

namespace QCAD {
  class SolverParamFn;
  class SolverResponseFn;
  class SolverSubSolver;

/** \brief Epetra-based Model Evaluator for QCAD solver
 *
 */

  class Solver : public EpetraExt::ModelEvaluator {
  public:

    /** \name Constructors/initializers */
    //@{

      Solver(const Teuchos::RCP<Teuchos::ParameterList>& appParams,
	     const Teuchos::RCP<const Epetra_Comm>& comm);
    //@}

    ~Solver();

    Teuchos::RCP<const Epetra_Map> get_x_map() const;
    Teuchos::RCP<const Epetra_Map> get_f_map() const;
    Teuchos::RCP<const Epetra_Map> get_p_map(int l) const;
    Teuchos::RCP<const Epetra_Map> get_g_map(int j) const;

    Teuchos::RCP<const Epetra_Vector> get_x_init() const;
    Teuchos::RCP<const Epetra_Vector> get_p_init(int l) const;

    EpetraExt::ModelEvaluator::InArgs createInArgs() const;
    EpetraExt::ModelEvaluator::OutArgs createOutArgs() const;

    void evalModel( const InArgs& inArgs, const OutArgs& outArgs ) const;    


  private:
    void evalPoissonSchrodingerModel(const InArgs& inArgs, const OutArgs& outArgs ) const;
    void evalPoissonCIModel(const InArgs& inArgs, const OutArgs& outArgs ) const;

    void setupParameterMapping(const Teuchos::ParameterList& list);
    void setupResponseMapping(const Teuchos::ParameterList& list);

    void preprocessParams(Teuchos::ParameterList& params, std::string preprocessType);
    SolverSubSolver CreateSubSolver(const std::string xmlfilename, 
				    const std::string& xmlPreprocessType, const Epetra_Comm& comm);
    void SetCoulombParams(const Teuchos::RCP<EpetraExt::ModelEvaluator::InArgs> inArgs, int i2, int i4) const;


    const SolverSubSolver& getSubSolver(const std::string& name) const;
    
  private:
    std::string problemName;

    std::map<std::string, SolverSubSolver> subSolvers;

    std::vector< std::vector<Teuchos::RCP<SolverParamFn> > > paramFnVecs;
    std::vector<Teuchos::RCP<SolverResponseFn> > responseFns;

    std::size_t maxIter;
    std::size_t nParameters;
    std::size_t nResponseDoubles;
    

    std::string iterationMethod;
    int nEigenvectors; //used in Poisson-CI coupling

    int num_p, num_g;
    Teuchos::RCP<Epetra_LocalMap> epetra_param_map;
    Teuchos::RCP<Epetra_LocalMap> epetra_response_map;
    Teuchos::RCP<Epetra_Map> epetra_x_map;

    Teuchos::RCP<const Epetra_Comm> solverComm;

    bool bVerbose;
    bool bSupportDpDg;

    double CONVERGE_TOL;
    double shiftPercentBelowMin;  // for eigensolver shift-invert: shift point == minPotential * (1 + shiftPercent/100)
    int    maxCIParticles;        // the maximum number of particles allowed to be used in CI calculation

    static void setRequestSensitivities(Teuchos::ParameterList &params, bool flag);
  };


  // helper classes - maybe nest inside Solver?
  class SolverParamFn {
  public:
    SolverParamFn(const std::string& fnString, 
		  const std::map<std::string, SolverSubSolver>& subSolvers);
    ~SolverParamFn() {};

    void fillSubSolverParams(double parameterValue, 
			const std::map<std::string, SolverSubSolver>& subSolvers) const;

    double getInitialParam(const std::map<std::string, SolverSubSolver>& subSolvers) const;

    std::string getTargetName() const { return targetName; }
    std::vector<int> getTargetIndices() const { return targetIndices; }
    std::size_t getNumFilters() const { return filters.size(); }
    double getFilterScaling() const;

  protected:
    std::string targetName;
    std::vector<int> targetIndices;
    std::vector< std::vector<std::string> > filters;
  };

  class SolverResponseFn {
  public:
    SolverResponseFn(const std::string& fnString,
		     const std::map<std::string, SolverSubSolver>& subSolvers);
    ~SolverResponseFn() {};

    void fillSolverResponses(Epetra_Vector& g, Teuchos::RCP<Epetra_MultiVector>& dgdp, int offset,
			     const std::map<std::string, SolverSubSolver>& subSolvers,
			     const std::vector<std::vector<Teuchos::RCP<SolverParamFn> > >& paramFnVecs,
			     bool bSupportDpDg) const;

    std::size_t getNumDoubles() const { return numDoubles; }

  protected:
    struct ArrayRef { 
      std::string name; 
      std::vector<int> indices;
    };

    std::string fnName;
    std::vector<ArrayRef> params;
    std::size_t numDoubles; //number of doubles produced by this response
  };


  class SolverSubSolver {
  public:
    Teuchos::RCP<Albany::Application> app;
    Teuchos::RCP<EpetraExt::ModelEvaluator> model;
    Teuchos::RCP<EpetraExt::ModelEvaluator::InArgs> params_in;
    Teuchos::RCP<EpetraExt::ModelEvaluator::OutArgs> responses_out;
  };

  
}
#endif

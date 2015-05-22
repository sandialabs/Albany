//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef QCADT_SOLVERT_H
#define QCADT_SOLVERT_H

#include <iostream>

#include "LOCA.H"
//#include "LOCA_Epetra.H"
#include "Tpetra_Vector.hpp"
#include "Tpetra_Map.hpp"
//#include "LOCA_Epetra_ModelEvaluatorInterface.H"
//#include <NOX_Epetra_MultiVector.H>

#include "Albany_ModelEvaluator.hpp"
#include "Albany_Utils.hpp"
//#include "Piro_Epetra_StokhosNOXObserver.hpp"

#include "QCAD_MultiSolutionObserver.hpp"

#ifdef ALBANY_CI
#include "AlbanyCI_Types.hpp"
#include "AlbanyCI_Tensor.hpp"
#include "AlbanyCI_BlockTensor.hpp"
#include "AlbanyCI_SingleParticleBasis.hpp"
#include "AlbanyCI_BasisFactory.hpp"
#include "AlbanyCI_ManyParticleBasis.hpp"
#include "AlbanyCI_ManyParticleBasisBlock.hpp"
#include "AlbanyCI_MatrixFactory.hpp"
#include "AlbanyCI_ManyParticleMatrix.hpp"
#include "AlbanyCI_Solver.hpp"
#include "AlbanyCI_Solution.hpp"
#include "AlbanyCI_qnumbers.hpp"
#endif



namespace QCADT {
  class SolverParamFn;
  class SolverResponseFn;
  class SolverSubSolver;
  class SolverSubSolverData;

/** \brief Epetra-based Model Evaluator for QCAD solver
 *
 */

  class Solver : Thyra::ModelEvaluatorDefaultBase<ST> {
  public:

    /** \name Constructors/initializers */
    //@{

      Solver(const Teuchos::RCP<Teuchos::ParameterList>& appParams,
              const Teuchos::RCP<const Teuchos_Comm>& commT,
              const Teuchos::RCP<const Tpetra_Vector>& initial_guessT); //OK
    //@}

    ~Solver(); //OK

    Teuchos::RCP<const Thyra::VectorSpaceBase<ST> > get_x_space() const;
    Teuchos::RCP<const Thyra::VectorSpaceBase<ST> > get_f_space() const;
    Teuchos::RCP<const Thyra::VectorSpaceBase<ST> > get_p_space(int l) const;
    Teuchos::RCP<const Thyra::VectorSpaceBase<ST> > get_g_space(int j) const;
    Thyra::ModelEvaluatorBase::InArgs<ST> getNominalValues() const;
    void allocateVectors();

   // Teuchos::RCP<const Epetra_Vector> get_x_init() const;
   // Teuchos::RCP<const Epetra_Vector> get_p_init(int l) const;

    Thyra::ModelEvaluatorBase::InArgs<ST> createInArgs() const;
    Thyra::ModelEvaluatorBase::OutArgs<ST> createOutArgsImpl() const;

    void evalModelImpl(
      const Thyra::ModelEvaluatorBase::InArgs<ST>& inArgs,
      const Thyra::ModelEvaluatorBase::OutArgs<ST>& outArgs) const;
 


  private:

    Thyra::ModelEvaluatorBase::InArgs<ST> createInArgsImpl() const;

    //! Cached nominal values
    Thyra::ModelEvaluatorBase::InArgs<ST> nominalValues;
    
    Teuchos::RCP<Teuchos::ParameterList> createPoissonInputFile(const Teuchos::RCP<Teuchos::ParameterList>& appParams,
								int numDims, int nEigen, const std::string& specialProcessing,
								const std::string& xmlOutputFile, const std::string& exoOutputFile) const; //OK
    Teuchos::RCP<Teuchos::ParameterList> createSchrodingerInputFile(const Teuchos::RCP<Teuchos::ParameterList>& appParams,
								    int numDims, int nEigen, const std::string& specialProcessing,
								    const std::string& xmlOutputFile, const std::string& exoOutputFile) const; //OK
    Teuchos::RCP<Teuchos::ParameterList> createPoissonSchrodingerInputFile(const Teuchos::RCP<Teuchos::ParameterList>& appParams,
									   int numDims, int nEigen, const std::string& xmlOutputFile,
									   const std::string& exoOutputFile) const; //OK

    /*void evalPoissonSchrodingerModel(const InArgs& inArgs, const OutArgs& outArgs,
				     std::vector<double>& eigenvalResponses, std::map<std::string, SolverSubSolver>& subSolvers) const;
    void evalPoissonCIModel(const InArgs& inArgs, const OutArgs& outArgs,
			    std::vector<double>& eigenvalResponses, std::map<std::string, SolverSubSolver>& subSolvers) const;
    void evalCIModel(const InArgs& inArgs, const OutArgs& outArgs, 
		     std::vector<double>& eigenvalResponses, std::map<std::string, SolverSubSolver>& subSolvers) const;

    bool doPSLoop(const std::string& mode, const InArgs& inArgs, std::map<std::string, SolverSubSolver>& subSolvers, 
		  Teuchos::RCP<Albany::EigendataStruct>& eigenDataResult, bool bPrintNumOfQuantumElectrons) const;
  */
    void setupParameterMapping(const Teuchos::ParameterList& list, const std::string& defaultSubSolver,
			       const std::map<std::string, SolverSubSolverData>& subSolversData);
    void setupResponseMapping(const Teuchos::ParameterList& list, const std::string& defaultSubSolver, int nEigenvalues,
			      const std::map<std::string, SolverSubSolverData>& subSolversData);

    void fillSingleSubSolverParams(const Thyra::ModelEvaluatorBase::InArgs<ST>& inArgsT, const std::string& name, 
				   QCADT::SolverSubSolver& subSolver, int nLeaveOffEnd=0) const;

    SolverSubSolver CreateSubSolver(const Teuchos::RCP<Teuchos::ParameterList> appParams, 
                                    const Teuchos::RCP<const Teuchos_Comm>& commT,
				    const Teuchos::RCP<const Tpetra_Vector>& initial_guess  = Teuchos::null) const;

    SolverSubSolverData CreateSubSolverData(const QCADT::SolverSubSolver& sub) const;


    const Teuchos::RCP<Teuchos::ParameterList>& getSubSolverParams(const std::string& name) const;
    Teuchos::RCP<const Teuchos::ParameterList> getValidProblemParameters() const;

    /*void printResponses(const QCADT::SolverSubSolver& solver, 
			const std::string& solverName, 
			Teuchos::RCP<Teuchos::FancyOStream> out) const;
    */

  private:
    int numDims;
    std::string problemNameBase;
    std::string defaultSubSolver;
    Teuchos::RCP<Teuchos::ParameterList> mainAppParams;
    std::map<std::string, Teuchos::RCP<Teuchos::ParameterList> > subProblemAppParams;

    std::vector< std::vector<Teuchos::RCP<SolverParamFn> > > paramFnVecs;
    std::vector<Teuchos::RCP<SolverResponseFn> > responseFns;

    std::size_t maxIter;
    std::size_t nParameters;
    std::size_t nResponseDoubles;    

    std::string iterationMethod;
    int  nEigenvectors; //used in Poisson-CI coupling

    int num_p, num_g;
    Teuchos::RCP<const Tpetra_Map> tpetra_param_map;
    Teuchos::RCP<const Tpetra_Map> tpetra_response_map;
    Teuchos::RCP<const Tpetra_Map> tpetra_x_map;

    Teuchos::RCP<Tpetra_Vector> tpetra_param_vec;
    DerivativeSupport deriv_support;

    Teuchos::RCP<const Teuchos_Comm> solverCommT;
    Teuchos::RCP<const Tpetra_Vector> saved_initial_guess;

    bool bVerbose;
    bool bSupportDpDg;
    bool bRealEvecs;

    std::string eigensolverName;
    double ps_converge_tol;
    double shiftPercentBelowMin;  // for eigensolver shift-invert: shift point == minPotential * (1 + shiftPercent/100)
    int    minCIParticles;        // the minimum number of particles allowed to be used in CI calculation
    int    maxCIParticles;        // the maximum number of particles allowed to be used in CI calculation
    int    nCIParticles;          // the number of particles used in CI calculation
    int    nCIExcitations;        // the number of excitations used in CI calculation
    double fixedPSOcc;
    bool   bUseIntegratedPS;
    bool   bUseTotalSpinSymmetry; // use S2 symmetry in CI calculation
  };


  // helper classes - maybe nest inside Solver?
  class SolverParamFn {
  public:
    SolverParamFn(const std::string& fnString, 
		  const std::map<std::string, SolverSubSolverData>& subSolversData);
    ~SolverParamFn() {};

    void fillSingleSubSolverParams(double parameterValue, const std::string& subSolverName,
				   SolverSubSolver& subSolver) const;

    void fillSubSolverParams(double parameterValue, 
			     const std::map<std::string, SolverSubSolver>& subSolvers) const;

    double getInitialParam(const std::map<std::string, SolverSubSolverData>& subSolversData) const;

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
		     const std::map<std::string, SolverSubSolverData>& subSolversData,
		     int nEigenvalues);
    ~SolverResponseFn() {};

    void fillSolverResponses(Tpetra_Vector& gT, Teuchos::RCP<Tpetra_MultiVector>& dgdpT, int offset,
			     const std::map<std::string, SolverSubSolver>& subSolvers,
			     const std::vector<std::vector<Teuchos::RCP<SolverParamFn> > >& paramFnVecs,
			     bool bSupportDpDg, const std::vector<double>& eigenvalueResponses) const;

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
    Teuchos::RCP<Thyra::ModelEvaluator<ST> > modelT;
    Teuchos::RCP<Thyra::ModelEvaluatorBase::InArgs<ST> > params_in;
    Teuchos::RCP<Thyra::ModelEvaluatorBase::OutArgs<ST> > responses_out;
    void freeUp() { app = Teuchos::null; modelT = Teuchos::null; }
  };

  class SolverSubSolverData {
  public:
    int Np, Ng;
    std::vector<int> pLength, gLength;
    Teuchos::RCP<const Tpetra_Vector> p_init;
    Thyra::ModelEvaluatorBase::DerivativeSupport deriv_support;
  };


#ifdef ALBANY_CI
/*  class CISolver {
  public:
    CISolver(int n1PSpinlessStates, Teuchos::RCP<const Epetra_Comm> eComm, 
	     Teuchos::RCP<Teuchos::FancyOStream> outStream);

    Teuchos::RCP<Teuchos::ParameterList> getDefaultParameterList() const;

    void fill1Pmx(const Teuchos::RCP<Albany::EigendataStruct>& eigenData1P);
    void fill1Pmx(const Teuchos::RCP<Albany::EigendataStruct>& eigenData1P,
		  const Teuchos::RCP<Epetra_Vector>& g_noCharge,
		  const Teuchos::RCP<Epetra_Vector>& g_delta,
		  double deltaScale, bool bRealEvecs, bool bVerbose);
    void fill2Pmx(Teuchos::RCP<Albany::EigendataStruct> eigenData1P,
		  const SolverSubSolver* coulombSolver, 
		  const SolverSubSolver* coulombSolver_ImPart,
		  const Teuchos::RCP<Epetra_Vector>& g_noCharge,
		  bool bRealEvecs, bool bVerbose);

    Teuchos::RCP<AlbanyCI::Solution> Solve(Teuchos::RCP<Teuchos::ParameterList> AlbanyCIList) const;

    Teuchos::RCP<Epetra_MultiVector> ComputeStateDensities(Teuchos::RCP<Albany::EigendataStruct> eigenData1P,
							   Teuchos::RCP<AlbanyCI::Solution> soln);


  private:
    void SetCoulombParams(const Teuchos::RCP<EpetraExt::ModelEvaluator::InArgs> inArgs, int i2, int i4) const;

  private:
    // number of single particle states of each type of spin (up / down)
    int n1PperBlock;

    // 1P Blocks, accessed individually or as a vector
    Teuchos::RCP<AlbanyCI::Tensor<AlbanyCI::dcmplx> > blockU,blockD;
    std::vector<Teuchos::RCP<AlbanyCI::Tensor<AlbanyCI::dcmplx> > > blocks1P;
    Teuchos::RCP<AlbanyCI::BlockTensor<AlbanyCI::dcmplx> > mx1P;

    // 2P Blocks, accessed individually or as a vector
    Teuchos::RCP<AlbanyCI::Tensor<AlbanyCI::dcmplx> > blockUU, blockUD, blockDU, blockDD;
    std::vector<Teuchos::RCP<AlbanyCI::Tensor<AlbanyCI::dcmplx> > > blocks2P;
    Teuchos::RCP<AlbanyCI::BlockTensor<AlbanyCI::dcmplx> > mx2P;

    // 1P basis
    Teuchos::RCP<AlbanyCI::SingleParticleBasis> basis1P;

    // MPI Comm
    Teuchos::RCP<Teuchos::Comm<int> > comm;    

    // Output stream
    Teuchos::RCP<Teuchos::FancyOStream> out;
  };*/
#endif
  
}
#endif

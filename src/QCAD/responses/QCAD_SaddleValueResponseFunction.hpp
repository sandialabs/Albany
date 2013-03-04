//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef QCAD_SADDLEVALUERESPONSEFUNCTION_HPP
#define QCAD_SADDLEVALUERESPONSEFUNCTION_HPP

#include "Albany_FieldManagerScalarResponseFunction.hpp"
#include "QCAD_MaterialDatabase.hpp"

#define MAX_DIMENSIONS 3

namespace QCAD {

  // Helper class: a vector with math operators
  class mathVector
  {
  public:
    mathVector();
    mathVector(int n);
    mathVector(const mathVector& copy);
    ~mathVector();

    void resize(std::size_t n);
    void fill(double d);
    void fill(const double* vec);
    double dot(const mathVector& v2) const;
    double distanceTo(const mathVector& v2) const;
    double distanceTo(const double* p) const;

    double norm() const;
    double norm2() const;
    void normalize();

    double* data();
    const double* data() const;
    std::size_t size() const;

    mathVector& operator=(const mathVector& rhs);

    mathVector operator+(const mathVector& v2) const;
    mathVector operator-(const mathVector& v2) const;
    mathVector operator*(double scale) const;

    mathVector& operator+=(const mathVector& v2);
    mathVector& operator-=(const mathVector& v2);
    mathVector& operator*=(double scale);
    mathVector& operator/=(double scale);

    double& operator[](int i);
    const double& operator[](int i) const;

  private:
    int dim_;
    std::vector<double> data_;
  };

  // Data Structure for an image point
  struct nebImagePt {
    void init(int nDims, double ptRadius) {
      coords.resize(nDims); coords.fill(0.0);
      velocity.resize(nDims); velocity.fill(0.0);
      grad.resize(nDims); grad.fill(0.0);
      value = weight = 0.0;
      radius = ptRadius;
    }

    void init(const mathVector& coordPt, double ptRadius) {
      init(coordPt.size(), ptRadius);
      coords = coordPt;
    }
      
    mathVector coords;
    mathVector velocity;
    mathVector grad;
    double value;
    double weight;
    double radius;
  };

  std::ostream& operator<<(std::ostream& os, const mathVector& mv);
  std::ostream& operator<<(std::ostream& os, const nebImagePt& np);

  // a double array with maximal dimension
  struct maxDimPt { 
    maxDimPt(const double* p, int numDims) {
      for(int i=0; i<numDims; i++) data[i] = p[i];
    }
    double data[MAX_DIMENSIONS];
  };

 
  /*!
   * \brief Reponse function for finding saddle point values of a field
   */
  class SaddleValueResponseFunction : 
    public Albany::FieldManagerScalarResponseFunction {
  public:
  
    //! Constructor
    SaddleValueResponseFunction(
      const Teuchos::RCP<Albany::Application>& application,
      const Teuchos::RCP<Albany::AbstractProblem>& problem,
      const Teuchos::RCP<Albany::MeshSpecsStruct>&  ms,
      const Teuchos::RCP<Albany::StateManager>& stateMgr,
      Teuchos::ParameterList& responseParams);

    //! Destructor
    virtual ~SaddleValueResponseFunction();

    //! Get the number of responses
    virtual unsigned int numResponses() const;

    //! Evaluate responses
    virtual void 
    evaluateResponse(const double current_time,
		     const Epetra_Vector* xdot,
		     const Epetra_Vector& x,
		     const Teuchos::Array<ParamVec>& p,
		     Epetra_Vector& g);

    //! Evaluate tangent = dg/dx*dx/dp + dg/dxdot*dxdot/dp + dg/dp
    virtual void 
    evaluateTangent(const double alpha, 
		    const double beta,
		    const double current_time,
		    bool sum_derivs,
		    const Epetra_Vector* xdot,
		    const Epetra_Vector& x,
		    const Teuchos::Array<ParamVec>& p,
		    ParamVec* deriv_p,
		    const Epetra_MultiVector* Vxdot,
		    const Epetra_MultiVector* Vx,
		    const Epetra_MultiVector* Vp,
		    Epetra_Vector* g,
		    Epetra_MultiVector* gx,
		    Epetra_MultiVector* gp);

    //! Evaluate gradient = dg/dx, dg/dxdot, dg/dp
    virtual void 
    evaluateGradient(const double current_time,
		     const Epetra_Vector* xdot,
		     const Epetra_Vector& x,
		     const Teuchos::Array<ParamVec>& p,
		     ParamVec* deriv_p,
		     Epetra_Vector* g,
		     Epetra_MultiVector* dg_dx,
		     Epetra_MultiVector* dg_dxdot,
		     Epetra_MultiVector* dg_dp);


    //! Post process responses
    virtual void 
    postProcessResponses(const Epetra_Comm& comm, const Teuchos::RCP<Epetra_Vector>& g);

    //! Post process response derivatives
    virtual void 
    postProcessResponseDerivatives(const Epetra_Comm& comm, const Teuchos::RCP<Epetra_MultiVector>& gt);

    //! Called by evaluator to interface with class data that persists across worksets
    std::string getMode();
    bool pointIsInImagePtRegion(const double* p, double refZ) const;
    bool pointIsInAccumRegion(const double* p, double refZ) const;
    bool pointIsInLevelSetRegion(const double* p, double refZ) const;
    void addBeginPointData(const std::string& elementBlock, const double* p, double value);
    void addEndPointData(const std::string& elementBlock, const double* p, double value);
    void addImagePointData(const double* p, double value, double* grad);
    void addFinalImagePointData(const double* p, double value);
    void accumulatePointData(const double* p, double value, double* grad);
    void accumulateLevelSetData(const double* p, double value, double cellArea);
    double getSaddlePointWeight(const double* p) const;
    double getTotalSaddlePointWeight() const;
    const double* getSaddlePointPosition() const;
    double getCurrent(const double& lattTemp, const Teuchos::RCP<QCAD::MaterialDatabase>& materialDB) const;
    
  private:

    //! Helper functions for Nudged Elastic Band (NEB) algorithm, performed in evaluateResponse
    void initializeImagePoints(const double current_time, const Epetra_Vector* xdot,
			       const Epetra_Vector& x, const Teuchos::Array<ParamVec>& p,
			       Epetra_Vector& g, int dbMode);
    void initializeFinalImagePoints(const double current_time, const Epetra_Vector* xdot,
			       const Epetra_Vector& x, const Teuchos::Array<ParamVec>& p,
			       Epetra_Vector& g, int dbMode);
    void doNudgedElasticBand(const double current_time, const Epetra_Vector* xdot,
			     const Epetra_Vector& x, const Teuchos::Array<ParamVec>& p,
			     Epetra_Vector& g, int dbMode);
    void fillSaddlePointData(const double current_time, const Epetra_Vector* xdot,
			     const Epetra_Vector& x, const Teuchos::Array<ParamVec>& p,
			     Epetra_Vector& g, int dbMode);

    //! Helper functions for level-set algorithm, performed in evaluateResponse
    void doLevelSet(const double current_time,  const Epetra_Vector* xdot,
		    const Epetra_Vector& x,  const Teuchos::Array<ParamVec>& p,
		    Epetra_Vector& g, int dbMode);
    int FindSaddlePoint_LevelSet(std::vector<double>& allFieldVals,
			     std::vector<double>* allCoords, std::vector<int>& ordering,
			     double cutoffDistance, double cutoffFieldVal, double minDepth, int dbMode,
			     Epetra_Vector& g);

    //! Helper functions for doNudgedElasticBand(...)
    void getImagePointValues(const double current_time, const Epetra_Vector* xdot,
			     const Epetra_Vector& x, const Teuchos::Array<ParamVec>& p,
			     Epetra_Vector& g, double* globalPtValues, double* globalPtWeights,
			     double* globalPtGrads, std::vector<mathVector> lastPositions, int dbMode);
    void getFinalImagePointValues(const double current_time, const Epetra_Vector* xdot,
			     const Epetra_Vector& x, const Teuchos::Array<ParamVec>& p,
			     Epetra_Vector& g, int dbMode);
    void writeOutput(int nIters);
    void initialIterationSetup(double& gradScale, double& springScale, int dbMode);
    void computeTangent(std::size_t i, mathVector& tangent, int dbMode);
    void computeClimbingForce(std::size_t i, const QCAD::mathVector& tangent, 
			      const double& gradScale, QCAD::mathVector& force, int dbMode);
    void computeForce(std::size_t i, const QCAD::mathVector& tangent, 
		      const std::vector<double>& springConstants,
		      const double& gradScale,  const double& springScale, 
		      QCAD::mathVector& force, double& dt, double& dt2, int dbMode);

    bool matchesCurrentResults(Epetra_Vector& g) const;


    //! Private to prohibit copying
    SaddleValueResponseFunction(const SaddleValueResponseFunction&);
    
    //! Private to prohibit copying
    SaddleValueResponseFunction& operator=(const SaddleValueResponseFunction&);

    //! function giving distribution of weights for "point"
    double pointFn(double d, double radius) const;

    //! helper function to get the highest image point (the one with the largest value)
    int getHighestPtIndex() const;

    //! Epetra Communicator
    Teuchos::RCP<const Epetra_Comm> comm;

    //! data used across worksets and processors in saddle point algorithm
    std::size_t numDims;
    std::size_t nImagePts;
    std::vector<nebImagePt> imagePts;
    std::vector<nebImagePt> finalPts;
    double imagePtSize;
    bool bClimbing, bAdaptivePointSize;
    double minAdaptivePointWt, maxAdaptivePointWt;
    double antiKinkFactor;

    //! data for memory-intensive but fast mode (hold entire proc's data)
    bool bAggregateWorksets;
    std::vector<double> vFieldValues;
    std::vector<maxDimPt> vCoords;
    std::vector<maxDimPt> vGrads;

    //! data for level set method
    std::vector<double> vlsFieldValues;
    std::vector<double> vlsCellAreas;
    std::vector<double> vlsCoords[MAX_DIMENSIONS];
    double fieldCutoffFctr;
    double minPoolDepthFctr;
    double distanceCutoffFctr;
    double levelSetRadius;

    // index into imagePts of the "found" saddle point
    int iSaddlePt;
    double returnFieldVal;  // value of the return field at the "found" saddle point

    double maxTimeStep, minTimeStep;
    double minSpringConstant, maxSpringConstant;
    std::size_t maxIterations;
    std::size_t backtraceAfterIters;
    double convergeTolerance;

    //! data for beginning and ending regions
    std::string beginRegionType, endRegionType; // "Point", "Element Block", or "Polygon"
    std::string beginElementBlock, endElementBlock;
    std::vector<mathVector> beginPolygon, endPolygon;
    bool saddleGuessGiven;
    mathVector saddlePointGuess;
    double shortenBeginPc, shortenEndPc;

    double zmin, zmax;  //defines lateral-volume region when numDims == 3
    double xmin, xmax, ymin, ymax; // dynamically adjusted box marking region containing image points
    bool bLockToPlane;
    double lockedZ;

    //! data for final points (just used at end to get more data pts along saddle path)
    int maxFinalPts;
    
    double gfGridSpacing;  // grid spacing for GB-CBR calculation
    double fieldScaling;   // unscale the field specified by Field Name by Field Scaling Factor
    
    double initVds;        // initial Vds value in [V]
    double finalVds;       // final Vds value in [V]
    int stepsVds;          // number of steps from initial to final Vds
    bool bSweepVds;        // true if sweeping Vds, false if only using finalVds
    
    std::string gfEigensolver;  // Anasazi or tql2 eigensolver used for the GF-CBR calculation

    bool bGetCurrent;
    double current_Ecutoff_offset_from_Emax;

    //! accumulation vectors for evaluator to fill
    mathVector imagePtValues;
    mathVector imagePtWeights;
    mathVector imagePtGradComps;

    mathVector finalPtValues;
    mathVector finalPtWeights;


    //! mode of current evaluator operation (maybe not thread safe?)
    std::string mode;

    int  debugMode;

    std::string outputFilename;
    std::string debugFilename;
    int nEvery;
    bool appendOutput; // true => output is just appended if files exist
  };

  


}

#endif // QCAD_SADDLEVALUERESPONSEFUNCTION_HPP

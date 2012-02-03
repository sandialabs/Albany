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


#ifndef QCAD_SADDLEVALUERESPONSEFUNCTION_HPP
#define QCAD_SADDLEVALUERESPONSEFUNCTION_HPP

#include "Albany_FieldManagerScalarResponseFunction.hpp"

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
    void normalize();

    double* data();
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
  struct nebPt {
    void init(int nDims) {
      coords.resize(nDims); coords.fill(0.0);
      velocity.resize(nDims); velocity.fill(0.0);
      grad.resize(nDims); grad.fill(0.0);
      value = weight = 0.0;
    }

    void init(const mathVector& coordPt) {
      init(coordPt.size());
      coords = coordPt;
    }
      
    mathVector coords;
    mathVector velocity;
    mathVector grad;
    double value;
    double weight;
  };


  // Data structure for a piece of the searched-region boundary
  // Now only holds line segments.  For non-lateral volume support 
  //  this needs to hold segments of planes.
  struct nebBoundaryPiece {
    nebBoundaryPiece(int dim) :p1(dim), p2(dim) {}
    mathVector p1;
    mathVector p2;
  };

  std::ostream& operator<<(std::ostream& os, const mathVector& mv);
  std::ostream& operator<<(std::ostream& os, const nebPt& np);
 
 
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
    int checkIfPointIsOnBoundary(const double* p);
    bool checkIfPointIsWithinBoundary(const double* p);

    void addBoundaryData(const double* p, double value);
    void addImagePointData(const double* p, double value, double* grad);
    double getSaddlePointWeight(const double* p);

    
  private:

    //! Private to prohibit copying
    SaddleValueResponseFunction(const SaddleValueResponseFunction&);
    
    //! Private to prohibit copying
    SaddleValueResponseFunction& operator=(const SaddleValueResponseFunction&);

    //! Setup boundary variables given input parameter list
    void setupBoundary(Teuchos::ParameterList& params);

    //! data used across worksets and processors in saddle point algorithm
    std::size_t numDims;
    std::size_t nImagePts;
    std::vector<nebPt> imagePts;
    double imagePtSize;
    mathVector saddlePt;

    double timeStep;
    double convergenceThreshold;
    double baseSpringConstant;
    std::size_t maxIterations;

    //! data for region to be searched and its boundary
    //std::string ebName;
    bool bLateralVolumes;

    double zmin, zmax;  //for lateral-volume regions (where z-coord is separated out)
    std::vector<nebBoundaryPiece> boundary;
    std::vector<nebPt> boundaryMinima;

    //! mode of current evaluator operation (maybe not thread safe?)
    std::string mode;

    int  debugMode;
    bool bPositiveOnly;
  };

  


}

#endif // QCAD_SADDLEVALUERESPONSEFUNCTION_HPP

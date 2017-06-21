//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include "Albany_SolutionFileResponseFunction.hpp"
#if defined(ALBANY_EPETRA)
#include "Epetra_Map.h"  
#include "EpetraExt_BlockMapIn.h"  
#include "Epetra_SerialComm.h"  ///HAQ
#endif
#include "Teuchos_CommHelpers.hpp"
#include "Tpetra_DistObject.hpp"

#if defined(ALBANY_EPETRA)
template<class Norm>
Albany::SolutionFileResponseFunction<Norm>::
SolutionFileResponseFunction(const Teuchos::RCP<const Teuchos_Comm>& commT)
  : SamplingBasedScalarResponseFunction(commT),
    RefSoln(NULL), RefSolnT(NULL), solutionLoaded(false)
{
}
#else
template<class Norm>
Albany::SolutionFileResponseFunction<Norm>::
SolutionFileResponseFunction(const Teuchos::RCP<const Teuchos_Comm>& commT)
  : SamplingBasedScalarResponseFunction(commT),
    RefSolnT(NULL), solutionLoaded(false)
{
}
#endif

template<class Norm>
Albany::SolutionFileResponseFunction<Norm>::
~SolutionFileResponseFunction()
{
  if (solutionLoaded) {
#if defined(ALBANY_EPETRA)
    delete RefSoln; 
#endif
    delete RefSolnT; 
  }
}

template<class Norm>
unsigned int
Albany::SolutionFileResponseFunction<Norm>::
numResponses() const 
{
  return 1;
}

template<class Norm>
void
Albany::SolutionFileResponseFunction<Norm>::
evaluateResponseT(const double current_time,
		 const Tpetra_Vector* xdotT,
		 const Tpetra_Vector* xdotdotT,
		 const Tpetra_Vector& xT,
		 const Teuchos::Array<ParamVec>& p,
		 Tpetra_Vector& gT)
{
  int MMFileStatus = 0;

  // Read the reference solution for comparison from "reference_solution.dat"

  // Note that this is of MatrixMarket array real general format

  if (!solutionLoaded) {
//    MMFileStatus = EpetraExt::MatrixMarketFileToVector("reference_solution.dat",x.Map(),RefSoln);
    MMFileStatus = MatrixMarketFileToTpetraVector("reference_solution.dat",*(xT.getMap()),RefSolnT);

    TEUCHOS_TEST_FOR_EXCEPTION(MMFileStatus, std::runtime_error,
      std::endl << "EpetraExt::MatrixMarketFileToVector, file " __FILE__
      " line " << __LINE__ << " returned " << MMFileStatus << std::endl);

    solutionLoaded = true;
  }


  // Build a vector to hold the difference between the actual and reference solutions
  Tpetra_Vector diffT(xT.getMap());

  double normval;
  Norm vec_op;

  // The diff vector equals 1.0 * soln + -1.0 * reference
  diffT.update(1.0,xT,-1.0,*RefSolnT,0.0); 

  // Print vectors for debugging
/*
  std::cout << "Difference from evaluate response" << std::endl;
  diff.Print(std::cout);
  std::cout << "Solution from evaluate response" << std::endl;
  x.Print(std::cout);
  std::cout << "Ref solution from evaluate response" << std::endl;
  RefSoln->Print(std::cout);
*/

  // Get the norm
  //normval = vec_op.Norm(diff);

  //g[0]=normval;
}

template<class Norm>
void
Albany::SolutionFileResponseFunction<Norm>::
evaluateTangentT(
	   const double alpha, 
	   const double beta,
	   const double omega,
	   const double current_time,
	   bool sum_derivs,
	   const Tpetra_Vector* xdot,
	   const Tpetra_Vector* xdotdot,
	   const Tpetra_Vector& x,
	   const Teuchos::Array<ParamVec>& p,
	   ParamVec* deriv_p,
	   const Tpetra_MultiVector* Vxdot,
	   const Tpetra_MultiVector* Vxdotdot,
	   const Tpetra_MultiVector* Vx,
	   const Tpetra_MultiVector* Vp,
	   Tpetra_Vector* g,
	   Tpetra_MultiVector* gx,
	   Tpetra_MultiVector* gp)
{
}

#if defined(ALBANY_EPETRA)
template<class Norm>
void
Albany::SolutionFileResponseFunction<Norm>::
evaluateGradient(const double current_time,
		 const Epetra_Vector* xdot,
		 const Epetra_Vector* xdotdot,
		 const Epetra_Vector& x,
		 const Teuchos::Array<ParamVec>& p,
		 ParamVec* deriv_p,
		 Epetra_Vector* g,
		 Epetra_MultiVector* dg_dx,
		 Epetra_MultiVector* dg_dxdot,
		 Epetra_MultiVector* dg_dxdotdot,
		 Epetra_MultiVector* dg_dp)
{
  int MMFileStatus = 0;

  if (!solutionLoaded) {
//    MMFileStatus = EpetraExt::MatrixMarketFileToVector("reference_solution.dat",x.Map(),RefSoln);
    MMFileStatus = MatrixMarketFileToVector("reference_solution.dat",x.Map(),RefSoln);

    TEUCHOS_TEST_FOR_EXCEPTION(MMFileStatus, std::runtime_error,
      std::endl << "EpetraExt::MatrixMarketFileToVector, file " __FILE__
      " line " << __LINE__ << " returned " << MMFileStatus << std::endl);

    solutionLoaded = true;
  }


  // Build a vector to hold the difference between the actual and reference solutions
  Epetra_Vector diff(x.Map());

  double normval;
  Norm vec_op;

  // Evaluate response g
  if (g != NULL) {

  // The diff vector equals 1.0 * soln + -1.0 * reference

    diff.Update(1.0,x,-1.0,*RefSoln,0.0);

    // Print vectors for debugging
/*
    std::cout << "Difference from evaluate gradient" << std::endl;
    diff.Print(std::cout);
    std::cout << "Solution from evaluate gradient" << std::endl;
    x.Print(std::cout);
    std::cout << "Ref solution from evaluate gradient" << std::endl;
    RefSoln->Print(std::cout);
*/

    normval = vec_op.Norm(diff);
    (*g)[0]=normval;
  }

  // Evaluate dg/dx
  if (dg_dx != NULL)
    dg_dx->Update(2.0,x,-2.0,*RefSoln,0.0);

  // Evaluate dg/dxdot
  if (dg_dxdot != NULL)
    dg_dxdot->PutScalar(0.0);
  if (dg_dxdotdot != NULL)
    dg_dxdotdot->PutScalar(0.0);

  // Evaluate dg/dp
  if (dg_dp != NULL)
    dg_dp->PutScalar(0.0);
}
#endif

template<class Norm>
void
Albany::SolutionFileResponseFunction<Norm>::
evaluateGradientT(const double current_time,
		 const Tpetra_Vector* xdotT,
		 const Tpetra_Vector* xdotdotT,
		 const Tpetra_Vector& xT,
		 const Teuchos::Array<ParamVec>& p,
		 ParamVec* deriv_p,
		 Tpetra_Vector* gT,
		 Tpetra_MultiVector* dg_dxT,
		 Tpetra_MultiVector* dg_dxdotT,
		 Tpetra_MultiVector* dg_dxdotdotT,
		 Tpetra_MultiVector* dg_dpT)
{
}

//! Evaluate distributed parameter derivative dg/dp
template<class Norm>
void
Albany::SolutionFileResponseFunction<Norm>::
evaluateDistParamDerivT(
    const double current_time,
    const Tpetra_Vector* xdotT,
    const Tpetra_Vector* xdotdotT,
    const Tpetra_Vector& xT,
    const Teuchos::Array<ParamVec>& param_array,
    const std::string& dist_param_name,
    Tpetra_MultiVector* dg_dpT) {
  if (dg_dpT != NULL)
    dg_dpT->putScalar(0.0);
}

// This is "borrowed" from EpetraExt because more explicit debugging information is needed than
//  is present in the EpetraExt version. TO DO: Move this back there

#if defined(ALBANY_EPETRA)
template<class Norm>
int 
Albany::SolutionFileResponseFunction<Norm>::
MatrixMarketFileToVector( const char *filename, const Epetra_BlockMap & map, Epetra_Vector * & A) {

  Epetra_MultiVector * A1;
  if (MatrixMarketFileToMultiVector(filename, map, A1)) return(-1);
  A = dynamic_cast<Epetra_Vector *>(A1);
  return(0);
}
#endif

template<class Norm>
int 
Albany::SolutionFileResponseFunction<Norm>::
MatrixMarketFileToTpetraVector( const char *filename, const Tpetra_Map & mapT, Tpetra_Vector * & AT) {

  Tpetra_MultiVector * A1;
  if (MatrixMarketFileToTpetraMultiVector(filename, mapT, A1)) return(-1);
  AT = dynamic_cast<Tpetra_Vector *>(A1);
  return(0);
}

#if defined(ALBANY_EPETRA)
template<class Norm>
int 
Albany::SolutionFileResponseFunction<Norm>::
MatrixMarketFileToMultiVector( const char *filename, const Epetra_BlockMap & map, Epetra_MultiVector * & A) {

  const int lineLength = 1025;
  const int tokenLength = 35;
  char line[lineLength];
  char token1[tokenLength];
  char token2[tokenLength];
  char token3[tokenLength];
  char token4[tokenLength];
  char token5[tokenLength];
  int M, N;

  FILE * handle = 0;

  handle = fopen(filename,"r");  // Open file
  if (handle == 0)
    // file not found
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
      std::endl << "Reference solution file \" " << filename << " \" not found"
      << std::endl);

  // Check first line, which should be "%%MatrixMarket matrix coordinate real general" (without quotes)
  if(fgets(line, lineLength, handle)==0) 

    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
      std::endl << "Reference solution: MatrixMarket file is not in the proper format."
      << std::endl);

  if(sscanf(line, "%s %s %s %s %s", token1, token2, token3, token4, token5 )==0)

    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
      std::endl << "Incorrect number of arguments on first line of reference solution file."
      << std::endl);

  if (strcmp(token1, "%%MatrixMarket") ||
      strcmp(token2, "matrix") ||
      strcmp(token3, "array") ||
      strcmp(token4, "real") ||
      strcmp(token5, "general")) 

    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
      std::endl << "Incorrect type of arguments on first line of reference solution file."
      << std::endl);

  // Next, strip off header lines (which start with "%")
  do {
    if(fgets(line, lineLength, handle)==0)
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
        std::endl << "Reference solution file: invalid comment line."
        << std::endl);
  } while (line[0] == '%');

  // Next get problem dimensions: M, N
  if(sscanf(line, "%d %d", &M, &N)==0)

      TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
        std::endl << "Reference solution file: cannot compute problem dimensions"
        << std::endl);

  // Compute the offset for each processor for when it should start storing values
  int numMyPoints = map.NumMyPoints();
  int offset;
  //map.Comm().ScanSum(&numMyPoints, &offset, 1); // ScanSum will compute offsets for us
  //offset -= numMyPoints; // readjust for my PE

  // Line to start reading in reference file
//  offset = map.MinMyGID();

  if(map.Comm().MyPID() == 0){
    std::cout << "Reading reference solution from file \"" << filename << "\"" << std::endl;
    std::cout << "Reference solution contains " << N << " vectors, each with " << M << " rows." << std::endl;
    std::cout << std::endl;
  }

  // Now construct vector/multivector
  if (N==1)
    A = new Epetra_Vector(map);
  else
    A = new Epetra_MultiVector(map, N);

  double ** Ap = A->Pointers();

/*
  for (int j=0; j<N; j++) {
    double * v = Ap[j];

    // Now read in lines that we will discard
    for (int i=0; i<offset; i++)
      if(fgets(line, lineLength, handle)==0)

        TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
          std::endl << "Reference solution file: invalid line number " << i << " found while reading file lead data."
          << std::endl);
    
    // Now read in each value and store to the local portion of the the  if the row is owned.
    double V;
    for (int i=0; i<numMyPoints; i++) {
      if(fgets(line, lineLength, handle)==0) 

        TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
          std::endl << "Reference solution file: cannot read line number " << i + offset << " in file."
          << std::endl);

      if(sscanf(line, "%lg\n", &V)==0)

        TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
          std::endl << "Reference solution file: cannot parse line number " << i << " in file."
          << std::endl);

      v[i] = V;
    }
    // Now read in the rest of the lines to discard
    for (int i=0; i < M-numMyPoints-offset; i++) {
      if(fgets(line, lineLength, handle)==0)

        TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
          std::endl << "Reference solution file: invalid line number " << i + offset + numMyPoints
          << " found parsing toward end of file."
          << std::endl);
    }
  }
*/

  for (int j=0; j<N; j++) {
    double * v = Ap[j];

    // Now read in each value and store to the local portion of the array if the row is owned.
    double V;
    for (int i=0; i<M; i++) { // i is rownumber in file, or the GID 
      if(fgets(line, lineLength, handle)==0)  // Can't read the line

        TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
          std::endl << "Reference solution file: cannot read line number " << i + offset << " in file."
          << std::endl);

      if(map.LID(i)>= 0){ // we own this data value
       if(sscanf(line, "%lg\n", &V)==0)

        TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
          std::endl << "Reference solution file: cannot parse line number " << i << " in file."
          << std::endl);

       v[map.LID(i)] = V;
      }
    }
  }

  if (fclose(handle))

        TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
          std::endl << "Cannot close reference solution file."
          << std::endl);
  
  return(0);
}
#endif

template<class Norm>
int 
Albany::SolutionFileResponseFunction<Norm>::
MatrixMarketFileToTpetraMultiVector( const char *filename, const Tpetra_Map & mapT, Tpetra_MultiVector * & AT) {

  const int lineLength = 1025;
  const int tokenLength = 35;
  char line[lineLength];
  char token1[tokenLength];
  char token2[tokenLength];
  char token3[tokenLength];
  char token4[tokenLength];
  char token5[tokenLength];
  int M, N;

  FILE * handle = 0;

  handle = fopen(filename,"r");  // Open file
  if (handle == 0)
    // file not found
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
      std::endl << "Reference solution file \" " << filename << " \" not found"
      << std::endl);

  // Check first line, which should be "%%MatrixMarket matrix coordinate real general" (without quotes)
  if(fgets(line, lineLength, handle)==0) 

    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
      std::endl << "Reference solution: MatrixMarket file is not in the proper format."
      << std::endl);

  if(sscanf(line, "%s %s %s %s %s", token1, token2, token3, token4, token5 )==0)

    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
      std::endl << "Incorrect number of arguments on first line of reference solution file."
      << std::endl);

  if (strcmp(token1, "%%MatrixMarket") ||
      strcmp(token2, "matrix") ||
      strcmp(token3, "array") ||
      strcmp(token4, "real") ||
      strcmp(token5, "general")) 

    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
      std::endl << "Incorrect type of arguments on first line of reference solution file."
      << std::endl);

  // Next, strip off header lines (which start with "%")
  do {
    if(fgets(line, lineLength, handle)==0)
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
        std::endl << "Reference solution file: invalid comment line."
        << std::endl);
  } while (line[0] == '%');

  // Next get problem dimensions: M, N
  if(sscanf(line, "%d %d", &M, &N)==0)

      TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
        std::endl << "Reference solution file: cannot compute problem dimensions"
        << std::endl);

  // Compute the offset for each processor for when it should start storing values
 // int numMyPoints = map.NumMyPoints();
  int numMyPoints = mapT.getNodeNumElements();
  int offset;
  //map.Comm().ScanSum(&numMyPoints, &offset, 1); // ScanSum will compute offsets for us
  //offset -= numMyPoints; // readjust for my PE

  // Line to start reading in reference file
//  offset = map.MinMyGID();

  if (mapT.getComm()->getRank() == 0) {
    std::cout << "Reading reference solution from file \"" << filename << "\"" << std::endl;
    std::cout << "Reference solution contains " << N << " vectors, each with " << M << " rows." << std::endl;
    std::cout << std::endl;
  }

  // Now construct vector/multivector
  if (N==1)
    AT = new Tpetra_Vector(Teuchos::rcpFromRef(mapT));
  else
    AT = new Tpetra_MultiVector(Teuchos::rcpFromRef(mapT), N);

  Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST> > ApT = AT->get2dViewNonConst(); 

/*
  for (int j=0; j<N; j++) {
    double * v = Ap[j];

    // Now read in lines that we will discard
    for (int i=0; i<offset; i++)
      if(fgets(line, lineLength, handle)==0)

        TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
          std::endl << "Reference solution file: invalid line number " << i << " found while reading file lead data."
          << std::endl);
    
    // Now read in each value and store to the local portion of the the  if the row is owned.
    double V;
    for (int i=0; i<numMyPoints; i++) {
      if(fgets(line, lineLength, handle)==0) 

        TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
          std::endl << "Reference solution file: cannot read line number " << i + offset << " in file."
          << std::endl);

      if(sscanf(line, "%lg\n", &V)==0)

        TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
          std::endl << "Reference solution file: cannot parse line number " << i << " in file."
          << std::endl);

      v[i] = V;
    }
    // Now read in the rest of the lines to discard
    for (int i=0; i < M-numMyPoints-offset; i++) {
      if(fgets(line, lineLength, handle)==0)

        TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
          std::endl << "Reference solution file: invalid line number " << i + offset + numMyPoints
          << " found parsing toward end of file."
          << std::endl);
    }
  }
*/

  for (int j=0; j<N; j++) {
    Teuchos::ArrayRCP<ST> vT = ApT[j];

    // Now read in each value and store to the local portion of the array if the row is owned.
    ST V;
    for (int i=0; i<M; i++) { // i is rownumber in file, or the GID 
      if(fgets(line, lineLength, handle)==0)  // Can't read the line

        TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
          std::endl << "Reference solution file: cannot read line number " << i + offset << " in file."
          << std::endl);

      if(mapT.getLocalElement(i)>= 0){ // we own this data value
       if(sscanf(line, "%lg\n", &V)==0)

        TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
          std::endl << "Reference solution file: cannot parse line number " << i << " in file."
          << std::endl);

       vT[mapT.getLocalElement(i)] = V;
      }
    }
  }

  if (fclose(handle))

        TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
          std::endl << "Cannot close reference solution file."
          << std::endl);
  
  return(0);
}


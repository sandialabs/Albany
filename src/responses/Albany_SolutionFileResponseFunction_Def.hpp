//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_SolutionFileResponseFunction.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_GlobalLocalIndexer.hpp"

#include "Teuchos_CommHelpers.hpp"
#include "Thyra_VectorStdOps.hpp"

namespace Albany
{

template<class Norm>
SolutionFileResponseFunction<Norm>::
SolutionFileResponseFunction(const Teuchos::RCP<const Teuchos_Comm>& comm)
 : SamplingBasedScalarResponseFunction(comm)
 , solutionLoaded(false)
{
  // Nothing to be done here
}

template<class Norm>
void SolutionFileResponseFunction<Norm>::
evaluateResponse(const double /*current_time*/,
  const Teuchos::RCP<const Thyra_Vector>& x,
  const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
  const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
  const Teuchos::Array<ParamVec>& /*p*/,
  const Teuchos::RCP<Thyra_Vector>& g)
{
  int MMFileStatus = 0;

  // Read the reference solution for comparison from "reference_solution.dat"

  // Note that this is of MatrixMarket array real general format

  if (!solutionLoaded) {
    RefSoln = Thyra::createMember(x->space());
    MMFileStatus = MatrixMarketFile("reference_solution.dat",RefSoln);

    TEUCHOS_TEST_FOR_EXCEPTION(MMFileStatus!=0, std::runtime_error,
      std::endl << "MatrixMarketFile, file " __FILE__
      " line " << __LINE__ << " returned " << MMFileStatus << std::endl);

    solutionLoaded = true;
  }

  if (diff.is_null()) {
    // Build a vector to hold the difference between the actual and reference solutions
    diff = Thyra::createMember(x->space());
    diff->assign(0.0);
  }

  // Thyra vectors do not support update method with 2 vectors, so we need to use 'linear_combination'
  Teuchos::Array<ST> coeffs(2);
  coeffs[0] = 1.0; coeffs[1] = -1.0;
  Teuchos::Array<Teuchos::Ptr<const Thyra_Vector>> vecs(2);
  vecs[0] = x.ptr();
  vecs[1] = RefSoln.ptr();
  diff->linear_combination(coeffs,vecs,0.0);

  // Get the norm
  g->assign(Norm::Norm(*diff));

  if (g_.is_null())
    g_ = Thyra::createMember(g->space());

  g_->assign(*g);
}

template<class Norm>
void SolutionFileResponseFunction<Norm>::
evaluateTangent(
  const double /*alpha*/, 
  const double /*beta*/,
  const double /*omega*/,
  const double /*current_time*/,
  bool /*sum_derivs*/,
  const Teuchos::RCP<const Thyra_Vector>& /*x*/,
  const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
  const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
  const Teuchos::Array<ParamVec>& /*p*/,
  ParamVec* /*deriv_p*/,
  const Teuchos::RCP<const Thyra_MultiVector>& /*Vx*/,
  const Teuchos::RCP<const Thyra_MultiVector>& /*Vxdot*/,
  const Teuchos::RCP<const Thyra_MultiVector>& /*Vxdotdot*/,
  const Teuchos::RCP<const Thyra_MultiVector>& /*Vp*/,
  const Teuchos::RCP<Thyra_Vector>& /*g*/,
  const Teuchos::RCP<Thyra_MultiVector>& /*gx*/,
  const Teuchos::RCP<Thyra_MultiVector>& /*gp*/)
{
  // Do nothing
}

template<class Norm>
void
SolutionFileResponseFunction<Norm>::
evaluateGradient(const double /*current_time*/,
  const Teuchos::RCP<const Thyra_Vector>& x,
  const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
  const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
	const Teuchos::Array<ParamVec>& /*p*/,
	ParamVec* /*deriv_p*/,
  const Teuchos::RCP<Thyra_Vector>& g,
  const Teuchos::RCP<Thyra_MultiVector>& dg_dx,
  const Teuchos::RCP<Thyra_MultiVector>& dg_dxdot,
  const Teuchos::RCP<Thyra_MultiVector>& dg_dxdotdot,
  const Teuchos::RCP<Thyra_MultiVector>& dg_dp)
{
  int MMFileStatus = 0;
  if (!solutionLoaded) {
    RefSoln = Thyra::createMember(x->space());
    MMFileStatus = MatrixMarketFile("reference_solution.dat",RefSoln);

    TEUCHOS_TEST_FOR_EXCEPTION(MMFileStatus!=0, std::runtime_error,
      std::endl << "MatrixMarketFile, file " __FILE__
      " line " << __LINE__ << " returned " << MMFileStatus << std::endl);

    solutionLoaded = true;
  }

  if (!g.is_null()) {
    if (diff.is_null()) {
      // Build a vector to hold the difference between the actual and reference solutions
      diff = Thyra::createMember(x->space());
      diff->assign(0.0);
    }

    // Thyra vectors do not support update method with 2 vectors, so we need to use 'linear_combination'
    Teuchos::Array<ST> coeffs(2);
    coeffs[0] = 1.0; coeffs[1] = -1.0;
    Teuchos::Array<Teuchos::Ptr<const Thyra_Vector>> vecs(2);
    vecs[0] = x.ptr();
    vecs[1] = RefSoln.ptr();
    diff->linear_combination(coeffs,vecs,0.0);

    // Get the norm
    g->assign(Norm::Norm(*diff));
  }

  // Evaluate dg/dx
  if (!dg_dx.is_null()) {
    TEUCHOS_TEST_FOR_EXCEPTION(dg_dx->domain()->dim()!=1, std::logic_error, "Error! dg_dx has more than one column.\n");
    Norm::NormDerivative(*x, *RefSoln, *dg_dx->col(0));
  }

  // Evaluate dg/dxdot
  if (!dg_dxdot.is_null()) {
    dg_dxdot->assign(0.0);
  }

  // Evaluate dg/dxdotdot
  if (!dg_dxdotdot.is_null()) {
    dg_dxdotdot->assign(0.0);
  }

  // Evaluate dg/dp
  if (!dg_dp.is_null()) {
    dg_dp->assign(0.0);
  }
}

//! Evaluate distributed parameter derivative dg/dp
template<class Norm>
void SolutionFileResponseFunction<Norm>::
evaluateDistParamDeriv(
  const double /*current_time*/,
  const Teuchos::RCP<const Thyra_Vector>& /* x */,
  const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
  const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
  const Teuchos::Array<ParamVec>& /*param_array*/,
  const std::string& /*dist_param_name*/,
  const Teuchos::RCP<Thyra_MultiVector>& dg_dp)
{
  if (!dg_dp.is_null()) {
    dg_dp->assign(0.0);
  }
}

template<class Norm>
void SolutionFileResponseFunction<Norm>::
evaluate_HessVecProd_xx(
    const double current_time,
    const Teuchos::RCP<const Thyra_MultiVector>& v,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec>& param_array,
    const Teuchos::RCP<Thyra_MultiVector>& Hv_dp)
{
  if (!Hv_dp.is_null()) {
    Hv_dp->assign(0.0);
  }
}

template<class Norm>
void SolutionFileResponseFunction<Norm>::
evaluate_HessVecProd_xp(
    const double current_time,
    const Teuchos::RCP<const Thyra_MultiVector>& v,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec>& param_array,
    const std::string& dist_param_direction_name,
    const Teuchos::RCP<Thyra_MultiVector>& Hv_dp)
{
  if (!Hv_dp.is_null()) {
    Hv_dp->assign(0.0);
  }
}

template<class Norm>
void SolutionFileResponseFunction<Norm>::
evaluate_HessVecProd_px(
    const double current_time,
    const Teuchos::RCP<const Thyra_MultiVector>& v,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec>& param_array,
    const std::string& dist_param_name,
    const Teuchos::RCP<Thyra_MultiVector>& Hv_dp)
{
  if (!Hv_dp.is_null()) {
    Hv_dp->assign(0.0);
  }
}

template<class Norm>
void SolutionFileResponseFunction<Norm>::
evaluate_HessVecProd_pp(
    const double current_time,
    const Teuchos::RCP<const Thyra_MultiVector>& v,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec>& param_array,
    const std::string& dist_param_name,
    const std::string& dist_param_direction_name,
    const Teuchos::RCP<Thyra_MultiVector>& Hv_dp)
{
  if (!Hv_dp.is_null()) {
    Hv_dp->assign(0.0);
  }
}

template<class Norm>
int SolutionFileResponseFunction<Norm>::
MatrixMarketFile (const char *filename, const Teuchos::RCP<Thyra_MultiVector>& mv)
{
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
  const auto spmd_vs = getSpmdVectorSpace(mv->range());
  int offset;
  //map.Comm().ScanSum(&numMyPoints, &offset, 1); // ScanSum will compute offsets for us
  //offset -= numMyPoints; // readjust for my PE

  // Line to start reading in reference file
//  offset = map.MinMyGID();

  if (spmd_vs->getComm()->getRank() == 0) {
    std::cout << "Reading reference solution from file \"" << filename << "\"" << std::endl;
    std::cout << "Reference solution contains " << N << " vectors, each with " << M << " rows." << std::endl;
    std::cout << std::endl;
  }

  // Now construct vector/multivector
  TEUCHOS_TEST_FOR_EXCEPTION (N!=static_cast<int>(mv->domain()->dim()), std::runtime_error,
                              "Error! Input file is storing a Thyra MultiVector with a number of vectors "
                              "different from the what was expected.\n");

  auto vals = getNonconstLocalData(mv);
  auto indexer = createGlobalLocalIndexer(spmd_vs);
  for (int j=0; j<N; j++) {
    Teuchos::ArrayRCP<ST> v = vals[j];

    // Now read in each value and store to the local portion of the array if the row is owned.
    ST V;
    for (int i=0; i<M; i++) { // i is rownumber in file, or the GID 
      if(fgets(line, lineLength, handle)==0)  // Can't read the line

        TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
          std::endl << "Reference solution file: cannot read line number " << i + offset << " in file."
          << std::endl);

      const LO lid = indexer->getLocalElement(i);
      if(lid>= 0) { // we own this data value
        if(sscanf(line, "%lg\n", &V)==0) {
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
                                     "Reference solution file: cannot parse line number " << i << " in file.\n");
        }
        v[lid] = V;
      }
    }
  }

  if (fclose(handle)) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
                               "Cannot close reference solution file.\n");
  }

  return 0;
}

template<class Norm>
void SolutionFileResponseFunction<Norm>::
printResponse(Teuchos::RCP<Teuchos::FancyOStream> out)
{
  if (g_.is_null()) {
    *out << " the response has not been evaluated yet!";
    return;
  }

  std::size_t precision = 8;
  std::size_t value_width = precision + 4;
  int gsize = g_->space()->dim();

  for (int j = 0; j < gsize; j++) {
    *out << std::setw(value_width) << Thyra::get_ele(*g_,j);
    if (j < gsize-1)
      *out << ", ";
  }
}

} // namespace Albany

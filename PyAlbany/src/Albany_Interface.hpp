//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <iostream>
#include <string>
#include <mpi.h>
#include <memory>

#include "Albany_Memory.hpp"
#include "Albany_SolverFactory.hpp"

#include "Teuchos_StackedTimer.hpp"

#include "Kokkos_Core.hpp"

#include "Albany_PyAlbanyTypes.hpp"
#include "Albany_PyUtils.hpp"

#include "Albany_Utils.hpp"
#include "Albany_Gather.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_YamlParameterListHelpers.hpp"

#include "Albany_CumulativeScalarResponseFunction.hpp"

#include "BelosConfigDefs.hpp"
#include "BelosOutputManager.hpp"

#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_StandardCatchMacros.hpp"
#include <Teuchos_StackedTimer.hpp>

#include "BelosTpetraAdapter.hpp"
#include "BelosTpetraOperator.hpp"
#include "Tpetra_Core.hpp"
#include "Tpetra_CrsMatrix.hpp"
#include <MatrixMarket_Tpetra.hpp>
#include "BelosOrthoManagerFactory.hpp"
#include "BelosOrthoManager.hpp"

using Teuchos::RCP;
using Teuchos::rcp;
using Teuchos::Array;

namespace PyAlbany
{
    /**
   * \brief scatterMVector function
   * 
   * This function is used to scatter a Tpetra::Multivector on different MPI processes.
   * 
   * \param inVector [in] Multivector which has to be scattered.
   * 
   * \param distributedMap [in] Map used to define how the entries have to be scattered.
   * 
   * The function returns an RCP to the scattered multivector.
   */
    Teuchos::RCP<PyTrilinosMultiVector> scatterMVector(Teuchos::RCP<PyTrilinosMultiVector> inVector, Teuchos::RCP<PyTrilinosMap> distributedMap);

    /**
   * \brief gatherMVector function
   * 
   * This function is used to gather a Tpetra::Multivector on rank 0.
   * 
   * \param inVector [in] Multivector which has to be gathered.
   * 
   * \param distributedMap [in] Map used to define how the entries have to be gathered.
   * 
   * The function returns an RCP to the gathered multivector.
   */
    Teuchos::RCP<PyTrilinosMultiVector> gatherMVector(Teuchos::RCP<PyTrilinosMultiVector> inVector, Teuchos::RCP<PyTrilinosMap> distributedMap);

    /**
   * \brief PyParallelEnv class
   * 
   * This class is used to communicate from Python to c++ the parallel environment information such as
   * a Teuchos communicator and the Kokkos arguments.
   * 
   * The constructor of this object calls Kokkos::initialize and its destructors calls finalize_all.
   * 
   */
    class PyParallelEnv
    {
    public:
        Teuchos::RCP<Teuchos::Comm<int>> comm;
        const int num_threads, num_numa, device_id;

        PyParallelEnv(Teuchos::RCP<Teuchos::Comm<int>> _comm, int _num_threads = -1, int _num_numa = -1, int _device_id = -1);
        ~PyParallelEnv()
        {
            Kokkos::finalize_all();
            std::cout << "~PyParallelEnv()\n";
        }
    };
   
    /**
    * \brief orthogTpMVecs function
    *
    * The function orthogonalizes the input MultiVector object.
    *
    */ 
    void orthogTpMVecs(Teuchos::RCP<PyAlbany::PyTrilinosMultiVector> inputVecs, int blkSize);
   
    /**
   * \brief getParameterList function
   * 
   * The function returns an RCP to the parameter list.
   */
    Teuchos::RCP<Teuchos::ParameterList> getParameterList(std::string filename, Teuchos::RCP<PyAlbany::PyParallelEnv> pyParallelEnv);

    /**
   * \brief getRankZeroMap function
   * 
   * The function returns an RCP to a map where all IDs are stored on Rank 0.
   */
    Teuchos::RCP<PyAlbany::PyTrilinosMap> getRankZeroMap(Teuchos::RCP<PyAlbany::PyTrilinosMap> distributedMap);

    /**
   * \brief PyProblem class
   * 
   * This class is used to drives Albany from Python.
   * 
   * This class is used to communicate directions, solutions, sensitivites, and reduced Hessian-vector products
   * between Python and Albany.
   */
    class PyProblem
    {
    private:
#ifndef SWIG
        bool forwardHasBeenSolved;
        bool inverseHasBeenSolved;
        Teuchos::RCP<PyParallelEnv> pyParallelEnv;

        Teuchos::RCP<Teuchos::StackedTimer> stackedTimer;
        Teuchos::RCP<const Teuchos_Comm> comm;
        Teuchos::RCP<Albany::SolverFactory> slvrfctry;
        Teuchos::RCP<Albany::Application> albanyApp;
        Teuchos::RCP<Albany::ModelEvaluator> albanyModel;
        Teuchos::RCP<Thyra::ResponseOnlyModelEvaluatorBase<double>> solver;

        Teuchos::Array<Teuchos::RCP<Thyra_Vector>> thyraResponses;
        Teuchos::Array<Teuchos::Array<Teuchos::RCP<Thyra_MultiVector>>> thyraSensitivities;
        Teuchos::Array<Teuchos::Array<Teuchos::RCP<Thyra_MultiVector>>> thyraReducedHessian;
        Teuchos::Array<Teuchos::RCP<Thyra_MultiVector>> thyraDirections;

        Teuchos::Array<Teuchos::RCP<Thyra_Vector>> thyraParameter;
#endif
    public:
        PyProblem(std::string filename, Teuchos::RCP<PyParallelEnv> _pyParallelEnv);

        PyProblem(Teuchos::RCP<Teuchos::ParameterList> params, Teuchos::RCP<PyParallelEnv> _pyParallelEnv);

        ~PyProblem()
        {
            std::cout << "~PyProblem()\n";
        }

        /**
         * \brief performSolve member function
         * 
         * This function is used to call the function Piro::PerformSolve and solve the
         * defined problem.
	     * \return false if solve converged, true otherwise.
         */
        bool performSolve();

        Teuchos::RCP<Teuchos::StackedTimer> getStackedTimer() { return stackedTimer; }

        /**
         * \brief performAnalysis member function
         * 
         * This function is used to call the function Piro::PerformAnalysis.
         * \return false if solve converged, true otherwise.
         */
        bool performAnalysis();

        /**
         * \brief getResponseMap member function
         * 
         * This function is used to communicate the map of one of the responses to Python.
         * 
         * \param g_index [in] Index of the response for which the map is requested.
         * 
         * The function returns an RCP to a map supported by PyTrilinos.
         * 
         * This function should ne be called before calling performSolve().
         */
        Teuchos::RCP<const PyTrilinosMap> getResponseMap(const int g_index);

	/**
         * \brief getStateMap member function
         * 
         * This function is used to communicate the map of the state to Python.
         * 
         * The function returns an RCP to a map supported by PyTrilinos.
         * 
         * This function should ne be called before calling performSolve().
         */
	Teuchos::RCP<const PyTrilinosMap> getStateMap();
        
        /**
         * \brief getParameterMap member function
         * 
         * This function is used to communicate the map of one of the parameters to Python.
         * 
         * \param p_index [in] Index of the parameter for which the map is requested.
         * 
         * The function returns an RCP to a map supported by PyTrilinos.
         */
        Teuchos::RCP<const PyTrilinosMap> getParameterMap(const int p_index);

        /**
         * \brief setDirections member function
         * 
         * This function is used to communicate the parameter directions from Python to Albany.
         * 
         * \param p_index [in] Index of the parameter for which the direction has to be set.
         * 
         * \param direction [in] A distributed multivector which stores the directions of the parameters.
         */
        void setDirections(const int p_index, Teuchos::RCP<PyTrilinosMultiVector> direction);

        /**
         * \brief setParameter member function
         * 
         * This function is used to communicate the parameter values from Python to Albany.
         * 
         * \param p_index [in] Index of the parameter for which the value has to be set.
         * 
         * \param p [in] A distributed vector which stores the values of the parameters.
         */
        void setParameter(const int p_index, Teuchos::RCP<PyTrilinosVector> p);

        /**
         * \brief getParameter member function
         * 
         * This function is used to communicate the parameter values from Albany to Python.
         * 
         * \param p_index [in] Index of the parameter for which the value has to be gotten.
         */
        Teuchos::RCP<PyTrilinosVector> getParameter(const int p_index);

        /**
         * \brief getResponse member function
         * 
         * This function is used to communicate the response from Albany to Python.
         * 
         * \param g_index [in] Index of the requested response.
         * 
         * \param response [out] A distributed vector which stores the requested response.
         *
         * This function should ne be called before calling performSolve().
         */
        Teuchos::RCP<PyTrilinosVector> getResponse(const int g_index);

	/**
         * \brief getState member function
         * 
         * This function is used to communicate the state from Albany to Python.
         * 
         * \param response [out] A distributed vector which stores the state.
         *
         * This function should ne be called before calling performSolve().
         */	
	Teuchos::RCP<PyTrilinosVector> getState();

        /**
         * \brief getSensitivity member function
         * 
         * This function is used to communicate the sensitivity of a given response
         * to a given parameter from Albany to Python.
         * 
         * \param g_index [in] Index of the response.
         * 
         * \param p_index [in] Index of the parameter.
         * 
         * \param sensitivity [out] A distributed multivector which stores the requested sensitivities.
         *
         * This function should ne be called before calling performSolve().
         */
        Teuchos::RCP<PyTrilinosMultiVector> getSensitivity(const int g_index, const int p_index);

        /**
         * \brief getReducedHessian member function
         * 
         * This function is used to communicate the reduced Hessian-vector products of a given response
         * w.r.t a given parameter from Albany to Python.
         * 
         * \param g_index [in] Index of the response.
         * 
         * \param p_index [in] Index of the parameter.
         * 
         * \param hessian [out] A distributed multivector which stores the requested reduced Hessian-vector products.
         *
         * This function should ne be called before calling performSolve().
         */
        Teuchos::RCP<PyTrilinosMultiVector> getReducedHessian(const int g_index, const int p_index);

        /**
         * \brief reportTimers member function
         * 
         * This function reports the Albany timers.
         */
        void reportTimers();

        double getCumulativeResponseContribution( int i, int j)
        {
            Teuchos::RCP<Albany::CumulativeScalarResponseFunction>  csrf = Teuchos::rcp_dynamic_cast<Albany::CumulativeScalarResponseFunction>(albanyApp->getResponse(i), false);
            if (csrf == Teuchos::null) {
                std::cout << "Warning: getCumulativeResponseContribution() response " << i << " is not a CumulativeScalarResponseFunction." << std::endl;
                return 0.;
            }
            else
                return csrf->getContribution(j);
        }

        void updateCumulativeResponseContributionWeigth( int i, int j, double weigth)
        {
            Teuchos::RCP<Albany::CumulativeScalarResponseFunction>  csrf = Teuchos::rcp_dynamic_cast<Albany::CumulativeScalarResponseFunction>(albanyApp->getResponse(i), false);
            if (csrf == Teuchos::null) {
                std::cout << "Warning: updateCumulativeResponseContributionWeigth() response " << i << " is not a CumulativeScalarResponseFunction." << std::endl;
            }
            else
                csrf->updateWeight(j, weigth);
        }
    };

} // namespace PyAlbany

Teuchos::RCP<PyAlbany::PyTrilinosMap> PyAlbany::getRankZeroMap(Teuchos::RCP<PyAlbany::PyTrilinosMap> distributedMap)
{
    int numGlobalElements = distributedMap->getGlobalNumElements();
    int numLocalElements = distributedMap->getNodeNumElements();

    // GO and PyAlbany::PyTrilinosMap::global_ordinal_type can be different
    auto nodes_gids_view = distributedMap->getMyGlobalIndices();
    Teuchos::Array<GO> nodes_gids(numLocalElements);
    for (int i=0; i<numLocalElements; ++i)
        nodes_gids[i] = nodes_gids_view(i);
    Teuchos::Array<GO> all_nodes_gids;
    Albany::gatherV(distributedMap->getComm(),nodes_gids(),all_nodes_gids,0);
    std::sort(all_nodes_gids.begin(),all_nodes_gids.end());
    auto it = std::unique(all_nodes_gids.begin(),all_nodes_gids.end());
    all_nodes_gids.erase(it,all_nodes_gids.end());

    Teuchos::Array<PyAlbany::PyTrilinosMap::global_ordinal_type> all_nodes_py_gids(all_nodes_gids.size());
    for (int i=0; i<all_nodes_gids.size(); ++i)
        all_nodes_py_gids[i] = all_nodes_gids[i];

    return rcp(new PyAlbany::PyTrilinosMap(numGlobalElements, all_nodes_py_gids, distributedMap->getIndexBase(), distributedMap->getComm()));
}

Teuchos::RCP<PyAlbany::PyTrilinosMultiVector> PyAlbany::scatterMVector(Teuchos::RCP<PyAlbany::PyTrilinosMultiVector> inVector, Teuchos::RCP<PyAlbany::PyTrilinosMap> distributedMap)
{
    int myRank = distributedMap->getComm()->getRank();
    auto rankZeroMap = getRankZeroMap(distributedMap);
    Teuchos::RCP<PyAlbany::PyTrilinosExport> exportZero = rcp(new PyAlbany::PyTrilinosExport(distributedMap, rankZeroMap));
    Teuchos::RCP<PyAlbany::PyTrilinosMultiVector> outVector = rcp(new PyAlbany::PyTrilinosMultiVector(distributedMap, inVector->getNumVectors()));
    outVector->doImport(*inVector, *exportZero, Tpetra::INSERT);
    return outVector;
}

Teuchos::RCP<PyAlbany::PyTrilinosMultiVector> PyAlbany::gatherMVector(Teuchos::RCP<PyAlbany::PyTrilinosMultiVector> inVector, Teuchos::RCP<PyAlbany::PyTrilinosMap> distributedMap)
{
    int myRank = distributedMap->getComm()->getRank();
    auto rankZeroMap = getRankZeroMap(distributedMap);
    Teuchos::RCP<PyAlbany::PyTrilinosExport> exportZero = rcp(new PyAlbany::PyTrilinosExport(distributedMap, rankZeroMap));
    Teuchos::RCP<PyAlbany::PyTrilinosMultiVector> outVector = rcp(new PyAlbany::PyTrilinosMultiVector(rankZeroMap, inVector->getNumVectors()));
    outVector->doExport(*inVector, *exportZero, Tpetra::ADD);
    return outVector;
}

void PyAlbany::orthogTpMVecs(Teuchos::RCP<PyAlbany::PyTrilinosMultiVector> inputVecs, int blkSize)
{
  typedef double                            ScalarType;
  typedef int                               OT;
  typedef typename Teuchos::SerialDenseMatrix<OT,ScalarType> MAT;
  typedef Tpetra::MultiVector<ScalarType>   MV;
  typedef Kokkos::DefaultExecutionSpace     EXSP;
  typedef Tpetra::Operator<ScalarType>             OP;
  typedef Belos::OperatorTraits<ScalarType,MV,OP> OPT;
  int numVecs = inputVecs->getNumVectors();
  int numRows = inputVecs->getGlobalLength();
  std::string orthogType("ICGS");

  RCP<MAT> B = rcp(new MAT(blkSize, blkSize)); //Matrix for coeffs of X
  Array<RCP<MAT>> C; 

  Belos::OrthoManagerFactory<ScalarType, MV, OP> factory;
  Teuchos::RCP<Teuchos::ParameterList> paramsOrtho;   // can be null

  //Default OutputManager is std::cout.
  Teuchos::RCP<Belos::OutputManager<ScalarType> > myOutputMgr = Teuchos::rcp( new Belos::OutputManager<ScalarType>() );
  const Teuchos::RCP<Belos::OrthoManager<ScalarType,MV>> orthoMgr = factory.makeOrthoManager (orthogType, Teuchos::null, myOutputMgr, "Tpetra OrthoMgr", paramsOrtho); 
  
  int numLoops = numVecs/blkSize;
  int remainder = numVecs % blkSize;

  RCP<MV> vecBlock = inputVecs->subViewNonConst(Teuchos::Range1D(0,blkSize-1));
  orthoMgr->normalize(*vecBlock, B);
  std::vector<RCP<const MV>> pastVecArray;
  pastVecArray.push_back(vecBlock);
  Teuchos::ArrayView<RCP<const MV>> pastVecArrayView; 

  for(int k=1; k<numLoops; k++){
    pastVecArrayView = arrayViewFromVector(pastVecArray);
    vecBlock = inputVecs->subViewNonConst(Teuchos::Range1D(k*blkSize,k*blkSize + blkSize - 1));
    C.append(rcp(new MAT(blkSize, blkSize)));
    int rank = orthoMgr->projectAndNormalize(*vecBlock, C, B, pastVecArrayView);
    pastVecArray.push_back(vecBlock);
  }
  if( remainder > 0){
    pastVecArrayView = arrayViewFromVector(pastVecArray);
    vecBlock = inputVecs->subViewNonConst(Teuchos::Range1D(numVecs-remainder, numVecs-1));
    B = rcp(new MAT(remainder, remainder));
    C.append(rcp(new MAT(remainder, remainder)));
    int rank = orthoMgr->projectAndNormalize(*vecBlock, C, B, pastVecArrayView);
  }
}

Teuchos::RCP<Teuchos::ParameterList> PyAlbany::getParameterList(std::string inputFile, Teuchos::RCP<PyAlbany::PyParallelEnv> pyParallelEnv)
{
    Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::createParameterList("Albany Parameters");

    std::string const input_extension = Albany::getFileExtension(inputFile);

    if (input_extension == "yaml" || input_extension == "yml")
    {
        Teuchos::updateParametersFromYamlFileAndBroadcast(
            inputFile, params.ptr(), *(pyParallelEnv->comm));
    }
    else
    {
        Teuchos::updateParametersFromXmlFileAndBroadcast(
            inputFile, params.ptr(), *(pyParallelEnv->comm));
    }

    return params;
}

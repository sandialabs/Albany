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
        std::string filename;
        bool hasBeenSolved;
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
#endif
    public:
        PyProblem(std::string _filename, Teuchos::RCP<PyParallelEnv> _pyParallelEnv);
        ~PyProblem()
        {
            std::cout << "~PyProblem()\n";
        }

        /**
         * \brief performSolve member function
         * 
         * This function is used to call the function Piro::PerformSolve and solve the
         * defined problem.
         */
        void performSolve();

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
         * This function is used to communicate the parameters directions from Python to Albany.
         * 
         * \param p_index [in] Index of the parameter for which the direction has to be set.
         * 
         * \param direction [in] A distributed multivector which stores the directions of the parameters.
         */
        void setDirections(const int p_index, Teuchos::RCP<PyTrilinosMultiVector> direction);

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
    };

} // namespace PyAlbany

Teuchos::RCP<PyAlbany::PyTrilinosMultiVector> PyAlbany::scatterMVector(Teuchos::RCP<PyAlbany::PyTrilinosMultiVector> inVector, Teuchos::RCP<PyAlbany::PyTrilinosMap> distributedMap)
{
    int myRank = distributedMap->getComm()->getRank();
    int numGlobalElements = distributedMap->getGlobalNumElements();
    int numLocalElements = (myRank == 0) ? numGlobalElements : 0;

    Teuchos::RCP<PyAlbany::PyTrilinosMap> rankZeroMap = rcp(new PyAlbany::PyTrilinosMap(numGlobalElements, numLocalElements, distributedMap->getIndexBase(), distributedMap->getComm()));
    Teuchos::RCP<PyAlbany::PyTrilinosExport> exportZero = rcp(new PyAlbany::PyTrilinosExport(distributedMap, rankZeroMap));
    Teuchos::RCP<PyAlbany::PyTrilinosMultiVector> outVector = rcp(new PyAlbany::PyTrilinosMultiVector(distributedMap, inVector->getNumVectors()));
    outVector->doImport(*inVector, *exportZero, Tpetra::INSERT);
    return outVector;
}

Teuchos::RCP<PyAlbany::PyTrilinosMultiVector> PyAlbany::gatherMVector(Teuchos::RCP<PyAlbany::PyTrilinosMultiVector> inVector, Teuchos::RCP<PyAlbany::PyTrilinosMap> distributedMap)
{
    int myRank = distributedMap->getComm()->getRank();
    int numGlobalElements = distributedMap->getGlobalNumElements();
    int numLocalElements = (myRank == 0) ? numGlobalElements : 0;

    Teuchos::RCP<PyAlbany::PyTrilinosMap> rankZeroMap = rcp(new PyAlbany::PyTrilinosMap(numGlobalElements, numLocalElements, distributedMap->getIndexBase(), distributedMap->getComm()));
    Teuchos::RCP<PyAlbany::PyTrilinosExport> exportZero = rcp(new PyAlbany::PyTrilinosExport(distributedMap, rankZeroMap));
    Teuchos::RCP<PyAlbany::PyTrilinosMultiVector> outVector = rcp(new PyAlbany::PyTrilinosMultiVector(rankZeroMap, inVector->getNumVectors()));
    outVector->doExport(*inVector, *exportZero, Tpetra::ADD);
    return outVector;
}
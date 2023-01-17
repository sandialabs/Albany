//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_PYINTERFACE_H
#define ALBANY_PYINTERFACE_H

#include <iostream>
#include <string>
#include <mpi.h>
#include <memory>

#include "Albany_Memory.hpp"
#include "Albany_SolverFactory.hpp"

#include "Teuchos_StackedTimer.hpp"

#include "Kokkos_Core.hpp"

#include "Albany_PyUtils.hpp"

#include "Albany_Utils.hpp"
#include "Albany_Gather.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_YamlParameterListHelpers.hpp"

#include "Albany_CumulativeScalarResponseFunction.hpp"
#include "Albany_ScalarResponsePower.hpp"

#include "BelosTpetraAdapter.hpp"
#include "BelosTpetraOperator.hpp"
#include "BelosOrthoManagerFactory.hpp"
#include "BelosOrthoManager.hpp"

#include <Teuchos_TwoDArray.hpp>

using Teuchos_Comm_PyAlbany = Teuchos::MpiComm<int>;
using RCP_Teuchos_Comm_PyAlbany = Teuchos::RCP<const Teuchos_Comm_PyAlbany >;

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
    Teuchos::RCP<Tpetra_MultiVector> scatterMVector(Teuchos::RCP<Tpetra_MultiVector> inVector, Teuchos::RCP<const Tpetra_Map> distributedMap);

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
    Teuchos::RCP<Tpetra_MultiVector> gatherMVector(Teuchos::RCP<Tpetra_MultiVector> inVector, Teuchos::RCP<const Tpetra_Map> distributedMap);

    /**
   * \brief PyParallelEnv class
   * 
   * This class is used to communicate from Python to c++ the parallel environment information such as
   * a Teuchos communicator and the Kokkos arguments.
   * 
   * The constructor of this object calls Kokkos::initialize and its destructors calls finalize.
    
   */
    class PyParallelEnv
    {
    private:
        RCP_Teuchos_Comm_PyAlbany comm;
        const int num_threads, num_devices, device_id;
        int rank;

    public:
        PyParallelEnv(RCP_Teuchos_Comm_PyAlbany _comm, int _num_threads = -1, int _num_devices = -1, int _device_id = -1);
        ~PyParallelEnv()
        {
            if (rank == 0)
                std::cout << "~PyParallelEnv()\n";
        }
        int getNumThreads() const { return num_threads; }
        int getNumDevices() const { return num_devices; }
        int getDeviceID() const { return device_id; }
        RCP_Teuchos_Comm_PyAlbany getComm() const { return comm; }
        void setComm(RCP_Teuchos_Comm_PyAlbany _comm) {comm = _comm;}
    };

    /**
    * \brief orthogTpMVecs function
    *
    * The function orthogonalizes the input MultiVector object.
    *
    */ 
    void orthogTpMVecs(Teuchos::RCP<Tpetra_MultiVector> inputVecs, int blkSize);

    /**
   * \brief getParameterList function
   * 
   * The function returns an RCP to the parameter list.
   */
    Teuchos::RCP<Teuchos::ParameterList> getParameterList(std::string filename, Teuchos::RCP<PyParallelEnv> pyParallelEnv);

    /**
   * \brief writeParameterList function
   * 
   * The function returns an RCP to the parameter list.
   */
    void writeParameterList(std::string filename, Teuchos::RCP<Teuchos::ParameterList> parameterList);

    /**
   * \brief finalizeKokkos function
   * 
   * The function finalizes Kokkos if it has been previously initialized.
   */
    void finalizeKokkos();

    /**
   * \brief getRankZeroMap function
   * 
   * The function returns an RCP to a map where all IDs are stored on Rank 0.
   */
    Teuchos::RCP<const Tpetra_Map> getRankZeroMap(Teuchos::RCP<const Tpetra_Map> distributedMap);

    /**
   * \brief PyProblem class
   * 
   * This class is used to drives Albany from Python.
   * 
   * This class is used to communicate directions, solutions, sensitivities, and reduced Hessian-vector products
   * between Python and Albany.
   */
    class PyProblem
    {
    private:
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
         * The function returns an RCP to a map.
         * 
         * This function should ne be called before calling performSolve().
         */
        Teuchos::RCP<const Tpetra_Map> getResponseMap(const int g_index);

	    /**
         * \brief getStateMap member function
         * 
         * This function is used to communicate the map of the state to Python.
         * 
         * The function returns an RCP to a map.
         * 
         * This function should ne be called before calling performSolve().
         */
	    Teuchos::RCP<const Tpetra_Map> getStateMap();
        
        /**
         * \brief getParameterMap member function
         * 
         * This function is used to communicate the map of one of the parameters to Python.
         * 
         * \param p_index [in] Index of the parameter for which the map is requested.
         * 
         * The function returns an RCP to a map.
         */
        Teuchos::RCP<const Tpetra_Map> getParameterMap(const int p_index);

        /**
         * \brief setDirections member function
         * 
         * This function is used to communicate the parameter directions from Python to Albany.
         * 
         * \param p_index [in] Index of the parameter for which the direction has to be set.
         * 
         * \param direction [in] A distributed multivector which stores the directions of the parameters.
         */
        void setDirections(const int p_index, Teuchos::RCP<Tpetra_MultiVector> direction);

        /**
         * \brief setParameter member function
         * 
         * This function is used to communicate the parameter values from Python to Albany.
         * 
         * \param p_index [in] Index of the parameter for which the value has to be set.
         * 
         * \param p [in] A distributed vector which stores the values of the parameters.
         */
        void setParameter(const int p_index, Teuchos::RCP<Tpetra_Vector> p);

        /**
         * \brief getParameter member function
         * 
         * This function is used to communicate the parameter values from Albany to Python.
         * 
         * \param p_index [in] Index of the parameter for which the value has to be gotten.
         */
        Teuchos::RCP<Tpetra_Vector> getParameter(const int p_index);

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
        Teuchos::RCP<Tpetra_Vector> getResponse(const int g_index);

	    /**
         * \brief getState member function
         * 
         * This function is used to communicate the state from Albany to Python.
         * 
         * \param response [out] A distributed vector which stores the state.
         *
         * This function should ne be called before calling performSolve().
         */	
	    Teuchos::RCP<Tpetra_Vector> getState();

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
        Teuchos::RCP<Tpetra_MultiVector> getSensitivity(const int g_index, const int p_index);

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
        Teuchos::RCP<Tpetra_MultiVector> getReducedHessian(const int g_index, const int p_index);

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

        void updateCumulativeResponseContributionTargetAndExponent( int i, int j, double target, double exponent)
        {
            Teuchos::RCP<Albany::CumulativeScalarResponseFunction>  csrf = Teuchos::rcp_dynamic_cast<Albany::CumulativeScalarResponseFunction>(albanyApp->getResponse(i), false);
            if (csrf == Teuchos::null) {
                std::cout << "Warning: updateCumulativeResponseContributionTarget() response " << i << " is not a CumulativeScalarResponseFunction." << std::endl;
            }
            else {
                Teuchos::RCP<Albany::ScalarResponsePower>  power = Teuchos::rcp_dynamic_cast<Albany::ScalarResponsePower>(csrf->getResponse(j), false);
                if (power == Teuchos::null) {
                    std::cout << "Warning: updateCumulativeResponseContributionTarget() response " << j << " is not a ScalarResponsePower." << std::endl;
                }
                else {
                    power->updateTarget(target);
                    power->updateExponent(exponent);
                }            
            }
        }

        void getCovarianceMatrix(double* C, int n, int m)
        {
            auto responseParams = albanyApp->getAppPL()->sublist("Problem").sublist("Responses").sublist("Response 0").sublist("Response 0");
            int total_dimension = n;

            Teuchos::TwoDArray<double> C_data(total_dimension, total_dimension, 0);
            if (responseParams.isParameter("Covariance Matrix"))
                C_data = responseParams.get<Teuchos::TwoDArray<double>>("Covariance Matrix");
            else
                for (int i=0; i<total_dimension; i++)
                    C_data(i,i) = 1.;
            for (int i=0; i<n; i++)
                for (int j=0; j<m; j++)
                    C[i * n + j] = C_data(i,j);
        }

        void setCovarianceMatrix(double* C, int n, int m)
        {
            auto responseParams = albanyApp->getAppPL()->sublist("Problem").sublist("Responses").sublist("Response 0").sublist("Response 0");
            int total_dimension = n;

            Teuchos::TwoDArray<double> C_data(total_dimension, total_dimension, 0);
            for (int i=0; i<n; i++)
                for (int j=0; j<m; j++)
                    C_data(i,j) = C[i * n + j];
            responseParams.set<Teuchos::TwoDArray<double>>("Covariance Matrix", C_data);                    
        }
    };
} // namespace PyAlbany

#endif

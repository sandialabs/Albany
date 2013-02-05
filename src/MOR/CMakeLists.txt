##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//
# MOR

SET(SOURCES 
  Albany_MORFacade.cpp
  Albany_BasisOps.cpp
  Albany_EpetraUtils.cpp
  Albany_EpetraMVDenseMatrixView.cpp
  Albany_EpetraSamplingOperator.cpp
  Albany_GaussNewtonOperatorFactory.cpp
  Albany_PetrovGalerkinOperatorFactory.cpp
  Albany_ReducedJacobianFactory.cpp
  Albany_ReducedSpace.cpp
  Albany_LinearReducedSpaceFactory.cpp
  Albany_ReducedBasisFactory.cpp
  Albany_ReducedBasisRepository.cpp
  Albany_StkBasisProvider.cpp
  Albany_SampleDofListFactory.cpp
  Albany_DefaultSampleDofListProviders.cpp
  Albany_DiscretizationDofListProvider.cpp
  Albany_ReducedOrderModelEvaluator.cpp
  Albany_ReducedOrderModelFactory.cpp
  Albany_MORObserverFactory.cpp
  Albany_FullStateReconstructor.cpp
  Albany_RythmosFullStateReconstructor.cpp
  Albany_RythmosStepperFullStateWrapper.cpp
  Albany_ProjectionError.cpp
  Albany_ProjectionErrorObserver.cpp
  Albany_RythmosProjectionErrorObserver.cpp
  Albany_SnapshotCollection.cpp
  Albany_SnapshotCollectionObserver.cpp
  Albany_RythmosSnapshotCollectionObserver.cpp
  Albany_BasisInputFile.cpp
  Albany_MultiVectorInputFile.cpp
  Albany_MultiVectorInputFileFactory.cpp
  Albany_MatrixMarketMVInputFile.cpp
  Albany_Hdf5MVInputFile.cpp
  Albany_MultiVectorOutputFileFactory.cpp
  Albany_MatrixMarketMVOutputFile.cpp
  Albany_Hdf5MVOutputFile.cpp
)
SET(HEADERS
  Albany_MORFacade.hpp
  Albany_MORUtils.hpp
  Albany_BasisOps.hpp
  Albany_EpetraUtils.hpp
  Albany_EpetraMVDenseMatrixView.hpp
  Albany_EpetraSamplingOperator.hpp
  Albany_ReducedOperatorFactory.hpp
  Albany_GaussNewtonOperatorFactory.hpp
  Albany_PetrovGalerkinOperatorFactory.hpp
  Albany_ReducedJacobianFactory.hpp
  Albany_ReducedSpace.hpp
  Albany_LinearReducedSpaceFactory.hpp
  Albany_ReducedBasisFactory.hpp
  Albany_ReducedBasisRepository.hpp
  Albany_StkBasisProvider.hpp
  Albany_SampleDofListFactory.hpp
  Albany_DefaultSampleDofListProviders.hpp
  Albany_DiscretizationDofListProvider.hpp
  Albany_ReducedOrderModelEvaluator.hpp
  Albany_ReducedOrderModelFactory.hpp
  Albany_MORObserverFactory.hpp
  Albany_FullStateReconstructor.hpp
  Albany_RythmosFullStateReconstructor.hpp
  Albany_RythmosStepperFullStateWrapper.hpp
  Albany_ProjectionError.hpp
  Albany_ProjectionErrorObserver.hpp
  Albany_RythmosProjectionErrorObserver.hpp
  Albany_SnapshotCollection.hpp
  Albany_SnapshotCollectionObserver.hpp
  Albany_RythmosSnapshotCollectionObserver.hpp
  Albany_RythmosUtils.hpp
  Albany_BasisInputFile.hpp
  Albany_MultiVectorInputFile.hpp
  Albany_MultiVectorInputFileFactory.hpp
  Albany_MatrixMarketMVInputFile.hpp
  Albany_Hdf5MVInputFile.hpp
  Albany_MultiVectorOutputFileFactory.hpp
  Albany_MultiVectorOutputFile.hpp
  Albany_MatrixMarketMVOutputFile.hpp
  Albany_Hdf5MVOutputFile.hpp
)


include_directories ($/Trilinos_INCLUDE_DIRS}  ${Trilinos_TPL_INCLUDE_DIRS}
                     ${Albany_SOURCE_DIR}/src 
                    )

add_library(MOR ${Albany_LIBRARY_TYPE} ${SOURCES} ${HEADERS})

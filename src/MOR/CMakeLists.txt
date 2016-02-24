##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//
# MOR

SET(MOR_SOURCES
  MOR_BasisOps.cpp
  MOR_EpetraUtils.cpp
  MOR_EpetraLocalMapMVMatrixMarketUtils.cpp
  MOR_EpetraMVDenseMatrixView.cpp
  MOR_EpetraSamplingOperator.cpp
  MOR_GaussNewtonOperatorFactory.cpp
  MOR_PetrovGalerkinOperatorFactory.cpp
  MOR_ReducedJacobianFactory.cpp
  MOR_ReducedSpace.cpp
  MOR_ReducedSpaceFactory.cpp
  MOR_ReducedBasisFactory.cpp
  MOR_DefaultReducedBasisFactory.cpp
  MOR_ReducedBasisRepository.cpp
  MOR_SampleDofListFactory.cpp
  MOR_DefaultSampleDofListProviders.cpp
  MOR_ReducedOrderModelEvaluator.cpp
  MOR_ReducedOrderModelFactory.cpp
  MOR_ObserverFactory.cpp
  MOR_NOXEpetraCompositeObserver.cpp
  MOR_FullStateReconstructor.cpp
  MOR_RythmosFullStateReconstructor.cpp
  MOR_RythmosStepperFullStateWrapper.cpp
  MOR_ProjectionError.cpp
  MOR_ProjectionErrorObserver.cpp
  MOR_RythmosProjectionErrorObserver.cpp
  MOR_GeneralizedCoordinatesOutput.cpp
  MOR_GeneralizedCoordinatesNOXObserver.cpp
  MOR_GeneralizedCoordinatesRythmosObserver.cpp
  MOR_SnapshotCollection.cpp
  MOR_SnapshotCollectionObserver.cpp
  MOR_RythmosSnapshotCollectionObserver.cpp
  MOR_EpetraMVSource.cpp
  MOR_ConcatenatedEpetraMVSource.cpp
  MOR_InputFileEpetraMVSource.cpp
  MOR_FileReducedBasisSource.cpp
  MOR_IdentityBasisSource.cpp
  MOR_MultiVectorInputFileFactory.cpp
  MOR_MatrixMarketMVInputFile.cpp
  MOR_Hdf5MVInputFile.cpp
  MOR_MultiVectorOutputFileFactory.cpp
  MOR_MatrixMarketMVOutputFile.cpp
  MOR_Hdf5MVOutputFile.cpp
  MOR_WindowedAtomicBasisSource.cpp
  MOR_GreedyAtomicBasisSample.cpp
  MOR_CollocationMetricCriterion.cpp
  MOR_CollocationMetricCriterionFactory.cpp
  MOR_ExtendedEpetraLapack.cpp
  MOR_StkNodalMeshReduction.cpp
  MOR_SnapshotPreprocessorFactory.cpp
  MOR_TrivialSnapshotPreprocessor.cpp
  MOR_SubstractingSnapshotPreprocessor.cpp
  MOR_MeanSubstractingSnapshotPreprocessor.cpp
  MOR_FirstVectorSubstractingSnapshotPreprocessor.cpp
  MOR_SingularValuesHelpers.cpp
  MOR_SnapshotBlockingUtils.cpp
  )

SET(MOR_HEADERS
  MOR_ContainerUtils.hpp
  MOR_BasisOps.hpp
  MOR_EpetraUtils.hpp
  MOR_EpetraLocalMapMVMatrixMarketUtils.hpp
  MOR_EpetraMVDenseMatrixView.hpp
  MOR_EpetraSamplingOperator.hpp
  MOR_ReducedOperatorFactory.hpp
  MOR_GaussNewtonOperatorFactory.hpp
  MOR_PetrovGalerkinOperatorFactory.hpp
  MOR_ReducedJacobianFactory.hpp
  MOR_ReducedSpace.hpp
  MOR_ReducedSpaceFactory.hpp
  MOR_ReducedBasisElements.hpp
  MOR_ReducedBasisSource.hpp
  MOR_TruncatedReducedBasisSource.hpp
  MOR_ReducedBasisFactory.hpp
  MOR_DefaultReducedBasisFactory.hpp
  MOR_ReducedBasisRepository.hpp
  MOR_SampleDofListFactory.hpp
  MOR_DefaultSampleDofListProviders.hpp
  MOR_ReducedOrderModelEvaluator.hpp
  MOR_ReducedOrderModelFactory.hpp
  MOR_ObserverFactory.hpp
  MOR_NOXEpetraCompositeObserver.hpp
  MOR_FullStateReconstructor.hpp
  MOR_RythmosFullStateReconstructor.hpp
  MOR_RythmosStepperFullStateWrapper.hpp
  MOR_ProjectionError.hpp
  MOR_ProjectionErrorObserver.hpp
  MOR_RythmosProjectionErrorObserver.hpp
  MOR_GeneralizedCoordinatesOutput.hpp
  MOR_GeneralizedCoordinatesNOXObserver.hpp
  MOR_GeneralizedCoordinatesRythmosObserver.hpp
  MOR_SnapshotCollection.hpp
  MOR_SnapshotCollectionObserver.hpp
  MOR_RythmosSnapshotCollectionObserver.hpp
  MOR_RythmosUtils.hpp
  MOR_EpetraMVSource.hpp
  MOR_ConcatenatedEpetraMVSource.hpp
  MOR_InputFileEpetraMVSource.hpp
  MOR_FileReducedBasisSource.hpp
  MOR_IdentityBasisSource.hpp
  MOR_MultiVectorInputFile.hpp
  MOR_MultiVectorInputFileFactory.hpp
  MOR_MatrixMarketMVInputFile.hpp
  MOR_Hdf5MVInputFile.hpp
  MOR_MultiVectorOutputFileFactory.hpp
  MOR_MultiVectorOutputFile.hpp
  MOR_MatrixMarketMVOutputFile.hpp
  MOR_Hdf5MVOutputFile.hpp
  MOR_AtomicBasisSource.hpp
  MOR_WindowedAtomicBasisSource.hpp
  MOR_GreedyAtomicBasisSample.hpp
  MOR_CollocationMetricCriterion.hpp
  MOR_CollocationMetricCriterionFactory.hpp
  MOR_MinMaxTools.hpp
  MOR_ExtendedEpetraLapack.hpp
  MOR_StkNodalMeshReduction.hpp
  MOR_SnapshotPreprocessor.hpp
  MOR_SnapshotPreprocessorFactory.hpp
  MOR_TrivialSnapshotPreprocessor.hpp
  MOR_SubstractingSnapshotPreprocessor.hpp
  MOR_MeanSubstractingSnapshotPreprocessor.hpp
  MOR_FirstVectorSubstractingSnapshotPreprocessor.hpp
  MOR_SingularValuesHelpers.hpp
  MOR_SnapshotBlockingUtils.hpp
  )


add_library(MOR ${Albany_LIBRARY_TYPE} ${MOR_SOURCES} ${MOR_HEADERS})
set_target_properties(MOR PROPERTIES PUBLIC_HEADER "${MOR_HEADERS}")

IF (INSTALL_ALBANY)
  install(TARGETS MOR EXPORT albany-export
    LIBRARY DESTINATION "${LIB_INSTALL_DIR}/"
    ARCHIVE DESTINATION "${LIB_INSTALL_DIR}/"
    PUBLIC_HEADER DESTINATION "${INCLUDE_INSTALL_DIR}")
ENDIF()

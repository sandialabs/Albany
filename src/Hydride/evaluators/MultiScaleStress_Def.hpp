//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace LCM {

//**********************************************************************
template<typename EvalT, typename Traits>
MultiScaleStressBase<EvalT, Traits>::
MultiScaleStressBase(const Teuchos::ParameterList& p) :
  strain(p.get<std::string> ("Strain Name"),
         p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout")),
  elasticModulus(p.get<std::string> ("Elastic Modulus Name"),
                 p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")),
  poissonsRatio(p.get<std::string> ("Poissons Ratio Name"),
                p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")),
  stressFieldRealType(std::string("stress_RealType"), 
         p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout")),
  stress(p.get<std::string> ("Stress Name"),
         p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout")) {

  // Pull out numQPs and numDims from a Layout
  Teuchos::RCP<PHX::DataLayout> tensor_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  tensor_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  this->addDependentField(strain);
  this->addDependentField(elasticModulus);

  Teuchos::ArrayRCP<RealType> s_mem(tensor_dl->size());
  stressFieldRealType.setFieldData(s_mem);

  numMesoPEs = p.get<int>("Num Meso PEs");
  interCommunicator = p.get< Teuchos::RCP<MPI_Comm> >("MPALE Intercommunicator");

  loc_data.resize(numMesoPEs);
  exchanged_stresses.resize(numDims * numDims);

  // PoissonRatio not used in 1D stress calc
  if(numDims > 1) this->addDependentField(poissonsRatio);

  this->addEvaluatedField(stress);

  this->setName("MultiScaleStress" + PHX::typeAsString<EvalT>());

}

//**********************************************************************
template<typename EvalT, typename Traits>
void MultiScaleStressBase<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm) {
  this->utils.setFieldData(stress, fm);
  this->utils.setFieldData(strain, fm);
  this->utils.setFieldData(elasticModulus, fm);

  if(numDims > 1) this->utils.setFieldData(poissonsRatio, fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void MultiScaleStressBase<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset) {
  TEUCHOS_TEST_FOR_EXCEPTION("MultiScaleStressBase::evaluateFields not implemented for this template type",
                             Teuchos::Exceptions::InvalidParameter, "Need specialization.");
}

template<typename EvalT, typename Traits>
void MultiScaleStressBase<EvalT, Traits>::
calcStress(typename Traits::EvalData workset) {

  ScalarT lambda, mu;

  // Calculate stresses

  switch(numDims) {
    case 1:
      Intrepid2::FunctionSpaceTools<PHX::Device>::tensorMultiplyDataData(stress.get_view(), elasticModulus.get_view(), strain.get_view());
      break;

    case 2:

      // Compute Stress (with the plane strain assumption for now)

      for(std::size_t cell = 0; cell < workset.numCells; ++cell) {
        for(std::size_t qp = 0; qp < numQPs; ++qp) {

          lambda = (elasticModulus(cell, qp) * poissonsRatio(cell, qp))
                   / ((1 + poissonsRatio(cell, qp)) * (1 - 2 * poissonsRatio(cell, qp)));
          mu = elasticModulus(cell, qp) / (2 * (1 + poissonsRatio(cell, qp)));
          stress(cell, qp, 0, 0) = 2.0 * mu * (strain(cell, qp, 0, 0))
                                   + lambda * (strain(cell, qp, 0, 0) + strain(cell, qp, 1, 1));
          stress(cell, qp, 1, 1) = 2.0 * mu * (strain(cell, qp, 1, 1))
                                   + lambda * (strain(cell, qp, 0, 0) + strain(cell, qp, 1, 1));
          stress(cell, qp, 0, 1) = 2.0 * mu * (strain(cell, qp, 0, 1));
          stress(cell, qp, 1, 0) = stress(cell, qp, 0, 1);

        }
      }

      break;

    case 3:

      // Compute Stress

      for(std::size_t cell = 0; cell < workset.numCells; ++cell) {
        for(std::size_t qp = 0; qp < numQPs; ++qp) {

          lambda = (elasticModulus(cell, qp) * poissonsRatio(cell, qp))
                   / ((1 + poissonsRatio(cell, qp)) * (1 - 2 * poissonsRatio(cell, qp)));
          mu = elasticModulus(cell, qp) / (2 * (1 + poissonsRatio(cell, qp)));
          stress(cell, qp, 0, 0) = 2.0 * mu * (strain(cell, qp, 0, 0))
                                   + lambda * (strain(cell, qp, 0, 0) + strain(cell, qp, 1, 1) + strain(cell, qp, 2, 2));
          stress(cell, qp, 1, 1) = 2.0 * mu * (strain(cell, qp, 1, 1))
                                   + lambda * (strain(cell, qp, 0, 0) + strain(cell, qp, 1, 1) + strain(cell, qp, 2, 2));
          stress(cell, qp, 2, 2) = 2.0 * mu * (strain(cell, qp, 2, 2))
                                   + lambda * (strain(cell, qp, 0, 0) + strain(cell, qp, 1, 1) + strain(cell, qp, 2, 2));
          stress(cell, qp, 0, 1) = 2.0 * mu * (strain(cell, qp, 0, 1));
          stress(cell, qp, 1, 2) = 2.0 * mu * (strain(cell, qp, 1, 2));
          stress(cell, qp, 2, 0) = 2.0 * mu * (strain(cell, qp, 2, 0));
          stress(cell, qp, 1, 0) = stress(cell, qp, 0, 1);
          stress(cell, qp, 2, 1) = stress(cell, qp, 1, 2);
          stress(cell, qp, 0, 2) = stress(cell, qp, 2, 0);

        }
      }

      break;
  }

  return;

}

template<typename EvalT, typename Traits>
void MultiScaleStressBase<EvalT, Traits>::
mesoBridgeStressRealType(PHX::MDField<RealType, Cell, QuadPoint, Dim, Dim>& stressFieldOut,
                         PHX::MDField<RealType, Cell, QuadPoint, Dim, Dim>& stressFieldIn,
                         typename Traits::EvalData workset) {

  // Communicate stress to and from MPALE

  /*
    Background: Here, we send the stress state for each QP for each cell in the
    workset to a waiting MPALE process. But, we limit the number of processes that
    can be used to numMesoPEs. So, we will process this in numMesoPEs sized chunks.
  */

  int procID = 0;

  for(std::size_t cell = 0; cell < workset.numCells; ++cell)
    for(std::size_t qp = 0; qp < numQPs; ++qp) {

      // send the stress tensor for this cell and qp to a processor (procID)

      sendCellQPData(cell, qp, procID, STRESS_TENSOR, stressFieldIn);

      procID++; // go to the next qp and processor (and possibly cell)

      // Chunk section: If we have sent all the data available (cell == workset.numCells - 1 AND
      // qp == numQPs - 1) OR we have sent requests to all the processors available, we receive the
      // data back.

      if(procID >= numMesoPEs || (cell == workset.numCells - 1 && qp == numQPs - 1)) {

        rcvCellQPData(procID, STRESS_TENSOR, stressFieldOut);

        procID = 0;

      }

    }

  return;

}

template<typename Traits>
void MultiScaleStress<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData workset) {

  this->calcStress(workset);

  this->mesoBridgeStressRealType(this->stress, this->stress, workset);

  return;

}

template<typename Traits>
void MultiScaleStress<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData workset) {

  this->calcStress(workset);

  // Begin Finite Difference
  // Do Base unperturbed case
  for(int cell = 0; cell < (int)workset.numCells; ++cell)
    for(int qp = 0; qp < (int)this->numQPs; ++qp)
      for(int i = 0; i < (int)this->numDims; ++i)
        for(int j = 0; j < (int)this->numDims; ++j)

          this->stressFieldRealType(cell, qp, i, j) = this->stress(cell, qp, i, j).val();

  this->mesoBridgeStressRealType(this->stressFieldRealType, this->stressFieldRealType, workset);

  for(int cell = 0; cell < (int)workset.numCells; ++cell)
    for(int qp = 0; qp < (int)this->numQPs; ++qp)
      for(int i = 0; i < (int)this->numDims; ++i)
        for(int j = 0; j < (int)this->numDims; ++j)

          this->stress(cell, qp, i, j).val() = this->stressFieldRealType(cell, qp, i, j);

  // Do Perturbations
  double pert = 1.0e-6;
  int numIVs = this->stress(0, 0, 0, 0).size();

  for(int iv = 0; iv < numIVs; ++iv) {
    for(int cell = 0; cell < (int)workset.numCells; ++cell)
      for(int qp = 0; qp < (int)this->numQPs; ++qp)
        for(int i = 0; i < (int)this->numDims; ++i)
          for(int j = 0; j < (int)this->numDims; ++j)

            this->stressFieldRealType(cell, qp, i, j) =
              this->stress(cell, qp, i, j).val() + pert * this->stress(cell, qp, i, j).fastAccessDx(iv);

    this->mesoBridgeStressRealType(this->stressFieldRealType, this->stressFieldRealType, workset);

    for(int cell = 0; cell < (int)workset.numCells; ++cell)
      for(int qp = 0; qp < (int)this->numQPs; ++qp)
        for(int i = 0; i < (int)this->numDims; ++i)
          for(int j = 0; j < (int)this->numDims; ++j)

            this->stress(cell, qp, i, j).fastAccessDx(iv) =
              (this->stressFieldRealType(cell, qp, i, j) - this->stress(cell, qp, i, j).val()) / pert;
  }



  return;

}

// Tangent implementation is Identical to Jacobian
template<typename Traits>
void MultiScaleStress<PHAL::AlbanyTraits::Tangent, Traits>::
evaluateFields(typename Traits::EvalData workset) {

  this->calcStress(workset);

  // Begin Finite Difference
  // Do Base unperturbed case
  for(int cell = 0; cell < (int)workset.numCells; ++cell)
    for(int qp = 0; qp < (int)this->numQPs; ++qp)
      for(int i = 0; i < (int)this->numDims; ++i)
        for(int j = 0; j < (int)this->numDims; ++j)

          this->stressFieldRealType(cell, qp, i, j) = this->stress(cell, qp, i, j).val();

  this->mesoBridgeStressRealType(this->stressFieldRealType, this->stressFieldRealType, workset);

  for(int cell = 0; cell < (int)workset.numCells; ++cell)
    for(int qp = 0; qp < (int)this->numQPs; ++qp)
      for(int i = 0; i < (int)this->numDims; ++i)
        for(int j = 0; j < (int)this->numDims; ++j)

          this->stress(cell, qp, i, j).val() = this->stressFieldRealType(cell, qp, i, j);

  // Do Perturbations
  double pert = 1.0e-6;
  int numIVs = this->stress(0, 0, 0, 0).size();

  for(int iv = 0; iv < numIVs; ++iv) {
    for(int cell = 0; cell < (int)workset.numCells; ++cell)
      for(int qp = 0; qp < (int)this->numQPs; ++qp)
        for(int i = 0; i < (int)this->numDims; ++i)
          for(int j = 0; j < (int)this->numDims; ++j)

            this->stressFieldRealType(cell, qp, i, j) =
              this->stress(cell, qp, i, j).val() + pert * this->stress(cell, qp, i, j).fastAccessDx(iv);

    this->mesoBridgeStressRealType(this->stressFieldRealType, this->stressFieldRealType, workset);

    for(int cell = 0; cell < (int)workset.numCells; ++cell)
      for(int qp = 0; qp < (int)this->numQPs; ++qp)
        for(int i = 0; i < (int)this->numDims; ++i)
          for(int j = 0; j < (int)this->numDims; ++j)

            this->stress(cell, qp, i, j).fastAccessDx(iv) =
              (this->stressFieldRealType(cell, qp, i, j) - this->stress(cell, qp, i, j).val()) / pert;
  }



  return;

}

//**********************************************************************


//**********************************************************************
template<typename EvalT, typename Traits>
void MultiScaleStressBase<EvalT, Traits>::
sendCellQPData(int cell, int qp, int toProc, MessageType type,
               PHX::MDField<RealType, Cell, QuadPoint, Dim, Dim>& stressFieldIn) {

  // Fill in array
  for(std::size_t i = 0; i < numDims; i++)
    for(std::size_t dim = 0; dim < numDims; dim++)

      // Store the stress tensor for this cell and qp

      exchanged_stresses[numDims * i + dim] = stressFieldIn(cell, qp, i, dim);


  // send it

  MPI_Send(&exchanged_stresses[0],     /* the stress tensor */
           exchanged_stresses.size(),         /* its length (9) */
           MPI_DOUBLE,                        /* data items are doubles */
           toProc,                            /* destination process */
           type,                              /* user chosen message tag */
           *interCommunicator.get());          /* the MPALE communicator */

  // Save the cell and qp ids for unpacking

  loc_data[toProc].cell = cell;
  loc_data[toProc].qp = qp;

  return;

}

template<typename EvalT, typename Traits>
void MultiScaleStressBase<EvalT, Traits>::
rcvCellQPData(int procIDReached, MessageType type,
              PHX::MDField<RealType, Cell, QuadPoint, Dim, Dim>& stressFieldOut) {

  MPI_Status status;

  // Loop over all the processors participating

  for(int proc_ctr = 0; proc_ctr < procIDReached; proc_ctr++) {

    /*
     * Receive the stress tensor from the first MPALE process to send one
     * No need to receive them in order
     */

    MPI_Recv(&exchanged_stresses[0],           /* incoming tensor */
             exchanged_stresses.size(),
             MPI_DOUBLE,                         /* of type double */
             MPI_ANY_SOURCE,                     /* receive from any sender */
             type,
             *interCommunicator.get(),           /* from one of the MPALE processes */
             &status);                          /* info about the received message */

    int proc = status.MPI_SOURCE; // What MPALE process sent the message?

    // Fill in the stress info back into the Albany array
    for(std::size_t i = 0; i < numDims; i++)
      for(std::size_t dim = 0; dim < numDims; dim++)

        // Store the stress tensor for this cell and qp

        stressFieldOut(loc_data[proc].cell, loc_data[proc].qp, i, dim) = exchanged_stresses[numDims * i + dim];

  }

  return;

}
//**********************************************************************

}


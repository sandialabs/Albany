//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef AADAPT_STKADAPTT_HPP
#define AADAPT_STKADAPTT_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "AAdapt_AbstractAdapterT.hpp"
#include "Albany_GenericSTKMeshStruct.hpp"
#include "Albany_STKDiscretization.hpp"
#ifdef ALBANY_FELIX
#include "Albany_STKDiscretizationStokesH.hpp"
#endif

#include "Phalanx.hpp"
#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"

#include "AAdapt_STKUnifSizeField.hpp"
#include "UniformRefinerPattern.hpp"

namespace AAdapt {

template<class SizeField>

class STKAdaptT : public AbstractAdapterT {
  public:

    STKAdaptT(const Teuchos::RCP<Teuchos::ParameterList>& params_,
              const Teuchos::RCP<ParamLib>& paramLib_,
              const Albany::StateManager& StateMgr_,
              const Teuchos::RCP<const Teuchos_Comm>& comm_);
    //! Destructor
    ~STKAdaptT();

    //! Check adaptation criteria to determine if the mesh needs adapting
    virtual bool queryAdaptationCriteria(int iteration);

    //! Apply adaptation method to mesh and problem. Returns true if adaptation is performed successfully.
    virtual bool adaptMesh(const Teuchos::RCP<const Tpetra_Vector>& solution,
                           const Teuchos::RCP<const Tpetra_Vector>& ovlp_solution);

    //! Each adapter must generate it's list of valid parameters
    Teuchos::RCP<const Teuchos::ParameterList> getValidAdapterParameters() const;

  private:

    // Disallow copy and assignment
    STKAdaptT(const STKAdaptT&);
    STKAdaptT& operator=(const STKAdaptT&);

    void printElementData();

    int numDim;
    int remeshFileIndex;
    std::string base_exo_filename;


    Teuchos::RCP<Albany::GenericSTKMeshStruct> genericMeshStruct;

    Teuchos::RCP<Albany::AbstractDiscretization> disc;

    Albany::STKDiscretization* stk_discretization;

    Teuchos::RCP<stk::percept::PerceptMesh> eMesh;
    Teuchos::RCP<stk::adapt::UniformRefinerPatternBase> refinerPattern;

    int num_iterations;

    const Teuchos::RCP<const Tpetra_Vector> solution;
    const Teuchos::RCP<const Tpetra_Vector> ovlp_solution;

};

}

// Define macros for explicit template instantiation
#define STKADAPTT_INSTANTIATE_TEMPLATE_CLASS_UNIFREFINE(name) \
  template class name<AAdapt::STKUnifRefineField>;

#define STKADAPTT_INSTANTIATE_TEMPLATE_CLASS(name) \
  STKADAPTT_INSTANTIATE_TEMPLATE_CLASS_UNIFREFINE(name)


#endif //ALBANY_STKADAPTT_HPP

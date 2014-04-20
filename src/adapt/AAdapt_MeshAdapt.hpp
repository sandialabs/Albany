//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef AADAPT_MESHADAPT_HPP
#define AADAPT_MESHADAPT_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "AAdapt_AbstractAdapter.hpp"
#include "AlbPUMI_FMDBMeshStruct.hpp"
#include "AlbPUMI_AbstractPUMIDiscretization.hpp"

#include "Phalanx.hpp"
#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"

#include "AAdapt_UnifSizeField.hpp"
#include "AAdapt_UnifRefSizeField.hpp"
#ifdef SCOREC_SPR
#include "AAdapt_SPRSizeField.hpp"
#endif

namespace AAdapt {

template<class SizeField>

class MeshAdapt : public AbstractAdapter {
  public:

    MeshAdapt(const Teuchos::RCP<Teuchos::ParameterList>& params_,
              const Teuchos::RCP<ParamLib>& paramLib_,
              Albany::StateManager& StateMgr_,
              const Teuchos::RCP<const Epetra_Comm>& comm_);
    //! Destructor
    ~MeshAdapt();

    //! Check adaptation criteria to determine if the mesh needs adapting
    virtual bool queryAdaptationCriteria();

    //! Apply adaptation method to mesh and problem. Returns true if adaptation is performed successfully.
    // Solution is needed to calculate the size function
    virtual bool adaptMesh(const Epetra_Vector& solution, const Epetra_Vector& ovlp_solution);

    //! Transfer solution between meshes.
    virtual void solutionTransfer(const Epetra_Vector& oldSolution,
                                  Epetra_Vector& newSolution);

    //! Each adapter must generate it's list of valid parameters
    Teuchos::RCP<const Teuchos::ParameterList> getValidAdapterParameters() const;

  private:

    // Disallow copy and assignment
    MeshAdapt(const MeshAdapt&);
    MeshAdapt& operator=(const MeshAdapt&);

    int remeshFileIndex;

    Teuchos::RCP<Albany::AbstractDiscretization> disc;
    Teuchos::RCP<AlbPUMI::AbstractPUMIDiscretization> pumi_discretization;

    apf::Mesh2* mesh;
    pMeshMdl pumiMesh;

    int num_iterations;

    static Teuchos::RCP<SizeField> szField;

    void printElementData();
  
    void checkValidStateVariable(const std::string name);

    std::string adaptation_method;
    std::string base_exo_filename;

};

}

// Define macros for explicit template instantiation
#define MESHADAPT_INSTANTIATE_TEMPLATE_CLASS_UNIF(name) \
  template class name<AAdapt::UnifSizeField>;
#define MESHADAPT_INSTANTIATE_TEMPLATE_CLASS_UNIFREF(name) \
  template class name<AAdapt::UnifRefSizeField>;

#ifdef SCOREC_SPR
#define MESHADAPT_INSTANTIATE_TEMPLATE_CLASS_SPR(name) \
  template class name<AAdapt::SPRSizeField>;
#endif

#ifdef SCOREC_SPR
#define MESHADAPT_INSTANTIATE_TEMPLATE_CLASS(name) \
  MESHADAPT_INSTANTIATE_TEMPLATE_CLASS_UNIF(name) \
  MESHADAPT_INSTANTIATE_TEMPLATE_CLASS_UNIFREF(name) \
  MESHADAPT_INSTANTIATE_TEMPLATE_CLASS_SPR(name)
#else
#define MESHADAPT_INSTANTIATE_TEMPLATE_CLASS(name) \
  MESHADAPT_INSTANTIATE_TEMPLATE_CLASS_UNIF(name) \
  MESHADAPT_INSTANTIATE_TEMPLATE_CLASS_UNIFREF(name)
#endif

#endif //ALBANY_MESHADAPT_HPP

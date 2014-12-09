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
#include "AAdapt_AbstractAdapterT.hpp"
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
#include "AAdapt_ReferenceConfigurationManager.hpp"

namespace AAdapt {

template<class SizeField>
class MeshAdapt {
  public:
    MeshAdapt(const Teuchos::RCP<Teuchos::ParameterList>& params_,
              const Albany::StateManager& StateMgr_,
              const Teuchos::RCP<ReferenceConfigurationManager>& refConfigMgr_);
    ~MeshAdapt();

    //! Check adaptation criteria to determine if the mesh needs adapting
    bool queryAdaptationCriteria(
        const Teuchos::RCP<Teuchos::ParameterList>& params,
        int iter);

    //! Apply adaptation method to mesh and problem. Returns true if adaptation is performed successfully.
    bool adaptMesh(const Teuchos::RCP<Teuchos::ParameterList>& adapt_params_,
        Teuchos::RCP<Teuchos::FancyOStream>& output_stream_);

    //! Each adapter must generate it's list of valid parameters
    Teuchos::RCP<const Teuchos::ParameterList> getValidAdapterParameters(
        Teuchos::RCP<Teuchos::ParameterList>& validPL) const;

    void beforeAdapt(const Teuchos::RCP<Teuchos::ParameterList>& adapt_params_,
        Teuchos::RCP<Teuchos::FancyOStream>& output_stream_);
    void adaptInPartition(const Teuchos::RCP<Teuchos::ParameterList>& adapt_params_);
    void afterAdapt(const Teuchos::RCP<Teuchos::ParameterList>& adapt_params_);

  private:

    // Disallow copy and assignment
    MeshAdapt(const MeshAdapt&);
    MeshAdapt& operator=(const MeshAdapt&);

    int remeshFileIndex;

    Teuchos::RCP<Albany::AbstractDiscretization> disc;
    Teuchos::RCP<AlbPUMI::AbstractPUMIDiscretization> pumi_discretization;

    apf::Mesh2* mesh;

    int num_iterations;

    Teuchos::RCP<SizeField> szField;
  
    void checkValidStateVariable(
        const Albany::StateManager& state_mgr_,
        const std::string name);

    std::string adaptation_method;
    std::string base_exo_filename;

    Teuchos::RCP<ReferenceConfigurationManager> rc_mgr;
};

template <class SizeField>
class MeshAdaptT : public AbstractAdapterT {
  public:
    MeshAdaptT(const Teuchos::RCP<Teuchos::ParameterList>& params_,
               const Teuchos::RCP<ParamLib>& paramLib_,
               const Albany::StateManager& StateMgr_,
               const Teuchos::RCP<ReferenceConfigurationManager>& refConfigMgr_,
               const Teuchos::RCP<const Teuchos_Comm>& commT_);
    virtual bool queryAdaptationCriteria(int iteration);
    virtual bool adaptMesh(
        const Teuchos::RCP<const Tpetra_Vector>& solution,
        const Teuchos::RCP<const Tpetra_Vector>& ovlp_solution);
    virtual Teuchos::RCP<const Teuchos::ParameterList>
        getValidAdapterParameters() const;
  private:
    MeshAdapt<SizeField> meshAdapt;
};

} //namespace AAdapt

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

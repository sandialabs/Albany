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
#include "Albany_PUMIMeshStruct.hpp"
#include "Albany_PUMIDiscretization.hpp"

#include "Phalanx.hpp"
#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"

struct Parma_GroupCode;
struct AdaptCallbackOf;

namespace AAdapt {
class MeshSizeField;
namespace rc { class Manager; }

class MeshAdapt : public AbstractAdapterT {
public:
  MeshAdapt(const Teuchos::RCP<Teuchos::ParameterList>& params_,
            const Teuchos::RCP<ParamLib>& paramLib_,
            const Albany::StateManager& StateMgr_,
            const Teuchos::RCP<AAdapt::rc::Manager>& refConfigMgr_,
            const Teuchos::RCP<const Teuchos_Comm>& commT_);
  ~MeshAdapt();

  //! Check adaptation criteria to determine if the mesh needs adapting
  bool queryAdaptationCriteria(int iteration);

  //! Apply adaptation method to mesh and problem. Returns true if adaptation is performed successfully.
  bool adaptMesh(
    const Teuchos::RCP<const Tpetra_Vector>& solution,
    const Teuchos::RCP<const Tpetra_Vector>& ovlp_solution);

  void adaptInPartition(const Teuchos::RCP<Teuchos::ParameterList>& adapt_params_);

  //! Each adapter must generate its list of valid parameters
  Teuchos::RCP<const Teuchos::ParameterList> getValidAdapterParameters() const;

private:

  // Disallow copy and assignment
  MeshAdapt(const MeshAdapt&);
  MeshAdapt& operator=(const MeshAdapt&);

  int remeshFileIndex;

  Teuchos::RCP<Albany::AbstractDiscretization> disc;
  Teuchos::RCP<Albany::PUMIDiscretization> pumi_discretization;

  apf::Mesh2* mesh;

  Teuchos::RCP<MeshSizeField> szField;
  
  std::string adaptation_method;
  std::string base_exo_filename;

  bool should_transfer_ip_data;

  Teuchos::RCP<rc::Manager> rc_mgr;

  void initRcMgr();
  void checkValidStateVariable(
    const Albany::StateManager& state_mgr,
    const std::string name);
  void initAdapt(const Teuchos::RCP<Teuchos::ParameterList>& adapt_params,
                 Teuchos::RCP<Teuchos::FancyOStream>& output_stream);
  void beforeAdapt();
  bool adaptMeshWithRc(const double min_part_density,
                       AdaptCallbackOf& callback);
  bool adaptMeshLoop(const double min_part_density, AdaptCallbackOf& callback);
  void afterAdapt(const Teuchos::RCP<Teuchos::ParameterList>& adapt_params);
};

} //namespace AAdapt

#endif //ALBANY_MESHADAPT_HPP

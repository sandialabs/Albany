//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AADAPT_MESH_ADAPT_HPP
#define AADAPT_MESH_ADAPT_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "AAdapt_AbstractAdapter.hpp"

#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"

// Forward declarations
struct Parma_GroupCode;
namespace Albany {
  class PUMIDiscretization;
}
namespace apf {
  class Mesh2;
}

namespace AAdapt {
  // Forward declarations
  class MeshAdaptMethod;
  namespace rc { class Manager; }

class MeshAdapt : public AbstractAdapter {
public:
  MeshAdapt(const Teuchos::RCP<Teuchos::ParameterList>& params_,
            const Teuchos::RCP<ParamLib>& paramLib_,
            const Albany::StateManager& StateMgr_,
            const Teuchos::RCP<AAdapt::rc::Manager>& refConfigMgr_,
            const Teuchos::RCP<const Teuchos_Comm>& comm_);

  ~MeshAdapt() = default;

  //! Check adaptation criteria to determine if the mesh needs adapting
  bool queryAdaptationCriteria(int iteration);

  //! Apply adaptation method to mesh and problem. Returns true if adaptation is performed successfully.
/*
  bool adaptMesh(
    const Teuchos::RCP<const Tpetra_Vector>& solution,
    const Teuchos::RCP<const Tpetra_Vector>& ovlp_solution);
*/
  bool adaptMesh();

  void adaptInPartition();

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

  Teuchos::RCP<MeshAdaptMethod> szField;
  
  std::string adaptation_method;
  std::string base_exo_filename;

  bool should_transfer_ip_data;

  Teuchos::RCP<rc::Manager> rc_mgr;

  void initRcMgr();
  void checkValidStateVariable(
    const Albany::StateManager& state_mgr,
    const std::string name);
  void initAdapt();
                
  void beforeAdapt();
  bool adaptMeshWithRc(const double min_part_density,
                       Parma_GroupCode& callback);
  bool adaptMeshLoop(const double min_part_density, Parma_GroupCode& callback);
  void afterAdapt();
};

} //namespace AAdapt

#endif //ALBANY_MESH_ADAPT_HPP

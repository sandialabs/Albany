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
#include "Albany_AbstractPUMIDiscretization.hpp"

#include "Phalanx.hpp"
#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"

struct Parma_GroupCode;

namespace AAdapt {
class MeshSizeField;
namespace rc { class Manager; }

class MeshAdapt {
public:
  MeshAdapt(const Teuchos::RCP<Teuchos::ParameterList>& params_,
            const Albany::StateManager& StateMgr_,
            const Teuchos::RCP<rc::Manager>& refConfigMgr_);
  ~MeshAdapt();

  //! Check adaptation criteria to determine if the mesh needs adapting
  bool queryAdaptationCriteria(
    const Teuchos::RCP<Teuchos::ParameterList>& params,
    int iter);

  //! Apply adaptation method to mesh and problem. Returns true if adaptation is performed successfully.
  bool adaptMesh(const Teuchos::RCP<Teuchos::ParameterList>& adapt_params_,
                 Teuchos::RCP<Teuchos::FancyOStream>& output_stream_);

  void adaptInPartition(const Teuchos::RCP<Teuchos::ParameterList>& adapt_params_);

  //! Each adapter must generate its list of valid parameters
  Teuchos::RCP<const Teuchos::ParameterList> getValidAdapterParameters(
    Teuchos::RCP<Teuchos::ParameterList>& validPL) const;

private:

  // Disallow copy and assignment
  MeshAdapt(const MeshAdapt&);
  MeshAdapt& operator=(const MeshAdapt&);

  int remeshFileIndex;

  Teuchos::RCP<Albany::AbstractDiscretization> disc;
  Teuchos::RCP<Albany::AbstractPUMIDiscretization> pumi_discretization;

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
                       Parma_GroupCode& callback);
  bool adaptMeshLoop(const double min_part_density, Parma_GroupCode& callback);
  void afterAdapt();
};

class MeshAdaptT : public AbstractAdapterT {
public:
  MeshAdaptT(const Teuchos::RCP<Teuchos::ParameterList>& params_,
             const Teuchos::RCP<ParamLib>& paramLib_,
             const Albany::StateManager& StateMgr_,
             const Teuchos::RCP<rc::Manager>& refConfigMgr_,
             const Teuchos::RCP<const Teuchos_Comm>& commT_);
  virtual bool queryAdaptationCriteria(int iteration);
  virtual bool adaptMesh(
    const Teuchos::RCP<const Tpetra_Vector>& solution,
    const Teuchos::RCP<const Tpetra_Vector>& ovlp_solution);
  virtual Teuchos::RCP<const Teuchos::ParameterList>
  getValidAdapterParameters() const;
private:
  MeshAdapt meshAdapt;
};

} //namespace AAdapt

#endif //ALBANY_MESHADAPT_HPP

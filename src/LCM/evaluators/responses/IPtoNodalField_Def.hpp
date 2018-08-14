//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <Teuchos_TestForException.hpp>
#include <fstream>
#include "Adapt_NodalDataVector.hpp"
#include "Albany_Utils.hpp"

namespace LCM {
class IPtoNodalFieldManager : public Adapt::NodalDataBase::Manager
{
 public:
  IPtoNodalFieldManager() : nwrkr_(0), prectr_(0), postctr_(0) {}

  void
  registerWorker()
  {
    ++nwrkr_;
  }
  int
  nWorker() const
  {
    return nwrkr_;
  }

  void
  initCounters()
  {
    prectr_ = postctr_ = 0;
  }
  int
  incrPreCounter()
  {
    return ++prectr_;
  }
  int
  incrPostCounter()
  {
    return ++postctr_;
  }

  // Start position in the nodal vector database, and number of vectors we're
  // using.
  int ndb_start, ndb_numvecs;
  // Multivector that will go into the nodal database.
  Teuchos::RCP<Tpetra_MultiVector> nodal_field;

 private:
  int nwrkr_, prectr_, postctr_;
};

template <typename EvalT, typename Traits>
IPtoNodalFieldBase<EvalT, Traits>::IPtoNodalFieldBase(
    Teuchos::ParameterList&              p,
    const Teuchos::RCP<Albany::Layouts>& dl,
    const Albany::MeshSpecsStruct*       mesh_specs)
    : weights_("Weights", dl->qp_scalar)
{
  Teuchos::ParameterList* plist =
      p.get<Teuchos::ParameterList*>("Parameter List");

  //! Register with state manager
  this->p_state_mgr_ = p.get<Albany::StateManager*>("State Manager Ptr");

  // loop over the number of fields and register
  number_of_fields_ = plist->get<int>("Number of Fields", 0);

  output_to_exodus_ = plist->get<bool>("Output to File", true);

  // Surface element prefix, if any.
  bool const is_surface_block = mesh_specs->ebName == "Surface Element";

  std::string const field_name_prefix = is_surface_block == true ? "surf_" : "";

  // resize field vectors
  ip_field_names_.resize(number_of_fields_);
  ip_field_layouts_.resize(number_of_fields_);
  nodal_field_names_.resize(number_of_fields_);
  ip_fields_.resize(number_of_fields_);

  for (int field(0); field < number_of_fields_; ++field) {
    ip_field_names_[field] =
        field_name_prefix +
        plist->get<std::string>(Albany::strint("IP Field Name", field));

    ip_field_layouts_[field] =
        plist->get<std::string>(Albany::strint("IP Field Layout", field));

    nodal_field_names_[field] = "nodal_" + ip_field_names_[field];

    if (ip_field_layouts_[field] == "Scalar") {
      PHX::MDField<ScalarT const> s(ip_field_names_[field], dl->qp_scalar);
      ip_fields_[field] = s;
    } else if (ip_field_layouts_[field] == "Vector") {
      PHX::MDField<ScalarT const> v(ip_field_names_[field], dl->qp_vector);
      ip_fields_[field] = v;
    } else if (ip_field_layouts_[field] == "Tensor") {
      PHX::MDField<ScalarT const> t(ip_field_names_[field], dl->qp_tensor);
      ip_fields_[field] = t;
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION(
          true, std::runtime_error, "Field Layout unknown");
    }

    this->addDependentField(ip_fields_[field].fieldTag());
  }

  this->addDependentField(weights_);

  // Create field tag
  field_tag_ =
      Teuchos::rcp(new PHX::Tag<ScalarT>("IP to Nodal Field", dl->dummy));

  this->addEvaluatedField(*field_tag_);
}

//------------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void
IPtoNodalFieldBase<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(weights_, fm);
  for (int field(0); field < number_of_fields_; ++field) {
    this->utils.setFieldData(ip_fields_[field], fm);
  }
}

//------------------------------------------------------------------------------
// Specialization: Residual
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
template <typename Traits>
IPtoNodalField<PHAL::AlbanyTraits::Residual, Traits>::IPtoNodalField(
    Teuchos::ParameterList&              p,
    const Teuchos::RCP<Albany::Layouts>& dl,
    const Albany::MeshSpecsStruct*       mesh_specs)
    : IPtoNodalFieldBase<PHAL::AlbanyTraits::Residual, Traits>(
          p,
          dl,
          mesh_specs)
{
  //! get and validate IPtoNodalField parameter list
  Teuchos::ParameterList* plist =
      p.get<Teuchos::ParameterList*>("Parameter List");
  Teuchos::RCP<const Teuchos::ParameterList> reflist =
      this->getValidIPtoNodalFieldParameters();
  plist->validateParameters(*reflist, 0);

  //! number of quad points per cell and dimension
  Teuchos::RCP<PHX::DataLayout> scalar_dl      = dl->qp_scalar;
  Teuchos::RCP<PHX::DataLayout> vector_dl      = dl->qp_vector;
  Teuchos::RCP<PHX::DataLayout> cell_dl        = dl->cell_scalar;
  Teuchos::RCP<PHX::DataLayout> node_dl        = dl->node_qp_vector;
  Teuchos::RCP<PHX::DataLayout> vert_vector_dl = dl->vertices_vector;
  num_pts_                                     = vector_dl->dimension(1);
  num_dims_                                    = vector_dl->dimension(2);
  num_nodes_                                   = node_dl->dimension(1);
  num_vertices_                                = vert_vector_dl->dimension(2);

  // Surface element prefix, if any.
  bool const is_surface_block = mesh_specs->ebName == "Surface Element";

  std::string const field_name_prefix = is_surface_block == true ? "surf_" : "";

  // Initialize manager.
  bool first;
  {
    const std::string key_suffix =
        field_name_prefix +
        (this->number_of_fields_ > 0 ?
             plist->get<std::string>(Albany::strint("IP Field Name", 0)) :
             "");
    const std::string key = "IPtoNodalField_" + key_suffix;
    const Teuchos::RCP<Adapt::NodalDataBase> ndb =
        this->p_state_mgr_->getNodalDataBase();
    first = !ndb->isManagerRegistered(key);
    if (first) {
      this->mgr_ = Teuchos::rcp(new IPtoNodalFieldManager());
      // Find out our starting position in the nodal database.
      this->mgr_->ndb_start = ndb->getVecsize();
      ndb->registerManager(key, this->mgr_);
    } else {
      this->mgr_ = Teuchos::rcp_dynamic_cast<IPtoNodalFieldManager>(
          ndb->getManager(key));
    }
    this->mgr_->registerWorker();
  }

  for (int field(0); field < this->number_of_fields_; ++field) {
    if (this->ip_field_layouts_[field] == "Scalar") {
      this->p_state_mgr_->registerNodalVectorStateVariable(
          this->nodal_field_names_[field],
          dl->node_node_scalar,
          dl->dummy,
          "all",
          "scalar",
          0.0,
          false,
          this->output_to_exodus_);
    } else if (this->ip_field_layouts_[field] == "Vector") {
      this->p_state_mgr_->registerNodalVectorStateVariable(
          this->nodal_field_names_[field],
          dl->node_node_vector,
          dl->dummy,
          "all",
          "scalar",
          0.0,
          false,
          this->output_to_exodus_);
    } else if (this->ip_field_layouts_[field] == "Tensor") {
      this->p_state_mgr_->registerNodalVectorStateVariable(
          this->nodal_field_names_[field],
          dl->node_node_tensor,
          dl->dummy,
          "all",
          "scalar",
          0.0,
          false,
          this->output_to_exodus_);
    }
  }

  // Register the nodal weights. Need a unique name so it doesn't conflict with
  // the weights vector of another IPtoNodalField response function. Even though
  // the weight vectors would be the same, coordination would be required to
  // prevent multiple sums.
  nodal_weights_name_ = "nw_" + this->ip_field_names_[0];
  this->p_state_mgr_->registerNodalVectorStateVariable(
      nodal_weights_name_,
      dl->node_node_scalar,
      dl->dummy,
      "all",
      "scalar",
      0.0,
      false,
      true);

  if (first) {
    this->mgr_->ndb_numvecs = this->p_state_mgr_->getStateInfoStruct()
                                  ->getNodalDataBase()
                                  ->getVecsize() -
                              this->mgr_->ndb_start;
  }
}

//------------------------------------------------------------------------------
template <typename Traits>
void
IPtoNodalField<PHAL::AlbanyTraits::Residual, Traits>::preEvaluate(
    typename Traits::PreEvalData workset)
{
  const int  ctr      = this->mgr_->incrPreCounter();
  const bool am_first = ctr == 1;
  if (!am_first) return;

  const Teuchos::RCP<Adapt::NodalDataVector> node_data =
      this->p_state_mgr_->getStateInfoStruct()
          ->getNodalDataBase()
          ->getNodalDataVector();
  this->mgr_->nodal_field = Teuchos::rcp(new Tpetra_MultiVector(
      node_data->getLocalMap(), this->mgr_->ndb_numvecs, true));
}

//------------------------------------------------------------------------------
template <typename Traits>
void
IPtoNodalField<PHAL::AlbanyTraits::Residual, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  // volume averaged field, store as nodal data that will be scattered
  // and summed

  // Get the node data block container
  const Teuchos::RCP<Tpetra_MultiVector>&  data       = this->mgr_->nodal_field;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO>> wsElNodeID = workset.wsElNodeID;
  Teuchos::RCP<const Tpetra_Map>           local_node_map = data->getMap();

  const Teuchos::RCP<Adapt::NodalDataVector> node_data =
      this->p_state_mgr_->getStateInfoStruct()
          ->getNodalDataBase()
          ->getNodalDataVector();

  int num_nodes = this->num_nodes_;
  int num_dims  = this->num_dims_;
  int num_pts   = this->num_pts_;

  // deal with weights
  int node_weight_offset;
  int node_weight_ndofs;
  node_data->getNDofsAndOffset(
      this->nodal_weights_name_, node_weight_offset, node_weight_ndofs);
  node_weight_offset -= this->mgr_->ndb_start;
  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int node = 0; node < num_nodes; ++node) {
      const GO global_row = wsElNodeID[cell][node];
      if (!local_node_map->isNodeGlobalElement(global_row)) continue;
      for (int pt = 0; pt < num_pts; ++pt)
        data->sumIntoGlobalValue(
            global_row, node_weight_offset, this->weights_(cell, pt));
    }
  }

  // deal with each of the fields

  for (int field(0); field < this->number_of_fields_; ++field) {
    int node_var_offset;
    int node_var_ndofs;
    node_data->getNDofsAndOffset(
        this->nodal_field_names_[field], node_var_offset, node_var_ndofs);
    node_var_offset -= this->mgr_->ndb_start;
    for (int cell = 0; cell < workset.numCells; ++cell) {
      for (int node = 0; node < num_nodes; ++node) {
        const GO global_row = wsElNodeID[cell][node];
        if (!local_node_map->isNodeGlobalElement(global_row)) continue;
        for (int pt = 0; pt < num_pts; ++pt) {
          if (this->ip_field_layouts_[field] == "Scalar") {
            // save the scalar component
            data->sumIntoGlobalValue(
                global_row,
                node_var_offset,
                this->ip_fields_[field](cell, pt) * this->weights_(cell, pt));
          } else if (this->ip_field_layouts_[field] == "Vector") {
            for (int dim0 = 0; dim0 < num_dims; ++dim0) {
              // save the vector component
              data->sumIntoGlobalValue(
                  global_row,
                  node_var_offset + dim0,
                  (this->ip_fields_[field](cell, pt, dim0) *
                   this->weights_(cell, pt)));
            }
          } else if (this->ip_field_layouts_[field] == "Tensor") {
            for (int dim0 = 0; dim0 < num_dims; ++dim0) {
              for (int dim1 = 0; dim1 < num_dims; ++dim1) {
                // save the tensor component
                PHX::MDField<ScalarT const>& tensor_field =
                    this->ip_fields_[field];
                ScalarT ipval  = tensor_field(cell, pt, dim0, dim1);
                ScalarT weight = this->weights_(cell, pt);
                data->sumIntoGlobalValue(
                    global_row,
                    node_var_offset + dim0 * num_dims + dim1,
                    ipval * weight);
              }
            }
          }
        }
      }
    }  // end cell loop
  }    // end field loop
}

//------------------------------------------------------------------------------
template <typename Traits>
void
IPtoNodalField<PHAL::AlbanyTraits::Residual, Traits>::postEvaluate(
    typename Traits::PostEvalData workset)
{
  const int  ctr     = this->mgr_->incrPostCounter();
  const bool am_last = ctr == this->mgr_->nWorker();
  if (!am_last) return;
  this->mgr_->initCounters();

  // Get the node data vector container.
  Teuchos::RCP<Adapt::NodalDataVector> node_data =
      this->p_state_mgr_->getStateInfoStruct()
          ->getNodalDataBase()
          ->getNodalDataVector();

  // Export the data from the local to overlapped decomposition.
  const Teuchos::RCP<const Tpetra_Import> importer =
      node_data->initializeExport();
  const Teuchos::RCP<const Tpetra_Map> overlap_node_map =
      node_data->getOverlapMap();
  const Teuchos::RCP<Tpetra_MultiVector> data = Teuchos::rcp(
      new Tpetra_MultiVector(overlap_node_map, this->mgr_->ndb_numvecs, true));
  data->doImport(*this->mgr_->nodal_field, *importer, Tpetra::ADD);

  const Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO>> wsElNodeID =
      workset.wsElNodeID;

  const int num_nodes = overlap_node_map->getNodeNumElements();
  const int blocksize = node_data->getVecSize();

  // Get weight info.
  int node_weight_offset;
  int node_weight_ndofs;
  node_data->getNDofsAndOffset(
      this->nodal_weights_name_, node_weight_offset, node_weight_ndofs);
  node_weight_offset -= this->mgr_->ndb_start;

  // Divide the overlap field through by the weights.
  Teuchos::ArrayRCP<const ST> weights = data->getData(node_weight_offset);
  for (int field(0); field < this->number_of_fields_; ++field) {
    int node_var_offset;
    int node_var_ndofs;
    node_data->getNDofsAndOffset(
        this->nodal_field_names_[field], node_var_offset, node_var_ndofs);
    node_var_offset -= this->mgr_->ndb_start;

    for (int k = 0; k < node_var_ndofs; ++k) {
      Teuchos::ArrayRCP<ST> v = data->getDataNonConst(node_var_offset + k);
      for (LO overlap_node = 0; overlap_node < num_nodes; ++overlap_node) {
        v[overlap_node] /= weights[overlap_node];
      }
    }
  }

  // Store the overlapped vector data in stk.
  node_data->saveNodalDataState(data, this->mgr_->ndb_start);
}

//------------------------------------------------------------------------------
template <typename EvalT, typename Traits>
Teuchos::RCP<const Teuchos::ParameterList>
IPtoNodalFieldBase<EvalT, Traits>::getValidIPtoNodalFieldParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
      rcp(new Teuchos::ParameterList("Valid IPtoNodalField Params"));

  validPL->set<std::string>("Name", "", "Name of field Evaluator");
  validPL->set<int>("Number of Fields", 0);
  validPL->set<std::string>("IP Field Name 0", "", "IP Field prefix");
  validPL->set<std::string>(
      "IP Field Layout 0", "", "IP Field Layout: Scalar, Vector, or Tensor");

  validPL->set<std::string>("IP Field Name 1", "", "IP Field prefix");
  validPL->set<std::string>(
      "IP Field Layout 1", "", "IP Field Layout: Scalar, Vector, or Tensor");

  validPL->set<std::string>("IP Field Name 2", "", "IP Field prefix");
  validPL->set<std::string>(
      "IP Field Layout 2", "", "IP Field Layout: Scalar, Vector, or Tensor");

  validPL->set<std::string>("IP Field Name 3", "", "IP Field prefix");
  validPL->set<std::string>(
      "IP Field Layout 3", "", "IP Field Layout: Scalar, Vector, or Tensor");

  validPL->set<std::string>("IP Field Name 4", "", "IP Field prefix");
  validPL->set<std::string>(
      "IP Field Layout 4", "", "IP Field Layout: Scalar, Vector, or Tensor");

  validPL->set<std::string>("IP Field Name 5", "", "IP Field prefix");
  validPL->set<std::string>(
      "IP Field Layout 5", "", "IP Field Layout: Scalar, Vector, or Tensor");

  validPL->set<std::string>("IP Field Name 6", "", "IP Field prefix");
  validPL->set<std::string>(
      "IP Field Layout 6", "", "IP Field Layout: Scalar, Vector, or Tensor");

  validPL->set<std::string>("IP Field Name 7", "", "IP Field prefix");
  validPL->set<std::string>(
      "IP Field Layout 7", "", "IP Field Layout: Scalar, Vector, or Tensor");

  validPL->set<std::string>("IP Field Name 8", "", "IP Field prefix");
  validPL->set<std::string>(
      "IP Field Layout 8", "", "IP Field Layout: Scalar, Vector, or Tensor");

  validPL->set<std::string>("IP Field Name 9", "", "IP Field prefix");
  validPL->set<std::string>(
      "IP Field Layout 9", "", "IP Field Layout: Scalar, Vector, or Tensor");

  validPL->set<bool>(
      "Output to File",
      true,
      "Whether nodal field info should be output to a file");
  validPL->set<bool>(
      "Generate Nodal Values",
      true,
      "Whether values at the nodes should be generated");

  return validPL;
}

//------------------------------------------------------------------------------
}  // namespace LCM

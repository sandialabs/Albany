//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/12/14: no Epetra!

#ifndef ALBANY_DISTRIBUTED_PARAMETER_LIBRARY_TPETRA_HPP
#define ALBANY_DISTRIBUTED_PARAMETER_LIBRARY_TPETRA_HPP

#include "Albany_DistributedParameterLibrary.hpp"

#include "Albany_DataTypes.hpp"

namespace Albany {

  //! Specialization of DistributedParameterTraits for Tpetra_Vector
  template <>
  struct DistributedParameterTraits<Tpetra_Vector, Tpetra_MultiVector, IDArray> {
    typedef Tpetra_Map map_type;
  };

  //! General distributed parameter storing an Tpetra_Vector
  class TpetraDistributedParameter :
    public DistributedParameter<Tpetra_Vector, Tpetra_MultiVector, IDArray> {
  public:

    typedef DistributedParameter<Tpetra_Vector, Tpetra_MultiVector, IDArray> base_type;
    typedef base_type::vector_type vector_type; // Tpetra_Vector
    typedef base_type::multi_vector_type multi_vector_type; // Tpetra_MultiVector
    typedef base_type::map_type map_type;       // Tpetra_Map
    typedef base_type::id_array_type id_array_type;  // IDArray
    typedef std::vector<id_array_type> id_array_vec_type; //vector of IDArray

    //! Constructor
    TpetraDistributedParameter(
      const std::string& param_name_,
      const Teuchos::RCP<Tpetra_Vector>& vec_,
      const Teuchos::RCP<const Tpetra_Map>& owned_map_,
      const Teuchos::RCP<const Tpetra_Map>& overlapped_map_) :
      param_name(param_name_),
      vec(vec_),
      lower_bounds_vec(new Tpetra_Vector(owned_map_, false)),
      upper_bounds_vec(new Tpetra_Vector(owned_map_, false)),
      owned_map(owned_map_),
      overlapped_map(overlapped_map_) {
      importer = Teuchos::rcp(new Tpetra_Import(owned_map, overlapped_map));
      exporter = Teuchos::rcp(new Tpetra_Export(overlapped_map, owned_map));
      overlapped_vec = Teuchos::rcp(new Tpetra_Vector(overlapped_map, false));
    }

    //! Destructor
    virtual ~TpetraDistributedParameter() {}

    //! Get name
    virtual std::string name() const { return param_name; }

    //! Get parallel map
    virtual Teuchos::RCP<const map_type> map() const {
      return owned_map;
    }

    //! Get overlap parallel map
    virtual Teuchos::RCP<const map_type> overlap_map() const {
      return overlapped_map;
    }

    //! Set workset_elem_dofs
    virtual void set_workset_elem_dofs(const Teuchos::RCP<const id_array_vec_type>& ws_elem_dofs_) {
      ws_elem_dofs = ws_elem_dofs_;
    }

    //! Return constant workset_elem_dofs.
    virtual const id_array_vec_type& workset_elem_dofs() const {
      return *ws_elem_dofs;
    }

    //! Get vector
    virtual Teuchos::RCP<vector_type> vector() const {
      return vec;
    }

    //! Get overlapped vector
    virtual Teuchos::RCP<vector_type> overlapped_vector() const {
      return overlapped_vec;
    }

    //! Get lower bounds vector
    virtual Teuchos::RCP<vector_type> lower_bounds_vector() const {
      return lower_bounds_vec;
    }

    //! Get upper bounds vector
    virtual Teuchos::RCP<vector_type> upper_bounds_vector() const {
      return upper_bounds_vec;
    }

    //! Import vector from owned to overlap maps
    virtual void import(multi_vector_type& dst,
                        const multi_vector_type& src) const {
      dst.doImport(src, *importer, Tpetra::INSERT);
    }

    //! Export vector from overlap to owned maps, CombineMode = Add
    virtual void export_add(multi_vector_type& dst,
                            const multi_vector_type& src) const {
      dst.doExport(src, *exporter, Tpetra::ADD);
    }

    //! Fill overlapped vector from owned vector
    virtual void scatter() const {
      overlapped_vec->doImport(*vec, *importer, Tpetra::INSERT);
    }

  protected:

    //! Name of parameter
    std::string param_name;

    //! Tpetra_Vector storing distributed parameter
    Teuchos::RCP<Tpetra_Vector> vec;

    Teuchos::RCP<Tpetra_Vector> lower_bounds_vec;

    Teuchos::RCP<Tpetra_Vector> upper_bounds_vec;

    //! Overlapped Tpetra_Vector storing distributed parameter
    Teuchos::RCP<Tpetra_Vector> overlapped_vec;

    //! Tpetra_Map storing distributed parameter vector's map
    Teuchos::RCP<const Tpetra_Map> owned_map;

    //! Tpetra_Map storing distributed parameter vector's overlap map
    Teuchos::RCP<const Tpetra_Map> overlapped_map;

    //! Importer from owned to overlap maps
    Teuchos::RCP<const Tpetra_Import> importer;

    //! Exporter from overlap to owned maps
    Teuchos::RCP<const Tpetra_Export> exporter;

    //! vector over worksets, containing DOF's map from (elem, node, nComp) into local id
    Teuchos::RCP<const id_array_vec_type> ws_elem_dofs;

  };

}

#endif

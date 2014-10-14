//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
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
  struct DistributedParameterTraits<Tpetra_Vector, Tpetra_MultiVector> {
    typedef Tpetra_Map map_type;
  };

  //! General distributed parameter storing an Tpetra_Vector
  class TpetraDistributedParameter :
    public DistributedParameter<Tpetra_Vector, Tpetra_MultiVector> {
  public:

    typedef DistributedParameter<Tpetra_Vector, Tpetra_MultiVector> base_type;
    typedef base_type::vector_type vector_type; // Tpetra_Vector
    typedef base_type::multi_vector_type multi_vector_type; // Tpetra_MultiVector
    typedef base_type::map_type map_type;       // Tpetra_Map

    //! Constructor
    TpetraDistributedParameter(
      const std::string& param_name_,
      const int num_per_cell_,
      const Teuchos::RCP<Tpetra_Vector>& vec_,
      const Teuchos::RCP<const Tpetra_Map>& owned_map_,
      const Teuchos::RCP<const Tpetra_Map>& overlapped_map_) :
      param_name(param_name_),
      num_per_cell(num_per_cell_),
      vec(vec_),
      owned_map(owned_map_),
      overlapped_map(overlapped_map_) {
      importer = Teuchos::rcp(new Tpetra_Import(overlapped_map, owned_map));
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

    //! Get vector
    virtual Teuchos::RCP<vector_type> vector() const {
      return vec;
    }

    //! Import vector from owned to overlap maps
    virtual void import(multi_vector_type& dst,
                        const multi_vector_type& src) const {
      dst.doImport(src, *importer, Tpetra::INSERT);
    }

    //! Get number of parameter entries per cell
    virtual int num_entries_per_cell() const {
      return num_per_cell;
    }

  protected:

    //! Name of parameter
    std::string param_name;

    //! Number of entries per cell
    int num_per_cell;

    //! Tpetra_Vector storing distributed parameter
    Teuchos::RCP<Tpetra_Vector> vec;

    //! Tpetra_Map storing distributed parameter vector's map
    Teuchos::RCP<const Tpetra_Map> owned_map;

    //! Tpetra_Map storing distributed parameter vector's overlap map
    Teuchos::RCP<const Tpetra_Map> overlapped_map;

    //! Importer from owned to overlap maps
    Teuchos::RCP<const Tpetra_Import> importer;

  };

}

#endif

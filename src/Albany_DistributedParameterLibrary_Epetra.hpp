//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef ALBANY_DISTRIBUTED_PARAMETER_LIBRARY_EPETRA_HPP
#define ALBANY_DISTRIBUTED_PARAMETER_LIBRARY_EPETRA_HPP

#include "Albany_DistributedParameterLibrary.hpp"

#include "Epetra_Map.h"
#include "Epetra_MultiVector.h"
#include "Epetra_Import.h"

namespace Albany {

  //! Specialization of DistributedParameterTraits for Epetra_Vector
  template <>
  struct DistributedParameterTraits<Epetra_Vector, Epetra_MultiVector> {
    typedef Epetra_Map map_type;
  };

  //! General distributed parameter storing an Epetra_Vector
  class EpetraDistributedParameter :
    public DistributedParameter<Epetra_Vector, Epetra_MultiVector> {
  public:

    typedef DistributedParameter<Epetra_Vector, Epetra_MultiVector> base_type;
    typedef base_type::vector_type vector_type; // Epetra_Vector
    typedef base_type::multi_vector_type multi_vector_type; // Epetra_MultiVector
    typedef base_type::map_type map_type;       // Epetra_Map

    //! Constructor
    EpetraDistributedParameter(
      const std::string& param_name_,
      const int num_per_cell_,
      const Teuchos::RCP<Epetra_Vector>& vec_,
      const Teuchos::RCP<const Epetra_Map>& owned_map_,
      const Teuchos::RCP<const Epetra_Map>& overlapoed_map_) :
      param_name(param_name_),
      num_per_cell(num_per_cell_),
      vec(vec_),
      owned_map(owned_map_),
      overlapped_map(overlapoed_map_) {
      importer = Teuchos::rcp(new Epetra_Import(*overlapped_map, *owned_map));
    }

    //! Destructor
    virtual ~EpetraDistributedParameter() {}

    //! Get name
    virtual std::string name() const { return param_name; }

    //! Get parallel map
    virtual Teuchos::RCP<const map_type> map() const {
      return owned_map;
    }

    //! Get overlap parallel map
    virtual Teuchos::RCP<const map_type> overlap_map() {
      return overlapped_map;
    }

    //! Get vector
    virtual Teuchos::RCP<vector_type> vector() const {
      return vec;
    }

    //! Import vector from owned to overlap maps
    virtual void import(multi_vector_type& dst,
                        const multi_vector_type& src) const {
      dst.Import(src, *importer, Insert);
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

    //! Epetra_Vector storing distributed parameter
    Teuchos::RCP<Epetra_Vector> vec;

    //! Epetra_Map storing distributed parameter vector's map
    Teuchos::RCP<const Epetra_Map> owned_map;

    //! Epetra_Map storing distributed parameter vector's overlap map
    Teuchos::RCP<const Epetra_Map> overlapped_map;

    //! Importer from owned to overlap maps
    Teuchos::RCP<const Epetra_Import> importer;

  };

}

#endif

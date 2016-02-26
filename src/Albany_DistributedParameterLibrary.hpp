//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/12/14: no Epetra!

#ifndef ALBANY_DISTRIBUTED_PARAMETER_LIBRARY_HPP
#define ALBANY_DISTRIBUTED_PARAMETER_LIBRARY_HPP

#include <map>
#include <string>

#include "Teuchos_RCP.hpp"
#include "Teuchos_TestForException.hpp"

namespace Albany {

  //! Traits class for extracting information out of a distributed parameter
  /*!
   * Should be specialized for a concrete vector implementation.
   */
  template <typename Vector, typename MultiVector, typename IDArray>
  struct DistributedParameterTraits {};

  //! Abstract base class for storing distributed parameters
  template <typename Vector, typename MultiVector, typename IDArray>
  class DistributedParameter {
  public:

    //! Vector type
    typedef Vector vector_type;

    //! Multi-vector type
    typedef MultiVector multi_vector_type;

    //! Id Array type
    typedef IDArray id_array_type;

    //! Parallel map type
    typedef typename DistributedParameterTraits<vector_type, multi_vector_type,id_array_type>::map_type map_type;

    //! Constructor
    DistributedParameter() {}

    //! Destructor
    virtual ~DistributedParameter() {}

    //! Get name
    virtual std::string name() const = 0;

    //! Set workset_elem_dofs map
    virtual void set_workset_elem_dofs(const Teuchos::RCP<const std::vector<id_array_type> >& ws_elem_dofs_) = 0;

    //! Return constant workset_elem_dofs. For each workset, workset_elem_dofs maps (elem, node, nComp) into local id
    virtual const std::vector<id_array_type>& workset_elem_dofs() const = 0;

    //! Get parallel map
    virtual Teuchos::RCP<const map_type> map() const = 0;

    //! Get overlap parallel map
    virtual Teuchos::RCP<const map_type> overlap_map() const = 0;

    //! Get vector
    virtual Teuchos::RCP<vector_type> vector() const = 0;

    //! Get lower bounds vector
    virtual Teuchos::RCP<vector_type> lower_bounds_vector() const = 0;

    //! Get upper bounds vector
    virtual Teuchos::RCP<vector_type> upper_bounds_vector() const = 0;

    //! Get overlapped vector
    virtual Teuchos::RCP<vector_type> overlapped_vector() const = 0;

    //! Fill overlapped vector from owned vector
    virtual void scatter() const = 0;

    //! Import vector from owned to overlap maps
    virtual void import(multi_vector_type& dst,
                        const multi_vector_type& src) const = 0;

    //! Export vector from overlap to owned maps, CombineMode = Add
    virtual void export_add(multi_vector_type& dst,
                            const multi_vector_type& src) const =0;
  };

  template <typename Vector, typename MultiVector, typename IDArray>
  class DistributedParameterLibrary {

    typedef const DistributedParameter<Vector,MultiVector,IDArray> param_type;
    typedef std::map< std::string, Teuchos::RCP<param_type> > param_map_type;

  public:

    typedef typename param_map_type::iterator iterator;
    typedef typename param_map_type::const_iterator const_iterator;

    //! Constructor
    DistributedParameterLibrary() : param_map() {}

    //! Destructor
    ~DistributedParameterLibrary() {}

    //! Number of parameters in the library
    size_t size() const { return param_map.size(); }

    //! Add parameter to library
    void add(const std::string& name,
             const Teuchos::RCP<param_type>& param) {
      param_map[name] = param;
    }

    //! Get parameter from library
    Teuchos::RCP<param_type> get(const std::string& name) const {
      const_iterator i = param_map.find(name);
      TEUCHOS_TEST_FOR_EXCEPTION(
        i == param_map.end(), std::logic_error,
        "Parameter " << name << " is not in the library");
      return i->second;
    }

    //! Return if library has parameter
    bool has(const std::string& name) const {
      return param_map.find(name) != param_map.end();
    }

    //! Loop through the stored parameters and scatter each of them
    void scatter() const
    {
      const_iterator it = param_map.begin();
      while(it != param_map.end())
        (it++)->second->scatter();
    }

    //! Iterator pointing at beginning of library
    iterator begin() { return param_map.begin(); }

    //! Iterator pointing at beginning of library
    const_iterator begin() const { return param_map.begin(); }

    //! Iterator pointing at end of library
    iterator end() { return param_map.end(); }

    //! Iterator pointing at end of library
    const_iterator end() const { return param_map.end(); }

  protected:

    //! Map between parameter name and parameter object
    param_map_type param_map;

  };

}

#endif

//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
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
  template <typename Vector, typename MultiVector>
  struct DistributedParameterTraits {};

  //! Abstract base class for storing distributed parameters
  template <typename Vector, typename MultiVector>
  class DistributedParameter {
  public:

    //! Vector type
    typedef Vector vector_type;

    //! Multi-vector type
    typedef MultiVector multi_vector_type;

    //! Parallel map type
    typedef typename DistributedParameterTraits<vector_type, multi_vector_type>::map_type map_type;

    //! Constructor
    DistributedParameter() {}

    //! Destructor
    virtual ~DistributedParameter() {}

    //! Get name
    virtual std::string name() const = 0;

    //! Get parallel map
    virtual Teuchos::RCP<const map_type> map() const = 0;

    //! Get overlap parallel map
    virtual Teuchos::RCP<const map_type> overlap_map() const = 0;

    //! Get vector
    virtual Teuchos::RCP<vector_type> vector() const = 0;

    //! Import vector from owned to overlap maps
    virtual void import(multi_vector_type& dst,
                        const multi_vector_type& src) const = 0;

    //! Get number of parameter entries per cell
    virtual int num_entries_per_cell() const = 0;

  };

  template <typename Vector, typename MultiVector>
  class DistributedParameterLibrary {

    typedef const DistributedParameter<Vector,MultiVector> param_type;
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
      typename param_map_type::const_iterator i = param_map.find(name);
      TEUCHOS_TEST_FOR_EXCEPTION(
        i == param_map.end(), std::logic_error,
        "Parameter " << name << " is not in the library");
      return i->second;
    }

    //! Return if library has parameter
    bool has(const std::string& name) const {
      return param_map.find(name) != param_map.end();
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

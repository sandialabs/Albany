//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_DISTRIBUTED_PARAMETER_LIBRARY_HPP
#define ALBANY_DISTRIBUTED_PARAMETER_LIBRARY_HPP

#include <map>

#include "Teuchos_RCP.hpp"
#include "Teuchos_TestForException.hpp"

#include "Albany_DistributedParameter.hpp"

namespace Albany {

class DistributedParameterLibrary {
public:

  typedef const DistributedParameter                        param_type;
  typedef std::map<std::string, Teuchos::RCP<param_type>>   param_map_type;

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

  //! Loop through the stored parameters and combine each of them
  void combine() const
  {
    const_iterator it = param_map.begin();
    while(it != param_map.end())
      (it++)->second->combine();
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

} // namespace Albany

#endif // ALBANY_DISTRIBUTED_PARAMETER_LIBRARY_HPP

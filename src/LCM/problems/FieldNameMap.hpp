//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_FieldNameMap_hpp)
#define LCM_FieldNameMap_hpp

#include <Teuchos_RCP.hpp>

namespace LCM {

  class FieldNameMap
  {
  public:
    ///
    /// Constructor
    ///
    FieldNameMap(bool surface_flag);

    ///
    /// Destructor
    ///
    virtual ~FieldNameMap();

    ///
    /// Return the map
    ///
    Teuchos::RCP<std::map<std::string, std::string>>
    getMap() { return field_name_map_; }

  private:
    ///
    /// Private and unimplemented
    ///
    FieldNameMap();

    ///
    /// Private to prohibit copying
    ///
    FieldNameMap(const FieldNameMap&);

    ///
    /// Private to prohibit copying
    ///
    FieldNameMap& operator=(const FieldNameMap&);

    ///
    /// Map data variable
    ///
    Teuchos::RCP<std::map<std::string, std::string>> field_name_map_;
  };
}
#endif

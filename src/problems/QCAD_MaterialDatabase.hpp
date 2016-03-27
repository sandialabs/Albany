//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef QCAD_MATERIALDATABASE_HPP
#define QCAD_MATERIALDATABASE_HPP

#include "Teuchos_ParameterList.hpp"
#include "Albany_Utils.hpp"

namespace QCAD {

  /*!
   * \brief Centralized collection of material parameters
   */
  class MaterialDatabase
  {
  public:
  
    //! Default constructor
    MaterialDatabase(const std::string& inputFile,
                     Teuchos::RCP<const Teuchos::Comm<int> >& tcomm);  

    //! Destructor
    ~MaterialDatabase();

    //! Get a parameter / check existence
    bool isParam(const std::string& paramName);
    template<typename T>
    T getParam(const std::string& paramName);
    template<typename T>
    T getParam(const std::string& paramName, T def_val);

    //! Get a parameter for a particular material
    bool isMaterialParam(const std::string& materialName, const std::string& paramName);
    template<typename T>
    T getMaterialParam(const std::string& materialName, const std::string& paramName);
    template<typename T>
    T getMaterialParam(const std::string& materialName, const std::string& paramName, T def_val);


    //! Get a parameter for a particular node set
    bool isNodeSetParam(const std::string& materialName, const std::string& paramName);
    template<typename T>
    T getNodeSetParam(const std::string& materialName, const std::string& paramName);
    template<typename T>
    T getNodeSetParam(const std::string& materialName, const std::string& paramName, T def_val);

    //! Get a parameter for a particular side set
    bool isSideSetParam(const std::string& materialName, const std::string& paramName);
    template<typename T>
    T getSideSetParam(const std::string& materialName, const std::string& paramName);
    template<typename T>
    T getSideSetParam(const std::string& materialName, const std::string& paramName, T def_val);

    //! Get a parameter for a particular element block (or assoc. material if paramName is not in element bloc)
    bool isElementBlockParam(const std::string& ebName, const std::string& paramName);
    template<typename T>
    T getElementBlockParam(const std::string& ebName, const std::string& paramName);
    template<typename T>
    T getElementBlockParam(const std::string& ebName, const std::string& paramName, T def_val);

    //! Get a sublist from a particular element block
    bool isElementBlockSublist(const std::string& ebName, const std::string& subListName);
    Teuchos::ParameterList&
    getElementBlockSublist(const std::string& ebName, const std::string& subListName);

    //! Get a vector of the value of all parameters in the entire list with name == paramName
    template<typename T> 
    std::vector<T> getAllMatchingParams(const std::string& paramName);

  private:
    template<typename T> 
    void getAllMatchingParams_helper(const std::string& paramName, 
				     std::vector<T>& results, Teuchos::ParameterList& pList);

    std::string translateDBSublistName(Teuchos::ParameterList*, const std::string&);

    

  private:

    //! Private to prohibit copying
    MaterialDatabase(const MaterialDatabase&);
    
    //! Private to prohibit copying
    MaterialDatabase& operator=(const MaterialDatabase&);

    //! Encapsulated parameter list which holds all the data
    Teuchos::ParameterList data_;
    Teuchos::ParameterList* pMaterialsList_;
    Teuchos::ParameterList* pEBList_;
    Teuchos::ParameterList* pNSList_;
    Teuchos::ParameterList* pSSList_;
  };

}

#endif

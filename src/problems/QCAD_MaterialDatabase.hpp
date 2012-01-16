/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


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
		     const Teuchos::RCP<const Epetra_Comm>& ecomm);

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
  };

}

#endif

//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef ALBANY_ABSTRACTFIELDCONT_HPP
#define ALBANY_ABSTRACTFIELDCONT_HPP

namespace Albany {

/*!
 * \brief Abstract interface for a field container
 *
 */
class AbstractFieldContainer {

  public:

    typedef std::vector<std::string> FieldContainerRequirements;

    //! Destructor
    virtual ~AbstractFieldContainer() {};

  protected:

};


}

#endif // ALBANY_ABSTRACTFIELDCONT_HPP

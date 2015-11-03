//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/12/14: no Epetra!

#ifndef PHAL_TYPE_KEY_MAP_HPP
#define PHAL_TYPE_KEY_MAP_HPP

#include "Teuchos_Array.hpp"
#include "Teuchos_any.hpp"

#include "boost/mpl/vector.hpp"
#include "boost/mpl/pair.hpp"
#include "boost/mpl/find_if.hpp"
#include "boost/mpl/size.hpp"
#include "boost/mpl/transform.hpp"
#include "boost/mpl/back_inserter.hpp"
#include "Sacado_mpl_placeholders.hpp"
#include "boost/type_traits.hpp"

namespace PHAL {

  //! Container for storing objects indexed by types
  /*!
   * This class provides a generic container class for storing objects
   * indexed by a type.  It's single template parameter should be an mpl::map
   * mapping the type index to the type of the object, which can then be 
   * retrieved from the container.
   *
   * Teuchos::any is used to store the objects, and thus they should have
   * value semantics.
   */
  template <typename TypeMap>
  class TypeKeyMap {

    public:

    //! Meta-function for getting the object type indexed by T
    template <typename T>
    struct GetObjectTypeAndPos {
      typedef typename Sacado::mpl::find_if<
	TypeMap, 
	boost::is_same<Sacado::mpl::first<Sacado::mpl::placeholders::_1>, T >
	>::type Iter;
      typedef typename Sacado::mpl::deref<Iter>::type Pair;
      typedef typename Sacado::mpl::second<Pair>::type type;
      typedef typename Iter::pos pos;
      static const int value = pos::value;
    };

    //! Meta-function for getting the object reference type indexed by T
    template <typename T>
    struct GetObjectRefType {
      typedef typename GetObjectTypeAndPos<T>::type object_type;
      typedef typename boost::add_reference<object_type>::type type;
    };

    //! Meta-function for getting the object const-reference type indexed by T
    template <typename T>
    struct GetObjectConstRefType {
      typedef typename GetObjectTypeAndPos<T>::type object_type;
      typedef typename boost::add_const<object_type>::type const_type;
      typedef typename boost::add_reference<const_type>::type type;
    };

    //! Typedef of container used
    typedef Teuchos::Array<Teuchos::any> container;

    //! Typedef for iterator
    typedef typename container::iterator iterator;

    //! Typedef for const_iterator
    typedef typename container::const_iterator const_iterator;

    //! Default constructor
    TypeKeyMap() : objects(Sacado::mpl::size<TypeMap>::value) {}

    //! Destructor
    ~TypeKeyMap() {}

    //! Get object indexed by T
    template<typename T> typename GetObjectRefType<T>::type 
    getValue() {
      typedef typename GetObjectTypeAndPos<T>::type type;
      const int pos = GetObjectTypeAndPos<T>::value;
      return Teuchos::any_cast<type>(objects[pos]);
    }

    //! Get object indexed by T
    template<typename T> typename GetObjectConstRefType<T>::type 
    getValue() const {
      typedef typename GetObjectTypeAndPos<T>::type type;
      const int pos = GetObjectTypeAndPos<T>::value;
      return Teuchos::any_cast<type>(objects[pos]);
    }

    //! Set object indexed by T
    template <typename T> 
    void setValue(typename GetObjectConstRefType<T>::type x) {
      const int pos = GetObjectTypeAndPos<T>::value;
      objects[pos] = x;
    }

    //! Return an iterator that points to the first object
    iterator begin() { return iterator(objects.begin()); }

    //! Return an iterator that points to the first object
    const_iterator begin() const { return const_iterator(objects.begin()); }

    //! Return an iterator that points one past the last object
    iterator end() { return iterator(objects.end()); }

    //! Return an iterator that points one past the last object
    const_iterator end() const { return const_iterator(objects.end()); }

  private:

    //! Stores array objects of each type
    container objects;

  };

  //! Zip a sequence of keys and elements into a sequence of pair<key,element>
  template <typename KeySeq, typename ElemSeq> 
  struct ZipMap {
    
    // Create pair<T1,T2>
    struct CreatePair {
      template <typename T1, typename T2> struct apply {
	typedef Sacado::mpl::pair<T1,T2> type;
      };
    };

    typedef typename 
    Sacado::mpl::transform< 
      KeySeq, 
      ElemSeq, 
      CreatePair,
      Sacado::mpl::back_inserter< Sacado::mpl::vector<> > >::type type;
  };

  /*! 
   * \brief Create map needed for TypeKeyMap from a key sequence and 
   * a lambda expression (meta-function class or placeholder expression).
   */
  template <typename KeySeq, typename F> 
  struct CreateLambdaKeyMap {
    typedef typename Sacado::mpl::transform<KeySeq, F>::type ElemSeq;
    typedef typename ZipMap<KeySeq,ElemSeq>::type type;
  };

}

#endif

#ifndef ALBANY_META_UTILS_HPP
#define ALBANY_META_UTILS_HPP

#include <tuple>
#include <type_traits>

// Wrote by Luca Bertagna

namespace Albany
{

// A list of types is just an alias for tuple
template<typename... Ts>
using TypeList = std::tuple<Ts...>;

template<typename TL>
struct type_list_size : std::tuple_size<TL> {};

// Concatenate two type lists
template<typename TL1, typename TL2>
struct CatTypeLists;

template<typename... T1s, typename... T2s>
struct CatTypeLists<TypeList<T1s...>,TypeList<T2s...>> {
  using type = TypeList<T1s...,T2s...>;
};

template<typename TL1, typename TL2>
using type_list_cat = typename CatTypeLists<TL1,TL2>::type;

// Find position of first occurrence of type T in a TypeList, store in 'pos'.
// If not found, set pos = -1.
template<typename T, typename TypeList>
struct FirstOf;

template<typename T, typename H>
struct FirstOf<T,TypeList<H>> {
  static constexpr int pos = std::is_same<H,T>::value ? 0 : -1;
};

template<typename T, typename H, typename... Ts>
struct FirstOf<T,TypeList<H,Ts...>> {
private:
  static constexpr int tail_pos = FirstOf<T,TypeList<Ts...>>::pos;
public:
  static constexpr int pos = std::is_same<H,T>::value ? 0 : (tail_pos>=0 ? 1+tail_pos : -1);
};

// Check if a TypeList contains unique types
template<typename TL>
struct is_type_list_unique;

template<typename T>
struct is_type_list_unique<TypeList<T>> : std::true_type {};

template<typename T, typename... Ts>
struct is_type_list_unique<TypeList<T,Ts...>> {
  static constexpr bool value = FirstOf<T,TypeList<Ts...>>::pos==-1 &&
                                is_type_list_unique<TypeList<Ts...>>::value;
};

// Access N-th entry of a type list
template<typename TL, int N>
using type_list_get = typename std::tuple_element<N,TL>::type;

// Iterate over a TypeList
// Requires a generic lambda that only accept an instance of the
// type list types. Can only use input to deduce its type.
//  auto l = [&](auto t) {
//    using T = decltype(t);
//    call_some_func<T>(...);
//  };
template<typename TL>
struct TypeListFor;

template<typename T>
struct TypeListFor<TypeList<T>> {
  template<typename Lambda>
  constexpr TypeListFor (Lambda&& l) {
    l(T());
  }
};

template<typename T, typename...Ts>
struct TypeListFor<TypeList<T,Ts...>> {
  template<typename Lambda>
  constexpr TypeListFor (Lambda&& l) {
    l(T());
    TypeListFor<TypeList<Ts...>> tlf(l);
  }
};

// A map of types. Much like a tuple, but can be accessed with a type,
// rather than an int.
template<typename Keys, typename Vals>
struct TypeMap;

template<typename... Ks, typename... Vs>
struct TypeMap<TypeList<Ks...>,TypeList<Vs...>> : public std::tuple<Vs...> {
  // Helper type. Since it's not exposed, there's no risk it can clash with
  // an actual entry of the keys or vals type lists (unlike something like 'void',
  // which could potentially be used by the user).
  struct NotFound {};
private:

  template<typename T>
  using base_t = typename std::remove_cv<typename std::remove_reference<T>::type>::type;

  template<typename V1, typename V2>
  static typename std::enable_if<std::is_same<base_t<V1>,base_t<V2>>::value,bool>::type
  check_eq (const V1& v1, const V2& v2) {
    return v1==v2;
  }
  template<typename V1, typename V2>
  static typename std::enable_if<!std::is_same<base_t<V1>,base_t<V2>>::value,bool>::type
  check_eq (const V1& /* v1 */, const V2& /* v2 */) {
    return false;
  }
public:

  using keys = TypeList<Ks...>;
  using vals = TypeList<Vs...>;
  using self = TypeMap<keys,vals>;
  static constexpr int size = type_list_size<keys>::value;
  static_assert (size>0,
                 "Error! TypeMap instantiated with a 0-length keys TypeList.\n");
  static_assert (size==type_list_size<vals>::value,
                 "Error! Keys and values have different sizes.\n");
  static_assert (is_type_list_unique<keys>::value, "Error! Keys are not unique.\n");

  template<typename K>
  struct AtHelper {
    static constexpr int pos = FirstOf<K,keys>::pos;
  private:
    // The second arg to std::conditional must be instantiated. If pos<0,
    // type_list_get does not compile. Hence, if pos<0, use 0, which will for
    // sure work (TypeList has size>=1). The corresponding type will not
    // be exposed anyways. It's just to get a valid instantiation of
    // std::conditional.
    struct GetSafe {
      static constexpr int safe_pos = pos>=0 ? pos : 0;
      using type = type_list_get<vals,safe_pos>;
    };
  public:
    using type = typename std::conditional<(pos<0),NotFound,typename GetSafe::type>::type;
  };

  template<typename K>
  static constexpr bool has_t () {
    return not std::is_same<at_t<K>,NotFound>::value;
  }

  template<typename V>
  bool has_v (const V& v) const {
    return has_v_impl<(FirstOf<V,vals>::pos>=0)>(v,*this);
  }


  template<typename K>
  using at_t = typename AtHelper<K>::type;

  template<typename K>
  at_t<K>& at () {
    static_assert (has_t<K>(), "Error! Type not found in this TypeMap's keys.\n");
    return std::get<AtHelper<K>::pos>(*this);
  }

  template<typename K>
  const at_t<K>& at () const {
    static_assert (has_t<K>(), "Error! Type not found in this TypeMap's keys.\n");
    return std::get<AtHelper<K>::pos>(*this);
  }

private:

  template<bool has_V_type, typename V>
  typename std::enable_if<has_V_type,bool>::type
  has_v_impl (const V& v, const self& map) const {
    bool found = false;
    TypeListFor<keys>([&](auto t){
      using key_t = decltype(t);

      const auto& value = map.template at<key_t>();
      if (check_eq(v,value)) {
        found = true;
        return;
      }
    });
    return found;
  }

  template<bool has_V_type, typename V>
  typename std::enable_if<!has_V_type,bool>::type
  has_v_impl (const V& /*v*/, const self& /*map*/) const {
    return false;
  }
};

// Given a class template C, and a type list with template args T1,...,Tn,
// create a type list containing the C<T1>,...,C<Tn> types. E.g., with
//
//   template<typename T> using vec = std::vector<T>;
//   using types = TypeList<int,float,double>;
//   using vec_types = ApplyTemplate<vec,types>::type;
//
// Then vec_type is TypeList<std::vector<int>,std::vector<float>,std::vector<double>>;
template<template<typename> class C, typename TL>
struct ApplyTemplate;

template<template<typename> class C, typename T>
struct ApplyTemplate<C,TypeList<T>> {
  using type = TypeList<C<T>>;
};

template<template<typename> class C, typename T, typename... Ts>
struct ApplyTemplate<C,TypeList<T,Ts...>> {
private:
  using head = TypeList<C<T>>;
  using tail = typename ApplyTemplate<C,TypeList<Ts...>>::type;
public:
  using type = type_list_cat<head,tail>;
};

} // namespace Albany

#endif // ALBANY_META_UTILS_HPP
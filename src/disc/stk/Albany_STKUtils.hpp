#ifndef ALBANY_STK_UTILS_HPP
#define ALBANY_STK_UTILS_HPP

#include <stk_mesh/base/Field.hpp>

namespace Albany
{

// Counts the number of non-void tags.
// WARNING: this assumes that as soon as a void is found,
//          the rest can be ignored (i.e., no <T,void,T,..>).
//          This is fine, cause that's how Field works.
template<typename Tag, typename... Tags>
struct NumNonVoid {
  enum { n = (std::is_same<Tag,void>::value
               ? 0
               : 1 + NumNonVoid<Tags...>::n) };
};

template<typename Tag>
struct NumNonVoid<Tag> {
  enum { n = (std::is_same<Tag,void>::value
               ? 0 : 1 ) };
};

// Get the Rank of a stk Field
template<typename F>
struct FieldRank;

template<typename Scalar, typename Tag1, typename Tag2, typename Tag3,
         typename Tag4,   typename Tag5, typename Tag6, typename Tag7>
struct FieldRank<stk::mesh::Field<Scalar,Tag1,Tag2,Tag3,Tag4,Tag5,Tag6,Tag7>> {
  enum {
    n = NumNonVoid<Tag1,Tag2,Tag3,Tag4,Tag5,Tag6,Tag7>::n
  };
};

// Get the scalar type of a stk field
template<typename F>
struct FieldScalar;

template<typename Scalar , typename Tag1 , typename Tag2 , typename Tag3 , typename Tag4 ,
         typename Tag5 , typename Tag6 , typename Tag7 >
struct FieldScalar<stk::mesh::Field<Scalar,Tag1,Tag2,Tag3,Tag4,Tag5,Tag6,Tag7>> {
  using type = Scalar;
};


} // namespace Albany

#endif // ALBANY_STK_UTILS_HPP

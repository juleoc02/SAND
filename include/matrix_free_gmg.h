#ifndef MATRIX_FREE_GMG_H
#define MATRIX_FREE_GMG_H

#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/sparse_direct.h>

using namespace dealii;

template <int dim, int fe_degree, typename number>
class ElasticityOperator
  : public MatrixFreeOperators::
      Base<dim, LinearAlgebra::distributed::Vector<number>>
{
public:

  using value_type = number;

  ElasticityOperator();

  void clear() override;

  void evaluate_coefficient(const Coefficient<dim> &coefficient_function);

  virtual void compute_diagonal() override;

private:

  virtual void apply_add(
    LinearAlgebra::distributed::Vector<number> &      dst,
    const LinearAlgebra::distributed::Vector<number> &src) const override;

  void
  local_apply(const MatrixFree<dim, number> &                   data,
              LinearAlgebra::distributed::Vector<number> &      dst,
              const LinearAlgebra::distributed::Vector<number> &src,
              const std::pair<unsigned int, unsigned int> &cell_range) const;

  void local_compute_diagonal(
    const MatrixFree<dim, number> &              data,
    LinearAlgebra::distributed::Vector<number> & dst,
    const unsigned int &                         dummy,
    const std::pair<unsigned int, unsigned int> &cell_range) const;

  Table<2, VectorizedArray<number>> coefficient;
};



#endif // MATRIX_FREE_GMG_H

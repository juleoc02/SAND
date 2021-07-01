//
// Created by justin on 5/13/21.
//

#ifndef SAND_DENSITY_FILTER_H
#define SAND_DENSITY_FILTER_H
#ifndef SAND_KKTSYSTEM_H
#define SAND_KKTSYSTEM_H


#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <iostream>
#include <fstream>
#include <algorithm>

using namespace dealii;

template <int dim>
class DensityFilter
{
    DensityFilter();
public:
    SparsityPattern get_filter_sparsity_pattern();
    SparseMatrix<double> filter_matrix;
    void initialize(Triangulation &triangulation);
    SparsityPattern fill_filter_matrix(Triangulation &triangulation);
private:
    SparsityPattern filter_sparsity_pattern;
    std::set<typename Triangulation<dim>::cell_iterator>
    std::set<typename Triangulation<dim>::cell_iterator> find_relevant_neighbors(Triangulation<dim> &triangulation,
    typename Triangulation<dim>::cell_iterator cell) const


};

#endif //SAND_DENSITY_FILTER_H
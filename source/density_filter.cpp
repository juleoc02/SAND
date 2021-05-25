//
// Created by justin on 5/13/21.
//
#include "../include/density_filter.h"

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

namespace SAND {
    using namespace dealii;
    template<int dim>
    DensityFilter<dim>::DensityFilter(triangulation)
            :
            filter_r(.251)
    {
    }

    template<int dim>
    void
    DensityFilter<dim>::initialize(Triangulation &triangulation)
    {
        DynamicSparsityPattern filter_dsp;
        filter_dsp.reinit(triangulation.n_active_cells(),
                          dtriangulation.n_active_cells());

        std::set<unsigned int> neighbor_ids;
        std::set<typename Triangulation<dim>::cell_iterator> cells_to_check;
        std::set<typename Triangulation<dim>::cell_iterator> cells_to_check_temp;
        double distance;

        /*finds neighbors-of-neighbors until it is out to specified radius*/
        for (const auto &cell : dof_handler.active_cell_iterators())
        {
            const unsigned int i = cell->active_cell_index();
            for(const auto &neighbor_cell : find_relevant_neighbors(triangulation, cell))
            {
                const unsigned int j = neighbor_cell->active_cell_index();
                filter_dsp.add(i, j);
            }
        }

        filter_sparsity_pattern.copy_from(filter_dsp);
        filter_matrix.reinit(filter_sparsity_pattern);

        for (const auto &cell : dof_handler.active_cell_iterators())
        {
            const unsigned int i = cell->active_cell_index();
            for(const auto &neighbor_cell : find_relevant_neighbors(triangulation, cell))
            {
                const unsigned int j = neighbor_cell->active_cell_index();
                /*value should be max radius - distance between cells*/
                filter_matrix.add(i, j, filter_r - distance);
        }

            //here we normalize the filter
        for (const auto &cell : dof_handler.active_cell_iterators()) {
            const unsigned int i = cell->active_cell_index();
            double denominator = 0;
            typename SparseMatrix<double>::iterator iter = filter_matrix.begin(
                    i);
            for (; iter != filter_matrix.end(i); iter++) {
                denominator = denominator + iter->value();
            }
            iter = filter_matrix.begin(i);
            for (; iter != filter_matrix.end(i); iter++) {
                iter->value() = iter->value() / denominator;
            }
        }

    }

    template<int dim>
    std::set<typename Triangulation<dim>::cell_iterator>
    DensityFilter<dim>::find_relevant_neighbors(Triangulation<dim> &triangulation,
    typename Triangulation<dim>::cell_iterator cell) const
    {
      std::set<unsigned int>                               neighbor_ids;
      std::set<typename Triangulation<dim>::cell_iterator> cells_to_check;
      neighbor_ids.insert(cell->active_cell_index());
      cells_to_check.insert(cell);
      bool new_neighbors_found;
      do
        {
          new_neighbors_found = false;
          for (const auto &check_cell :
               std::vector<typename Triangulation<dim>::cell_iterator>(
                 cells_to_check.begin(), cells_to_check.end()))
            {
              for (const auto n : check_cell->face_indices())
                {
                  if (!(check_cell->face(n)->at_boundary()))
                    {
                      const auto & neighbor = check_cell->neighbor(n);
                      const double distance =
                        cell->center().distance(neighbor->center());
                      if ((distance < filter_r) &&
                          !(neighbor_ids.count(neighbor->active_cell_index())))
                        {
                          cells_to_check.insert(neighbor);
                          neighbor_ids.insert(neighbor->active_cell_index());
                          new_neighbors_found = true;
                        }
                    }
                }
            }
        }
      while (new_neighbors_found);
      return cells_to_check;
    }




}//SAND namespace
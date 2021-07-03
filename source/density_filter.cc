//
// Created by justin on 5/13/21.
//
#include "../include/density_filter.h"

#include <deal.II/base/tensor.h>

#include <deal.II/lac/block_sparse_matrix.h>

#include <deal.II/lac/precondition.h>


#include <deal.II/grid/tria.h>

#include <deal.II/grid/grid_refinement.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

namespace SAND {
    using namespace dealii;

    template<int dim>
    DensityFilter<dim>::DensityFilter(const double filter_r_in)
            :
            filter_r(filter_r_in)
    {
    }

    template<int dim>
    void
    DensityFilter<dim>::initialize(Triangulation<dim> &triangulation) {
        DynamicSparsityPattern filter_dsp;
        filter_dsp.reinit(triangulation.n_active_cells(),
                          triangulation.n_active_cells());

        std::set<unsigned int> neighbor_ids;
        std::set<typename Triangulation<dim>::cell_iterator> cells_to_check;
        std::set<typename Triangulation<dim>::cell_iterator> cells_to_check_temp;
        double distance;

        /*finds neighbors-of-neighbors until it is out to specified radius*/
        for (const auto &cell : triangulation.active_cell_iterators()) {
            const unsigned int i = cell->active_cell_index();
            for (const auto &neighbor_cell : find_relevant_neighbors(triangulation, cell)) {
                const unsigned int j = neighbor_cell->active_cell_index();
                filter_dsp.add(i, j);
            }
        }

        filter_sparsity_pattern.copy_from(filter_dsp);
        filter_matrix.reinit(filter_sparsity_pattern);

        for (const auto &cell : triangulation.active_cell_iterators())
        {
            const unsigned int i = cell->active_cell_index();
            for (const auto &neighbor_cell : find_relevant_neighbors(triangulation, cell)) {
                const unsigned int j = neighbor_cell->active_cell_index();
                const double distance =
                        cell->center().distance(neighbor_cell->center());
                /*value should be max radius - distance between cells*/
                double value = filter_r - distance;
                std::cout << "i,j:   " << i << "  " << j << std::endl;
                std::cout << "filter m value:   " << value << std::endl;
                filter_matrix.add(i, j, value);
            }
        }

        //here we normalize the filter
        for (const auto &cell : triangulation.active_cell_iterators())
        {
            const unsigned int i = cell->active_cell_index();
            double denominator = 0;
            typename SparseMatrix<double>::iterator iter = filter_matrix.begin(
                    i);
            for (; iter != filter_matrix.end(i); iter++)
            {
                denominator = denominator + iter->value();
            }
            iter = filter_matrix.begin(i);
            for (; iter != filter_matrix.end(i); iter++)
            {
                iter->value() = iter->value() / denominator;
                std::cout << iter->value() << std::endl;
            }
        }
    }

    template<int dim>
    std::set<typename Triangulation<dim>::cell_iterator>
    DensityFilter<dim>::find_relevant_neighbors(Triangulation<dim> &triangulation,
                                                typename Triangulation<dim>::cell_iterator cell) const {
        std::set<unsigned int> neighbor_ids;
        std::set<typename Triangulation<dim>::cell_iterator> cells_to_check;
        neighbor_ids.insert(cell->active_cell_index());
        cells_to_check.insert(cell);
        bool new_neighbors_found;
        do {
            new_neighbors_found = false;
            for (const auto &check_cell :
                    std::vector<typename Triangulation<dim>::cell_iterator>(
                            cells_to_check.begin(), cells_to_check.end())) {
                for (const auto n : check_cell->face_indices()) {
                    if (!(check_cell->face(n)->at_boundary())) {
                        const auto &neighbor = check_cell->neighbor(n);
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
        } while (new_neighbors_found);
        return cells_to_check;
    }

}//SAND namespace
    template class SAND::DensityFilter<2>;
    template class SAND::DensityFilter<3>;

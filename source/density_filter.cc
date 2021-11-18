//
// Created by justin on 5/13/21.
//
#include "../include/density_filter.h"
#include "../include/input_information.h"
#include <deal.II/base/tensor.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/grid/tria.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

namespace SAND {
    using namespace dealii;

    /* When initialized, this function takes the current triangulation and creates a matrix corresponding to a
     * convolution being applied to a piecewise constant function on that triangulation  */
    template<int dim>
    void
    DensityFilter<dim>::initialize(DoFHandler<dim> &dof_handler) {
        const std::vector<types::global_dof_index> dofs_per_block = DoFTools::count_dofs_per_fe_block(dof_handler);
        filter_dsp.reinit(dofs_per_block[0],
                          dofs_per_block[0]);
        std::cout << "initialized with " << dofs_per_block[0] << std::endl;
        std::set<unsigned int> neighbor_ids;
        std::set<typename DoFHandler<dim>::cell_iterator> cells_to_check;
        std::set<typename DoFHandler<dim>::cell_iterator> cells_to_check_temp;

        /*finds neighbors whose values would be relevant, and adds them to the sparsity pattern of the matrix*/
        for (const auto &cell : dof_handler.active_cell_iterators()) {
            if(cell->is_locally_owned())
            {
                std::vector<types::global_dof_index> i(cell->get_fe().n_dofs_per_cell());
                cell->get_dof_indices(i);
                for (const auto &neighbor_cell : find_relevant_neighbors(cell))
                {
                    std::vector<types::global_dof_index> j(neighbor_cell->get_fe().n_dofs_per_cell());
                    neighbor_cell->get_dof_indices(j);
                    filter_dsp.add(i[16], j[16]);
                }
            }

        }
        filter_sparsity_pattern.copy_from(filter_dsp);

        const auto owned_dofs = dof_handler.locally_owned_dofs().get_view(0, dofs_per_block[0]);

        filter_matrix.reinit(owned_dofs, filter_sparsity_pattern, MPI_COMM_WORLD);

        /*adds values to the matrix corresponding to the max radius - */
        for (const auto &cell : dof_handler.active_cell_iterators())
        {
            if(cell->is_locally_owned())
            {
                std::vector<types::global_dof_index> i(cell->get_fe().n_dofs_per_cell());
                cell->get_dof_indices(i);
                for (const auto &neighbor_cell : find_relevant_neighbors(cell)) {
                    std::vector<types::global_dof_index> j(neighbor_cell->get_fe().n_dofs_per_cell());
                    neighbor_cell->get_dof_indices(j);
                    const double d =
                            cell->center().distance(neighbor_cell->center());
                    /*value should be (max radius - distance between cells)*cell measure */
                    double value = (Input::filter_r - d)*neighbor_cell->measure();
                    filter_matrix.add(i[16], j[16], value);
                }
            }
        }

        //here we normalize the filter so it computes an average. Sum of values in a row should be 1
        for (const auto &cell : dof_handler.active_cell_iterators())
        {
            if(cell->is_locally_owned())
            {
                std::vector<types::global_dof_index> i(cell->get_fe().n_dofs_per_cell());
                cell->get_dof_indices(i);
                double denominator = 0;
                typename LA::MPI::SparseMatrix::iterator iter = filter_matrix.begin(
                        i[16]);
                for (; iter != filter_matrix.end(i[16]); iter++)
                {
                    denominator = denominator + iter->value();
                }
                iter = filter_matrix.begin(i[16]);
                for (; iter != filter_matrix.end(i[16]); iter++)
                {
                    iter->value() = iter->value() / denominator;
                }
            }
        }
        std::cout << "density nonzero entries" << filter_matrix.n_nonzero_elements() << std::endl;
    }

    /*This function finds which neighbors are within a certain radius of the initial cell.*/
    template<int dim>
    std::set<typename DoFHandler<dim>::cell_iterator>
    DensityFilter<dim>::find_relevant_neighbors(typename DoFHandler<dim>::cell_iterator cell) const {
        std::set<unsigned int> neighbor_ids;
        std::set<typename DoFHandler<dim>::cell_iterator> cells_to_check;
        neighbor_ids.insert(cell->active_cell_index());
        cells_to_check.insert(cell);
        bool new_neighbors_found;
        do {
            new_neighbors_found = false;
            for (const auto &check_cell :
                    std::vector<typename DoFHandler<dim>::cell_iterator>(
                            cells_to_check.begin(), cells_to_check.end())) {
                for (const auto n : check_cell->face_indices()) {
                    if (!(check_cell->face(n)->at_boundary())) {
                        const auto &neighbor = check_cell->neighbor(n);
                        const double distance =
                                cell->center().distance(neighbor->center());
                        if ((distance < Input::filter_r) &&
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

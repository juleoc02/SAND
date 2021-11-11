//
// Created by justin on 2/17/21.
//

#ifndef SAND_KKT_SYSTEM_H
#define SAND_KKT_SYSTEM_H
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_vector.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>

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
#include <deal.II/hp/fe_collection.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include "../include/schur_preconditioner.h"
#include "../include/density_filter.h"

#include <iostream>
#include <fstream>
#include <algorithm>
namespace SAND {
    namespace LA
    {
        using namespace dealii::LinearAlgebraTrilinos;
    }
    using namespace dealii;

    template<int dim>
    class KktSystem {

    public:
        MPI_Comm  mpi_communicator;
        std::vector<IndexSet> owned_partitioning;
        std::vector<IndexSet> relevant_partitioning;

        KktSystem();

        void
        create_triangulation();

        void
        setup_boundary_values();

        void
        setup_filter_matrix();

        void
        setup_block_system();

        void
        assemble_block_system(const LA::MPI::BlockVector &state, const double barrier_size);

        LA::MPI::BlockVector
        solve(const LA::MPI::BlockVector &state);

        LA::MPI::BlockVector
        get_initial_state();

        double
        calculate_objective_value(const LA::MPI::BlockVector &state) const;

        double
        calculate_barrier_distance(const LA::MPI::BlockVector &state) const;

        double
        calculate_feasibility(const LA::MPI::BlockVector &state, const double barrier_size) const;

        double
        calculate_convergence(const LA::MPI::BlockVector &state) const;

        void
        output(const LA::MPI::BlockVector &state, const unsigned int j) const;

        void
        calculate_initial_rhs_error();

        double
        calculate_rhs_norm(const LA::MPI::BlockVector &state, const double barrier_size) const;

        void
        output_stl(const LA::MPI::BlockVector &state);

    private:

        LA::MPI::BlockVector
        calculate_rhs(const LA::MPI::BlockVector &test_solution, const double barrier_size) const;

        BlockDynamicSparsityPattern dsp;
        BlockSparsityPattern sparsity_pattern;
        mutable LA::MPI::BlockSparseMatrix system_matrix;
        mutable LA::MPI::BlockVector locally_relevant_solution;
        mutable LA::MPI::BlockVector distributed_solution;
        LA::MPI::BlockVector system_rhs;
        parallel::distributed::Triangulation<dim> triangulation;
        DoFHandler<dim> dof_handler;
        AffineConstraints<double> constraints;
        FESystem<dim> fe_nine;
        FESystem<dim> fe_ten;
        hp::FECollection<dim> fe_collection;
        const double density_ratio;
        const double density_penalty_exponent;

        mutable DensityFilter<dim> density_filter;

        std::map<types::global_dof_index, double> boundary_values;


        double initial_rhs_error;

    };
}

#endif //SAND_KKT_SYSTEM_H

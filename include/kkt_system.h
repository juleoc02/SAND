//
// Created by justin on 2/17/21.
//

#ifndef SAND_KKT_SYSTEM_H
#define SAND_KKT_SYSTEM_H
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
#include <deal.II/hp/fe_collection.h>

#include "../include/schur_preconditioner.h"
#include "../include/density_filter.h"

#include <iostream>
#include <fstream>
#include <algorithm>
namespace SAND {
    using namespace dealii;

    template<int dim>
    class KktSystem {

    public:
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
        assemble_block_system(const BlockVector<double> &state, const double barrier_size);

        BlockVector<double>
        solve(const BlockVector<double> &state);

        BlockVector<double>
        get_initial_state();

        double
        calculate_objective_value(const BlockVector<double> &state) const;

        double
        calculate_barrier_distance(const BlockVector<double> &state) const;

        double
        calculate_feasibility(const BlockVector<double> &state, const double barrier_size) const;

        double
        calculate_convergence(const BlockVector<double> &state) const;

        void
        output(const BlockVector<double> &state, const unsigned int j) const;

        void
        calculate_initial_rhs_error();



    private:

        BlockVector<double>
        calculate_rhs(const BlockVector<double> &test_solution, const double barrier_size) const;

        BlockSparsityPattern sparsity_pattern;
        BlockSparseMatrix<double> system_matrix;
        BlockVector<double> linear_solution;
        BlockVector<double> system_rhs;
        Triangulation<dim> triangulation;
        DoFHandler<dim> dof_handler;
        AffineConstraints<double> constraints;
        FESystem<dim> fe_nine;
        FESystem<dim> fe_ten;
        hp::FECollection<dim> fe_collection;
        const double density_ratio;
        const double density_penalty_exponent;

        DensityFilter<dim> density_filter;

        std::map<types::global_dof_index, double> boundary_values;


        double initial_rhs_error;

    };
}

#endif //SAND_KKT_SYSTEM_H

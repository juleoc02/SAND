//
// Created by justin on 3/2/21.
//

#ifndef SAND_SCHUR_PRECONDITIONER_H
#define SAND_SCHUR_PRECONDITIONER_H
#include "../include/kkt_system.h"
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_bicgstab.h>

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
#include <deal.II/base/config.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/table.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>

#include <deal.II/differentiation/ad/ad_number_traits.h>

#include <deal.II/lac/exceptions.h>
#include <deal.II/lac/identity_matrix.h>


#include <cstring>
#include <iomanip>
#include <vector>


#include "../include/parameters_and_components.h"

#include <iostream>
#include <algorithm>


namespace SAND
{
    using namespace dealii;

    class TopOptSchurPreconditioner: public Subscriptor {
    public:
        TopOptSchurPreconditioner();
        void initialize (BlockSparseMatrix<double> &matrix,  std::map<types::global_dof_index, double> boundary_values);
        void vmult(BlockVector<double> &dst, const BlockVector<double> &src);
        void Tvmult(BlockVector<double> &dst, const BlockVector<double> &src) ;
        void vmult_add(BlockVector<double> &dst, const BlockVector<double> &src) ;
        void Tvmult_add(BlockVector<double> &dst, const BlockVector<double> &src) ;
        void clear();
        unsigned int m() const;
        unsigned int n() const;

    private:
        unsigned int n_rows;
        unsigned int n_columns;
        unsigned int n_block_rows;
        unsigned int n_block_columns;
        void vmult_step_1(BlockVector<double> &dst, const BlockVector<double> &src);
        void vmult_step_2(BlockVector<double> &dst, const BlockVector<double> &src);
        void vmult_step_3(BlockVector<double> &dst, const BlockVector<double> &src);
        void vmult_step_4(BlockVector<double> &dst, const BlockVector<double> &src);

        SparseDirectUMFPACK elastic_direct;
        decltype(linear_operator(std::declval<BlockSparseMatrix<double>>().block(0,0))) op_elastic_inverse;
        SparseDirectUMFPACK scaled_direct;
        decltype(linear_operator(std::declval<BlockSparseMatrix<double>>().block(0,0))) op_scaled_inverse;
        SparseDirectUMFPACK diag_sum_direct;
        decltype(linear_operator(std::declval<BlockSparseMatrix<double>>().block(0,0))) op_diag_sum_inverse;

        decltype(linear_operator(std::declval<BlockSparseMatrix<double>>().block(0,0))) op_elastic;
        decltype(linear_operator(std::declval<BlockSparseMatrix<double>>().block(0,0))) op_filter;
        decltype(linear_operator(std::declval<BlockSparseMatrix<double>>().block(0,0))) op_diag_1;
        decltype(linear_operator(std::declval<BlockSparseMatrix<double>>().block(0,0))) op_diag_2;
        decltype(linear_operator(std::declval<BlockSparseMatrix<double>>().block(0,0))) op_density_density;
        decltype(linear_operator(std::declval<BlockSparseMatrix<double>>().block(0,0))) op_displacement_density;
        decltype(linear_operator(std::declval<BlockSparseMatrix<double>>().block(0,0))) op_displacement_multiplier_density;
        decltype(linear_operator(std::declval<BlockSparseMatrix<double>>().block(0,0))) op_scaled_identity;
        decltype(linear_operator(std::declval<BlockSparseMatrix<double>>().block(0,0))) op_fddf_chunk;
        decltype(linear_operator(std::declval<BlockSparseMatrix<double>>().block(0,0))) op_bcaeeac_chunk;
        decltype(linear_operator(std::declval<BlockSparseMatrix<double>>().block(0,0))) op_top_big_inverse;
        decltype(linear_operator(std::declval<BlockSparseMatrix<double>>().block(0,0))) op_bot_big_inverse;

        SolverControl diag_solver_control;
        SolverCG<Vector<double>> diag_cg;

        SolverControl other_solver_control;
        SolverBicgstab<Vector<double>> other_bicgstab;

        TimerOutput timer;
    };

}
#endif //SAND_SCHUR_PRECONDITIONER_H

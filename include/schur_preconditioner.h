//
// Created by justin on 3/2/21.
//

#ifndef SAND_SCHUR_PRECONDITIONER_H
#define SAND_SCHUR_PRECONDITIONER_H
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
    template<int dim>
    class TopOptSchurPreconditioner: public Subscriptor {
    public:
        TopOptSchurPreconditioner(BlockSparseMatrix<double> &matrix_in);
        void initialize (BlockSparseMatrix<double> &matrix, const std::map<types::global_dof_index, double> &boundary_values, const DoFHandler<dim> &dof_handler, const double barrier_size, const BlockVector<double> &state);
        void vmult(BlockVector<double> &dst, const BlockVector<double> &src) const;
        void Tvmult(BlockVector<double> &dst, const BlockVector<double> &src) const;
        void vmult_add(BlockVector<double> &dst, const BlockVector<double> &src) const;
        void Tvmult_add(BlockVector<double> &dst, const BlockVector<double> &src) const;
        void clear();
        unsigned int m() const;
        unsigned int n() const;
        void get_sparsity_pattern(BlockDynamicSparsityPattern &bdsp);

        void assemble_mass_matrix(const BlockVector<double> &state, const hp::FECollection<dim> &fe_system, const DoFHandler<dim> &dof_handler, const AffineConstraints<double> &constraints,   const BlockSparsityPattern &bsp);

        void print_stuff(const BlockSparseMatrix<double> &matrix);

        BlockSparseMatrix<double> &system_matrix;

    private:
        unsigned int n_rows;
        unsigned int n_columns;
        unsigned int n_block_rows;
        unsigned int n_block_columns;
        void vmult_step_1(BlockVector<double> &dst, const BlockVector<double> &src) const;
        void vmult_step_2(BlockVector<double> &dst, const BlockVector<double> &src) const;
        void vmult_step_3(BlockVector<double> &dst, const BlockVector<double> &src) const;
        void vmult_step_4(BlockVector<double> &dst, const BlockVector<double> &src) const;
        void vmult_step_5(BlockVector<double> &dst, const BlockVector<double> &src) const;

        BlockSparsityPattern mass_sparsity;
        BlockSparseMatrix<double> approx_h_mat;

        SolverControl other_solver_control;
        mutable SolverBicgstab<Vector<double>> other_bicgstab;
        mutable SolverGMRES<Vector<double>> other_gmres;
        mutable SolverCG<Vector<double>> other_cg;

        SparseMatrix<double> &a_mat;
        const SparseMatrix<double> &b_mat;
        const SparseMatrix<double> &c_mat;
        const SparseMatrix<double> &e_mat;
        const SparseMatrix<double> &f_mat;
        const SparseMatrix<double> &d_m_mat;
        const SparseMatrix<double> &d_1_mat;
        const SparseMatrix<double> &d_2_mat;
        const SparseMatrix<double> &m_vect;

        SparseMatrix<double> d_3_mat;
        SparseMatrix<double> d_4_mat;
        SparseMatrix<double> d_5_mat;
        SparseMatrix<double> d_6_mat;
        SparseMatrix<double> d_7_mat;
        SparseMatrix<double> d_8_mat;
        SparseMatrix<double> d_m_inv_mat;

        FullMatrix<double> g_mat;
        FullMatrix<double> h_mat;
        FullMatrix<double> k_inv_mat;
        LAPACKFullMatrix<double> k_mat;

        mutable Vector<double> pre_j;
        mutable Vector<double> pre_k;
        mutable Vector<double> g_d_m_inv_density;
        mutable Vector<double> k_g_d_m_inv_density;

        SparseDirectUMFPACK a_inv_direct;

        mutable TimerOutput timer;

    };

}
#endif //SAND_SCHUR_PRECONDITIONER_H

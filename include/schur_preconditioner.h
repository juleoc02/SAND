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
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/packaged_operation.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/matrix_out.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>

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
    namespace LA
    {
        using namespace dealii::LinearAlgebraTrilinos;
    }
    using namespace dealii;

    class VmultTrilinosSolverDirect : public TrilinosWrappers::SparseMatrix {
        public:
            VmultTrilinosSolverDirect(SolverControl &cn,
                                      const TrilinosWrappers::SolverDirect::AdditionalData &data
                                      );
            void vmult(LA::MPI::Vector &dst, const LA::MPI::Vector &src) const;
            void vmult(LinearAlgebra::distributed::Vector<double> &dst, const LinearAlgebra::distributed::Vector<double> &src) const;
            void Tvmult(LA::MPI::Vector &dst, const LA::MPI::Vector &src) const;
            void Tvmult(LinearAlgebra::distributed::Vector<double> &dst, const LinearAlgebra::distributed::Vector<double> &src) const;
            void initialize(LA::MPI::SparseMatrix &a_mat);
            unsigned int m() const;
            unsigned int n() const;
            int get_size()
            {
                return size;
            }
        private:
            mutable TrilinosWrappers::SolverDirect solver_direct;
            int size;
    };

    template<int dim>
    class TopOptSchurPreconditioner: public Subscriptor {
    public:
        TopOptSchurPreconditioner(LA::MPI::BlockSparseMatrix &matrix_in);
        void initialize (LA::MPI::BlockSparseMatrix &matrix, const std::map<types::global_dof_index, double> &boundary_values, const DoFHandler<dim> &dof_handler, const LA::MPI::BlockVector &state, const LA::MPI::BlockVector &distributed_state);
        void vmult(LA::MPI::BlockVector &dst, const LA::MPI::BlockVector &src) const;
        void Tvmult(LA::MPI::BlockVector &dst, const LA::MPI::BlockVector &src) const;
        void vmult_add(LA::MPI::BlockVector &dst, const LA::MPI::BlockVector &src) const;
        void Tvmult_add(LA::MPI::BlockVector &dst, const LA::MPI::BlockVector &src) const;
        void clear();
        unsigned int m() const;
        unsigned int n() const;
        void get_sparsity_pattern(BlockDynamicSparsityPattern &bdsp);

        void assemble_mass_matrix(const LA::MPI::BlockVector &state, const hp::FECollection<dim> &fe_system, const DoFHandler<dim> &dof_handler, const AffineConstraints<double> &constraints,   const BlockSparsityPattern &bsp);

        void print_stuff();

        LA::MPI::BlockSparseMatrix &system_matrix;

    private:
        unsigned int n_rows;
        unsigned int n_columns;
        unsigned int n_block_rows;
        unsigned int n_block_columns;
        void vmult_step_1(LA::MPI::BlockVector &dst, const LA::MPI::BlockVector &src) const;
        void vmult_step_2(LA::MPI::BlockVector &dst, const LA::MPI::BlockVector &src) const;
        void vmult_step_3(LA::MPI::BlockVector &dst, const LA::MPI::BlockVector &src) const;
        void vmult_step_4(LA::MPI::BlockVector &dst, const LA::MPI::BlockVector &src) const;
        void vmult_step_5(LA::MPI::BlockVector &dst, const LA::MPI::BlockVector &src) const;

        BlockSparsityPattern mass_sparsity;
        LA::MPI::BlockSparseMatrix approx_h_mat;

        SolverControl other_solver_control;
        mutable SolverBicgstab<Vector<double>> other_bicgstab;
        mutable SolverGMRES<Vector<double>> other_gmres;
        mutable SolverCG<Vector<double>> other_cg;

        LA::MPI::SparseMatrix &a_mat;
        const LA::MPI::SparseMatrix &b_mat;
        const LA::MPI::SparseMatrix &c_mat;
        const LA::MPI::SparseMatrix &e_mat;
        const LA::MPI::SparseMatrix &f_mat;
        const LA::MPI::SparseMatrix &d_m_mat;
        const LA::MPI::SparseMatrix &d_1_mat;
        const LA::MPI::SparseMatrix &d_2_mat;
        const LA::MPI::SparseMatrix &m_vect;

        LA::MPI::SparseMatrix d_3_mat;
        LA::MPI::SparseMatrix d_4_mat;
        LA::MPI::SparseMatrix d_5_mat;
        LA::MPI::SparseMatrix d_6_mat;
        LA::MPI::SparseMatrix d_7_mat;
        LA::MPI::SparseMatrix d_8_mat;
        LA::MPI::SparseMatrix d_m_inv_mat;

        FullMatrix<double> g_mat;
        FullMatrix<double> h_mat;
        FullMatrix<double> k_inv_mat;
        LAPACKFullMatrix<double> k_mat;

        mutable LA::MPI::Vector pre_j;
        mutable LA::MPI::Vector pre_k;
        mutable LA::MPI::Vector g_d_m_inv_density;
        mutable LA::MPI::Vector k_g_d_m_inv_density;

        std::string solver_type;
        TrilinosWrappers::SolverDirect::AdditionalData additional_data;
        SolverControl direct_solver_control;
        mutable VmultTrilinosSolverDirect a_inv_direct;
//        TrilinosWrappers::SolverDirect a_inv_direct;
        mutable TimerOutput timer;

    };

}
#endif //SAND_SCHUR_PRECONDITIONER_H

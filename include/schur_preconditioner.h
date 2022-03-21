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

#include <deal.II/base/conditional_ostream.h>

#include <iostream>
#include <algorithm>


namespace SAND
{
    using MatrixType  = dealii::TrilinosWrappers::SparseMatrix;
    using VectorType  = dealii::TrilinosWrappers::MPI::Vector;
    using PayloadType = dealii::TrilinosWrappers::internal::LinearOperatorImplementation::TrilinosPayload;
    using PayloadVectorType = typename PayloadType::VectorType;
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


    class GMatrix : public TrilinosWrappers::SparseMatrix {
        public:
            GMatrix(const LA::MPI::SparseMatrix &f_mat_in, LA::MPI::SparseMatrix &d_8_mat_in);
            void vmult(LA::MPI::Vector &dst, const LA::MPI::Vector &src) const;
            void Tvmult(LA::MPI::Vector &dst, const LA::MPI::Vector &src) const;
            void initialize(LA::MPI::Vector &exemplar_density_vector);
            unsigned int m() const;
            unsigned int n() const;
        private:
            const LA::MPI::SparseMatrix &f_mat;
            LA::MPI::SparseMatrix &d_8_mat;
            mutable LA::MPI::Vector temp_vect_1;
            mutable LA::MPI::Vector temp_vect_2;


    };

    class HMatrix : public TrilinosWrappers::SparseMatrix {
        public:
            HMatrix(LA::MPI::SparseMatrix &a_mat_in, const LA::MPI::SparseMatrix &b_mat_in, const LA::MPI::SparseMatrix &c_mat_in, const LA::MPI::SparseMatrix &e_mat_in,TrilinosWrappers::PreconditionAMG &pre_amg_in);
            void vmult(LA::MPI::Vector &dst, const LA::MPI::Vector &src) const;
            void Tvmult(LA::MPI::Vector &dst, const LA::MPI::Vector &src) const;
            void initialize(LA::MPI::Vector &exemplar_density_vector,  LA::MPI::Vector &exemplar_displacement_vector);
            unsigned int m() const;
            unsigned int n() const;
        private:
            LA::MPI::SparseMatrix &a_mat;
            const LA::MPI::SparseMatrix &b_mat;
            const LA::MPI::SparseMatrix &c_mat;
            const LA::MPI::SparseMatrix &e_mat;
            TrilinosWrappers::PreconditionAMG &pre_amg;
            mutable LA::MPI::Vector temp_vect_1;
            mutable LA::MPI::Vector temp_vect_2;
            mutable LA::MPI::Vector temp_vect_3;
            mutable LA::MPI::Vector temp_vect_4;
            mutable LA::MPI::Vector temp_vect_5;
            mutable LA::MPI::Vector temp_vect_6;
            mutable LA::MPI::Vector temp_vect_7;

    };

    class KinvMatrix : public TrilinosWrappers::SparseMatrix {
        public:
            KinvMatrix(HMatrix &h_mat_in, GMatrix &g_mat_in, const LA::MPI::SparseMatrix &d_m_mat_in, LA::MPI::SparseMatrix &d_m_inv_mat_in);
            void vmult(LA::MPI::Vector &dst, const LA::MPI::Vector &src) const;
            void Tvmult(LA::MPI::Vector &dst, const LA::MPI::Vector &src) const;
            void initialize(LA::MPI::Vector &exemplar_density_vector);
            unsigned int m() const;
            unsigned int n() const;
        private:
            HMatrix &h_mat;
            GMatrix &g_mat;
            const LA::MPI::SparseMatrix &d_m_mat;
            LA::MPI::SparseMatrix &d_m_inv_mat;
            mutable LA::MPI::Vector temp_vect_1;
            mutable LA::MPI::Vector temp_vect_2;
            mutable LA::MPI::Vector temp_vect_3;
            mutable LA::MPI::Vector temp_vect_4;
    };

    class JinvMatrix : public TrilinosWrappers::SparseMatrix {
        public:
            JinvMatrix(HMatrix &h_mat_in, GMatrix &g_mat_in, const LA::MPI::SparseMatrix &d_m_mat_in, LA::MPI::SparseMatrix &d_m_inv_mat_in);
            void vmult(LA::MPI::Vector &dst, const LA::MPI::Vector &src) const;
            void Tvmult(LA::MPI::Vector &dst, const LA::MPI::Vector &src) const;
            void initialize(LA::MPI::Vector &exemplar_density_vector);
            unsigned int m() const;
            unsigned int n() const;
        private:
            HMatrix &h_mat;
            GMatrix &g_mat;
            const LA::MPI::SparseMatrix &d_m_mat;
            LA::MPI::SparseMatrix &d_m_inv_mat;
            mutable LA::MPI::Vector temp_vect_1;
            mutable LA::MPI::Vector temp_vect_2;
            mutable LA::MPI::Vector temp_vect_3;
            mutable LA::MPI::Vector temp_vect_4;
    };

    template<int dim>
    class TopOptSchurPreconditioner: public Subscriptor {
    public:
        TopOptSchurPreconditioner(LA::MPI::BlockSparseMatrix &matrix_in);
        void initialize (LA::MPI::BlockSparseMatrix &matrix, const std::map<types::global_dof_index, double> &boundary_values, const DoFHandler<dim> &dof_handler, const LA::MPI::BlockVector &distributed_state);
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
        MPI_Comm  mpi_communicator;
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
        mutable SolverBicgstab<LA::MPI::Vector> other_bicgstab;
        mutable SolverGMRES<LA::MPI::Vector> other_gmres;
        mutable SolverCG<LA::MPI::Vector> other_cg;

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

        mutable LA::MPI::Vector pre_j;
        mutable LA::MPI::Vector pre_k;
        mutable LA::MPI::Vector g_d_m_inv_density;
        mutable LA::MPI::Vector k_g_d_m_inv_density;

        std::string solver_type;
        TrilinosWrappers::SolverDirect::AdditionalData additional_data;
        SolverControl direct_solver_control;
        mutable VmultTrilinosSolverDirect a_inv_direct;
        ConditionalOStream pcout;
        mutable TimerOutput timer;

        mutable TrilinosWrappers::PreconditionAMG pre_amg;

        GMatrix g_mat;
        HMatrix h_mat;

        JinvMatrix j_inv_mat;
        KinvMatrix k_inv_mat;



        LinearOperator<VectorType,VectorType,PayloadType> op_d_8;
        LinearOperator<VectorType,VectorType,PayloadType> op_f;
        LinearOperator<VectorType,VectorType,PayloadType> op_b;
        LinearOperator<VectorType,VectorType,PayloadType> op_c;
        LinearOperator<VectorType,VectorType,PayloadType> op_a_inv;
        LinearOperator<VectorType,VectorType,PayloadType> op_a_inv_ind;
        LinearOperator<VectorType,VectorType,PayloadType> op_e;
        LinearOperator<VectorType,VectorType,PayloadType> op_d_m;
        LinearOperator<VectorType,VectorType,PayloadType> op_d_m_inv;
        LinearOperator<VectorType,VectorType,PayloadType> op_g;
        LinearOperator<VectorType,VectorType,PayloadType> op_h;
        LinearOperator<VectorType,VectorType,PayloadType> op_k_inv;
        LinearOperator<VectorType,VectorType,PayloadType> op_j_inv;
    };


}
#endif //SAND_SCHUR_PRECONDITIONER_H

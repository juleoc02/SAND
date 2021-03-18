//
// Created by justin on 3/2/21.
//

#ifndef SAND_LINEAR_SOLVER_H
#define SAND_LINEAR_SOLVER_H
#include "../include/kktSystem.h"
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
        void initialize (const BlockSparseMatrix<double> &matrix);
        void vmult(BlockVector<double> &dst, const BlockVector<double> &src) const;
        void Tvmult(BlockVector<double> &dst, const BlockVector<double> &src) const;
        void vmult_add(BlockVector<double> &dst, const BlockVector<double> &src) const;
        void Tvmult_add(BlockVector<double> &dst, const BlockVector<double> &src) const;
        void clear();
        unsigned int m() const;
        unsigned int n() const;
    private:
        unsigned int n_rows;
        unsigned int n_columns;
        unsigned int n_block_rows;
        unsigned int n_block_columns;
        void vmult_step_1(BlockVector<double> &dst, const BlockVector<double> &src) const;
        void vmult_step_2(BlockVector<double> &dst, const BlockVector<double> &src) const;
        void vmult_step_3(BlockVector<double> &dst, const BlockVector<double> &src) const;
        void vmult_step_4(BlockVector<double> &dst, const BlockVector<double> &src) const;

//        op_A;
//        op_F;
//        op_D1;
//        op_D2;
//        op_A_inv;

    };

}
#endif //SAND_LINEAR_SOLVER_H

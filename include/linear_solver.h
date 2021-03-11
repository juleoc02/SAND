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

#include "../include/parameters_and_components.h"

#include <iostream>
#include <algorithm>


namespace SAND
{
    using namespace dealii;
//    class BigPreconditioner: public Subscriptor {
//    public:
//        BigPreconditioner();
//        void initialize (const MatrixType &matrix, const AdditionalData &additional_data=AdditionalData());
//        void vmult(BlockVector<double> &dst, const BlockVector<double> &src) const;
//        void Tvmult(BlockVector<double> &dst, const BlockVector<double> &src) const;
//        void vmult_add(BlockVector<double> &dst, const BlockVector<double> &src) const;
//        void Tvmult_add(BlockVector<double> &dst, const BlockVector<double> &src) const;
//        void clear();
//        size_type m() const;
//        size_type n() const;
//    private:
//        size_type n_rows;
//        size_type n_columns;
//    };
//
//
//    BigPreconditioner::BigPreconditioner()
//    :
//    n_rows(0),
//    n_columns(0)
//    {
//    }

}
#endif //SAND_LINEAR_SOLVER_H

//
// Created by justin on 2/17/21.
//
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_precondition.h>
#include "../include/schur_preconditioner.h"
#include "../include/input_information.h"
#include "../include/sand_tools.h"
#include <fstream>

namespace SAND {
    using MatrixType  = dealii::TrilinosWrappers::SparseMatrix;
    using VectorType  = dealii::TrilinosWrappers::MPI::Vector;
    using PayloadType = dealii::TrilinosWrappers::internal::LinearOperatorImplementation::TrilinosPayload;
    using PayloadVectorType = typename PayloadType::VectorType;
    using size_type         = dealii::types::global_dof_index;

    namespace ChangeVectorTypes
    {
    template <typename number>
    void copy(LA::MPI::Vector &                                         out,
              const dealii::LinearAlgebra::distributed::Vector<number> &in)
    {
        dealii::LinearAlgebra::ReadWriteVector<double> rwv(
                    out.locally_owned_elements());
        rwv.import(in, VectorOperation::insert);
        out.import(rwv, VectorOperation::insert);
    }
    template <typename number>
    void copy(dealii::LinearAlgebra::distributed::Vector<number> &out,
              const LA::MPI::Vector &                             in)
    {
        dealii::LinearAlgebra::ReadWriteVector<double> rwv;
        rwv.reinit(in);
        out.import(rwv, VectorOperation::insert);
    }
    } // namespace ChangeVectorTypes



    using namespace dealii;
    template<int dim>
    TopOptSchurPreconditioner<dim>::TopOptSchurPreconditioner(LA::MPI::BlockSparseMatrix &matrix_in, DoFHandler<dim> &big_dof_handler_in, MF_Elasticity_Operator<dim,1,double> &mf_elasticity_operator_in , PreconditionMG<dim,LinearAlgebra::distributed::Vector<double> ,MGTransferMatrixFree<dim, double>>
                                                              &mf_gmg_preconditioner_in)
            :
            system_matrix(matrix_in),
            mpi_communicator(MPI_COMM_WORLD),
            n_rows(0),
            n_columns(0),
            n_block_rows(0),
            n_block_columns(0),
            other_solver_control(1000000, 1e-6),
            other_bicgstab(other_solver_control),
            other_gmres(other_solver_control),
            other_cg(other_solver_control),
            a_mat(matrix_in.block(SolutionBlocks::displacement, SolutionBlocks::displacement_multiplier)),
            b_mat(matrix_in.block(SolutionBlocks::density, SolutionBlocks::density)),
            c_mat(matrix_in.block(SolutionBlocks::displacement,SolutionBlocks::density)),
            e_mat(matrix_in.block(SolutionBlocks::displacement_multiplier,SolutionBlocks::density)),
            f_mat(matrix_in.block(SolutionBlocks::unfiltered_density_multiplier,SolutionBlocks::unfiltered_density)),
            d_m_mat(matrix_in.block(SolutionBlocks::density_upper_slack_multiplier, SolutionBlocks::density_upper_slack)),
            d_1_mat(matrix_in.block(SolutionBlocks::density_lower_slack, SolutionBlocks::density_lower_slack)),
            d_2_mat(matrix_in.block(SolutionBlocks::density_upper_slack, SolutionBlocks::density_upper_slack)),
            m_vect(matrix_in.block(SolutionBlocks::density, SolutionBlocks::total_volume_multiplier)),
            solver_type("Amesos_Klu"),
            additional_data(false, solver_type),
            direct_solver_control(1, 0),
            a_inv_direct(direct_solver_control, additional_data),
            a_inv_mf_gmg(mf_elasticity_operator_in, mf_gmg_preconditioner_in),
            pcout(std::cout,(Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
            timer(pcout, TimerOutput::summary, TimerOutput::wall_times),
            g_mat(f_mat, d_8_mat),
            h_mat(a_mat, b_mat, c_mat, e_mat, pre_amg, a_inv_direct, a_inv_mf_gmg),
            j_inv_mat(h_mat, g_mat, d_m_mat, d_m_inv_mat),
            k_inv_mat(h_mat, g_mat, d_m_mat, d_m_inv_mat),
            big_dof_handler(big_dof_handler_in)
    {

    }

    template<int dim>
    void TopOptSchurPreconditioner<dim>::initialize(LA::MPI::BlockSparseMatrix &matrix, const std::map<types::global_dof_index, double> &boundary_values,const DoFHandler<dim> &dof_handler, const LA::MPI::BlockVector &distributed_state)
    {
        TimerOutput::Scope t(timer, "initialize");
        {
                TimerOutput::Scope t(timer, "diag stuff");
                const types::global_dof_index disp_start_index = system_matrix.get_row_indices().block_start(
                        SolutionBlocks::displacement);
                const types::global_dof_index disp_mult_start_index = system_matrix.get_row_indices().block_start(
                        SolutionBlocks::displacement_multiplier);

                for (auto &pair: boundary_values) {
                    const auto dof_index=pair.first;
                    const types::global_dof_index n_u = system_matrix.block(SolutionBlocks::displacement,
                                                                            SolutionBlocks::displacement).m();
                    if ((dof_index >= disp_start_index) && (dof_index < disp_start_index + n_u)) {
                        double diag_val = system_matrix.block(SolutionBlocks::displacement,
                                                              SolutionBlocks::displacement).el(
                                dof_index - disp_start_index, dof_index - disp_start_index);

                        system_matrix.block(SolutionBlocks::displacement, SolutionBlocks::displacement_multiplier).set(
                                dof_index - disp_start_index, dof_index - disp_start_index, diag_val);

                    }
                    else if ((dof_index >= disp_mult_start_index) && (dof_index < disp_mult_start_index + n_u))
                    {
                        double diag_val = system_matrix.block(SolutionBlocks::displacement_multiplier,
                                                              SolutionBlocks::displacement_multiplier).el(
                                dof_index - disp_mult_start_index, dof_index - disp_mult_start_index);
                        system_matrix.block(SolutionBlocks::displacement_multiplier, SolutionBlocks::displacement).set(
                                dof_index - disp_mult_start_index, dof_index - disp_mult_start_index, diag_val);
                    }
                }

                //set diagonal to 0?
                for (auto &pair: boundary_values) {
                    const auto dof_index=pair.first;
                    const types::global_dof_index disp_start_index = system_matrix.get_row_indices().block_start(
                            SolutionBlocks::displacement);
                    const types::global_dof_index disp_mult_start_index = system_matrix.get_row_indices().block_start(
                            SolutionBlocks::displacement_multiplier);
                    const types::global_dof_index n_u = system_matrix.block(SolutionBlocks::displacement,
                                                                            SolutionBlocks::displacement).m();
                    if ((dof_index >= disp_start_index) && (dof_index < disp_start_index + n_u)) {
                        system_matrix.block(SolutionBlocks::displacement, SolutionBlocks::displacement).set(
                                dof_index - disp_start_index, dof_index - disp_start_index, 0);
                    } else if ((dof_index >= disp_mult_start_index) && (dof_index < disp_mult_start_index + n_u)) {
                        system_matrix.block(SolutionBlocks::displacement_multiplier,
                                            SolutionBlocks::displacement_multiplier).set(
                                dof_index - disp_mult_start_index, dof_index - disp_mult_start_index, 0);
                    }
                }

            system_matrix.compress(VectorOperation::insert);
        }



//        if (Input::solver_choice==SolverOptions::inexact_K_with_inexact_A_gmres)
//        {
//            TimerOutput::Scope t(timer, "build A inv");

//             std::vector<std::vector<bool>> constant_modes;
//             FEValuesExtractors::Vector     displacement_components(SolutionComponents::displacement<dim>);
//             DoFTools::extract_constant_modes(dof_handler,
//                                              dof_handler.get_fe_collection()
//                                              .component_mask(displacement_components),
//                                              constant_modes);
//             TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
//             amg_data.constant_modes = constant_modes;
////             amg_data.n_cycles = 1;
////             amg_data.w_cycle = true;

//             pre_amg.initialize(a_mat,amg_data);

//        }
//        else
//        {
//             a_inv_direct.initialize(a_mat);
//        }
        {
            TimerOutput::Scope t(timer, "reinit diag matrices");
            d_3_mat.reinit(matrix.block(SolutionBlocks::density, SolutionBlocks::density));
            d_4_mat.reinit(matrix.block(SolutionBlocks::density, SolutionBlocks::density));
            d_5_mat.reinit(matrix.block(SolutionBlocks::density, SolutionBlocks::density));
            d_6_mat.reinit(matrix.block(SolutionBlocks::density, SolutionBlocks::density));
            d_7_mat.reinit(matrix.block(SolutionBlocks::density, SolutionBlocks::density));
            d_8_mat.reinit(matrix.block(SolutionBlocks::density, SolutionBlocks::density));
            d_m_inv_mat.reinit(matrix.block(SolutionBlocks::density, SolutionBlocks::density));
            d_3_mat=0;
            d_4_mat=0;
            d_5_mat=0;
            d_6_mat=0;
            d_7_mat=0;
            d_8_mat=0;
            d_m_inv_mat=0;
            pcout << "diag reinit" << std::endl;
        }
        {
            const types::global_dof_index n_p = system_matrix.block(SolutionBlocks::density,
                                                                    SolutionBlocks::density).m();

            pcout << "np: " << n_p << std::endl;

            std::vector<double> l_global(n_p);
            std::vector<double> lm_global(n_p);
            std::vector<double> u_global(n_p);
            std::vector<double> um_global(n_p);
            std::vector<double> m_global(n_p);

            std::vector<double> l(n_p);
            std::vector<double> lm(n_p);
            std::vector<double> u(n_p);
            std::vector<double> um(n_p);
            std::vector<double> m(n_p);

            TimerOutput::Scope t(timer, "build diag matrices");
            for (const auto cell: dof_handler.active_cell_iterators())
            {
                if(cell->is_locally_owned())
                {
                    std::vector<types::global_dof_index> i(cell->get_fe().n_dofs_per_cell());
                    cell->get_dof_indices(i);

                    const int i_ind = cell->get_fe().component_to_system_index(0,0);

                    if(distributed_state.block(SolutionBlocks::density_lower_slack_multiplier).in_local_range(i[i_ind]))
                    {
                        lm[i[i_ind]] = distributed_state.block(SolutionBlocks::density_lower_slack_multiplier)[i[i_ind]];
                        m[i[i_ind]] = cell->measure();
                    }

                    if(distributed_state.block(SolutionBlocks::density_lower_slack).in_local_range(i[i_ind]))
                    {
                        l[i[i_ind]] =  distributed_state.block(SolutionBlocks::density_lower_slack)[i[i_ind]];
                    }

                    if(distributed_state.block(SolutionBlocks::density_upper_slack_multiplier).in_local_range(i[i_ind]))
                    {
                        um[i[i_ind]]= distributed_state.block(SolutionBlocks::density_upper_slack_multiplier)[i[i_ind]];
                    }

                    if(distributed_state.block(SolutionBlocks::density_upper_slack).in_local_range(i[i_ind]))
                    {
                        u[i[i_ind]] = distributed_state.block(SolutionBlocks::density_upper_slack)[i[i_ind]];
                    }
                }
            }

            MPI_Allreduce(lm.data(), lm_global.data(), n_p, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(l.data(), l_global.data(), n_p, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(um.data(), um_global.data(), n_p, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(u.data(), u_global.data(), n_p, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(m.data(), m_global.data(), n_p, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            for (unsigned int k=0; k< n_p; k++)
            {
                    if(distributed_state.block(0).in_local_range(k))
                    {
                        d_3_mat.set(k, k, -1 * lm_global[k]/(m_global[k]*l_global[k]));
                        d_4_mat.set(k, k, -1 * um_global[k]/(m_global[k]*u_global[k]));
                        d_5_mat.set(k, k, lm_global[k]/l_global[k]);
                        d_6_mat.set(k, k, um_global[k]/u_global[k]);
                        d_7_mat.set(k, k, m_global[k]*(lm_global[k]*u_global[k] + um_global[k]*l_global[k])/(l_global[k]*u_global[k]));
                        d_8_mat.set(k, k, l_global[k]*u_global[k]/(m_global[k]*(lm_global[k]*u_global[k] + um_global[k]*l_global[k])));
                        d_m_inv_mat.set(k, k, 1 / m_global[k]);
                    }
            }
        }
        d_3_mat.compress(VectorOperation::insert);
        d_4_mat.compress(VectorOperation::insert);
        d_5_mat.compress(VectorOperation::insert);
        d_6_mat.compress(VectorOperation::insert);
        d_7_mat.compress(VectorOperation::insert);
        d_8_mat.compress(VectorOperation::insert);
        d_m_inv_mat.compress(VectorOperation::insert);

        pre_j=distributed_state.block(SolutionBlocks::density);
        pre_k=distributed_state.block(SolutionBlocks::density);
        g_d_m_inv_density=distributed_state.block(SolutionBlocks::density);
        k_g_d_m_inv_density=distributed_state.block(SolutionBlocks::density);

        LA::MPI::Vector density_exemplar = distributed_state.block(SolutionBlocks::density);
        LA::MPI::Vector displacement_exemplar = distributed_state.block(SolutionBlocks::displacement);

        g_mat.initialize(density_exemplar);
        h_mat.initialize(density_exemplar, displacement_exemplar);
        j_inv_mat.initialize(density_exemplar);
        k_inv_mat.initialize(density_exemplar);


    }


    template<int dim>
    void TopOptSchurPreconditioner<dim>::vmult(LA::MPI::BlockVector &dst, const LA::MPI::BlockVector &src) const {
        LA::MPI::BlockVector temp_src;
        {
            TimerOutput::Scope t(timer, "part 1");
            vmult_step_1(dst, src);
            temp_src = dst;
        }

        {
            TimerOutput::Scope t(timer, "part 2");
            vmult_step_2(dst, temp_src);
            temp_src = dst;

        }

        {
            TimerOutput::Scope t(timer, "part 3");
            vmult_step_3(dst, temp_src);
            temp_src = dst;
        }
        {
            TimerOutput::Scope t(timer, "part 4");
            vmult_step_4(dst, temp_src);
            temp_src = dst;
        }
        vmult_step_5(dst, temp_src);
    }

    template<int dim>
    void TopOptSchurPreconditioner<dim>::Tvmult(LA::MPI::BlockVector &dst, const LA::MPI::BlockVector &src) const {
        dst = src;
    }

    template<int dim>
    void TopOptSchurPreconditioner<dim>::vmult_add(LA::MPI::BlockVector &dst, const LA::MPI::BlockVector &src) const {
        LA::MPI::BlockVector dst_temp = dst;
        vmult(dst_temp, src);
        dst += dst_temp;
    }

    template<int dim>
    void TopOptSchurPreconditioner<dim>::Tvmult_add(LA::MPI::BlockVector &dst, const LA::MPI::BlockVector &src) const {
        dst = src;
    }

    template<int dim>
    void TopOptSchurPreconditioner<dim>::vmult_step_1(LA::MPI::BlockVector &dst, const LA::MPI::BlockVector &src) const {
        dst = src;
        auto dst_temp = dst;
        dst.block(SolutionBlocks::unfiltered_density) = dst_temp.block(SolutionBlocks::unfiltered_density)
                - linear_operator<VectorType,VectorType,PayloadType>(d_5_mat) * src.block(SolutionBlocks::density_lower_slack_multiplier)
                + linear_operator<VectorType,VectorType,PayloadType>(d_6_mat) * src.block(SolutionBlocks::density_upper_slack_multiplier)
                + src.block(SolutionBlocks::density_lower_slack)
                - src.block(SolutionBlocks::density_upper_slack);
    }

    template<int dim>
    void TopOptSchurPreconditioner<dim>::vmult_step_2(LA::MPI::BlockVector &dst, const LA::MPI::BlockVector &src) const {
        dst = src;
        auto dst_temp = dst;

        dst.block(SolutionBlocks::unfiltered_density_multiplier) = dst_temp.block(SolutionBlocks::unfiltered_density_multiplier)
                - linear_operator<VectorType,VectorType,PayloadType>(f_mat)*linear_operator<VectorType,VectorType,PayloadType>(d_8_mat) * src.block(SolutionBlocks::unfiltered_density);

    }

    template<int dim>
    void TopOptSchurPreconditioner<dim>::vmult_step_3(LA::MPI::BlockVector &dst, const LA::MPI::BlockVector &src) const {

        dst = src;
        auto dst_temp = dst;

        if (Input::solver_choice==SolverOptions::inexact_K_with_inexact_A_gmres)
        {
//            SolverControl            solver_control_1(100000, 1e-6);
//            SolverControl            solver_control_2(100000, 1e-6);
//            SolverCG<LA::MPI::Vector> a_solver_cg_1(solver_control_1);
//            SolverCG<LA::MPI::Vector> a_solver_cg_2(solver_control_2);
//            a_solver_cg_1.solve(a_mat,dst_temp.block(SolutionBlocks::displacement_multiplier),src.block(SolutionBlocks::displacement_multiplier),pre_amg);
//            a_solver_cg_2.solve(a_mat,dst_temp.block(SolutionBlocks::displacement),src.block(SolutionBlocks::displacement),pre_amg);

            a_inv_mf_gmg.vmult(dst_temp.block(SolutionBlocks::displacement_multiplier),src.block(SolutionBlocks::displacement_multiplier));

            a_inv_mf_gmg.vmult(dst_temp.block(SolutionBlocks::displacement),src.block(SolutionBlocks::displacement));
            c_mat.Tvmult(dst_temp.block(SolutionBlocks::density_upper_slack),dst_temp.block(SolutionBlocks::displacement_multiplier));
            e_mat.Tvmult(dst_temp.block(SolutionBlocks::density_lower_slack),dst_temp.block(SolutionBlocks::displacement));

            dst.block(SolutionBlocks::density) = dst_temp.block(SolutionBlocks::density) - dst_temp.block(SolutionBlocks::density_upper_slack) - dst_temp.block(SolutionBlocks::density_lower_slack);

        }
        else
        {
             a_inv_direct.vmult(dst_temp.block(SolutionBlocks::displacement_multiplier),src.block(SolutionBlocks::displacement_multiplier));
             a_inv_direct.vmult(dst_temp.block(SolutionBlocks::displacement),src.block(SolutionBlocks::displacement));
             c_mat.Tvmult(dst_temp.block(SolutionBlocks::density_upper_slack),dst_temp.block(SolutionBlocks::displacement_multiplier));
             e_mat.Tvmult(dst_temp.block(SolutionBlocks::density_lower_slack),dst_temp.block(SolutionBlocks::displacement));
             dst.block(SolutionBlocks::density) = dst_temp.block(SolutionBlocks::density) - dst_temp.block(SolutionBlocks::density_upper_slack) - dst_temp.block(SolutionBlocks::density_lower_slack);
        }



    }

    template<int dim>
    void TopOptSchurPreconditioner<dim>::vmult_step_4(LA::MPI::BlockVector &dst, const LA::MPI::BlockVector &src) const {
        dst = src;
        auto dst_temp = dst;
        auto k_density_mult =  src.block(SolutionBlocks::density);

        TrilinosWrappers::PreconditionIdentity preconditioner;
        preconditioner.initialize(b_mat);

            auto d_m_inv_density = g_d_m_inv_density;
            d_m_inv_mat.vmult(d_m_inv_density,src.block(SolutionBlocks::density));
            g_mat.vmult(g_d_m_inv_density,d_m_inv_density);
            SolverControl step_4_gmres_control_1 (Input::k_inv_iterations, g_d_m_inv_density.l2_norm()*1e-12);
            SolverFGMRES<LA::MPI::Vector> step_4_gmres_1 (step_4_gmres_control_1);
            try {
                step_4_gmres_1.solve(k_inv_mat,k_g_d_m_inv_density,g_d_m_inv_density, preconditioner );
            }
            catch (std::exception &exc)
            {
                pcout << "Failure of linear solver step_4_gmres_1" << std::endl;
                pcout << "first residual: " << step_4_gmres_control_1.initial_value() << std::endl;
                pcout << "last residual: " << step_4_gmres_control_1.last_value() << std::endl;
//                throw;
            }
            SolverControl step_4_gmres_control_2 (Input::k_inv_iterations, 1e-12 * src.block(SolutionBlocks::unfiltered_density_multiplier).l2_norm() );
            SolverFGMRES<LA::MPI::Vector> step_4_gmres_2 (step_4_gmres_control_2);
            try {
                step_4_gmres_2.solve(k_inv_mat,k_density_mult,src.block(SolutionBlocks::unfiltered_density_multiplier), PreconditionIdentity());
            } catch (std::exception &exc)
            {
                pcout << "Failure of linear solver step_4_gmres_2" << std::endl;
                pcout << "first residual: " << step_4_gmres_control_2.initial_value() << std::endl;
                pcout << "last residual: " << step_4_gmres_control_2.last_value() << std::endl;
//                throw;
            }


        dst.block(SolutionBlocks::total_volume_multiplier) = transpose_operator<VectorType, VectorType, PayloadType>(m_vect)*k_g_d_m_inv_density
                                                        - transpose_operator<VectorType, VectorType, PayloadType>(m_vect)*k_density_mult
                                                        +dst_temp.block(SolutionBlocks::total_volume_multiplier);
    }




    template<int dim>
    void TopOptSchurPreconditioner<dim>::vmult_step_5(LA::MPI::BlockVector &dst, const LA::MPI::BlockVector &src) const {
        {
            //First Block Inverse
            TimerOutput::Scope t(timer, "inverse 1");
            dst.block(SolutionBlocks::density_lower_slack_multiplier) = linear_operator<VectorType,VectorType,PayloadType>(d_3_mat) * src.block(SolutionBlocks::density_lower_slack_multiplier) +
                    linear_operator<VectorType,VectorType,PayloadType>(d_m_inv_mat) * src.block(SolutionBlocks::density_lower_slack);
            dst.block(SolutionBlocks::density_upper_slack_multiplier) = linear_operator<VectorType,VectorType,PayloadType>(d_4_mat) * src.block(SolutionBlocks::density_upper_slack_multiplier) +
                    linear_operator<VectorType,VectorType,PayloadType>(d_m_inv_mat) * src.block(SolutionBlocks::density_upper_slack);
            dst.block(SolutionBlocks::density_lower_slack) = linear_operator<VectorType,VectorType,PayloadType>(d_m_inv_mat) * src.block(SolutionBlocks::density_lower_slack_multiplier);
            dst.block(SolutionBlocks::density_upper_slack) = linear_operator<VectorType,VectorType,PayloadType>(d_m_inv_mat) * src.block(SolutionBlocks::density_upper_slack_multiplier);
        }

        {
            //Second Block Inverse
            TimerOutput::Scope t(timer, "inverse 2");
            dst.block(SolutionBlocks::unfiltered_density) =
                    linear_operator<VectorType,VectorType,PayloadType>(d_8_mat) * src.block(SolutionBlocks::unfiltered_density);
        }


        {
            //Third Block Inverse
            TimerOutput::Scope t(timer, "inverse 3");
            if (Input::solver_choice==SolverOptions::inexact_K_with_inexact_A_gmres)
            {
//                SolverControl            solver_control(100000, 1e-6);
//                SolverCG<LA::MPI::Vector> a_solver_cg(solver_control);

//                auto a_inv_op = inverse_operator(linear_operator<VectorType,VectorType,PayloadType>(a_mat),a_solver_cg, pre_amg);

//                dst.block(SolutionBlocks::displacement) = a_inv_op * src.block(SolutionBlocks::displacement_multiplier);
//                dst.block(SolutionBlocks::displacement_multiplier) = a_inv_op * src.block(SolutionBlocks::displacement);

                  a_inv_mf_gmg.vmult( dst.block(SolutionBlocks::displacement), src.block(SolutionBlocks::displacement_multiplier));
                  a_inv_mf_gmg.vmult( dst.block(SolutionBlocks::displacement_multiplier), src.block(SolutionBlocks::displacement));


            }
            else
            {
                a_inv_direct.vmult( dst.block(SolutionBlocks::displacement), src.block(SolutionBlocks::displacement_multiplier));
                a_inv_direct.vmult( dst.block(SolutionBlocks::displacement_multiplier), src.block(SolutionBlocks::displacement));
            }


        }


        {
            //Fourth (ugly) Block Inverse
            TimerOutput::Scope t(timer, "inverse 4");



            if (Input::solver_choice == SolverOptions::exact_preconditioner_with_gmres)
            {
//                pre_j = src.block(SolutionBlocks::density) + linear_operator<VectorType,VectorType,PayloadType>(h_mat) * linear_operator<VectorType,VectorType,PayloadType>(d_m_inv_mat) * src.block(SolutionBlocks::unfiltered_density_multiplier);
//                pre_k = -1* linear_operator<VectorType,VectorType,PayloadType>(g_mat) * linear_operator<VectorType,VectorType,PayloadType>(d_m_inv_mat) * src.block(SolutionBlocks::density) + src.block(SolutionBlocks::unfiltered_density_multiplier);
//                dst.block(SolutionBlocks::unfiltered_density_multiplier) = transpose_operator<VectorType, VectorType, PayloadType>(k_mat) * pre_j;
//                dst.block(SolutionBlocks::density) = linear_operator<VectorType,VectorType,PayloadType>(k_mat) * pre_k;
            }

            else if (Input::solver_choice == SolverOptions::inexact_K_with_inexact_A_gmres)
            {

                TrilinosWrappers::PreconditionIdentity preconditioner;
                preconditioner.initialize(b_mat);
                auto pre_pre_k = pre_k;
                auto pre_pre_pre_k = pre_k;
                auto d_m_inv_unfil_density_mult = pre_k;
                auto h_d_m_inv_unfil_density_mult = pre_k;

                {
                    TimerOutput::Scope t(timer, "not inverse 5.1");
                    d_m_inv_mat.vmult(d_m_inv_unfil_density_mult, src.block(SolutionBlocks::unfiltered_density_multiplier));
                    h_mat.vmult(h_d_m_inv_unfil_density_mult,d_m_inv_unfil_density_mult);
                    pre_j = src.block(SolutionBlocks::density) + h_d_m_inv_unfil_density_mult;
                    d_m_inv_mat.vmult(pre_pre_pre_k,src.block(SolutionBlocks::density));
                    g_mat.vmult(pre_pre_k,pre_pre_pre_k);
                    pre_k =  -1 * pre_pre_k + src.block(SolutionBlocks::unfiltered_density_multiplier);
                }
                SolverControl step_5_gmres_control_1 (Input::k_inv_iterations, 1e-12*pre_j.l2_norm());
                SolverFGMRES<LA::MPI::Vector> step_5_gmres_1 (step_5_gmres_control_1);
                try {
                    TimerOutput::Scope t(timer, "actual inverse 5.1");
                    step_5_gmres_1.solve(j_inv_mat, dst.block(SolutionBlocks::unfiltered_density_multiplier), pre_j , preconditioner);
                } catch (std::exception &exc)
                {
                    pcout << "Failure of linear solver step_5_gmres_1" << std::endl;
                    pcout << "first residual: " << step_5_gmres_control_1.initial_value() << std::endl;
                    pcout << "last residual: " << step_5_gmres_control_1.last_value() << std::endl;
//                    throw;
                }


                SolverControl step_5_gmres_control_2 (Input::k_inv_iterations, 1e-12*pre_k.l2_norm());
                SolverFGMRES<LA::MPI::Vector> step_5_gmres_2 (step_5_gmres_control_2);
                try {
                    TimerOutput::Scope t(timer, "actual inverse 5.2");
                    step_5_gmres_2.solve(k_inv_mat,dst.block(SolutionBlocks::density), pre_k , preconditioner);
                } catch (std::exception &exc)
                {
                    pcout << "Failure of linear solver step_5_gmres_2" << std::endl;
                    pcout << "first residual: " << step_5_gmres_control_2.initial_value() << std::endl;
                    pcout << "last residual: " << step_5_gmres_control_2.last_value() << std::endl;
//                    throw;
                }
            }
            else if (Input::solver_choice == SolverOptions::inexact_K_with_exact_A_gmres)
            {

                TrilinosWrappers::PreconditionIdentity preconditioner;
                preconditioner.initialize(b_mat);
                auto pre_pre_k = pre_k;
                auto pre_pre_pre_k = pre_k;
                auto d_m_inv_unfil_density_mult = pre_k;
                auto h_d_m_inv_unfil_density_mult = pre_k;

                {
                    TimerOutput::Scope t(timer, "not inverse 5.1");
                    d_m_inv_mat.vmult(d_m_inv_unfil_density_mult, src.block(SolutionBlocks::unfiltered_density_multiplier));
                    h_mat.vmult(h_d_m_inv_unfil_density_mult,d_m_inv_unfil_density_mult);
                    pre_j = src.block(SolutionBlocks::density) + h_d_m_inv_unfil_density_mult;
                    d_m_inv_mat.vmult(pre_pre_pre_k,src.block(SolutionBlocks::density));
                    g_mat.vmult(pre_pre_k,pre_pre_pre_k);
                    pre_k =  -1 * pre_pre_k + src.block(SolutionBlocks::unfiltered_density_multiplier);
                }
                SolverControl step_5_gmres_control_1 (Input::k_inv_iterations, 1e-12*pre_j.l2_norm());
                SolverFGMRES<LA::MPI::Vector> step_5_gmres_1 (step_5_gmres_control_1);
                try {
                    TimerOutput::Scope t(timer, "actual inverse 5.1");
                    step_5_gmres_1.solve(j_inv_mat, dst.block(SolutionBlocks::unfiltered_density_multiplier), pre_j , preconditioner);
                } catch (std::exception &exc)
                {
                    pcout << "Failure of linear solver step_5_gmres_1" << std::endl;
                    pcout << "first residual: " << step_5_gmres_control_1.initial_value() << std::endl;
                    pcout << "last residual: " << step_5_gmres_control_1.last_value() << std::endl;
//                    throw;
                }


                SolverControl step_5_gmres_control_2 (Input::k_inv_iterations, 1e-12*pre_k.l2_norm());
                SolverFGMRES<LA::MPI::Vector> step_5_gmres_2 (step_5_gmres_control_2);
                try {
                    TimerOutput::Scope t(timer, "actual inverse 5.2");
                    step_5_gmres_2.solve(k_inv_mat,dst.block(SolutionBlocks::density), pre_k , preconditioner);
                } catch (std::exception &exc)
                {
                    pcout << "Failure of linear solver step_5_gmres_2" << std::endl;
                    pcout << "first residual: " << step_5_gmres_control_2.initial_value() << std::endl;
                    pcout << "last residual: " << step_5_gmres_control_2.last_value() << std::endl;
//                    throw;
                }
            }
            else
            {
                pcout << "shouldn't get here";
                throw;
            }

        }
        {
            dst.block(SolutionBlocks::total_volume_multiplier) = src.block(SolutionBlocks::total_volume_multiplier);
        }
    }


    template<int dim>
    void TopOptSchurPreconditioner<dim>::print_stuff()
    {


    }



    void VmultTrilinosSolverDirect::initialize(LA::MPI::SparseMatrix &a_mat)
    {
        reinit(a_mat);
        solver_direct.initialize(a_mat);
        size = a_mat.n();
    }

    void
    VmultTrilinosSolverDirect::vmult(LinearAlgebra::distributed::Vector<double> &dst, const LinearAlgebra::distributed::Vector<double> &src) const
    {
        solver_direct.solve(dst, src);
    }

    void VmultTrilinosSolverDirect::vmult(LA::MPI::Vector &dst, const LA::MPI::Vector &src) const
    {
        solver_direct.solve(dst, src);
    }

    void
    VmultTrilinosSolverDirect::Tvmult(LinearAlgebra::distributed::Vector<double> &dst, const LinearAlgebra::distributed::Vector<double> &src) const
    {
        solver_direct.solve(dst, src);
    }

    void VmultTrilinosSolverDirect::Tvmult(LA::MPI::Vector &dst, const LA::MPI::Vector &src) const
    {
        solver_direct.solve(dst, src);
    }


    VmultTrilinosSolverDirect::VmultTrilinosSolverDirect(SolverControl &cn,
                     const TrilinosWrappers::SolverDirect::AdditionalData &data)
    : solver_direct(cn, data)
    {

    }

    // **************** A inv MF GMG **********************

    template<int dim>
    AInvMatMFGMG<dim>::AInvMatMFGMG(MF_Elasticity_Operator<dim,1,double> &mf_elasticity_operator_in , PreconditionMG<dim, LinearAlgebra::distributed::Vector<double>, MGTransferMatrixFree<dim, double> > &mf_gmg_preconditioner_in)
        : mf_elasticity_operator(mf_elasticity_operator_in),
          mf_gmg_preconditioner(mf_gmg_preconditioner_in)
    {
        mf_elasticity_operator.initialize_dof_vector(temp_dst);
        mf_elasticity_operator.initialize_dof_vector(temp_src);
    }

    template<int dim>
    void AInvMatMFGMG<dim>::vmult(LA::MPI::Vector &dst, const LA::MPI::Vector &src) const
    {

        SolverControl            solver_control(25, 1e-12);
        SolverCG<LinearAlgebra::distributed::Vector<double>> a_solver_cg(solver_control);

        ChangeVectorTypes::copy(temp_src, src);
        try{
            a_solver_cg.solve(mf_elasticity_operator,temp_dst,temp_src, mf_gmg_preconditioner);
        } catch (std::exception &exc)
        {
            std::cout << "failed with a reduction of: " << solver_control.initial_value()/solver_control.last_value() << std::endl;
        }

        ChangeVectorTypes::copy(dst,temp_dst);

    }

    template<int dim>
    void AInvMatMFGMG<dim>::Tvmult(LA::MPI::Vector &dst, const LA::MPI::Vector &src) const {
        SolverControl            solver_control(iterations, src.l2_norm()*tolerance);
        SolverCG<LinearAlgebra::distributed::Vector<double>> a_solver_cg(solver_control);

        ChangeVectorTypes::copy(temp_src, src);
        temp_dst = 0.;

        a_solver_cg.solve(mf_elasticity_operator,temp_dst,temp_src, mf_gmg_preconditioner);

        ChangeVectorTypes::copy(dst,temp_dst);
    }

    template<int dim>
    void AInvMatMFGMG<dim>::set_tol(double tol_in)
    {
        tolerance = tol_in;
    }
    template<int dim>
    void AInvMatMFGMG<dim>::set_iter(unsigned int iterations_in)
    {
        iterations = iterations_in;
    }


    // ******************     GMATRIX     ***********************

    GMatrix::GMatrix(const LA::MPI::SparseMatrix &f_mat_in, LA::MPI::SparseMatrix &d_8_mat_in)
            :
            f_mat(f_mat_in),
            d_8_mat(d_8_mat_in)
    {

    }

    void
    GMatrix::initialize(LA::MPI::Vector &exemplar_density_vector)
    {
        temp_vect_1 = exemplar_density_vector;
        temp_vect_2 = exemplar_density_vector;
    }


    void GMatrix::vmult(LA::MPI::Vector &dst, const LA::MPI::Vector &src) const
    {
        f_mat.Tvmult(temp_vect_1,src);
        d_8_mat.vmult(temp_vect_2, temp_vect_1);
        f_mat.vmult(dst,temp_vect_2);
    }



    void GMatrix::Tvmult(LA::MPI::Vector &dst, const LA::MPI::Vector &src) const
    {
        f_mat.Tvmult(temp_vect_1,src);
        d_8_mat.vmult(temp_vect_2, temp_vect_1);
        f_mat.vmult(dst,temp_vect_2);
    }


    // ******************     HMatrix     ***********************

    template<int dim>
    HMatrix<dim>::HMatrix(LA::MPI::SparseMatrix &a_mat_in, const LA::MPI::SparseMatrix &b_mat_in, const LA::MPI::SparseMatrix &c_mat_in, const LA::MPI::SparseMatrix &e_mat_in,TrilinosWrappers::PreconditionAMG &pre_amg_in,VmultTrilinosSolverDirect &a_inv_direct_in, AInvMatMFGMG<dim> &a_inv_mf_gmg_in)
            :
            a_mat(a_mat_in),
            b_mat(b_mat_in),
            c_mat(c_mat_in),
            e_mat(e_mat_in),
            pre_amg(pre_amg_in),
            a_inv_direct(a_inv_direct_in),
            a_inv_mf_gmg(a_inv_mf_gmg_in)
    {

    }

    template<int dim>
    void
    HMatrix<dim>::initialize(LA::MPI::Vector &exemplar_density_vector,  LA::MPI::Vector &exemplar_displacement_vector)
    {
        temp_vect_1 = exemplar_displacement_vector;
        temp_vect_2 = exemplar_displacement_vector;
        temp_vect_3 = exemplar_displacement_vector;
        temp_vect_4 = exemplar_displacement_vector;
        temp_vect_5 = exemplar_density_vector;
        temp_vect_6 = exemplar_density_vector;
        temp_vect_7 = exemplar_density_vector;


    }

    template<int dim>
    void HMatrix<dim>::vmult(LA::MPI::Vector &dst, const LA::MPI::Vector &src) const
    {
        if (Input::solver_choice==SolverOptions::inexact_K_with_inexact_A_gmres)
        {
            c_mat.vmult(temp_vect_1,src);
            e_mat.vmult(temp_vect_2,src);

            SolverControl            solver_control_1(Input::a_inv_iterations, temp_vect_1.l2_norm()*1e-6);
            SolverCG<LA::MPI::Vector> a_solver_cg_1(solver_control_1);

            try
            {
//                a_solver_cg_1.solve(a_mat,temp_vect_3,temp_vect_1,pre_amg);
                a_inv_mf_gmg.vmult(temp_vect_3,temp_vect_1);
            } catch (std::exception &exc)
            {

            }
            SolverControl            solver_control_2(Input::a_inv_iterations, temp_vect_2.l2_norm()*1e-6);
            SolverCG<LA::MPI::Vector> a_solver_cg_2(solver_control_2);
            try
            {
//                a_solver_cg_2.solve(a_mat,temp_vect_4,temp_vect_2,pre_amg);
                a_inv_mf_gmg.vmult(temp_vect_4,temp_vect_2);
            } catch (std::exception &exc)
            {

            }

            c_mat.Tvmult(temp_vect_6,temp_vect_4);
            e_mat.Tvmult(temp_vect_5,temp_vect_3);

            b_mat.vmult(temp_vect_7,src);
            dst =  temp_vect_7 - temp_vect_6 - temp_vect_5;
        }
        else
        {
            c_mat.vmult(temp_vect_1,src);
            e_mat.vmult(temp_vect_2,src);

            a_inv_direct.vmult(temp_vect_3,temp_vect_1);
            a_inv_direct.vmult(temp_vect_4,temp_vect_2);

            c_mat.Tvmult(temp_vect_6,temp_vect_4);
            e_mat.Tvmult(temp_vect_5,temp_vect_3);

            b_mat.vmult(temp_vect_7,src);
            dst =  temp_vect_7 - temp_vect_6 - temp_vect_5;
        }



    }


    template<int dim>
    void HMatrix<dim>::Tvmult(LA::MPI::Vector &dst, const LA::MPI::Vector &src) const
    {
        if (Input::solver_choice==SolverOptions::inexact_K_with_inexact_A_gmres)
        {
            c_mat.vmult(temp_vect_1,src);
            e_mat.vmult(temp_vect_2,src);

            SolverControl            solver_control_1(Input::a_inv_iterations, temp_vect_1.l2_norm()*1e-6);
            SolverCG<LA::MPI::Vector> a_solver_cg_1(solver_control_1);

            try
            {
//                a_solver_cg_1.solve(a_mat,temp_vect_3,temp_vect_1,pre_amg);
                a_inv_mf_gmg.vmult(temp_vect_3,temp_vect_1);

            } catch (std::exception &exc)
            {

            }
            SolverControl            solver_control_2(Input::a_inv_iterations, temp_vect_2.l2_norm()*1e-6);
            SolverCG<LA::MPI::Vector> a_solver_cg_2(solver_control_2);
            try
            {
//                a_solver_cg_2.solve(a_mat,temp_vect_4,temp_vect_2,pre_amg);
                a_inv_mf_gmg.vmult(temp_vect_4,temp_vect_2);

            } catch (std::exception &exc)
            {

            }

            c_mat.Tvmult(temp_vect_6,temp_vect_4);
            e_mat.Tvmult(temp_vect_5,temp_vect_3);

            b_mat.vmult(temp_vect_7,src);
            dst =  temp_vect_7 - temp_vect_6 - temp_vect_5;
        }
        else
        {
            c_mat.vmult(temp_vect_1,src);
            e_mat.vmult(temp_vect_2,src);

            a_inv_direct.vmult(temp_vect_3,temp_vect_1);
            a_inv_direct.vmult(temp_vect_4,temp_vect_2);

            c_mat.Tvmult(temp_vect_6,temp_vect_4);
            e_mat.Tvmult(temp_vect_5,temp_vect_3);

            b_mat.vmult(temp_vect_7,src);
            dst =  temp_vect_7 - temp_vect_6 - temp_vect_5;
        }

    }

    // ******************     JinvMatrix     ***********************
    template<int dim>
    JinvMatrix<dim>::JinvMatrix(HMatrix<dim> &h_mat_in, GMatrix &g_mat_in, const LA::MPI::SparseMatrix &d_m_mat_in, LA::MPI::SparseMatrix &d_m_inv_mat_in)
            :
            h_mat(h_mat_in),
            g_mat(g_mat_in),
            d_m_mat(d_m_mat_in),
            d_m_inv_mat(d_m_inv_mat_in)
    {

    }

    template<int dim>
    void
    JinvMatrix<dim>::initialize(LA::MPI::Vector &exemplar_density_vector)
    {
        temp_vect_1 = exemplar_density_vector;
        temp_vect_2 = exemplar_density_vector;
        temp_vect_3 = exemplar_density_vector;
        temp_vect_4 = exemplar_density_vector;

    }

    template<int dim>
    void JinvMatrix<dim>::vmult(LA::MPI::Vector &dst, const LA::MPI::Vector &src) const
    {
        g_mat.vmult(temp_vect_1,src);
        d_m_inv_mat.vmult(temp_vect_2,temp_vect_1);
        h_mat.vmult(temp_vect_3,temp_vect_2);
        d_m_mat.vmult(temp_vect_4,src);

        dst = -1*temp_vect_4 - temp_vect_3;
    }

    template<int dim>
    void JinvMatrix<dim>::Tvmult(LA::MPI::Vector &dst, const LA::MPI::Vector &src) const
    {
        h_mat.vmult(temp_vect_1,src);
        d_m_inv_mat.vmult(temp_vect_2,temp_vect_1);
        g_mat.vmult(temp_vect_3,temp_vect_2);
        d_m_mat.vmult(temp_vect_4,src);

        dst = -1*temp_vect_4 - temp_vect_3;
    }

    // ******************     KinvMatrix     ***********************
    template<int dim>
    KinvMatrix<dim>::KinvMatrix(HMatrix<dim> &h_mat_in, GMatrix &g_mat_in, const LA::MPI::SparseMatrix &d_m_mat_in, LA::MPI::SparseMatrix &d_m_inv_mat_in)
            :
            h_mat(h_mat_in),
            g_mat(g_mat_in),
            d_m_mat(d_m_mat_in),
            d_m_inv_mat(d_m_inv_mat_in)
    {

    }

    template<int dim>
    void
    KinvMatrix<dim>::initialize(LA::MPI::Vector &exemplar_density_vector)
    {
        temp_vect_1 = exemplar_density_vector;
        temp_vect_2 = exemplar_density_vector;
        temp_vect_3 = exemplar_density_vector;
        temp_vect_4 = exemplar_density_vector;

    }

    template<int dim>
    void KinvMatrix<dim>::vmult(LA::MPI::Vector &dst, const LA::MPI::Vector &src) const
    {
        h_mat.vmult(temp_vect_1,src);
        d_m_inv_mat.vmult(temp_vect_2,temp_vect_1);
        g_mat.vmult(temp_vect_3,temp_vect_2);
        d_m_mat.vmult(temp_vect_4,src);

        dst = -1*temp_vect_4 - temp_vect_3;
    }

    template<int dim>
    void KinvMatrix<dim>::Tvmult(LA::MPI::Vector &dst, const LA::MPI::Vector &src) const
    {
        g_mat.vmult(temp_vect_1,src);
        d_m_inv_mat.vmult(temp_vect_2,temp_vect_1);
        h_mat.vmult(temp_vect_3,temp_vect_2);
        d_m_mat.vmult(temp_vect_4,src);

        dst = -1*temp_vect_4 - temp_vect_3;
    }

}



template class SAND::TopOptSchurPreconditioner<2>;
template class SAND::TopOptSchurPreconditioner<3>;

template class SAND::AInvMatMFGMG<2>;
template class SAND::AInvMatMFGMG<3>;

template class SAND::JinvMatrix<2>;
template class SAND::JinvMatrix<3>;

template class SAND::KinvMatrix<2>;
template class SAND::KinvMatrix<3>;

template class SAND::HMatrix<2>;
template class SAND::HMatrix<3>;

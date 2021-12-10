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
    using namespace dealii;
    template<int dim>
    TopOptSchurPreconditioner<dim>::TopOptSchurPreconditioner(LA::MPI::BlockSparseMatrix &matrix_in)
            :
            mpi_communicator(MPI_COMM_WORLD),
            system_matrix(matrix_in),
            n_rows(0),
            n_columns(0),
            n_block_rows(0),
            n_block_columns(0),
            other_solver_control(100000, 1e-6),
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
            pcout(std::cout,(Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
            timer(std::cout, TimerOutput::summary, TimerOutput::wall_times)
    {

    }

    template<int dim>
    void TopOptSchurPreconditioner<dim>::initialize(LA::MPI::BlockSparseMatrix &matrix, const std::map<types::global_dof_index, double> &boundary_values,const DoFHandler<dim> &dof_handler, const LA::MPI::BlockVector &locally_relevant_state, const LA::MPI::BlockVector &distributed_state)
    {
        TimerOutput::Scope t(timer, "initialize");
        {

            TimerOutput::Scope t(timer, "diag stuff");
            const types::global_dof_index disp_start_index = system_matrix.get_row_indices().block_start(
                    SolutionBlocks::displacement);
            const types::global_dof_index disp_mult_start_index = system_matrix.get_row_indices().block_start(
                    SolutionBlocks::displacement_multiplier);
            for (auto&[dof_index, boundary_value]: boundary_values) {

                const types::global_dof_index n_u = system_matrix.block(SolutionBlocks::displacement,
                                                                        SolutionBlocks::displacement).m();
                if ((dof_index >= disp_start_index) && (dof_index < disp_start_index + n_u)) {
                    double diag_val = system_matrix.block(SolutionBlocks::displacement,
                                                          SolutionBlocks::displacement).el(
                            dof_index - disp_start_index, dof_index - disp_start_index);
                    system_matrix.block(SolutionBlocks::displacement, SolutionBlocks::displacement_multiplier).set(
                            dof_index - disp_start_index, dof_index - disp_start_index, diag_val);
                } else if ((dof_index >= disp_mult_start_index) && (dof_index < disp_mult_start_index + n_u)) {
                    double diag_val = system_matrix.block(SolutionBlocks::displacement_multiplier,
                                                          SolutionBlocks::displacement_multiplier).el(
                            dof_index - disp_mult_start_index, dof_index - disp_mult_start_index);
                    system_matrix.block(SolutionBlocks::displacement_multiplier, SolutionBlocks::displacement).set(
                            dof_index - disp_mult_start_index, dof_index - disp_mult_start_index, diag_val);
                }
            }

            //set diagonal to 0?
            for (auto&[dof_index, boundary_value]: boundary_values) {
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

//            system_matrix.compress(VectorOperation::insert);

//            const unsigned int m = a_mat.m();
//            const unsigned int n = a_mat.n();
//            std::ofstream Xmat("a_mat_par.csv");
//            for (unsigned int i = 0; i < m; i++)
//            {
//                Xmat << a_mat.el(i, 0);
//                for (unsigned int j = 1; j < n; j++)
//                {
//                    Xmat << "," << a_mat.el(i, j);
//                }
//                Xmat << "\n";
//            }
//            Xmat.close();

        }
        if (Input::solver_choice==SolverOptions::inexact_K_with_inexact_A_gmres)
        {

        }
        else
        {
            TimerOutput::Scope t(timer, "build A inv");
            a_inv_direct.initialize(a_mat);
        }
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




            double l_global[n_p]= {0};
            double lm_global[n_p]= {0};
            double u_global[n_p]= {0};
            double um_global[n_p]= {0};
            double m_global[n_p]= {0};

            double l[n_p]= {0};
            double lm[n_p]= {0};
            double u[n_p]= {0};
            double um[n_p]= {0};
            double m[n_p]= {0};

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

            MPI_Allreduce(&lm, &lm_global, n_p, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&l, &l_global, n_p, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&um, &um_global, n_p, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&u, &u_global, n_p, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&m, &m_global, n_p, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

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

        pcout << "compressed" << std::endl;

        pre_j=distributed_state.block(SolutionBlocks::density);
        pre_k=distributed_state.block(SolutionBlocks::density);
        g_d_m_inv_density=distributed_state.block(SolutionBlocks::density);
        k_g_d_m_inv_density=distributed_state.block(SolutionBlocks::density);
        LinearOperator<VectorType,VectorType,PayloadType> op_g;
        LinearOperator<VectorType,VectorType,PayloadType> op_h;
        LinearOperator<VectorType,VectorType,PayloadType> op_f;
        LinearOperator<VectorType,VectorType,PayloadType> op_d_8;
        op_f = linear_operator<VectorType,VectorType,PayloadType>(f_mat);
        op_d_8 = linear_operator<VectorType,VectorType,PayloadType>(d_8_mat);
        op_g = linear_operator<VectorType,VectorType,PayloadType>(f_mat) * linear_operator<VectorType,VectorType,PayloadType>(d_8_mat) * transpose_operator<VectorType, VectorType, PayloadType>(linear_operator<VectorType,VectorType,PayloadType>(f_mat));

        pcout << "ops made" << std::endl;

        if (Input::solver_choice==SolverOptions::inexact_K_with_inexact_A_gmres)
        {
            SolverControl            solver_control(100000, 1e-6);
            SolverCG<LA::MPI::Vector> a_solver_cg(solver_control);
            auto a_inv_op = inverse_operator(linear_operator<VectorType,VectorType,PayloadType>(a_mat),a_solver_cg,PreconditionIdentity());
            op_h = linear_operator<VectorType,VectorType,PayloadType>(b_mat)
                   - transpose_operator<VectorType, VectorType, PayloadType>(linear_operator<VectorType, VectorType, PayloadType>(c_mat)) * a_inv_op * linear_operator<VectorType,VectorType,PayloadType>(e_mat)
                   - transpose_operator<VectorType, VectorType, PayloadType>(linear_operator<VectorType, VectorType, PayloadType>(e_mat)) * a_inv_op * linear_operator<VectorType,VectorType,PayloadType>(c_mat);
        }
        else
        {
            auto a_inv_op =linear_operator<VectorType,VectorType,PayloadType>(a_inv_direct);
            op_h = linear_operator<VectorType,VectorType,PayloadType>(b_mat)
                   - transpose_operator<VectorType, VectorType, PayloadType>(c_mat) * a_inv_op * linear_operator<VectorType, VectorType, PayloadType>(e_mat)
                   - transpose_operator<VectorType, VectorType, PayloadType>(e_mat) * a_inv_op * linear_operator<VectorType, VectorType, PayloadType>(c_mat);
        }


        if(Input::solver_choice == SolverOptions::inexact_K_with_inexact_A_gmres || Input::solver_choice == SolverOptions::inexact_K_with_exact_A_gmres)
        {

        }
        else
        {
//            {
//                TimerOutput::Scope t(timer, "build g_mat");
//                g_mat.reinit(b_mat.n(), b_mat.n());
//                build_matrix_element_by_element(op_g, g_mat);
//            }
//
//
//            {
//                TimerOutput::Scope t(timer, "build h_mat");
//                h_mat.reinit(b_mat.n(), b_mat.n());
//                build_matrix_element_by_element(op_h, h_mat);
//            }
//
//            {
//                TimerOutput::Scope t(timer, "build k_inv_mat");
//                auto op_k_inv = -1 * op_g * linear_operator<VectorType,VectorType,PayloadType>(d_m_inv_mat) * op_h -
//                                linear_operator<VectorType,VectorType,PayloadType>(d_m_mat);
//                k_inv_mat.reinit(b_mat.n(), b_mat.n());
//                build_matrix_element_by_element(op_k_inv, k_inv_mat);
//            }
        }

        if (Input::solver_choice == SolverOptions::exact_preconditioner_with_gmres)
        {
            TimerOutput::Scope t(timer, "invert k_mat");
            k_mat.copy_from(k_inv_mat);
            k_mat.invert();
        }
    }


    template<int dim>
    void TopOptSchurPreconditioner<dim>::vmult(LA::MPI::BlockVector &dst, const LA::MPI::BlockVector &src) const {
        LA::MPI::BlockVector temp_src;
        pcout << "vmult" << std::endl;
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
//        auto temp = src.block(SolutionBlocks::unfiltered_density);
//        temp = linear_operator<VectorType,VectorType,PayloadType>(f_mat) * linear_operator<VectorType,VectorType,PayloadType>(d_8_mat) * src.block(SolutionBlocks::unfiltered_density);
//        pcout <<std::endl <<std::endl<< "temp output:" << std::endl;
//        temp.print(pcout);
//        pcout << std::endl;


        dst.block(SolutionBlocks::unfiltered_density_multiplier) = dst_temp.block(SolutionBlocks::unfiltered_density_multiplier)
                - linear_operator<VectorType,VectorType,PayloadType>(f_mat)*linear_operator<VectorType,VectorType,PayloadType>(d_8_mat) * src.block(SolutionBlocks::unfiltered_density);

    }

    template<int dim>
    void TopOptSchurPreconditioner<dim>::vmult_step_3(LA::MPI::BlockVector &dst, const LA::MPI::BlockVector &src) const {
        dst = src;

        if (Input::solver_choice == SolverOptions::inexact_K_with_inexact_A_gmres)
        {
            SolverControl            solver_control_1(100000, 1e-6);
            SolverControl            solver_control_2(100000, 1e-6);
            SolverCG<LA::MPI::Vector> a_solver_cg_1(solver_control_1);
            SolverCG<LA::MPI::Vector> a_solver_cg_2(solver_control_2);
            auto dst_temp = dst;
            a_solver_cg_1.solve(a_mat,dst_temp.block(SolutionBlocks::displacement_multiplier),src.block(SolutionBlocks::displacement_multiplier),PreconditionIdentity());
            a_solver_cg_2.solve(a_mat,dst_temp.block(SolutionBlocks::displacement),src.block(SolutionBlocks::displacement),PreconditionIdentity());
            c_mat.Tvmult(dst_temp.block(SolutionBlocks::density_upper_slack),dst_temp.block(SolutionBlocks::displacement_multiplier));
            e_mat.Tvmult(dst_temp.block(SolutionBlocks::density_lower_slack),dst_temp.block(SolutionBlocks::displacement));

            dst.block(SolutionBlocks::density) = dst_temp.block(SolutionBlocks::density) - dst_temp.block(SolutionBlocks::density_upper_slack) - dst_temp.block(SolutionBlocks::density_lower_slack);


        }
        else
        {
            auto dst_temp = dst;
            dst.block(SolutionBlocks::density) = dst_temp.block(SolutionBlocks::density)
                    - transpose_operator<VectorType,VectorType,PayloadType>(e_mat) * linear_operator<VectorType,VectorType,PayloadType>(a_inv_direct)* src.block(SolutionBlocks::displacement)
                    - transpose_operator<VectorType,VectorType,PayloadType>(c_mat) * linear_operator<VectorType,VectorType,PayloadType>(a_inv_direct)* src.block(SolutionBlocks::displacement_multiplier);
        }

    }

    template<int dim>
    void TopOptSchurPreconditioner<dim>::vmult_step_4(LA::MPI::BlockVector &dst, const LA::MPI::BlockVector &src) const {
        dst = src;
        auto dst_temp = dst;
        auto k_density_mult =  src.block(SolutionBlocks::density);



        if (Input::solver_choice == SolverOptions::exact_preconditioner_with_gmres)
        {
//            g_d_m_inv_density = linear_operator<VectorType,VectorType,PayloadType>(g_mat) * linear_operator<VectorType,VectorType,PayloadType>(d_m_inv_mat) * src.block(SolutionBlocks::density);
//            k_g_d_m_inv_density = linear_operator<VectorType,VectorType,PayloadType>(k_mat) * g_d_m_inv_density;
//            k_density_mult = linear_operator<VectorType,VectorType,PayloadType>(k_mat) * src.block(SolutionBlocks::unfiltered_density_multiplier);
        }

        else if (Input::solver_choice == SolverOptions::inexact_K_with_exact_A_gmres)
        {
            auto op_d_8 = linear_operator<VectorType,VectorType,PayloadType>(d_8_mat);
            auto op_f = linear_operator<VectorType,VectorType,PayloadType>(f_mat);
            auto op_b = linear_operator<VectorType,VectorType,PayloadType>(b_mat);
            auto op_c = linear_operator<VectorType,VectorType,PayloadType>(c_mat);
            auto op_a_inv = linear_operator<VectorType,VectorType,PayloadType>(a_inv_direct);
            auto op_e = linear_operator<VectorType,VectorType,PayloadType>(e_mat);
            auto op_d_m= linear_operator<VectorType,VectorType,PayloadType>(d_m_mat);
            auto op_d_m_inv= linear_operator<VectorType,VectorType,PayloadType>(d_m_inv_mat);

            auto op_g = op_f * op_d_8 *
                    transpose_operator(op_f);


            auto op_h = op_b
                   - transpose_operator(op_c) * op_a_inv * op_e
                   - transpose_operator(op_e) * op_a_inv * op_c;

            auto op_k_inv = -1 * op_g *op_d_m_inv * op_h - op_d_m;

            g_d_m_inv_density = op_g * op_d_m_inv * src.block(SolutionBlocks::density);
            SolverControl step_4_gmres_control_1 (100000, g_d_m_inv_density.l2_norm()*1e-6);
            SolverGMRES<LA::MPI::Vector> step_4_gmres_1 (step_4_gmres_control_1);
            try {
                k_g_d_m_inv_density = inverse_operator(op_k_inv, step_4_gmres_1, PreconditionIdentity()) *
                                      g_d_m_inv_density;
            }
            catch (std::exception &exc)
            {
                std::cerr << "Failure of linear solver step_4_gmres_1" << std::endl;
                pcout << "first residual: " << step_4_gmres_control_1.initial_value() << std::endl;
                pcout << "last residual: " << step_4_gmres_control_1.last_value() << std::endl;
                throw;
            }
            SolverControl step_4_gmres_control_2 (100000, src.block(SolutionBlocks::unfiltered_density_multiplier).l2_norm()*1e-6);
            SolverGMRES<LA::MPI::Vector> step_4_gmres_2 (step_4_gmres_control_2);
            try {
                k_density_mult = inverse_operator(op_k_inv,step_4_gmres_2, PreconditionIdentity()) *
                                 src.block(SolutionBlocks::unfiltered_density_multiplier);
            } catch (std::exception &exc)
            {
                std::cerr << "Failure of linear solver step_4_gmres_2" << std::endl;
                pcout << "first residual: " << step_4_gmres_control_2.initial_value() << std::endl;
                pcout << "last residual: " << step_4_gmres_control_2.last_value() << std::endl;
                throw;
            }
        }
        else if (Input::solver_choice == SolverOptions::inexact_K_with_inexact_A_gmres)
        {

            SolverControl            solver_control(100000, 1e-6);
            SolverCG<LA::MPI::Vector> a_solver_cg(solver_control);

            auto preconditioner = PreconditionIdentity();

            auto a_inv_op = inverse_operator(linear_operator<VectorType,VectorType,PayloadType>(a_mat),a_solver_cg, preconditioner);


            auto op_g = linear_operator<VectorType,VectorType,PayloadType>(f_mat) * linear_operator<VectorType,VectorType,PayloadType>(d_8_mat) *
                        transpose_operator<VectorType, VectorType, PayloadType>(f_mat);

            auto op_h = linear_operator<VectorType,VectorType,PayloadType>(b_mat)
                        - transpose_operator<VectorType, VectorType, PayloadType>(c_mat) * a_inv_op * linear_operator<VectorType,VectorType,PayloadType>(e_mat)
                        - transpose_operator<VectorType, VectorType, PayloadType>(e_mat) * a_inv_op * linear_operator<VectorType,VectorType,PayloadType>(c_mat);

            auto op_k_inv = -1 * op_g * linear_operator<VectorType,VectorType,PayloadType>(d_m_inv_mat) * op_h - linear_operator<VectorType,VectorType,PayloadType>(d_m_mat);

            SolverControl step_4_gmres_control_1 (10000, std::max(g_d_m_inv_density.l2_norm()*1e-6,1e-6));
            SolverGMRES<LA::MPI::Vector> step_4_gmres_1 (step_4_gmres_control_1);
            try {
                k_g_d_m_inv_density = inverse_operator(op_k_inv, step_4_gmres_1, PreconditionIdentity()) *
                                      g_d_m_inv_density;
            } catch (std::exception &exc)
            {
                std::cerr << "Failure of linear solver step_4_gmres_1" << std::endl;
                pcout << "first residual: " << step_4_gmres_control_1.initial_value() << std::endl;
                pcout << "last residual: " << step_4_gmres_control_1.last_value() << std::endl;
                throw;
            }

            SolverControl step_4_gmres_control_2 (10000, std::max(src.block(SolutionBlocks::unfiltered_density_multiplier).l2_norm()*1e-6,1e-6));
            SolverGMRES<LA::MPI::Vector> step_4_gmres_2 (step_4_gmres_control_2);
            try {

                k_density_mult = inverse_operator(op_k_inv,step_4_gmres_2, PreconditionIdentity()) *
                                 src.block(SolutionBlocks::unfiltered_density_multiplier);
            } catch (std::exception &exc)
            {
                std::cerr << "Failure of linear solver step_4_gmres_2" << std::endl;
                pcout << "first residual: " << step_4_gmres_control_2.initial_value() << std::endl;
                pcout << "last residual: " << step_4_gmres_control_2.last_value() << std::endl;
                throw;
            }
        }
        else
        {
            pcout << "shouldn't get here";
            throw;
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
            if(Input::solver_choice == SolverOptions::inexact_K_with_inexact_A_gmres)
            {
                SolverControl            solver_control(100000, 1e-6);
                SolverCG<LA::MPI::Vector> a_solver_cg(solver_control);

                auto a_inv_op = inverse_operator(linear_operator<VectorType,VectorType,PayloadType>(a_mat),a_solver_cg, PreconditionIdentity());

                dst.block(SolutionBlocks::displacement) = a_inv_op * src.block(SolutionBlocks::displacement_multiplier);
                dst.block(SolutionBlocks::displacement_multiplier) = a_inv_op * src.block(SolutionBlocks::displacement);
            } else
            {
                dst.block(SolutionBlocks::displacement) = linear_operator<VectorType,VectorType,PayloadType>(a_inv_direct) * src.block(SolutionBlocks::displacement_multiplier);
                dst.block(SolutionBlocks::displacement_multiplier) = linear_operator<VectorType,VectorType,PayloadType>(a_inv_direct) * src.block(SolutionBlocks::displacement);
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

            else if (Input::solver_choice == SolverOptions::inexact_K_with_exact_A_gmres)
            {
                auto op_g = linear_operator<VectorType,VectorType,PayloadType>(f_mat) * linear_operator<VectorType,VectorType,PayloadType>(d_8_mat) *
                            transpose_operator(linear_operator<VectorType,VectorType,PayloadType>(f_mat));

                auto op_h = linear_operator<VectorType,VectorType,PayloadType>(b_mat)
                            - transpose_operator<VectorType, VectorType, PayloadType>(c_mat) *
                                    linear_operator<VectorType,VectorType,PayloadType>(a_inv_direct) * linear_operator<VectorType,VectorType,PayloadType>(e_mat)
                            - transpose_operator<VectorType, VectorType, PayloadType>(e_mat) * linear_operator<VectorType,VectorType,PayloadType>(a_inv_direct) * linear_operator<VectorType,VectorType,PayloadType>(c_mat);

                auto op_k_inv = -1 * op_g * linear_operator<VectorType,VectorType,PayloadType>(d_m_inv_mat) * op_h - linear_operator<VectorType,VectorType,PayloadType>(d_m_mat);

                pre_j = src.block(SolutionBlocks::density) + op_h * linear_operator<VectorType,VectorType,PayloadType>(d_m_inv_mat) * src.block(SolutionBlocks::unfiltered_density_multiplier);
                auto pre_pre_k = pre_k;
                pre_pre_k = -1 * op_g * linear_operator<VectorType,VectorType,PayloadType>(d_m_inv_mat) * src.block(SolutionBlocks::density);
                pre_k =  pre_pre_k + src.block(SolutionBlocks::unfiltered_density_multiplier);


                SolverControl step_5_gmres_control_1 (100000, pre_j.l2_norm()*1e-6);
                SolverGMRES<LA::MPI::Vector> step_5_gmres_1 (step_5_gmres_control_1);
                try {
                    dst.block(SolutionBlocks::unfiltered_density_multiplier) = inverse_operator(transpose_operator<VectorType, VectorType, PayloadType>(op_k_inv), step_5_gmres_1, PreconditionIdentity()) *
                                                                               pre_j;
                } catch (std::exception &exc)
                {
                    std::cerr << "Failure of linear solver step_5_gmres_1" << std::endl;
                    pcout << "first residual: " << step_5_gmres_control_1.initial_value() << std::endl;
                    pcout << "last residual: " << step_5_gmres_control_1.last_value() << std::endl;
                    throw;
                }


                SolverControl step_5_gmres_control_2 (100000, pre_k.l2_norm()*1e-6);
                SolverGMRES<LA::MPI::Vector> step_5_gmres_2 (step_5_gmres_control_2);
                try {
                    dst.block(SolutionBlocks::density) = inverse_operator(op_k_inv, step_5_gmres_2, PreconditionIdentity()) * pre_k;
                } catch (std::exception &exc)
                {
                    std::cerr << "Failure of linear solver step_5_gmres_2" << std::endl;
                    pcout << "first residual: " << step_5_gmres_control_2.initial_value() << std::endl;
                    pcout << "last residual: " << step_5_gmres_control_2.last_value() << std::endl;
                    throw;
                }
            }
            else if (Input::solver_choice == SolverOptions::inexact_K_with_inexact_A_gmres)
            {

                SolverControl            solver_control(100000, 1e-6);
                SolverCG<LA::MPI::Vector> a_solver_cg (solver_control);
                auto a_inv_op = inverse_operator(linear_operator<VectorType,VectorType,PayloadType>(a_mat),a_solver_cg, PreconditionIdentity());

                auto op_g = linear_operator<VectorType,VectorType,PayloadType>(f_mat) * linear_operator<VectorType,VectorType,PayloadType>(d_8_mat) *
                            transpose_operator<VectorType, VectorType, PayloadType>(f_mat);

                auto op_h = linear_operator<VectorType,VectorType,PayloadType>(b_mat)
                            - transpose_operator<VectorType, VectorType, PayloadType>(c_mat) * a_inv_op * linear_operator<VectorType,VectorType,PayloadType>(e_mat)
                            - transpose_operator<VectorType, VectorType, PayloadType>(e_mat) * a_inv_op * linear_operator<VectorType,VectorType,PayloadType>(c_mat);

                auto op_k_inv = -1 * op_g * linear_operator<VectorType,VectorType,PayloadType>(d_m_inv_mat) * op_h - linear_operator<VectorType,VectorType,PayloadType>(d_m_mat);

                pre_j = src.block(SolutionBlocks::density) + op_h * linear_operator<VectorType,VectorType,PayloadType>(d_m_inv_mat) * src.block(SolutionBlocks::unfiltered_density_multiplier);
                pre_k = -1* op_g * linear_operator<VectorType,VectorType,PayloadType>(d_m_inv_mat) * src.block(SolutionBlocks::density) + src.block(SolutionBlocks::unfiltered_density_multiplier);

                SolverControl step_5_gmres_control_1 (10000, std::max(pre_j.l2_norm()*1e-6,1e-6));
                SolverGMRES<LA::MPI::Vector> step_5_gmres_1 (step_5_gmres_control_1);
                try {
                    dst.block(SolutionBlocks::unfiltered_density_multiplier) = inverse_operator(transpose_operator<VectorType,VectorType,PayloadType>(op_k_inv), step_5_gmres_1, PreconditionIdentity()) *
                                                                               pre_j;
                } catch (std::exception &exc)
                {
                    std::cerr << "Failure of linear solver step_5_gmres_1" << std::endl;
                    pcout << "first residual: " << step_5_gmres_control_1.initial_value() << std::endl;
                    pcout << "last residual: " << step_5_gmres_control_1.last_value() << std::endl;
                    throw;
                }

                SolverControl step_5_gmres_control_2 (10000, std::max(pre_k.l2_norm()*1e-6,1e-6));
                SolverGMRES<LA::MPI::Vector> step_5_gmres_2 (step_5_gmres_control_2);
                try {
                    dst.block(SolutionBlocks::density) = inverse_operator(op_k_inv, step_5_gmres_2, PreconditionIdentity()) *
                                                         pre_k;
                } catch (std::exception &exc)
                {
                    std::cerr << "Failure of linear solver step_5_gmres_2" << std::endl;
                    pcout << "first residual: " << step_5_gmres_control_2.initial_value() << std::endl;
                    pcout << "last residual: " << step_5_gmres_control_2.last_value() << std::endl;
                    throw;
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

//        print_matrix(std::string("OAmat.csv"),a_mat);
//        print_matrix(std::string("OBmat.csv"),b_mat);
//        print_matrix(std::string("OCmat.csv"),c_mat);
//        print_matrix(std::string("OEmat.csv"),e_mat);
//        print_matrix(std::string("OFmat.csv"),f_mat);
        FullMatrix<double> g_mat;
        FullMatrix<double> h_mat;
        FullMatrix<double> k_inv_mat;
        g_mat.reinit(b_mat.m(),b_mat.n());
        h_mat.reinit(b_mat.m(),b_mat.n());
        k_inv_mat.reinit(b_mat.m(),b_mat.n());
        auto op_g = linear_operator<VectorType,VectorType,PayloadType>(f_mat) * linear_operator<VectorType,VectorType,PayloadType>(d_8_mat) *
                    transpose_operator(linear_operator<VectorType,VectorType,PayloadType>(f_mat));

        auto op_h = linear_operator<VectorType,VectorType,PayloadType>(b_mat)
                    - transpose_operator(linear_operator<VectorType,VectorType,PayloadType>(c_mat)) * linear_operator<VectorType,VectorType,PayloadType>(a_inv_direct) * linear_operator<VectorType,VectorType,PayloadType>(e_mat)
                    - transpose_operator(linear_operator<VectorType,VectorType,PayloadType>(e_mat)) * linear_operator<VectorType,VectorType,PayloadType>(a_inv_direct) * linear_operator<VectorType,VectorType,PayloadType>(c_mat);

        auto op_k_inv = -1 * op_g * linear_operator<VectorType,VectorType,PayloadType>(d_m_inv_mat) * op_h - linear_operator<VectorType,VectorType,PayloadType>(d_m_mat);
//        build_matrix_element_by_element(op_g,g_mat, src.block(SolutionBlocks::density));
//        build_matrix_element_by_element(op_h,h_mat, src.block(SolutionBlocks::density));
//        build_matrix_element_by_element(op_k_inv,k_inv_mat, src.block(SolutionBlocks::density));
//        print_matrix(std::string("OGmat.csv"),g_mat);
//        print_matrix(std::string("OHmat.csv"),h_mat);
//        print_matrix(std::string("OKinvmat.csv"),k_inv_mat);


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

}
template class SAND::TopOptSchurPreconditioner<2>;
template class SAND::TopOptSchurPreconditioner<3>;

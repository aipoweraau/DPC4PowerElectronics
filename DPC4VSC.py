# """
# Learning to optimize parametric Quadratic Programming (pQP) and
# parametric Quadratically Constrained Quadratic Programming (pQCQP)
# problems using Neuromancer.

# Problem formulation pQP:
#     minimize     x^2 + y^2
#     subject to
#                 -x - y + p1 <= 0,
#                 x + y - p1 - 5 <= 0,
#                 x - y + p2 - 5 <= 0,
#                 -x + y - p2 <= 0

# Problem formulation pQCQP:
#     minimize     x^2 + y^2
#     subject to
#             -x - y + p1 <= 0,
#             x^2 + y^2 <= p2^2

#     problem parameters:            p1, p2
#     problem decition variables:    x, y
# """
# import sys
# # 手动添加 Neuromancer 的安装路径
# sys.path.append(r"c:\users\du47tw\appdata\local\anaconda3\envs\neuromancer\lib\site-packages")


# import cvxpy as cp
# import numpy as np
# import time
# import torch
# import torch.nn as nn
# import neuromancer.slim as slim
# import matplotlib.pyplot as plt
# import matplotlib.patheffects as patheffects

# from neuromancer.trainer import Trainer
# from neuromancer.problem import Problem
# from neuromancer.constraint import variable
# from neuromancer.dataset import DictDataset
# from neuromancer.loss import PenaltyLoss
# from neuromancer.modules import blocks
# from neuromancer.system import Node


# if __name__ == "__main__":

#     problem_type = 'pQP'   # select from 'pQP' or 'pQCQP'

#     """
#     # # #  Dataset
#     """
#     data_seed = 408
#     np.random.seed(data_seed)
#     nsim = 3000  # number of datapoints: increase sample density for more robust results
#     # create dictionaries with sampled datapoints with uniform distribution
#     p_low, p_high = 1.0, 11.0
#     samples_train = {"p1": torch.FloatTensor(nsim, 1).uniform_(p_low, p_high),
#                       "p2": torch.FloatTensor(nsim, 1).uniform_(p_low, p_high)}
#     samples_dev = {"p1": torch.FloatTensor(nsim, 1).uniform_(p_low, p_high),
#                     "p2": torch.FloatTensor(nsim, 1).uniform_(p_low, p_high)}
#     samples_test = {"p1": torch.FloatTensor(nsim, 1).uniform_(p_low, p_high),
#                     "p2": torch.FloatTensor(nsim, 1).uniform_(p_low, p_high)}
#     # create named dictionary datasets
#     train_data = DictDataset(samples_train, name='train')
#     dev_data = DictDataset(samples_dev, name='dev')
#     test_data = DictDataset(samples_test, name='test')
#     # create torch dataloaders for the Trainer
#     train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, num_workers=0,
#                                                 collate_fn=train_data.collate_fn, shuffle=True)
#     dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=32, num_workers=0,
#                                               collate_fn=dev_data.collate_fn, shuffle=True)
#     test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, num_workers=0,
#                                               collate_fn=test_data.collate_fn, shuffle=True)
#     # note: training quality will depend on the DataLoader parameters such as batch size and shuffle

#     # visualize taining and test samples for 2D parametric space
#     a_train = samples_train['p1'].numpy()
#     p_train = samples_train['p2'].numpy()
#     a_dev = samples_dev['p1'].numpy()
#     p_dev = samples_dev['p2'].numpy()
#     plt.figure()
#     plt.scatter(a_train, p_train, s=2., c='blue', marker='o')
#     plt.scatter(a_dev, p_dev, s=2., c='red', marker='o')
#     plt.title('Sampled parametric space for training')
#     plt.xlim(p_low, p_high)
#     plt.ylim(p_low, p_high)
#     plt.grid(True)
#     plt.xlabel('p1')
#     plt.ylabel('p2')
#     plt.legend(['train', 'test'], loc='upper right')
#     plt.show()
#     plt.show(block=True)


#     """
#     # # #  pQP primal solution map architecture
#     """
#     # define neural architecture for the solution map
#     func = blocks.MLP(insize=2, outsize=2,
#                     bias=True,
#                     linear_map=slim.maps['linear'],
#                     nonlin=nn.ReLU,
#                     hsizes=[80] * 4)
#     # define symbolic solution map
#     sol_map = Node(func, ['p1', 'p2'], ['x'], name='map')
#     # trainable components of the problem solution
#     components = [sol_map]

#     """
#     # # #  mpQP objective and constraints formulation in Neuromancer
#     """
#     # variables
#     x = variable("x")[:, [0]]
#     y = variable("x")[:, [1]]
#     # sampled parameters
#     p1 = variable('p1')
#     p2 = variable('p2')

#     # objective function
#     f = x ** 2 + y ** 2
#     obj = f.minimize(weight=1.0, name='obj')
#     objectives = [obj]

#     # constraints
#     Q_con = 100.
#     g1 = -x - y + p1
#     con_1 = Q_con * (g1 <= 0)
#     con_1.name = 'c1'
#     if problem_type == 'pQP':  # constraints for QP
#         g2 = x + y - p1 - 5
#         con_2 = Q_con*(g2 <= 0)
#         con_2.name = 'c2'
#         g3 = x - y + p2 - 5
#         con_3 = Q_con*(g3 <= 0)
#         con_3.name = 'c3'
#         g4 = -x + y - p2
#         con_4 = Q_con*(g4 <= 0)
#         con_4.name = 'c4'
#         constraints = [con_1, con_2, con_3, con_4]
#     elif problem_type == 'pQCQP':  # constraints for QCQP
#         g2 = x**2+y**2 - p2**2
#         con_2 = Q_con*(g2 <= 0)
#         con_2.name = 'c2'
#         constraints = [con_1, con_2]

#     """
#     # # #  pQP problem formulation in Neuromancer
#     """
#     # create penalty method loss function
#     loss = PenaltyLoss(objectives, constraints)
#     # construct constrained optimization problem
#     problem = Problem(components, loss)

#     """
#     # # #  pQP problem solution in Neuromancer
#     """
#     optimizer = torch.optim.AdamW(problem.parameters(), lr=0.001)
#     # define trainer
#     trainer = Trainer(
#         problem,
#         train_loader,
#         dev_loader,
#         test_loader,
#         optimizer,
#         epochs=400,
#         patience=100,
#         warmup=100,
#         train_metric="train_loss",
#         dev_metric="dev_loss",
#         test_metric="test_loss",
#         eval_metric="dev_loss",
#     )

#     # Train solution map
#     best_model = trainer.train()
#     best_outputs = trainer.test(best_model)
#     # load best model dict
#     problem.load_state_dict(best_model)

#     """
#     CVXPY benchmarks
#     """
#     # Define the CVXPY problems.

#     def QP_param(p1, p2):
#         x = cp.Variable(1)
#         y = cp.Variable(1)
#         prob = cp.Problem(cp.Minimize(x ** 2 + y ** 2),
#                           [-x - y + p1 <= 0,
#                             x + y - p1 - 5 <= 0,
#                             x - y + p2 - 5 <= 0,
#                             -x + y - p2 <= 0])
#         return prob, x, y

#     def QCQP_param(p1, p2):
#         x = cp.Variable(1)
#         y = cp.Variable(1)
#         prob = cp.Problem(cp.Minimize(x ** 2 + y ** 2),
#                       [-x - y + p1 <= 0,
#                         x ** 2 + y ** 2 - p2 ** 2 <= 0])
#         return prob, x, y

#     """
#     Plots
#     """
#     # test problem parameters
#     params = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
#     x1 = np.arange(-1.0, 10.0, 0.05)
#     y1 = np.arange(-1.0, 10.0, 0.05)
#     xx, yy = np.meshgrid(x1, y1)
#     fig, ax = plt.subplots(3,3)
#     row_id = 0
#     column_id = 0
#     for i, p in enumerate(params):
#         if i % 3 == 0 and i != 0:
#             row_id += 1
#             column_id = 0

#         # eval and plot objective and constraints
#         J = xx ** 2 + yy ** 2
#         cp_plot = ax[row_id, column_id].contourf(xx, yy, J, 50, alpha=0.4)
#         ax[row_id, column_id].set_title(f'QP p={p}')
#         if problem_type == 'pQP':  # constraints for QP
#             c1 = xx + yy - p
#             c2 = -xx - yy + p + 5
#             c3 = -xx + yy - p + 5
#             c4 = xx - yy + p
#             cg1 = ax[row_id, column_id].contour(xx, yy, c1, [0], colors='mediumblue', alpha=0.7)
#             cg2 = ax[row_id, column_id].contour(xx, yy, c2, [0], colors='mediumblue', alpha=0.7)
#             cg3 = ax[row_id, column_id].contour(xx, yy, c3, [0], colors='mediumblue', alpha=0.7)
#             cg4 = ax[row_id, column_id].contour(xx, yy, c4, [0], colors='mediumblue', alpha=0.7)
#             plt.setp(cg1.collections,
#                       path_effects=[patheffects.withTickedStroke()], alpha=0.7)
#             plt.setp(cg2.collections,
#                       path_effects=[patheffects.withTickedStroke()], alpha=0.7)
#             plt.setp(cg3.collections,
#                       path_effects=[patheffects.withTickedStroke()], alpha=0.7)
#             plt.setp(cg4.collections,
#                       path_effects=[patheffects.withTickedStroke()], alpha=0.7)
#         if problem_type == 'pQCQP':  # constraints for QCQP
#             c1 = xx + yy - p
#             c2 = - xx**2 - yy**2 + p**2
#             cg1 = ax[row_id, column_id].contour(xx, yy, c1, [0], colors='mediumblue', alpha=0.7)
#             cg2 = ax[row_id, column_id].contour(xx, yy, c2, [0], colors='mediumblue', alpha=0.7)
#             plt.setp(cg1.collections,
#                       path_effects=[patheffects.withTickedStroke()], alpha=0.7)
#             plt.setp(cg2.collections,
#                       path_effects=[patheffects.withTickedStroke()], alpha=0.7)
#         fig.colorbar(cp_plot, ax=ax[row_id,column_id])

#         # Solve CVXPY problem
#         if problem_type == 'pQP':
#             prob, x, y = QP_param(p, p)
#         elif problem_type == 'pQCQP':
#             prob, x, y = QCQP_param(p, p)
#         prob.solve()

#         # Solve via neuromancer
#         datapoint = {'p1': torch.tensor([[p]]), 'p2': torch.tensor([[p]]),
#                       'name': 'test'}
#         model_out = problem(datapoint)
#         x_nm = model_out['test_' + "x"][0, 0].detach().numpy()
#         y_nm = model_out['test_' + "x"][0, 1].detach().numpy()

#         print(f'primal solution {problem_type} x={x.value}, y={y.value}')
#         print(f'parameter p={p, p}')
#         print(f'primal solution Neuromancer x1={x_nm}, x2={y_nm}')
#         print(f' f: {model_out["test_" + f.key]}')
#         print(f' g1: {model_out["test_" + g1.key]}')
#         print(f' g2: {model_out["test_" + g2.key]}')
#         if problem_type == 'pQP':
#             print(f' g3: {model_out["test_" + g3.key]}')
#             print(f' g4: {model_out["test_" + g4.key]}')

#         # Plot optimal solutions
#         ax[row_id, column_id].plot(x.value, y.value, 'g*', markersize=10)
#         ax[row_id, column_id].plot(x_nm, y_nm, 'r*', markersize=10)
#         column_id += 1
#     plt.show()
#     plt.show(block=True)
#     plt.interactive(False)

#     """
#     Benchmark Solution
#     """

#     def eval_constraints(x, y, p1, p2):
#         """
#         evaluate mean constraints violations
#         """
#         con_1_viol = np.maximum(0, -x - y + p1)
#         con_2_viol = np.maximum(0, x + y - p1 - 5)
#         con_3_viol = np.maximum(0, x - y + p2 - 5)
#         con_4_viol = np.maximum(0, -x + y - p2)
#         con_viol = con_1_viol + con_2_viol + con_3_viol + con_4_viol
#         con_viol_mean = np.mean(con_viol)
#         return con_viol_mean

#     def eval_objective(x, y, a1=1, a2=1):
#         obj_value_mean = np.mean(a1 * x**2 + a2 * y**2)
#         return obj_value_mean

#     # Solve via neuromancer
#     t = time.time()
#     samples_test['name'] = 'test'
#     model_out = problem(samples_test)
#     nm_time = time.time() - t
#     x_nm = model_out['test_' + "x"][:, [0]].detach().numpy()
#     y_nm = model_out['test_' + "x"][:, [1]].detach().numpy()

#     # Solve via solver
#     t = time.time()
#     x_solver, y_solver = [], []
#     for i in range(0, nsim):
#         p1 = samples_test['p1'][i].detach().numpy()
#         p2 = samples_test['p2'][i].detach().numpy()
#         prob, x, y = QP_param(p1, p2)
#         prob.solve(solver='ECOS_BB', verbose=False)
#         prob.solve()
#         x_solver.append(x.value)
#         y_solver.append(y.value)
#     solver_time = time.time() - t
#     x_solver = np.asarray(x_solver)
#     y_solver = np.asarray(y_solver)

#     # Evaluate neuromancer solution
#     print(f'Solution for {nsim} problems via Neuromancer obtained in {nm_time:.4f} seconds')
#     nm_con_viol_mean = eval_constraints(x_nm, y_nm, p1, p2)
#     print(f'Neuromancer mean constraints violation {nm_con_viol_mean:.4f}')
#     nm_obj_mean = eval_objective(x_nm, y_nm)
#     print(f'Neuromancer mean objective value {nm_obj_mean:.4f}')

#     # Evaluate solver solution
#     print(f'Solution for {nsim} problems via solver obtained in {solver_time:.4f} seconds')
#     solver_con_viol_mean = eval_constraints(x_solver, y_solver, p1, p2)
#     print(f'Solver mean constraints violation {solver_con_viol_mean:.4f}')
#     solver_obj_mean = eval_objective(x_solver, y_solver)
#     print(f'Solver mean objective value {solver_obj_mean:.4f}')

#     # neuromancer solver comparison
#     speedup_factor = solver_time/nm_time
#     print(f'Solution speedup factor {speedup_factor:.4f}')

#     # Difference in primal optimizers
#     dx = (x_solver - x_nm)[:,0]
#     dy = (y_solver - y_nm)[:,0]
#     err_x = np.mean(dx**2)
#     err_y = np.mean(dy**2)
#     err_primal = err_x + err_y
#     print('MSE primal optimizers:', err_primal)

#     # Difference in objective
#     err_obj = np.abs(solver_obj_mean - nm_obj_mean) / solver_obj_mean * 100
#     print(f'mean objective value discrepancy: {err_obj:.2f} %')

#     # stats to log
#     stats = {"nsim": nsim,
#               "nm_time": nm_time,
#               "nm_con_viol_mean": nm_con_viol_mean,
#               "nm_obj_mean": nm_obj_mean,
#               "solver_time": solver_time,
#               "solver_con_viol_mean": solver_con_viol_mean,
#               "solver_obj_mean": solver_obj_mean,
#               "speedup_factor": speedup_factor,
#               "err_primal": err_primal,
#               "err_obj": err_obj}






"""
Learning to optimize parametric Quadratic Programming (pQP) and
parametric Quadratically Constrained Quadratic Programming (pQCQP)
problems using Neuromancer.

Problem formulation pQP:
    minimize     x^2 + y^2
    subject to
                -x - y + p1 <= 0,
                x + y - p1 - 5 <= 0,
                x - y + p2 - 5 <= 0,
                -x + y - p2 <= 0

Problem formulation pQCQP:
    minimize     x^2 + y^2
    subject to
            -x - y + p1 <= 0,
            x^2 + y^2 <= p2^2

    problem parameters:            p1, p2
    problem decition variables:    x, y
"""
import sys
# 手动添加 Neuromancer 的安装路径
sys.path.append(r"c:\users\du47tw\appdata\local\anaconda3\envs\neuromancer\lib\site-packages")


import cvxpy as cp
import numpy as np
import time
import torch
import torch.nn as nn
import neuromancer.slim as slim
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects

from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
from neuromancer.constraint import variable
from neuromancer.dataset import DictDataset
from neuromancer.loss import PenaltyLoss
from neuromancer.modules import blocks
from neuromancer.system import Node


if __name__ == "__main__":

    problem_type = 'pQCQP'   # select from 'pQP' or 'pQCQP'

    """
    # # #  Dataset
    """
    data_seed = 408
    np.random.seed(data_seed)
    nsim = 5000  # number of datapoints: increase sample density for more robust results
# create dictionaries with sampled datapoints with uniform distribution
    if_alpha_low, if_alpha_high = -16, 16
    if_beta_low, if_beta_high = -16, 16
    vc_alpha_low, vc_alpha_high = -10, 10
    vc_beta_low, vc_beta_high = -10, 10
    vref_alpha_low, vref_alpha_high = -325, 325
    vref_beta_low, vref_beta_high = -325, 325
    # t_low, t_high = 0, 0.01
    R_low, R_high = 30, 60
    samples_train = {"if_alpha": torch.FloatTensor(nsim, 1).uniform_(if_alpha_low, if_alpha_high),
                      "if_beta": torch.FloatTensor(nsim, 1).uniform_(if_beta_low, if_beta_high),
                      "vc_alpha": torch.FloatTensor(nsim, 1).uniform_(vc_alpha_low, vc_alpha_high),
                      "vc_beta": torch.FloatTensor(nsim, 1).uniform_(vc_beta_low, vc_beta_high),
                      "vref_alpha": torch.FloatTensor(nsim, 1).uniform_(vref_alpha_low, vref_alpha_high),
                      "vref_beta": torch.FloatTensor(nsim, 1).uniform_(vref_beta_low, vref_beta_high),
    #                  "t": torch.FloatTensor(nsim, 1).uniform_(t_low, t_high),
                      "R": torch.FloatTensor(nsim, 1).uniform_(R_low, R_high)}
    samples_dev = {"if_alpha": torch.FloatTensor(nsim, 1).uniform_(if_alpha_low, if_alpha_high),
                      "if_beta": torch.FloatTensor(nsim, 1).uniform_(if_beta_low, if_beta_high),
                      "vc_alpha": torch.FloatTensor(nsim, 1).uniform_(vc_alpha_low, vc_alpha_high),
                      "vc_beta": torch.FloatTensor(nsim, 1).uniform_(vc_beta_low, vc_beta_high),
                      "vref_alpha": torch.FloatTensor(nsim, 1).uniform_(vref_alpha_low, vref_alpha_high),
                      "vref_beta": torch.FloatTensor(nsim, 1).uniform_(vref_beta_low, vref_beta_high),
    #                  "t": torch.FloatTensor(nsim, 1).uniform_(t_low, t_high),
                      "R": torch.FloatTensor(nsim, 1).uniform_(R_low, R_high)}
    samples_test = {"if_alpha": torch.FloatTensor(nsim, 1).uniform_(if_alpha_low, if_alpha_high),
                      "if_beta": torch.FloatTensor(nsim, 1).uniform_(if_beta_low, if_beta_high),
                      "vc_alpha": torch.FloatTensor(nsim, 1).uniform_(vc_alpha_low, vc_alpha_high),
                      "vc_beta": torch.FloatTensor(nsim, 1).uniform_(vc_beta_low, vc_beta_high),
                      "vref_alpha": torch.FloatTensor(nsim, 1).uniform_(vref_alpha_low, vref_alpha_high),
                      "vref_beta": torch.FloatTensor(nsim, 1).uniform_(vref_beta_low, vref_beta_high),
    #                  "t": torch.FloatTensor(nsim, 1).uniform_(t_low, t_high),
                      "R": torch.FloatTensor(nsim, 1).uniform_(R_low, R_high)}
#     # note: training quality will depend on the DataLoader parameters such as batch size and shuffle
    # create named dictionary datasets
    train_data = DictDataset(samples_train, name='train')
    dev_data = DictDataset(samples_dev, name='dev')
    test_data = DictDataset(samples_test, name='test')
    # create torch dataloaders for the Trainer
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, num_workers=0,
                                                            collate_fn=train_data.collate_fn, shuffle=True)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=32, num_workers=0,
                                                          collate_fn=dev_data.collate_fn, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, num_workers=0,
                                                          collate_fn=test_data.collate_fn, shuffle=True)
    
    
        
    # import torch
    
    # # Number of general samples
    # nsim = 5000
    # # Number of additional samples in the specific range
    # extra_samples = 1000
    
    # Define ranges for each variable
    # if_alpha_low, if_alpha_high = -16, 16
    # if_beta_low, if_beta_high = -16, 16
    # # vc_alpha_low, vc_alpha_high = -10, 10
    # # vc_beta_low, vc_beta_high = -10, 10
    # vc_alpha_low, vc_alpha_high = -330, 330
    # vc_beta_low, vc_beta_high = -330, 330
    # vref_alpha_low, vref_alpha_high = -330, 330
    # vref_beta_low, vref_beta_high = -330, 330
    # R_low, R_high = 30, 60
    
    # # Generate general samples
    # samples_train = {
    #     "if_alpha": torch.FloatTensor(nsim, 1).uniform_(if_alpha_low, if_alpha_high),
    #     "if_beta": torch.FloatTensor(nsim, 1).uniform_(if_beta_low, if_beta_high),
    #     "vc_alpha": torch.FloatTensor(nsim, 1).uniform_(vc_alpha_low, vc_alpha_high),
    #     "vc_beta": torch.FloatTensor(nsim, 1).uniform_(vc_beta_low, vc_beta_high),
    #     "vref_alpha": torch.FloatTensor(nsim, 1).uniform_(vref_alpha_low, vref_alpha_high),
    #     "vref_beta": torch.FloatTensor(nsim, 1).uniform_(vref_beta_low, vref_beta_high),
    #     "R": torch.FloatTensor(nsim, 1).uniform_(R_low, R_high)
    # }
    
    # # Generate additional samples with |vref| > 300
    # additional_samples = {
    #     "if_alpha": torch.FloatTensor(extra_samples, 1).uniform_(if_alpha_low, if_alpha_high),
    #     "if_beta": torch.FloatTensor(extra_samples, 1).uniform_(if_beta_low, if_beta_high),
    #     "vc_alpha": torch.FloatTensor(extra_samples, 1).uniform_(vc_alpha_low, vc_alpha_high),
    #     "vc_beta": torch.FloatTensor(extra_samples, 1).uniform_(vc_beta_low, vc_beta_high),
    #     "vref_alpha": torch.cat([
    #         torch.FloatTensor(extra_samples // 2, 1).uniform_(300, vref_alpha_high),  # Values > 300
    #         torch.FloatTensor(extra_samples // 2, 1).uniform_(vref_alpha_low, -300)   # Values < -300
    #     ]),
    #     "vref_beta": torch.cat([
    #         torch.FloatTensor(extra_samples // 2, 1).uniform_(300, vref_beta_high),  # Values > 300
    #         torch.FloatTensor(extra_samples // 2, 1).uniform_(vref_beta_low, -300)   # Values < -300
    #     ]),
    #     "R": torch.FloatTensor(extra_samples, 1).uniform_(R_low, R_high)
    # }
    
    # # Combine general samples and additional samples
    # samples_train["if_alpha"] = torch.cat([samples_train["if_alpha"], additional_samples["if_alpha"]], dim=0)
    # samples_train["if_beta"] = torch.cat([samples_train["if_beta"], additional_samples["if_beta"]], dim=0)
    # samples_train["vc_alpha"] = torch.cat([samples_train["vc_alpha"], additional_samples["vc_alpha"]], dim=0)
    # samples_train["vc_beta"] = torch.cat([samples_train["vc_beta"], additional_samples["vc_beta"]], dim=0)
    # samples_train["vref_alpha"] = torch.cat([samples_train["vref_alpha"], additional_samples["vref_alpha"]], dim=0)
    # samples_train["vref_beta"] = torch.cat([samples_train["vref_beta"], additional_samples["vref_beta"]], dim=0)
    # samples_train["R"] = torch.cat([samples_train["R"], additional_samples["R"]], dim=0)

    
    # samples_dev = {"if_alpha": torch.FloatTensor(nsim, 1).uniform_(if_alpha_low, if_alpha_high),
    #                   "if_beta": torch.FloatTensor(nsim, 1).uniform_(if_beta_low, if_beta_high),
    #                   "vc_alpha": torch.FloatTensor(nsim, 1).uniform_(vc_alpha_low, vc_alpha_high),
    #                   "vc_beta": torch.FloatTensor(nsim, 1).uniform_(vc_beta_low, vc_beta_high),
    #                   "vref_alpha": torch.FloatTensor(nsim, 1).uniform_(vref_alpha_low, vref_alpha_high),
    #                   "vref_beta": torch.FloatTensor(nsim, 1).uniform_(vref_beta_low, vref_beta_high),
    # #                  "t": torch.FloatTensor(nsim, 1).uniform_(t_low, t_high),
    #                   "R": torch.FloatTensor(nsim, 1).uniform_(R_low, R_high)}
    # samples_test = {"if_alpha": torch.FloatTensor(nsim, 1).uniform_(if_alpha_low, if_alpha_high),
    #                   "if_beta": torch.FloatTensor(nsim, 1).uniform_(if_beta_low, if_beta_high),
    #                   "vc_alpha": torch.FloatTensor(nsim, 1).uniform_(vc_alpha_low, vc_alpha_high),
    #                   "vc_beta": torch.FloatTensor(nsim, 1).uniform_(vc_beta_low, vc_beta_high),
    #                   "vref_alpha": torch.FloatTensor(nsim, 1).uniform_(vref_alpha_low, vref_alpha_high),
    #                   "vref_beta": torch.FloatTensor(nsim, 1).uniform_(vref_beta_low, vref_beta_high),
    #                   # "t": torch.FloatTensor(nsim, 1).uniform_(t_low, t_high),
    #                   "R": torch.FloatTensor(nsim, 1).uniform_(R_low, R_high)}
    # # Now samples_train has more samples where |vref_alpha| and |vref_beta| are greater than 300
    # # note: training quality will depend on the DataLoader parameters such as batch size and shuffle
    # # create named dictionary datasets
    # train_data = DictDataset(samples_train, name='train')
    # dev_data = DictDataset(samples_dev, name='dev')
    # test_data = DictDataset(samples_test, name='test')
    # # create torch dataloaders for the Trainer
    # train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, num_workers=0,
    #                                                         collate_fn=train_data.collate_fn, shuffle=True)
    # dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=32, num_workers=0,
    #                                                       collate_fn=dev_data.collate_fn, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, num_workers=0,
    #                                                       collate_fn=test_data.collate_fn, shuffle=True)
    
    
    

    
    
    
    
    

    # from scipy.io import loadmat
    
    # # 加载.mat文件
    # data = loadmat('datatrainingv7.mat')['DPCdata']
    
    # # data = loadmat('datafortraining.mat')
    # # print(data.keys())

    
    # # 将数据分配到各个变量，假设列的顺序是确定的
    # if_alpha = data[:, 0]
    # if_beta = data[:, 1]
    # # print(if_beta)
    # vc_alpha = data[:, 2]
    # vc_beta = data[:, 3]
    # vref_alpha = data[:, 4]
    # vref_beta = data[:, 5]
    # R = data[:, 6]
    
    # # 将变量转换成适合torch的Tensor
    # samples_train = {
    #     "if_alpha": torch.tensor(if_alpha[:4000]).float().reshape(-1, 1),
    #     "if_beta": torch.tensor(if_beta[:4000]).float().reshape(-1, 1),
    #     "vc_alpha": torch.tensor(vc_alpha[:4000]).float().reshape(-1, 1),
    #     "vc_beta": torch.tensor(vc_beta[:4000]).float().reshape(-1, 1),
    #     "vref_alpha": torch.tensor(vref_alpha[:4000]).float().reshape(-1, 1),
    #     "vref_beta": torch.tensor(vref_beta[:4000]).float().reshape(-1, 1),
    #     "R": torch.tensor(R[:4000]).float().reshape(-1, 1)
    # }
    
    # samples_dev = {
    #     "if_alpha": torch.tensor(if_alpha[4000:5000]).float().reshape(-1, 1),
    #     "if_beta": torch.tensor(if_beta[4000:5000]).float().reshape(-1, 1),
    #     "vc_alpha": torch.tensor(vc_alpha[4000:5000]).float().reshape(-1, 1),
    #     "vc_beta": torch.tensor(vc_beta[4000:5000]).float().reshape(-1, 1),
    #     "vref_alpha": torch.tensor(vref_alpha[4000:5000]).float().reshape(-1, 1),
    #     "vref_beta": torch.tensor(vref_beta[4000:5000]).float().reshape(-1, 1),
    #     "R": torch.tensor(R[4000:5000]).float().reshape(-1, 1)
    # }
    
    # samples_test = {
    #     "if_alpha": torch.tensor(if_alpha[4000:5000]).float().reshape(-1, 1),
    #     "if_beta": torch.tensor(if_beta[4000:5000]).float().reshape(-1, 1),
    #     "vc_alpha": torch.tensor(vc_alpha[4000:5000]).float().reshape(-1, 1),
    #     "vc_beta": torch.tensor(vc_beta[4000:5000]).float().reshape(-1, 1),
    #     "vref_alpha": torch.tensor(vref_alpha[4000:5000]).float().reshape(-1, 1),
    #     "vref_beta": torch.tensor(vref_beta[4000:5000]).float().reshape(-1, 1),
    #     "R": torch.tensor(R[4000:5000]).float().reshape(-1, 1)
    # }
    
    # # 创建数据集和数据加载器
    # train_data = DictDataset(samples_train, name='train')
    # dev_data = DictDataset(samples_dev, name='dev')
    # test_data = DictDataset(samples_test, name='test')
    
    # train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, num_workers=0,
    #                                             collate_fn=train_data.collate_fn, shuffle=True)
    # dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=32, num_workers=0,
    #                                           collate_fn=dev_data.collate_fn, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, num_workers=0,
    #                                           collate_fn=test_data.collate_fn, shuffle=True)

    # note: training quality will depend on the DataLoader parameters such as batch size and shuffle

    # visualize taining and test samples for 2D parametric space
    # a_train = samples_train['p1'].numpy()
    # p_train = samples_train['p2'].numpy()
    # a_dev = samples_dev['p1'].numpy()
    # p_dev = samples_dev['p2'].numpy()
    # plt.figure()
    # plt.scatter(a_train, p_train, s=2., c='blue', marker='o')
    # plt.scatter(a_dev, p_dev, s=2., c='red', marker='o')
    # plt.title('Sampled parametric space for training')
    # plt.xlim(p_low, p_high)
    # plt.ylim(p_low, p_high)
    # plt.grid(True)
    # plt.xlabel('p1')
    # plt.ylabel('p2')
    # plt.legend(['train', 'test'], loc='upper right')
    # plt.show()
    # plt.show(block=True)


    """
    # # #  pQP primal solution map architecture
    """
    # define neural architecture for the solution map
    func = blocks.MLP(insize=7, outsize=2,
                    bias=True,
                    linear_map=slim.maps['linear'],
                    nonlin=nn.ReLU,
                    hsizes=[80] * 4)
    # define symbolic solution map
    #sol_map = Node(func, ['if_alpha', 'if_beta','vc_alpha', 'vc_beta','t', 'R'], ['x'], name='map')
    sol_map = Node(func, ['if_alpha', 'if_beta','vc_alpha', 'vc_beta','vref_alpha', 'vref_beta', 'R'], ['x'], name='map')
    # trainable components of the problem solution
    components = [sol_map]

    """
    # # #  mpQP objective and constraints formulation in Neuromancer
    """
    # variables
    x1 = variable("x")[:, [0]]
    x2 = variable("x")[:, [1]]
    # sampled parameters
    if_alpha = variable('if_alpha')
    if_beta = variable('if_beta')
    vc_alpha = variable('vc_alpha')
    vc_beta = variable('vc_beta')
    vref_alpha = variable('vref_alpha')
    vref_beta = variable('vref_beta')
    #t = variable('t')
    R = variable('R')

    # loss function calculation
    
    import numpy as np
    #import matplotlib.pyplot as plt
    import torch
    # 假设 vref_alpha, vref_beta, vc_alpha, vc_beta, if_alpha, if_beta, R, C, wref, Adf, Bdf 和 s_states 已经定义
    
    #vref_alpha = 325*torch.sin(2 * torch.pi * 50 * t)  # 实部
    #vref_beta = 325*torch.sin(2 * torch.pi * 50 * t - 3 * torch.pi / 2)  # 虚部
    v_ref_real = vref_alpha
    v_ref_imag = vref_beta
    
    # 计算测量电压和电流 (实部和虚部)
    v_meas_real = vref_alpha + vc_alpha
    v_meas_imag = vref_beta + vc_beta
    # v_meas_real = vc_alpha
    # v_meas_imag = vc_beta
    i_load_real = (vref_alpha + vc_alpha) / R
    i_load_imag = (vref_beta + vc_beta) / R
    # i_load_real = (vc_alpha) / R
    # i_load_imag = (vc_beta) / R
    i_meas_real = if_alpha
    i_meas_imag = if_beta
    
    # 初始化加权电压矢量
    v_o1_real = x1
    v_o1_imag = x2
    
    
    
    # # 根据电压矢量计算系统的预测输出 (实部和虚部)
    # ift1_real = 0.960998 * i_meas_real + (-0.020555) * v_meas_real + 0.020555 * v_o1_real + 0.03695 * i_load_real
    # ift1_imag = 0.960998  * i_meas_imag + (-0.020555) * v_meas_imag + 0.020555  * v_o1_imag + 0.03695 * i_load_imag
    
    # vc1_real = 3.52363 * i_meas_real + 0.96305 * v_meas_real + 0.03695* v_o1_real + (-3.52733) * i_load_real
    # vc1_imag = 3.52363 * i_meas_imag + 0.96305 * v_meas_imag + 0.03695 * v_o1_imag + (-3.52733) * i_load_imag
    
    # # 计算电流误差和电压误差的实部和虚部差异
    # real_diff = (v_ref_real - vc1_real) ** 2
    # imag_diff = (v_ref_imag - vc1_imag) ** 2
    
    # ift1_real_diff = (ift1_real - i_load_real + 0.000014 * 100*(np.pi) * v_ref_imag) ** 2
    # ift1_imag_diff = (ift1_imag - i_load_imag - 0.000014 * 100*(np.pi) * v_ref_real) ** 2
    
    # 根据电压矢量计算系统的预测输出 (实部和虚部)
    ift1_real = 0.9932 * i_meas_real + (-0.0083) * v_meas_real + 0.0083 * v_o1_real + 0.0059 * i_load_real
    ift1_imag = 0.9932 * i_meas_imag + (-0.0083) * v_meas_imag + 0.0083 * v_o1_imag + 0.0059 * i_load_imag
    
    vc1_real = 1.4251 * i_meas_real + 0.9941 * v_meas_real + 0.0059 * v_o1_real + (-1.4257) * i_load_real
    vc1_imag = 1.4251 * i_meas_imag + 0.9941 * v_meas_imag + 0.0059 * v_o1_imag + (-1.4257) * i_load_imag
    
    # 计算电流误差和电压误差的实部和虚部差异
    real_diff = (v_ref_real - vc1_real) ** 2
    imag_diff = (v_ref_imag - vc1_imag) ** 2
    
    ift1_real_diff = (ift1_real - i_load_real + 0.000014 * 100*(np.pi) * v_ref_imag) ** 2
    ift1_imag_diff = (ift1_imag - i_load_imag - 0.000014 * 100*(np.pi) * v_ref_real) ** 2

    # objective function
    f = 1*real_diff + 1*imag_diff + 1*ift1_real_diff + 1*ift1_imag_diff
    obj = f.minimize(weight=1.0, name='obj')
    objectives = [obj]



    # constraints
    Q_con = 0
    g1 = x1-467 
    con_1 = Q_con * (g1 <=0)
    con_1.name = 'c1'
    if problem_type == 'pQP':  # constraints for QP
        g2 = -467-x1
        con_2 = Q_con*(g2 <=0)
        con_2.name = 'c2'
        g3 = x2-467
        con_3 = Q_con*(g3 <= 0)
        con_3.name = 'c3'
        g4 = -467-x2
        con_4 = Q_con*(g4 <= 0)
        con_4.name = 'c4'
        constraints = [con_1, con_2, con_3, con_4]
    elif problem_type == 'pQCQP':  # constraints for QCQP
        g2 = torch.sqrt(x1**2+x2**2) - 467
        con_2 = Q_con*(g2 == 0)
        con_2.name = 'c2'
        g3 = x1+x2-1 
        con_3 = Q_con*(g3 == 0)
        con_3.name = 'c3'
        g4 = - x1+x2 
        con_4 = Q_con*(g4 <= 0)
        con_4.name = 'c4'
        g5 = - x1+x2 
        con_5 = Q_con*(g5 <= 0)
        con_5.name = 'c5'
        g6 = - x1+x2 
        con_6 = Q_con*(g6 <= 0)
        con_6.name = 'c6'
        g7 = - x1+x2 
        con_7 = Q_con*(g7 <= 0)
        con_7.name = 'c7'
        g8 = - x1+x2 
        con_8 = Q_con*(g8 <= 0)
        con_8.name = 'c8'
        g9 = - x1+x2 
        con_9 = Q_con*(g9 <= 0)
        con_9.name = 'c9'
        g10 = - x1+x2 
        con_10 = Q_con*(g10 <= 0)
        con_10.name = 'c10'
        #constraints = [con_2, con_3, con_4, con_5, con_6, con_7, con_8, con_9,con_10]
        constraints = [con_2]

    """
    # # #  pQP problem formulation in Neuromancer
    """
    # create penalty method loss function
    loss = PenaltyLoss(objectives, constraints)
    # construct constrained optimization problem
    problem = Problem(components, loss)

    """
    # # #  pQP problem solution in Neuromancer
    """
    optimizer = torch.optim.AdamW(problem.parameters(), lr=0.001)
    # define trainer
    trainer = Trainer(
        problem,
        train_loader,
        dev_loader,
        test_loader,
        optimizer,
        epochs=3000,
        patience=2000,
        warmup=100,
        train_metric="train_loss",
        dev_metric="dev_loss",
        test_metric="test_loss",
        eval_metric="dev_loss",
    )

    # Train solution map
    best_model = trainer.train()
    best_outputs = trainer.test(best_model)
    # load best model dict
    problem.load_state_dict(best_model)





#     """
#     # # #  pQP problem formulation in Neuromancer
#     """
#     # create penalty method loss function
#     loss = PenaltyLoss(objectives, constraints)
#     # construct constrained optimization problem
#     problem = Problem(components, loss)

#     """
#     # # #  pQP problem solution in Neuromancer
#     """
#     optimizer = torch.optim.AdamW(problem.parameters(), lr=0.001)
#     # define trainer
#     trainer = Trainer(
#         problem,
#         train_loader,
#         dev_loader,
#         test_loader,
#         optimizer,
#         epochs=400,
#         patience=100,
#         warmup=100,
#         train_metric="train_loss",
#         dev_metric="dev_loss",
#         test_metric="test_loss",
#         eval_metric="dev_loss",
#     )

#     # Train solution map
#     best_model = trainer.train()
#     best_outputs = trainer.test(best_model)
#     # load best model dict
#     problem.load_state_dict(best_model)

#     """
    
    
    
    # # 修改学习率，不重新创建优化器
    # new_lr = 0.0001  # 设定新的学习率
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = new_lr
    
    # # 创建新的 Trainer，用于继续训练
    # trainer = Trainer(
    #     problem,
    #     train_loader,
    #     dev_loader,
    #     test_loader, 
    #     optimizer,
    #     epochs=1000,  # 新的训练轮数
    #     patience=100,
    #     warmup=50,
    #     train_metric="train_loss",
    #     dev_metric="dev_loss",
    #     test_metric="test_loss",
    #     eval_metric="dev_loss",
    # )
    
    # # 继续训练
    # best_model_continued = trainer.train()
    # best_outputs_continued = trainer.test(best_model_continued)

    # 将输入字典格式数据传递给神经网络
    # with torch.no_grad():  # 禁用梯度计算，因为这是推理过程
    #     # 使用字典来传递输入，同时添加 'name' 键
    #     model_input = {
    #         "if_alpha": torch.FloatTensor(nsim, 1).uniform_(10, 10),
    #         "if_beta": torch.FloatTensor(nsim, 1).uniform_(1, 1),
    #         "vc_alpha": torch.FloatTensor(nsim, 1).uniform_(0, 0),
    #         "vc_beta": torch.FloatTensor(nsim, 1).uniform_(0, 0),
    #         "vref_alpha": torch.FloatTensor(nsim, 1).uniform_(-10, -10),
    #         "vref_beta": torch.FloatTensor(nsim, 1).uniform_(320, 320),
    #     #     "t": torch.FloatTensor(nsim, 1).uniform_(0.005, 0.005),
    #         "R": torch.FloatTensor(nsim, 1).uniform_(30, 60),
    #         "name": "test"  # 添加 'name' 键
    #     }
    
    #     # 将输入传递给模型
    #     model_output = problem(model_input)
    
    # # 现在 `model_output` 是网络输出的结果，假设它输出的包含 s1 到 s7 的变量。
    # # 从模型输出中提取对应的值（例如 s1 到 s7）
    # s1_output = model_output['test_x'][1, [0]]
    # s2_output = model_output['test_x'][1, [1]]
    # #s3_output = model_output['test_x'][1, [2]]
    # #s4_output = model_output['test_x'][1, [3]]
    # #s5_output = model_output['test_x'][1, [4]]
    # #s6_output = model_output['test_x'][1, [5]]
    # #s7_output = model_output['test_x'][1, [6]]
    
    # # 打印或存储输出值
    # print("s1:", s1_output)
    # print("s2:", s2_output)
    #print("s3:", s3_output)
    #print("s4:", s4_output)
    #print("s5:", s5_output)
    #print("s6:", s6_output)
    #print("s7:", s7_output)
    
    # import torch
    # import onnx
    
    # # 定义一个示例输入，用于模型导出
    # dummy_input = torch.randn(1, 7)  # 确保 input_size 与模型的输入尺寸匹配
    
    # # 导出模型到 ONNX 格式
    # torch.onnx.export(problem, dummy_input, "problem_model.onnx")


    # for name, param in problem.state_dict().items():
    #     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]}")  # 查看前两个参数值
        
    # # # !!保存到 .pth 文件
    # torch.save(problem.state_dict(), "model_parameters.pth")
    
    # # import torch
    # import scipy.io as sio
    
    # # 加载模型参数
    # model_parameters = torch.load("model_parameters.pth")
    
    # # 替换字段名称中的点号为下划线
    # mat_dict = {}
    # for key, param in model_parameters.items():
    #     new_key = key.replace(".", "_")  # 替换点号为下划线
    #     mat_dict[new_key] = param.numpy()  # 转换为 NumPy 格式
    
    # # 保存为 .mat 文件
    # sio.savemat("model_parametersMPCDPC1v27.mat", mat_dict)

    



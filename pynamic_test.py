import time
import matplotlib.pyplot as plt
from pynamic import pynamic_ode
import numpy as np

def pynamic_test():
    """
    Validate the ode integrator with this function. This function will plot the
    function dy(t)/dt=-2y(t), which has the solution y(t)=3e^(-2t) for t>=0.
    The test will plot the results of pynamic compared to the true solution.
    """

    sol_times = np.linspace(0, 3, 25)

    #the true solution to the ODE
    sol_ys = 3*np.exp(-2*sol_times)

    #the parameters for pynamic_ode
    def test_func(_, y_vals):
        return -2*y_vals

    start_time = 0
    max_time = 2
    initial_val = [3]
    param_to_monitor = 0
    max_param_delta = 0.1
    base_time_step = 0.5
    min_step_time = 0.01

    def end_condition(_, y_vals):
        if y_vals[0] < 0.1:
            return 1
        return 0

    start = time.time()
    times, y_vals, status = pynamic_ode(test_func, start_time, max_time, 
                                        initial_val, param_to_monitor, 
                                        max_param_delta, base_time_step, 
                                        min_step_time, end_condition, 
                                        use_rk4=False)
    end = time.time()
    print("time to run with RK1: %2.3e"%(end-start))

    start = time.time()
    t_rk4, y_rk4s, st_rk4 = pynamic_ode(test_func, start_time, max_time, 
                                        initial_val, param_to_monitor, 
                                        max_param_delta, base_time_step, 
                                        min_step_time, end_condition, 
                                        use_rk4=True)
    end = time.time()
    print("time to run with RK4: %2.3e"%(end-start))

    sol_times = np.linspace(0, max_time, max_time*15)

    #the true solution to the ODE
    sol_ys = 3*np.exp(-2*sol_times)


    print("status of solver: %d, status with RK4: %d"%(status, st_rk4))
    plt.plot(sol_times, sol_ys, 'k', label="True Sol.")

    plt.plot(times, y_vals, 'r+', label="RK1")
    plt.plot(times, y_vals, 'r:')
    plt.plot(t_rk4, y_rk4s, 'bx', label="RK4")
    plt.plot(t_rk4, y_rk4s, 'b:')

    plt.legend()

    plt.savefig("test_results.png")

    return 0

pynamic_test()

"""
Pynamic ODE - an ODE integrator with dynamic time steps. Can be run as RK1 
(faster) or RK4 (slower, but more accurate).
"""
import numpy as np
import matplotlib.pyplot as plt

def dynamic_ode_solver(func, start_time, max_time, initial_guess, 
                       param_to_monitor, max_param_delta,
                       base_time_step, min_step_time, end_condition,
                       use_rk4=True):
    """
    Solves a system of ordinary differential equations with dynamic time steps.
    The passed in function (func) should take two arguments, a time step value,
    and the current parameter values. It will have the form:
        func(time_step, y_vals)
    The func provided should return the derivatives of each value considered.
    This routine will start at time start_time, and run until max_time is 
    reached, or the end condition is met. The end condition is set by the
    end_condition paramter, which should be a function like func that takes
    both the current time (NOT TIME STEP) and the system values like:
        end_condition(current_time, y_vals)
    and should return not 0 if the integration should terminate, or 0 if it 
    should stop. 
    This solver will attempt each time step and if the % change in the monitored 
    parameter is greater than the provided max_param_delta (as a fraction), then
    the time step will be reduced by half (repeatedly if necessary) until the
    minimum step time is reached (min_step_time). The step time will slowly
    relax back to the base time step (base_time_step) specified by the input. 
    The parameter to monitor should be an integer in the y_vals array. For example,
    if the initial guess has values:
        initial_guess = [position, velocity]
    and you want to make sure the position never changes by more than 1% you'd
    set param_to_monitor=0 and max_param_delta=0.01. 
    Inputs:
        func             - function that takes time step and values, returns 
                           derivatives
        start_time       - simulation start time [s]
        max_time         - maximum time to run simulation before returning an 
                           error [s]
        initial_guess    - array with initial parameter values
        param_to_monitor - index of which parameter to track
        max_param_delta  - fractional difference to tolerate in param_to_monitor. 
                           A value of 0.01 means changes must be less than 1%. 
                           A value of 0.2 would mean values must be less than 20%.
        base_time_step   - the default step size to use in integration [s]
        min_step_time    - the smallest time step to allow. If max_param_delta 
                           is exceeded at the smallest allowed time step an 
                           error status will be returned.
        end_condition    - function that takes current time and values, returns 
                           not 0 if the integration should end, 0 otherwise. The
                           value of end condition will be passed out of this 
                           function.
        use_rk4          - default is True. If true, use the RK4 method, if 
                           false, use the RK1 (Euler) method.
    Returns:
        times  - the array of time values [s] calculated 
        y_vals     - the array of parameter values at each time step
        status - the status of the solver. Values are:
                    -1 : failure because max_param_delta was exceeded at the 
                         smallest allowed time step.
                     0 : simulation ended without meeting end condition (it hit
                         max_time).
                    >0 : simulation reached end condition successfully. Return
                         the result of end_condition.
    """

    y_cur = np.array(initial_guess)

    times = []
    y_vals = []
    current_time = start_time
    cur_step_size = base_time_step
    next_relax = -1 #if >0 this is the next time to increase the time step

    end_cond_val = True #set to false if the end condition is met 

    status = 0 #status of the solver
    not_failed = True #set to false if the solver fails

    while current_time < max_time and end_cond_val and not_failed:
        end_val = end_condition(current_time, y_cur)
        if  end_val != 0:
            end_cond_val = False #we hit the end condition
            status = abs(end_val) #success!
        else:
            #first check if the time step should relax
            if cur_step_size < base_time_step and current_time >= next_relax:
                #the step should relax, double it
                next_relax = -1
                cur_step_size = cur_step_size*2
                if cur_step_size > base_time_step:
                    #make sure the base time step isn't exceeded
                    cur_step_size = base_time_step

            #get the current value of the param to monitor
            monitor_cur = y_cur[param_to_monitor]

            #run the function to get the new derivative values
            deltas = np.array(func(cur_step_size, y_cur))

            if use_rk4:
                #calculate the derivative multiple times and take the weighted 
                #average.
                rk4_k1 = deltas

                #calculate k2
                y_for_k2 = y_cur + rk4_k1*(cur_step_size/2)
                rk4_k2 = np.array(func(cur_step_size/2, y_for_k2))

                #calculate k3
                y_for_k3 = y_cur + rk4_k2*(cur_step_size/2)
                rk4_k3 = np.array(func(cur_step_size/2, y_for_k3))

                #calculate k4
                y_for_k4 = y_cur + rk4_k3*(cur_step_size)
                rk4_k4 = np.array(func(cur_step_size, y_for_k4))

                deltas = (rk4_k1 + 2*rk4_k2 + 2*rk4_k3 + rk4_k4)/6

            #calculate the new values
            y_new = y_cur + deltas*cur_step_size

            #get the new monitor parameter
            monitor_new = y_new[param_to_monitor]

            #check the percent change in the monitor
            if abs(monitor_new - monitor_cur)/monitor_new > max_param_delta:
                #the change was larger than allowed. Reduce the step size and
                #try again

                #first check if the step size is already minimal, fail if so
                if cur_step_size == min_step_time:
                    not_failed = False
                    status = -1 #time step fail
                else:
                    #not at the smallest allowed step, so halve the step size
                    cur_step_size = cur_step_size/2
                    if cur_step_size < min_step_time:
                        cur_step_size = min_step_time

                    if next_relax < 0:
                        #the time to try increasing step size isn't set, set it
                        next_relax = current_time + base_time_step
                
            else:
                #the step succeeded, add the new values and increment
                current_time += cur_step_size
                y_vals.append(y_new)
                times.append(current_time)
                y_cur = y_new

    return times, y_vals, status

def pynamic_test():
    """
    Validate the ode integrator with this function. This function will plot the
    function dy(t)/dt=-2y(t), which has the solution y(t)=3e^(-2t) for t>=0.
    The test will plot the results of pynamic compared to the true solution.
    """

    sol_times = np.linspace(0, 3, 25)
    sol_ys = 3*np.exp(-2*sol_times)

    plt.plot(sol_times, sol_ys)

    plt.savefig("test_results.png")

    return 0

pynamic_test()

#' ---
#' title : CMiE Project
#' author : Saqib M. Choudhary
#' date : 12th May 2018
#' ---

#' # Introduction
#' I am using an AR1 data generating process to simulate stocks for N entites over a time period T.
#' The first part deals with Markowitz Portfolio Optimiation problem. I choose the optimal weights 
#' to be assigned to N stock options based on their performance in time period 1-T. This assignment
#' is for time-period T+1 

#' In the second part, I simulate the weights assigned with Time. I start with time 2, assign weights 
#' to stock options based on performance on time period 1. For time period 3, I look into performance 
#' in time periods 1 & 2 and so on.

#' The final part of the project combines this with savings rate problem we did in the class
#' I use expected value of returns from N stocks to arrive at a savings rate.

#' # Defining Functions

#' 
#' Defining an AR1 process for Data generation


using Gadfly, Plots, Distributions, JuMP, Ipopt

struct AR1
    p::Float64
    sigma::Float64
    phi::Distributions.Normal{Float64}
end

#'
#'

#' A function that takes an object of type AR1 as input along with Time and Initail Value and generates consiquent values

function simulateData(m::AR1, n::Integer, x0::Real)
    X = Array{Float64}(n)
    X[1] = x0
    for t in 1:(n-1)
        X[t+1] = m.p * X[t] + m.sigma * rand(m.phi)/10
    end
    return X
end

#' A function that simulates an AR1 process for N stocks over T time period

function simulateTimeData(T,N)
    Data = zeros(T,N)
    dgp = Array{AR1}(N)

    for n=1:N
        dgp[n] = AR1(0.9,0.1,Normal(rand(),rand()))
    end

    for n = 1:N
        Data[:,n] = simulateData(dgp[n],T,rand()/10)
    end
    return Data
end

#' Function to assign random weights to N stock options
weights_rand = function (N)
    k = rand(N)
    return (k/sum(k))
end

#' Function to return the mean and covriance matrix of the Data

data_mean_cov = function(Data)
    p = [mean(Data[:,i]) for i in 1:N]
    C = cov(Data)

    return p,C
end

#' Function to calcute mean returns and variance for a given pre-defined weight among stock and 
#' a time series data of stocks

portfolio_return = function(returns,weights)
    p = [mean(returns[:,i]) for i in 1:N]
    C = cov(returns)

    mu = transpose(p)*weights
    sigma = (transpose(weights) * C * (weights))

    return mu,sigma
end

#' # Markowitz Portfolio Optimiation 

#' Generating Data

#+ term=true

N = 5
T = 100
Data = simulateTimeData(T,N)
using Plotly,Rsvg
P1 = Plotly.plot(Data)
Plotly.savefig(P1, "figure1.png")
#+

#' Using the simulated data to plot mean and standard devitions of return
#Numner of Portfolios

function generate_mean_sd(Data,NumberOfPort)
    mean = zeros(NumberOfPort)
    sd = zeros(NumberOfPort)

    for i in 1:NumberOfPort 
        weight = weights_rand(N)
        mean[i],sd[i] = portfolio_return(Data,weight)
    end
    return mean,sd
end


npf = 10000
mean1,sd1 = generate_mean_sd(Data,npf)

#+ term = true
Plots.scatter(sd1, mean1, title= "Means and Standard deviation of returns of randomly generated portfolios",xlabel = "standard deviation",ylab ="mean", markersize = 0.2,markeralpha = 0.6)
#+

#' #Getting Optimum Weights
#' Using two solvers to find optimum weights. Given a series of data from 1 to T, the solver provider an 
#' an optimum weight to be used for T+1

#' ##Minimising Variance 
#' In case an optimum solution is not found, equal weights are given to all portfolios

getOptimWeights = function(Data)
    r_min = 0.035
    val = [1/N for i = 1:N]

    m = Model(solver = IpoptSolver())
    mean, C = data_mean_cov(Data)
    @variable(m, 0 <= x[i = 1:N] <= 1)
    @objective(m, Min, sum{x[j]* sum{x[i]*C[i,j], i=1:N},j = 1:N})
    @constraint(m, sum{x[i]*mean[i],i=1:N}>=r_min)
    @constraint(m, sum(x[i = 1:N]) == 1.0)
    status = solve(m)

        
   if(status==:Optimal)
        val =  getvalue(x)
   end
   return val
end

#' ##Maximising Utlility while keeping variance below a thershold

getOptimWeights2 = function(Data)
    v_max = 0.00001
    val = [1/N for i = 1:N]

    m = Model(solver = IpoptSolver())
    mean, C = data_mean_cov(Data)
    @variable(m, 0 <= x[i = 1:N] <= 1)
    #@objective(m, Min, sum{x[j]* sum{x[i]*C[i,j], i=1:N},j = 1:N})
    @objective(m, Max, sum{x[i]*mean[i],i=1:N})
    @constraint(m, sum{x[j]* sum{x[i]*C[i,j], i=1:N},j = 1:N}<=v_max)
    @constraint(m, sum(x[i = 1:N]) == 1.0)
    status = solve(m)

    if(status==:Optimal)
        val =  getvalue(x)
   end
   return val
end

#' Adding Optimal Weight to Scatter Plot
optimWeights = getOptimWeights(Data)
meanReturn,SDReturn = portfolio_return(Data,optimWeights)
scatter!([SDReturn],[meanReturn],markersize = 5, markercolor = :red)

#' Getting optimal weights as a series of Time. 
weights_plot1 = [getOptimWeights(Data[1:t,:]) for t=2:T]
weights_plot2 = [getOptimWeights2(Data[1:t,:]) for t=2:T]

#' The returned objects are not easily plotable. Redefining them for plots

w_plot1 = zeros(T,N)
w_plot2 = zeros(T,N)
for t=1:T-1 
    for n=1:N 
        w_plot1[t,n] = weights_plot1[t][n]
        w_plot2[t,n] = weights_plot2[t][n]

    end
    
end
#+ term = true
Plots.plot(w_plot1,title = "Weights assigned to stocks for Variance Minimisation",xlab = "Time",ylab = "Weight")
Plots.plot(w_plot2,title = "Weights assigned to stocks for Value Maximisation",xlab = "Time",ylab = "Weight")
#+

#' #Saving Rate
#' Building on the model used in class. Using gains from stocks as capital gains for the next time period.

function solve_jump_version(Data, getOptimWeights)
    m = Model(solver = IpoptSolver())
    B = 0.97

    w = [getOptimWeights(Data[1:t,:]) for t=2:T]
    Rt = [transpose(w[t])*Data[t+1,:] for t=1:T-1]

    @variable(m, k[1:T] >= 0)
    @variable(m, 0 <= s[1:T] <= 1)
    @NLobjective(m, Max, sum(log((1-s[t])*k[t]*(1+Rt[t-1])*B^(t)) for t=2:T))
    @constraint(m, k[1]==1)
    @NLconstraint(m, eq_motion[t=1:T-1], k[t+1] == 0.9*k[t]+s[t]*(1+Rt[t]))
    
    solve(m)
    s_solution = getvalue(s)
    k_solution = getvalue(k)
    #c_solution = (1-s_solution).*k_solution.^α
    
    return k_solution, s_solution
end

k1,s1 = solve_jump_version(Data, getOptimWeights)
k2,s2 = solve_jump_version(Data, getOptimWeights2)


Plots.plot(s1, title = "Savings rate over time period T", xlab = "Time", ylab = "Savings rate")
Plots.plot!(s2)

Plots.plot(k1, title = "Capital over time period T", xlab = "Time", ylab = "Capital")

Plots.plot!(k2)

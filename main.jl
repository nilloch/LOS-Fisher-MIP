# ASEN 6519 Final Project
# Collin Hudson 12/07/2024
using Plots
using Reel
using Printf
using Distributions
using LinearAlgebra
using Zygote
using LazySets
using JuMP, SCIP, Juniper, Ipopt
using TickTock
using ParticleFilters
using Distances: Euclidean, colwise
ENV["TICKTOCK_MESSAGES"] = false
if isdefined(@__MODULE__, :workspace)
    workspace isa Module || error("workspace is present and it is not a Module")
else
    include("workspace.jl")
end
using .workspace
# Workspace parameters (xlim, ylim, number of obstacles, seed)
    # seed 2300 = squeeze
    # 353 broken somehow
    # e = env([-5,5],[-5,5],6,62)
    # e = env([-5,5],[-5,5],5,62)
    # global e = env([-5,5],[-5,5],5)
    xTrue = [[-6.0;0;0.25;0.25]]
    # gaPosX = [[-1;5]]
    # gaPosY = [[-1;-4]]
    gs = [2;-2]
    endK = 20
    Nobs = 0
    e = env([-5,5],[-5,5],Nobs,100)
    # e = env([-5,5],[-5,5],Nobs)
    # while !inLOS(xTrue[1][1],xTrue[1][2],xTrue[1][1] + endK*xTrue[1][3],xTrue[1][2] + endK*xTrue[1][4],e.obs) || inObs(gs[1],gs[2],e.obs)
    #     global e = env([-5,5],[-5,5],Nobs)
    # end
    print("Attempting to Solve:")
# 

# # Solve for optimal M values for obstacle avoidance
    #     ks = zeros(Nverts,1)
    #     hs = zeros(Nverts,2*Nverts)
    #     row = 1
    #     for ob in e.obs
    #         for h in LazySets.constraints_list(ob)
    #             hs[row,:] = hcat(zeros(1,2*(row-1)),h.a',zeros(1,2*(Nverts - row)))
    #             ks[row] = h.b
    #             global row += 1
    #         end
    #     end
    #     model = Model(HiGHS.Optimizer)
    #     set_silent(model)
    #     @variable(model, x[i=1:2*Nverts])
    #     for j in 1:2:2*Nverts
    #         @constraint(model,e.xlim[1] <= x[j] <= e.xlim[2])
    #         @constraint(model,e.ylim[1] <= x[j+1] <= e.ylim[2])
    #     end
    #     @objective(model, Max, sum(ks - hs*x))
    #     optimize!(model)
    #     # Mobs = maximum(ks - hs*value.(x))*ones(length(value.(x)),1)
    #     Mobs = ks - hs*value.(x)
# # 

# Problem parameters
    Nga = 2
    maxUAS = 4
    Ntot = Nga+maxUAS
    rCon = 2.5
    rMin = max(0.0,rCon - 3)
    rUAS = 2
    vMax = 2.5
    samples = 0.5
    Mcon = 1.1*((e.xlim[2] - e.xlim[1])^2 + (e.ylim[2] - e.ylim[1])^2)
    R = 0.01.*I(maxUAS)
    A =  [1 0 1 0;0 1 0 1; 0 0 1 0; 0 0 0 1]
    wiggleCov = [1 0 0 0;0 1 0 0;0 0 0.001 0; 0 0 0 0.001]
    Nparticles = 25000
    xPath = []
    yPath = []
    xcPath = []
    cPath = []
    xEst = []
    plots = []
    Xmin = [e.xlim[1];e.ylim[1];0;0]
    Xmax = [e.xlim[2];e.ylim[2];sqrt(vMax)/4;sqrt(vMax)/4]
#

# Information functions
    function FIMtrace(x,y,maxUAS)
        FIM = zeros(maxUAS,maxUAS)
        val = 0
        for j in 3:maxUAS+2
            FIM = FIM .+ [(x[2] - x[j])^2 (x[2] - x[j])*(y[2] - y[j]);(x[2] - x[j])*(y[2] - y[j]) (y[2] - y[j])^2]./((x[2] - x[j])^2 + (y[2] - y[j])^2)
            # val += tr([(x[2] - x[j])^2 (x[2] - x[j])*(y[2] - y[j]);(x[2] - x[j])*(y[2] - y[j]) (y[2] - y[j])^2]./((x[2] - x[j])^2 + (y[2] - y[j])^2))
        end
        val = tr(FIM)
        return val
    end

    d = 0.1 #added to reduce degeneracy
    function FIMdet(x,y,maxUAS)
        val = 0
        FIM = zeros(2,2)
        for j in 3:maxUAS+2
            FIM = FIM + [(x[2] - x[j])^2 (x[2] - x[j])*(y[2] - y[j]);(x[2] - x[j])*(y[2] - y[j]) (y[2] - y[j])^2]./((x[2] - x[j])^2 + (y[2] - y[j])^2)
        end
        val = FIM[1,1]*FIM[2,2] - FIM[1,2]*FIM[2,1]
        return val
    end
    function FIMdet2(xt,yt,x,y,maxUAS)
        val = 0
        FIM = zeros(2,2)
        for j in 3:maxUAS+2
            FIM = FIM + [(xt - x[j])^2 (xt - x[j])*(yt - y[j]);(xt - x[j])*(yt - y[j]) (yt - y[j])^2]./((xt - x[j])^2 + (yt - y[j])^2)
        end
        val = FIM[1,1]*FIM[2,2] - FIM[1,2]*FIM[2,1]
        return val
    end

    function getFIM(x,y,maxUAS)
        H = zeros(maxUAS,2)
        for j = 3:maxUAS+2
            H[j,:] = [x[2] - x[j], y[2] - y[j]]./(sqrt(((x[2] - x[j])^2 + (y[2] - y[j])^2)))
        end
        return H'*H
    end
# 

# Optimizer function
    function findStep(e, xLast, yLast, xcLast, k, xt)
        # model = Model(SCIP.Optimizer)
        ipopt = optimizer_with_attributes(Ipopt.Optimizer, "print_level"=>0)
        optimizer = optimizer_with_attributes(Juniper.Optimizer, "nl_solver"=>ipopt)
        model = Model(optimizer)
        set_silent(model)
        # set_time_limit_sec(model,45)
        # Variables and workspace boundary constraints
            @variable(model, e.xlim[1] <= x[i=1:Ntot] <= e.xlim[2]) #Agent positions x
            @variable(model, e.ylim[1] <= y[i=1:Ntot] <= e.ylim[2]) #Agent positions y
            @variable(model, c[i=1:Ntot,j=1:Ntot], Bin) #Network edge choice (which nodes are connected)
            @variable(model, o[i=1:e.Nverts,j=1:Ntot],Bin) #Slack variables for obstacle avoidance
            @variable(model, l[r=1:e.Nverts,i=1:Ntot,j=1:Ntot,k=1:length(samples)],Bin) #Slack variables for LOS
            @variable(model, xc[i=1:Ntot],Bin) #Network cluster (active connection set)
            @variable(model, f[i=1:Ntot,j=1:Ntot] >= 0,Int) #Network flow
            @objective(model, Max, FIMdet2(x[2],y[2],x,y,maxUAS) + FIMdet2(x[2]+xt[3],y[2]+xt[4],x,y,maxUAS) + FIMdet2(x[2]+2*xt[3],y[2]+2*xt[4],x,y,maxUAS))
        #

        # Ground agent constraints
            for j in 1:Nga
                fix(x[j], xLast[j];force=true) #Ground agents must be at given x
                fix(y[j], yLast[j];force=true) #Ground agents must be at given y
                # fix(xc[j], 1;force=true) #Ground agents must be in network
            end
            fix(xc[1], 1;force=true) #Ground station must be in network
            # fix(c[1,2], 0;force=true) #Ground station can't connect to target
            for j in 3:Ntot
                fix(xc[j], 1;force=true) #all sensors must be in network
            end

            # for i in 1:Ntot
            #     for j in 1:Ntot
            #         if i < j
            #             @constraint(model, (x[i] - x[j])^2 + (y[i] - y[j])^2 >= xc[j]*rMin^2)
            #         end
            #     end
            # end
        # 
        
        # Timestep constraints
            if k != 1
                for i in (Nga+1):Ntot
                    # @constraint(model, (x[i] - xLast[i])^2 + (y[i] - yLast[i])^2 <= (rUAS^2 + Mcon*(1-xcLast[i])))
                    @constraint(model, (x[i] - xLast[i])^2 + (y[i] - yLast[i])^2 <= (rUAS^2))
                    # fix(xc[i], xcLast[i];force=true)
                end
                # Add LOS constraint by adding columns? to o for each last position uas
            end
        # 

        # Network constraints
            # @constraint(model, sum(c*ones(Ntot,1)) >= (sum(xc)-1)) #Network must have at least (number of cluster agents - 1) edges
            # @constraint(model, sum(c*ones(Ntot,1)) >= (Ntot-1)) #Network must have at least (number of cluster agents - 1) edges
            for i in 1:Ntot
                # @constraint(model, sum(c[i,:]) <= Ntot*xc[i]) #Only agents in cluster can be connected
                # @constraint(model, sum(c[:,i]) <= Ntot*xc[i]) #Only agents in cluster can be connected
                @constraint(model, sum(f[i,:]) <= Ntot*xc[i]) #Only agents in cluster can have flow
                
                # @constraint(model, sum(f[:,i]) <= Ntot*xc[i]) #Only agents in cluster can have flow
                if i != 1
                    @constraint(model,(sum(f[i,:]) - sum(f[:,i])) == -1*xc[i]) #Non-source cluster agents must consume one unit of flow
                    # @constraint(model,(sum(f[i,:]) - sum(f[:,i])) == -1) #Non-source cluster agents must consume one unit of flow
                end
                for j in 1:Ntot
                    if i < j
                        @constraint(model, (f[i,j] + f[j,i]) <= ((sum(xc) - 1)*c[i,j]))
                        # @constraint(model, (f[i,j] + f[j,i]) <= ((Ntot - 1)*c[i,j]))
                    elseif i == j
                        fix(f[i,j], 0;force=true) #Flow must be between two different agents
                    end
                end
            end
            @constraint(model, sum(f[:,2]) == 0) #estimated target node cannot have flow out
        # 

        # Connection radius constraint
            for i in 1:Ntot
                for j in 1:Ntot
                    if i < j
                        # @constraint(model, (x[i] - x[j])^2 + (y[i] - y[j])^2 <= (rCon^2 + Mcon*(1-c[i,j]) + Mcon*(1-xc[j])))
                        @constraint(model, (x[i] - x[j])^2 + (y[i] - y[j])^2 <= (rCon^2 + Mcon*(1-c[i,j])))
                        # @constraint(model, (x[i] - x[j])^2 + (y[i] - y[j])^2 >= xc[j]*rMin^2)
                        @constraint(model, (x[i] - x[j])^2 + (y[i] - y[j])^2 >= rMin^2)
                    else
                        fix(c[i,j], 0;force=true)
                    end
                end
            end
        # 

        # Obstacle avoidance and LOS constraints
            row = 1
            obV = 1
            for ob in e.obs
                for con in constraints_list(ob)
                    for j in 1:Ntot
                        @constraint(model,-con.a'*[x[j];y[j]] <= (-con.b + e.Mobs[row]*o[row,j]))
                    end
                    for i in 1:Ntot
                        for j in 1:Ntot
                            if i < j
                                for k in eachindex(samples)
                                    si = (1-samples[k])*[x[i];y[i]] + samples[k]*[x[j];y[j]]
                                    @constraint(model, -con.a'*si <= (-con.b + e.Mobs[row]*o[row,i] + e.Mobs[row]*l[row,i,j,k]))
                                    @constraint(model, -con.a'*si <= (-con.b + e.Mobs[row]*o[row,j] + e.Mobs[row]*l[row,i,j,k]))
                                end
                                @constraint(model, c[i,j] <= sum(1 .- l[row,i,j,1:length(samples)])) #At least one sample point must be valid if connected
                            end
                        end
                    end
                    row += 1
                end
                for j in 1:Ntot
                    @constraint(model,sum(o[obV:(obV+length(ob.vertices)-1),j]) <= (length(ob.vertices)-1)) #At least one constraint must be active per obstacle
                end
                obV += length(ob.vertices)
            end
        # 
        
        optimize!(model)
        if !is_solved_and_feasible(model)
            print("Infeasible workspace at k = " * string(k)*"\n")
            # p = plot(size = (400,400))
            # for j in eachindex(e.obs)
            #     plot!(p,e.obs[j])
            # end
            # xlims!(p,e.xlim[1]-1,e.xlim[2]+1)
            # ylims!(p,e.ylim[1]-1,e.ylim[2]+1)
            # for j in 1:Ntot
            #     if j < Nga+1
            #         scatter!([xLast[j]],[yLast[j]],mc=:green,label=nothing)
            #         plot!(xLast[j] .+ rCon*cos.(range(0,2*π,500)),yLast[j] .+ rCon*sin.(range(0,2*π,500)),linecolor=:green,label=nothing)
            #     else
            #         scatter!([xLast[j]],[yLast[j]],mc=:black,label=nothing)
            #         plot!(xLast[j] .+ rUAS*cos.(range(0,2*π,500)),yLast[j] .+ rUAS*sin.(range(0,2*π,500)),linecolor=:gray,label=nothing)
            #     end
            # end
            # title!(p,"Infeasible workspace")
            # display(p);
        end
        if is_solved_and_feasible(model)
            # @show value.(H)
            # @show value.(iFIM)
            return [value.(x),value.(y),ceil.(value.(xc)),ceil.(value.(c))]
        else
            return [xLast,yLast,xcLast,zeros(Ntot,Ntot)]
        end
    end
# 
    
# Particle filter
    function dynPrediction(x,u,rng)
        # dynamics model for a constant velocity
        # input x = state at timestep k(column vector)
        # input u = UAS/sensor positions [unused]
        # input rng = rng seed (required by package)[unused]
        # returns x' = predicted state at timestep k + 1
        # f(x) = [1 0 1 0;0 1 0 1; 0 0 1 0; 0 0 0 1][xpos;ypos;xvel;yvel]
        return A*x + rand(MvNormal(wiggleCov))
        # return A*x + rand(MvNormal(wiggleCov)).*[1;1;0;0]
    end

    function obsWeight(xold,u,x,y)
        # weighting function given range only measurements from N sensors
        # input xold = state at timestep k(4x1 vector) [unused]
        # input u = UAS/sensor positions (Nx2 matrix)
        # input x = state at timestep k+1(4x1 vector)
        # input y = received measurement at timestep k+1 (Nx1 vector)
        # returns w = weight for particle with state x given recieved measurements y

        # Independent sensors (UAS) so probability of received measurement vector is 
        #   product of probabilies of each sensor receiving measurement y[i]
        # w = 1
        if any(x .> Xmax) || any(x .< Xmin) || inObs(x[1],x[2],e.obs)
            return 0 # reject out-of-bounds particles
        end
        w = pdf(MvNormal([0;0],[0.05 0;0 0.05]),(x[1:2] - xold[1:2]) - xold[3:4]) #velocity error weight
        for j = axes(u,2)
            r = Euclidean()(x[1:2],u[:,j])
            measTrunc = truncated(Normal(r,R[j,j]); lower=0, upper=rCon)  #Gaussian truncated to the interval [l, u]
            if y[j] == -1 #check for NOT receiving a measurement (i.e. target is outside communication range)
                w *= (r > rCon)||(!inLOS(x[1],x[2],u[1,j],u[2,j],e.obs)) #reject particles that are inside the measurement range and in LOS of sensor j
            elseif r <= rCon
                w *= pdf(measTrunc,y[j])
            else
                return 0 # reject particles outside sensor range when valid measurement received
            end
        end
        return w
    end

    function initBelief(Xmin,Xmax,Nparticles)
        len = size(Xmin,1)
        p0 = zeros(Float64,len,Nparticles)
        for j = 1:Nparticles
            temp = rand(len)
            p0[:,j] = Xmin.*temp + Xmax.*(1 .- temp)
            while inObs(p0[1,j],p0[2,j],e.obs)
                temp = rand(len)
                p0[:,j] = Xmin.*temp + Xmax.*(1 .- temp)
            end
        end
        return ParticleCollection([p0[:,k] for k in 1:Nparticles])
    end

    function plotStep!(b,u,x,plots)
        plt = scatter([x[1]], [x[2]], color=:blue, xlim=(Xmin[1],Xmax[1]), ylim=(Xmin[2],Xmax[2]), label="GT", legend=:bottomleft)
        scatter!(plt, u[1,:], u[2,:], color=:red, label="UAS")
        for j in 1:maxUAS
            plot!(plt, u[1,j] .+ rCon*cos.(range(0,2*π,500)),u[2,j] .+ rCon*sin.(range(0,2*π,500)),linecolor=:green,label=nothing)
        end
        scatter!(plt,[p[1] for p in particles(b)], [p[2] for p in particles(b)], color=:black, markersize=1, label="")
        scatter!(plt,[mode(b)[1]],[mode(b)[2]], color=:purple, markersize=2, label="Mode")
        push!(plots, plt)
    end
    bootPF = BootstrapFilter(ParticleFilterModel{Vector{Float64}}(dynPrediction,obsWeight), Nparticles)

# Simulation loop
    b = initBelief(Xmin,Xmax,Nparticles) #initialize particle collection (belief) from uniform distribution
    measNoise = MvNormal(R) #AWGN distribution to sample from for measurement noise
    push!(xEst,rand(b)) # draw initial estimate from initial belief
    xEst2 = copy(xEst)
    tick()
    for k in 1:endK
        # plot workspace and prior particle distribution
            p = plot(dpi=500)
            for j in eachindex(e.obs)
                plot!(p,e.obs[j])
            end
            scatter!(p,[p[1] for p in particles(b)], [p[2] for p in particles(b)], color=:black, markersize=1, label="")
            scatter!(p,[ParticleFilters.mode(b)[1]],[ParticleFilters.mode(b)[2]], color=:purple, markersize=2, label="Mode")
            scatter!(p,[ParticleFilters.mean(b)[1]],[ParticleFilters.mean(b)[2]], color=:pink, markersize=2, label="Mean")
        # 

        # Get action (network configuration)
            if k == 1
                (xk,yk,xck,ck) = findStep(e,[gs[1];xEst[1][1];zeros(maxUAS,1)],[gs[2];xEst[1][2];zeros(maxUAS,1)],zeros(Ntot,1),1,xEst[1])
            else
                (xk,yk,xck,ck) = findStep(e,[gs[1];xEst[k][1] + xEst[k][3];xPath[k-1][(Nga+1):Ntot]],[gs[2];xEst[k][2] + xEst[k][4];yPath[k-1][(Nga+1):Ntot]],xcPath[k-1],k,xEst[k])
            end
        # 

        # take measurement
            y = colwise(Euclidean(),xTrue[k][1:2],[xk[(Nga+1):Ntot]';yk[(Nga+1):Ntot]']) + 0 .*rand(measNoise)
            # send -1 measurement if out of range or LOS
            for j in 1:maxUAS
                if !inLOS(xTrue[k][1],xTrue[k][2],xk[j+Nga],yk[j+Nga],e.obs) || y[j] > rCon
                    y[j] = -1.0
                end
            end
            @show y
        # 

        # update belief
        global b = update(bootPF,b,[xk[(Nga+1):Ntot]';yk[(Nga+1):Ntot]'],y)

        #Move forward in time and store timestep k values
            push!(xTrue,A*xTrue[k])
            push!(xPath,xk)
            push!(yPath,yk)
            push!(xcPath,xck)
            push!(cPath,ck)
            push!(xEst,ParticleFilters.mode(b))
            push!(xEst2,ParticleFilters.mean(b))
        # 

        # additional plotting
            scatter!(p,[xTrue[k][1]], [xTrue[k][2]], color=:red, label="GT", legend=:bottomleft)
            scatter!(p,[xPath[k][1]], [yPath[k][1]], color=:green, label="GS")
            scatter!(p,[xEst[k][1] + xEst[k][3]], [xEst[k][2] + xEst[k][4]], color=:blue, label="GT Prior")
            for j in 3:(maxUAS+2)
                scatter!(p, [xPath[k][j]], [yPath[k][j]], text = string(j-2), color=:yellow, label=nothing)
            end
            for j in 1:2
                plot!(p, xPath[k][j] .+ rCon*cos.(range(0,2*π,500)),yPath[k][j] .+ rCon*sin.(range(0,2*π,500)),linecolor=:green,label=nothing)
            end
            for j in 1:Ntot
                for i in 1:Ntot
                    if(cPath[k][j,i] != 0) && xcPath[k][j] != 0
                        plot!(p,[xPath[k][j],xPath[k][i]], [yPath[k][j],yPath[k][i]],linecolor=:gray,label=nothing)
                    end
                end
            end
            xlims!(p,e.xlim[1]-1,e.xlim[2]+1)
            ylims!(p,e.ylim[1]-1,e.ylim[2]+1)
            title!(p,"Timestep: "*string(k))
            # display(p);
            push!(plots,p)
            savefig(p,"finalProject/plots/" * string(k) * ".png")
        # 
    end
    tock()
    # frames = Frames(MIME("image/png"), fps=1)
    # for plt in plots
    #     print(".")
    #     push!(frames, plt)
    # end
    # write("finalProject/output.gif", frames)
# 

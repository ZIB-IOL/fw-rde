indices = 0:49
rates = [2000, 4000, 6000, 8000, 10000, 14000, 18000, 22000, 26000, 30000, 34000, 38000, 42000, 46000, 50000]
max_iter = 2000

fw_arguments = (
    line_search=FrankWolfe.MonotonousNonConvexStepSize(),
    max_iteration=max_iter,
    print_iter=max_iter / 10,
    verbose=true,
    #lazy=true,
)

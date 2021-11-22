indices = 0:49
rates = [50, 100, 150, 200, 250, 300, 350, 400]
max_iter = 2000

fw_arguments = (
    line_search=FrankWolfe.MonotonousNonConvexStepSize(),
    max_iteration=max_iter,
    print_iter=max_iter / 10,
    verbose=true,
    #lazy=true,
)

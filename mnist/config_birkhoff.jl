indices = 0:49
rates = [50, 100, 150, 200, 250, 300, 350, 400]
all_rates = 1:784
max_iter = 200

fw_arguments = (
    line_search=FrankWolfe.Nonconvex(),
    max_iteration=max_iter,
    print_iter=max_iter / 10,
    verbose=true,
    #lazy=true,
    #momentum_iterator=FrankWolfe.ConstantMomentumIterator(0.5),
    #momentum_iterator=FrankWolfe.ExpMomentumIterator(),
    #batch_iterator=FrankWolfe.ConstantBatchIterator(80),
    #batch_iterator=FrankWolfe.IncrementBatchIterator(40, 100),
)

class Optimizer:
    def __init__(self, params, defaults):
        self.defaults = defaults
        self.param_groups = []

        # make sure params is a list 
        param_list = list(params)
        # convert to list of dicts 
        if not isinstance(param_list[0], dict):
            param_list = [{'params': param_list}]

        for param_group in param_list:
            self.add_param_group(param_group)

    def add_param_group(self, param_group):
        # merge it with the global defaults
        group = self.defaults.copy()
        group.update(param_group)
        self.param_groups.append(group)

    def zero_grad(self):
        for param_group in self.param_groups:
            for p in param_group['params']:
                if p.grad is not None:
                    p.grad = None 
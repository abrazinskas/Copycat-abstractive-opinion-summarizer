def get_init_func(multi_dim_init_func, single_dim_init_func):
    """Returns an initialization function for multi and single dim. params."""

    def init(m):
        params = m.parameters()
        for p in params:
            try:
                if len(p.data.shape) > 1:
                    multi_dim_init_func(p.data)
                else:
                    single_dim_init_func(p.data)
            except Exception:
                raise ValueError("Could not initialize the parameter `%s`."
                                 % m.__class__.__name__)

    return init

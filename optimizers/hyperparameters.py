from optimizers.distribution import Choice, Normal, QNormal, LogNormal, Uniform, QUniform

DDPG = {
    'gamma': Uniform(0.8, 0.99),
    'actor_learning_rate': Normal(1e-3, 5e-4),
    'critic_learning_rate': Normal(1e-3, 5e-4),
    'n_layers_actor': Choice([2, 3]),
    'n_layers_critic': Choice([2, 3]),
    'units_per_layer_actor': QNormal(230, 20, 1),
    'units_per_layer_critic': QNormal(230, 20, 1),
    'tau': LogNormal(-6, 1),
    'buffer_size': QNormal(10000, 2000, 10),
    'batch_size': QUniform(32, 512, 10),
    'start_training': QUniform(1, 1000, 10),
    'noise_std': LogNormal(-1.61, 0.5)
}

A2C = {
    'gamma': Uniform(0.8, 0.99),
    'actor_learning_rate': Normal(1e-3, 5e-4),
    'critic_learning_rate': Normal(1e-3, 5e-4),
    'n_layers_actor': Choice([2, 3]),
    'n_layers_critic': Choice([2, 3]),
    'units_per_layer_actor': QNormal(230, 20, 1),
    'units_per_layer_critic': QNormal(230, 20, 1),
    'std_state_dependent': Choice([True, False]),
    'log_std_init': Normal(0.5, 0.1)
}

DDPG_BOUNDARIES = {
    'gamma': (0.8, 0.99),
    'actor_learning_rate': (1e-4, 2e-3),
    'critic_learning_rate': (1e-4, 2e-3),
    'n_layers_actor': (2, 4),
    'n_layers_critic': (2, 4),
    'units_per_layer_actor': (64, 1024),
    'units_per_layer_critic': (64, 1024),
    'tau': (0.001, 0.01),
    'buffer_size': (10_000, 1_000_000),
    'batch_size': (16, 256),
    'start_training': (1, 10_000),
    'noise_std': (0.01, 0.5)
}

A2C_BOUNDARIES = {
    'gamma': (0.8, 0.99),
    'actor_learning_rate': (1e-4, 2e-3),
    'critic_learning_rate': (1e-4, 2e-3),
    'n_layers_actor': (2, 4),
    'n_layers_critic': (2, 4),
    'units_per_layer_actor': (64, 1024),
    'units_per_layer_critic': (64, 1024),
    'std_state_dependent': (0, 1),
    'log_std_init': (0.1, 1)
}


def get_hyp(algo):
    if algo == 'A2C':
        return A2C
    elif algo == 'DDPG':
        return DDPG
    return None


def get_boundaries(algo):
    bounds = []
    if algo == 'A2C':
        for b in A2C_BOUNDARIES:
            bounds.append(b)
    elif algo == 'A2C':
        for b in DDPG_BOUNDARIES:
            bounds.append(b)
    else:
        return None
    return bounds

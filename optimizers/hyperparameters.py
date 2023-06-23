from tpe import Choice, Normal, QNormal, LogNormal, Uniform, QUniform

DDPG = {
    'gamma': Uniform(0.8, 0.99),
    'actor_learning_rate': Normal(1e-3, 5e-4),
    'critic_learning_rate': Normal(1e-3, 5e-4),
    'n_layers_actor': Choice([2, 3]),
    'n_layers_critic': Choice([2, 3]),
    'units_per_layer_actor': QNormal(230, 20, 1),
    'units_per_layer_critic': QNormal(230, 20, 1),
    'activation_actor': Choice(['relu', 'relu6']),
    'activation_critic': Choice(['relu', 'relu6']),
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
    'activation_actor': Choice(['relu', 'relu6']),
    'activation_critic': Choice(['relu', 'relu6']),
    'std_state_dependent': Choice([True, False]),
    'log_std_init': Normal(0.5, 0.1)
}


def get_hyp(algo):
    if algo == 'A2C':
        return A2C
    elif algo == 'DDPG':
        return DDPG
    return None

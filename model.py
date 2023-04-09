import flax.linen as nn
import jax.numpy as jnp

class NN(nn.Module):
    """ Shared-param model for policy and value function """

    hidden_layer_sizes: tuple[int]
    n_actions: int
    single_input_shape: tuple[int]
    activation: str

    @nn.compact
    def __call__(self, x: jnp.array):
        """ x: (n_features,) """

        if self.activation == "relu":
            activation = nn.relu
        elif self.activation == "tanh":
            activation = nn.tanh
        else: 
            raise NotImplementedError
        
        # Flatten input features if required
        if x.shape == self.single_input_shape:
            x = jnp.ravel(x)
        elif x.shape[1:] == self.single_input_shape:
            x = jnp.reshape(x, (x.shape[0], -1))
        else: 
            raise NotImplementedError

        # Shared layers
        for l, size in enumerate(self.hidden_layer_sizes):
            x = nn.Dense(features=size, 
                         kernel_init=nn.initializers.orthogonal(scale=jnp.sqrt(2.0)),
                         name=f"dense_{l+1}")(x)
            x = activation(x)

        # Output layers
        policy_logits = nn.Dense(features=self.n_actions, 
                                 kernel_init=nn.initializers.orthogonal(scale=0.01),
                                 name="logits")(x)
        policy_log_probs = nn.log_softmax(policy_logits)

        value = nn.Dense(features=1, 
                         kernel_init=nn.initializers.orthogonal(scale=1.0),
                         name="value")(x)
        
        return policy_log_probs, value  # (n_actions,), (1,)


class NNSeparate(nn.Module):
    """ Separate models for policy and value function """

    hidden_layer_sizes: tuple[int]
    n_actions: int
    single_input_shape: tuple[int]
    activation: str

    @nn.compact
    def __call__(self, x: jnp.array):
        """ x: (n_features,) """

        if self.activation == "relu":
            activation = nn.relu
        elif self.activation == "tanh":
            activation = nn.tanh
        else: 
            raise NotImplementedError
        
        # Flatten input features if required
        if x.shape == self.single_input_shape:
            x = jnp.ravel(x)
        elif x.shape[1:] == self.single_input_shape:
            x = jnp.reshape(x, (x.shape[0], -1))
        else: 
            raise NotImplementedError

        actor_x = x
        critic_x = x


        # Actor
        for l, size in enumerate(self.hidden_layer_sizes):
            actor_x = nn.Dense(features=size, 
                         kernel_init=nn.initializers.orthogonal(scale=jnp.sqrt(2.0)),
                         name=f"dense_{l+1}_policy")(actor_x)
            actor_x = activation(actor_x)

        policy_logits = nn.Dense(features=self.n_actions, 
                                 kernel_init=nn.initializers.orthogonal(scale=0.01),
                                 name="logits")(actor_x)
        policy_log_probs = nn.log_softmax(policy_logits)


        # Critic
        for l, size in enumerate(self.hidden_layer_sizes):
            critic_x = nn.Dense(features=size, 
                         kernel_init=nn.initializers.orthogonal(scale=jnp.sqrt(2.0)),
                         name=f"dense_{l+1}_value")(critic_x)
            critic_x = activation(critic_x)

        value = nn.Dense(features=1, 
                         kernel_init=nn.initializers.orthogonal(scale=1.0),
                         name="value")(critic_x)
        
        # Return
        return policy_log_probs, value  # (n_actions,), (1,)
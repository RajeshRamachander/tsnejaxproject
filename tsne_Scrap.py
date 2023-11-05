import jax
import jax.numpy as jnp
from jax import lax

@jax.jit
def update_beta_until_convergence(i, beta_values, distances_i, log_target_perplexity, entropy_difference, tolerance=1e-5, max_tries=50):
    num_tries = 0
    beta_value = beta_values[i]
    beta_min = -jnp.inf
    beta_max = jnp.inf

    def loop_cond(carry):
        _, _, _, entropy_difference, num_tries, _ = carry
        return jnp.logical_and(jnp.abs(entropy_difference) > tolerance, num_tries < max_tries)

    def loop_body(carry):
        beta_value, beta_min, beta_max, entropy_difference, num_tries, beta_values = carry
        # beta_value, beta_min, beta_max = update_beta(beta_value, entropy_difference, beta_min, beta_max)
        beta_values = beta_values.at[i].set(beta_value)
        entropy, this_probabilities = 0.1, .2
        entropy_difference = entropy - log_target_perplexity
        num_tries += 1
        return (beta_value, beta_min, beta_max, entropy_difference, num_tries, beta_values, this_probabilities)

    initial_loop_state = (beta_value, beta_min, beta_max, entropy_difference, num_tries, beta_values)
    final_loop_state = lax.while_loop(loop_cond, loop_body, initial_loop_state)

    # Extract updated beta_values from the final state
    updated_beta_values = final_loop_state[-1]

    return updated_beta_values

# Example usage:
i = 0  # Replace with the desired value of 'i'
beta_values = jnp.array([0.1, 0.2, 0.3])  # Replace with your initial beta values
distances_i = jnp.array([0.5, 0.6, 0.7])  # Replace with your distances_i array
log_target_perplexity = jnp.log(30.0)  # Replace with your target perplexity
entropy_difference = jnp.inf  # Replace with your initial entropy difference
updated_beta_values = update_beta_until_convergence(i, beta_values, distances_i, log_target_perplexity, entropy_difference)
print("Updated beta_values:", updated_beta_values)

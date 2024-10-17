import functools
import jax
from flax import struct
import jax.numpy as jnp
import time
import pdb

@functools.partial(jax.jit, static_argnames=["k", "recall_target"])
def l2_ann(qy, db, half_db_norms, k=5, recall_target=0.95):
    """Compute k-nearest neighbors using approximate L2 distance."""
    dists = half_db_norms - jax.lax.dot(qy, db.transpose())
    return jax.lax.approx_min_k(dists, k=k, recall_target=recall_target)

@struct.dataclass
class KNN(struct.PyTreeNode):
    capacity: int
    data: jnp.ndarray = struct.field(pytree_node=True)
    norms: jnp.ndarray = struct.field(pytree_node=True)
    index: int = struct.field(pytree_node=False)
    size: int = struct.field(pytree_node=False)

    @classmethod
    def create(cls, capacity, d=48):
        """Factory method to create a KNN instance."""
        data = jnp.zeros((capacity, d))
        norms = jnp.full((capacity,), 1e20)
        return cls(capacity=capacity, data=data, norms=norms, index=0, size=0)

    def get_state(self):
        return {
            "capacity": self.capacity,
            "data": self.data,
            "norms": self.norms,
            "index": self.index,
            "size": self.size,
        }

    def restore(self, state):
        return self.replace(
            capacity=state["capacity"],
            data=state["data"],
            norms=state["norms"],
            index=state["index"],
            size=state["size"]
        )

    def add_batch(self, batch):
        """Add a batch of vectors to the dataset, cycling through when exceeding capacity."""
        batch_size = batch.shape[0]

        # Calculate new indices for the batch
        indices = jnp.arange(self.index, self.index + batch_size) % self.capacity

        # Update data and norms in a vectorized manner
        updated_data = self.data.at[indices].set(batch)
        updated_norms = self.norms.at[indices].set(0.5 * (jnp.linalg.norm(batch, axis=-1) ** 2))

        # Update the index and size
        new_index = (self.index + batch_size) % self.capacity
        new_size = min(self.size + batch_size, self.capacity)

        # Return a new instance with updated fields
        return self.replace(
            data=updated_data,
            norms=updated_norms,
            index=new_index,
            size=new_size
        )

    def query(self, query_points, k):
        valid_data = self.data
        valid_norms = self.norms
        half_distances, indices = l2_ann(query_points, valid_data, valid_norms, k=k)
        valid_mask = jnp.less(indices, self.size).astype(jnp.float32)
        distances = jnp.sqrt(2 * half_distances + jnp.expand_dims(jnp.linalg.norm(query_points, axis=-1) ** 2, axis=1))
        avg = jnp.sum(distances * valid_mask, axis=-1) / (jnp.sum(valid_mask, axis=-1) + 1e-10)
        return avg

# Benchmarking
if __name__ == "__main__":
    # Create a KNN instance with capacity of 100,000
    dim = 512
    knn = KNN.create(capacity=1000000, d=dim)

    num_iterations = 2000
    batch_size = 4

    # Precompile the l2_ann function with dummy data
    dummy_data = jax.random.normal(jax.random.PRNGKey(0), (1000000, dim))  # Dummy data
    knn = knn.add_batch(dummy_data)  # Pre-add some dummy data
    dummy_query = jax.random.normal(jax.random.PRNGKey(1), (1, dim))  # Dummy query
    l2_ann(dummy_query, knn.data, knn.norms, k=5)  # JIT compile the function

    # Start benchmarking
    start_time = time.time()

    for i in range(num_iterations):
        # Add a batch of vectors
        print(f"iter {i}")
        batch = jax.random.normal(jax.random.PRNGKey(2), (batch_size, dim))  # Batch of 4 vectors
        knn = knn.add_batch(batch)

        # Perform a KNN query
        query_batch = jax.random.normal(jax.random.PRNGKey(3), (8, dim))  # 1 query vector
        distances = knn.query(query_batch, 5)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Total time for {num_iterations} iterations: {elapsed_time:.4f} seconds")
    print(f"Average time per iteration: {elapsed_time / num_iterations:.6f} seconds")

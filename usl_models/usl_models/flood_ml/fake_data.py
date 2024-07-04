from usl_models.flood_ml import constants
from usl_models.flood_ml import model
import functools
from typing import Iterator, Tuple, Dict, Any
import dataclasses
import jax
import jax.numpy as jnp
import tensorflow as tf


@dataclasses.dataclass
class Example:
    geospatial: jax.Array
    temporal: jax.Array
    spatiotemporal: jax.Array


@dataclasses.dataclass
class DatasetConfig:
    width: int
    height: int
    batch_size: int
    storm_duration: int

    def geospatial_shape(self) -> tuple:
        return (self.batch_size, self.height, self.width, constants.GEO_FEATURES)

    def temporal_shape(self) -> tuple:
        # return (self.batch_size, constants.MAX_RAINFALL_DURATION, constants.M_RAINFALL)
        return (self.batch_size, constants.MAX_TIMESTEPS, constants.M_RAINFALL)

    def spaciotemporal_shape(self) -> tuple:
        return (self.batch_size, self.height, self.width, 1)

    def label_shape(self) -> tuple:
        return (self.batch_size, constants.MAX_TIMESTEPS, self.height, self.width)


def fake_example_generator(config: DatasetConfig) -> Iterator[Tuple[list, tf.Tensor]]:
    key = jax.random.key(123)
    for i in range(10):
        key, *subkeys = jax.random.split(key, 5)
        print("\n---------------------------------------------------------")
        print(f"Generate sample {i}.")

        geospatial = jax.random.normal(subkeys[0], shape=config.geospatial_shape())
        temporal = jax.random.normal(subkeys[1], shape=config.temporal_shape())
        spatiotemporal = jax.random.normal(subkeys[2], shape=config.spaciotemporal_shape())
        x = [geospatial, temporal, spatiotemporal]
        y = jax.random.normal(subkeys[3], config.label_shape())
        yield (x, y)


def get_fake_dataset(config: DatasetConfig) -> Tuple[tf.data.Dataset, int]:
    """
    Get the dataset for training.
    """
    # Create the dataset
    dataset = tf.data.Dataset.from_generator(
        functools.partial(
            fake_example_generator, config
        ),  # Pass 'sim_name' to 'combined_generator'
        output_signature=(
            dict(
                geospatial=tf.TensorSpec(shape=config.geospatial_shape(), dtype=tf.float32),
                temporal=tf.TensorSpec(
                    shape=config.temporal_shape(), dtype=tf.float32
                ),
                spatiotemporal=tf.TensorSpec(
                    shape=config.spaciotemporal_shape(), dtype=tf.float32
                ),
            ),
            tf.TensorSpec(shape=config.label_shape(), dtype=tf.float32),
        )
    )
    return (dataset, config.storm_duration)

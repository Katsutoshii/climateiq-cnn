from usl_models.flood_ml import constants
from usl_models.flood_ml import model
import functools
from typing import Iterator, Tuple, Dict, Any
import dataclasses
import tensorflow as tf


@dataclasses.dataclass
class Example:
    geospatial: tf.Tensor
    temporal: tf.Tensor
    spatiotemporal: tf.Tensor


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
    for i in range(10):
        print("\n---------------------------------------------------------")
        print(f"Generate sample {i}.")

        x = dict(geospatial=tf.random.normal(shape=config.geospatial_shape()),
                 temporal=tf.random.normal(shape=config.temporal_shape()),
                 spatiotemporal=tf.random.normal(shape=config.spaciotemporal_shape()))
        y = tf.random.normal(config.label_shape())
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

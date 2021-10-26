import numpy as np
import tensorflow as tf


def log_histogram(summary, tag, values, bins=1000):
    """Logs the histogram of a list/vector of values."""
    # Convert to a numpy array
    values = np.array(values)

    # Create histogram using numpy
    counts, bin_edges = np.histogram(values, bins=bins)

    # Fill fields of histogram proto
    hist = tf.HistogramProto()
    hist.min = float(np.min(values))
    hist.max = float(np.max(values))
    hist.num = int(np.prod(values.shape))
    hist.sum = float(np.sum(values))
    hist.sum_squares = float(np.sum(values ** 2))

    # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
    # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
    # Thus, we drop the start of the first bin
    bin_edges = bin_edges[1:]

    # Add bin edges and counts
    for edge in bin_edges:
        hist.bucket_limit.append(edge)
    for c in counts:
        hist.bucket.append(c)

    summary.value.add(tag=tag, histo=hist)


def log_text(writer, tag, text):
    text_tensor = tf.make_tensor_proto(text, dtype=tf.string)
    meta = tf.SummaryMetadata()
    meta.plugin_data.plugin_name = "text"
    summary = tf.Summary()
    summary.value.add(tag=tag, metadata=meta, tensor=text_tensor)
    writer.add_summary(summary)


def log_scalar(summary: tf.Summary, tag, value):
    summary.value.add(tag=tag, simple_value=value)


def log_scalars(**kwargs):
    for key, value in kwargs.items():
        tf.summary.scalar(key, value)
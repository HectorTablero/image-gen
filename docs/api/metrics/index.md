# Metrics

Metrics for evaluating the quality of generative models. These metrics help quantify the performance of the models.

## Base Metric

The base class for all evaluation metrics. It defines the interface that all metric implementations must follow.

**Main Methods:**
- `__call__(real, generated)`: Computes the metric value between real and generated samples.

[View Implementation](base.md)

## Bits Per Dimension

Evaluates probabilistic generative models based on their log-likelihood. Lower values indicate better models.

**Main Methods:**
- `__call__(real, _)`: Computes bits per dimension for the real data.

[View Implementation](bpd.md)

## Fréchet Inception Distance

Measures the distance between feature representations of real and generated images using the Inception-v3 model. Lower values indicate better quality and diversity.

**Main Methods:**
- `_get_inception()`: Creates and prepares the Inception-v3 model for feature extraction.
- `_get_activations(images)`: Extracts Inception features from input images.
- `_calculate_fid(real, gen)`: Calculates the FID score from feature activations.
- `__call__(real, generated)`: Computes the FID score between real and generated images.

[View Implementation](fid.md)

## Inception Score

Evaluates the quality and diversity of generated images using the Inception-v3 model. Higher values indicate better quality and diversity.

**Main Methods:**
- `_get_inception()`: Creates and prepares the Inception-v3 model for feature extraction.
- `_get_predictions(images)`: Gets softmax predictions from the Inception model.
- `_calculate_is(predictions)`: Calculates the Inception Score from softmax predictions.
- `__call__(_, generated)`: Computes the Inception Score for generated images.

[View Implementation](inception.md)
import os
import time
import csv
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import sys
sys.path.append("./..")
if True:
    from image_gen.metrics import BitsPerDimension, FrechetInceptionDistance, InceptionScore
    from image_gen import GenerativeModel

NUM_EPOCHS = 100
SAMPLES_PER_RUN = 25
TRAIN_PERCENT = 0.8
SEED = 0
BATCH_SIZE = 64
MNIST_CLASS_LABEL = 3
CIFAR_CLASS_LABEL = 1


def load_full_dataset(dataset_name):
    print(f"Loading full {dataset_name} dataset...")

    if dataset_name == 'mnist':
        transform = transforms.ToTensor()
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    dataset_class = datasets.MNIST if dataset_name == 'mnist' else datasets.CIFAR10
    data = dataset_class(
        root='data',
        train=True,
        download=True,
        transform=transform
    )

    train_size = int(TRAIN_PERCENT * len(data))
    test_size = len(data) - train_size
    train_data, test_data = torch.utils.data.random_split(
        data, [train_size, test_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    test_loader = DataLoader(test_data, batch_size=SAMPLES_PER_RUN)
    test_batch = next(iter(test_loader))[0]

    return {
        'train': train_data,
        'test': test_batch
    }


def load_class_dataset(dataset_name, class_label):
    print(f"Loading {dataset_name} dataset for class {class_label}...")

    if dataset_name == 'mnist':
        transform = transforms.ToTensor()
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    dataset_class = datasets.MNIST if dataset_name == 'mnist' else datasets.CIFAR10
    full_data = dataset_class(
        root='data',
        train=True,
        download=True,
        transform=transform
    )

    if dataset_name == 'mnist':
        indices = torch.where(full_data.targets == class_label)[0]
    else:
        targets = torch.tensor(full_data.targets)
        indices = torch.where(targets == class_label)[0]

    class_data = Subset(full_data, indices)

    train_size = int(TRAIN_PERCENT * len(class_data))
    test_size = len(class_data) - train_size
    train_data, test_data = torch.utils.data.random_split(
        class_data, [train_size, test_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    test_loader = DataLoader(test_data, batch_size=min(
        SAMPLES_PER_RUN, len(test_data)))
    test_batch = next(iter(test_loader))[0]

    return {
        'train': train_data,
        'test': test_batch
    }


print("Loading datasets...")
datasets = {
    'mnist': {
        'full': load_full_dataset('mnist'),
        'class': load_class_dataset('mnist', MNIST_CLASS_LABEL)
    },
    'cifar10': {
        'full': load_full_dataset('cifar10'),
        'class': load_class_dataset('cifar10', CIFAR_CLASS_LABEL)
    }
}

configurations = [
    {'name': 've', 'diffusion': 've', 'schedule': None, 'full': True},
    {'name': 'vp-lin', 'diffusion': 'vp', 'schedule': 'lin', 'full': True},
    {'name': 'vp-cos', 'diffusion': 'vp', 'schedule': 'cos', 'full': True},
    {'name': 'svp-lin', 'diffusion': 'svp', 'schedule': 'lin', 'full': True},
    {'name': 'svp-cos', 'diffusion': 'svp', 'schedule': 'cos', 'full': True},
    {'name': 've', 'diffusion': 've', 'schedule': None, 'full': False},
    {'name': 'vp-lin', 'diffusion': 'vp', 'schedule': 'lin', 'full': False},
    {'name': 'vp-cos', 'diffusion': 'vp', 'schedule': 'cos', 'full': False},
    {'name': 'svp-lin', 'diffusion': 'svp', 'schedule': 'lin', 'full': False},
    {'name': 'svp-cos', 'diffusion': 'svp', 'schedule': 'cos', 'full': False},
]

samplers = [
    'euler-maruyama',
    'exponential',
    'ode',
    'predictor-corrector'
]

methods = {
    'generate': ['mnist', 'cifar10'],
    'colorize': ['cifar10'],
    'imputation': ['mnist', 'cifar10']
}

dummy_model = GenerativeModel()
metrics = {
    'bpd': BitsPerDimension(dummy_model),
    'fid': FrechetInceptionDistance(dummy_model),
    'is': InceptionScore(dummy_model)
}


def create_single_channel_mask(batch: torch.Tensor) -> torch.Tensor:
    _, _, h, w = batch.shape
    mask = torch.zeros(batch.shape[0], 1, h, w, device=batch.device)
    mask[:, :, h//4:3*h//4, w//4:3*w//4] = 1
    return mask


def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_model(model, dataset, dataset_type, method, conditional, sampler_name, writer, class_label=None):
    test_batch = datasets[dataset][dataset_type]['test'].to(model.device)

    if method == 'colorize':
        input_data = test_batch.mean(dim=1, keepdim=True)
    elif method == 'imputation':
        mask = create_single_channel_mask(test_batch)
        input_data = test_batch * (1 - mask)

    if conditional and class_label is not None:
        class_labels = class_label
    else:
        class_labels = None

    start_time = time.time()
    try:
        if method == 'generate':
            generated = model.generate(
                num_samples=SAMPLES_PER_RUN,
                seed=SEED,
                class_labels=class_labels
            )
        elif method == 'colorize':
            generated = model.colorize(
                input_data,
                seed=SEED,
                class_labels=class_labels
            )
        elif method == 'imputation':
            generated = model.imputation(
                input_data,
                mask,
                seed=SEED,
                class_labels=class_labels
            )

        elapsed = time.time() - start_time
        print(f"Generation completed in {elapsed:.2f}s")

        clear_gpu_memory()

        scores = {}
        for metric_name, metric in metrics.items():
            try:
                print(f"Calculating {metric_name}...", end=' ')
                metric.model = model
                score = metric(test_batch, generated)
                scores[metric_name] = score
                print(score)
            except Exception as e:
                print(f"Error calculating {metric_name}: {str(e)}")
                scores[metric_name] = float('nan')

            clear_gpu_memory()

        return {
            'Time (seconds)': round(elapsed, 2),
            'BPD': round(scores.get('bpd', float('nan')), 5),
            'FID': round(scores.get('fid', float('nan')), 5),
            'Inception': round(scores.get('is', float('nan')), 5)
        }

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        return None
    finally:
        clear_gpu_memory()


with open('results.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=[
        'Configuration', 'Sampler', 'Dataset', 'Full Dataset', 'Method', 'Conditional',
        'Time (seconds)', 'BPD', 'FID', 'Inception',
    ])
    writer.writeheader()

    total_configs = sum(
        len([m for m in methods if dataset in methods[m]]) *
        (2 if config['full'] else 1) * len(samplers)
        for config in configurations
        for dataset in ['mnist', 'cifar10']
    )
    current_config = 0

    for config in configurations:
        config_name = config['name']
        is_full_dataset = config['full']
        dataset_type = 'full' if is_full_dataset else 'class'

        for dataset in ['mnist', 'cifar10']:
            available_methods = [m for m in methods if dataset in methods[m]]

            class_label = MNIST_CLASS_LABEL if dataset == 'mnist' else CIFAR_CLASS_LABEL

            if is_full_dataset:
                model_filename = f"saved_models/{dataset}_train_{config_name}_{NUM_EPOCHS}e.pth"
            else:
                model_filename = f"saved_models/{dataset}_{class_label}_train_{config_name}_{NUM_EPOCHS}e.pth"

            if os.path.exists(model_filename):
                print(f"\nLoading model from {model_filename}...")
                model = GenerativeModel()
                model.load(model_filename)
            else:
                print(f"\nTraining new model {config_name} for {dataset}" +
                      (f" class {class_label}" if not is_full_dataset else "") + "...")

                model = GenerativeModel(
                    diffusion=config['diffusion'],
                    noise_schedule=config['schedule']
                )

                model.train(
                    dataset=datasets[dataset][dataset_type]['train'],
                    epochs=NUM_EPOCHS,
                    batch_size=BATCH_SIZE
                )
                model.save(model_filename)

            conditional_options = [True, False] if is_full_dataset else [False]

            for method in available_methods:
                for conditional in conditional_options:
                    for sampler_name in samplers:
                        current_config += 1
                        print(f"\n[{current_config}/{total_configs}] "
                              f"Evaluating {config_name} on {dataset}" +
                              (f" class {class_label}" if not is_full_dataset else "") +
                              f" with {method} (conditional={conditional}) using {sampler_name}")

                        model.sampler = sampler_name

                        results = evaluate_model(
                            model, dataset, dataset_type, method, conditional,
                            sampler_name, writer, class_label if conditional else None
                        )

                        if results:
                            row = {
                                'Configuration': config_name,
                                'Sampler': sampler_name,
                                'Dataset': dataset,
                                'Full Dataset': 'yes' if is_full_dataset else 'no',
                                'Method': method,
                                'Conditional': 'yes' if conditional else 'no',
                                **results
                            }
                            writer.writerow(row)
                            csvfile.flush()

                        torch.cuda.empty_cache()

            model = None
            clear_gpu_memory()

print("\nEvaluation complete. Results saved to results.csv")

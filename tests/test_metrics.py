import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..')))
from image_gen.base import GenerativeModel
import torch
from image_gen.metrics import InceptionScore, BitsPerDimension, FrechetInceptionDistance
from torch.utils.data import Subset
from torchvision import datasets
from torchvision.transforms import ToTensor
from image_gen.diffusion import VarianceExploding
from image_gen.samplers import EulerMaruyama


def evaluate_generative_model(model, test_dataset, num_samples=1000, batch_size=32):
    """
    Evalúa un modelo generativo usando múltiples métricas
    
    Args:
        model: El modelo generativo (GenerativeModel)
        test_dataset: Dataset de imágenes reales para comparación
        num_samples: Número de muestras a generar
        batch_size: Tamaño del batch para evaluación
        
    Returns:
        Dict con los resultados de las métricas
    """
    # Obtener imágenes reales del dataset
    dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    real_images = []
    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            # Si el dataset devuelve (imagen, etiqueta), tomar solo la imagen
            x = batch[0]
        else:
            x = batch
        real_images.append(x)
        if len(torch.cat(real_images)) >= num_samples:
            break
    
    real_images = torch.cat(real_images)[:num_samples]
    
    # Generar imágenes con el modelo
    generated_images = []
    for i in range(0, num_samples, batch_size):
        samples = model.generate(
            num_samples=min(batch_size, num_samples - i),
            n_steps=200  # Ajustar según la configuración de tu modelo
        )
        generated_images.append(samples)
    
    generated_images = torch.cat(generated_images)
    
    # Inicializar métricas
    # Para BPD, necesitamos pasar el modelo de difusión
    bpd_metric = BitsPerDimension(diffusion_model=model)
    fid_metric = FrechetInceptionDistance()
    is_metric = InceptionScore()
    
    # Calcular métricas
    results = {}
    
    # Para BPD, sólo necesitamos imágenes reales
    try:
        bpd = bpd_metric(real_images)
        results["BPD"] = bpd
        print(f"BPD: {bpd:.4f} (menor es mejor)")
    except Exception as e:
        print(f"Error al calcular BPD: {e}")
    
    # Para FID, necesitamos tanto imágenes reales como generadas
    try:
        fid = fid_metric(real_images, generated_images)
        results["FID"] = fid
        print(f"FID: {fid:.4f} (menor es mejor)")
    except Exception as e:
        print(f"Error al calcular FID: {e}")
    
    # Para IS, sólo necesitamos imágenes generadas
    try:
        is_mean, is_std = is_metric.calculate_with_std(generated_images)
        results["IS"] = is_mean
        results["IS_std"] = is_std
        print(f"IS: {is_mean:.4f} ± {is_std:.4f} (mayor es mejor)")
    except Exception as e:
        print(f"Error al calcular IS: {e}")
    
    return results

model = GenerativeModel(
    diffusion=VarianceExploding,
    sampler=EulerMaruyama
)  # Reemplaza con tu modelo generativo

# Load the dataset
data = datasets.MNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor()
)

# Select a subset to speed up the training process
digit = 3
epochs = 25
indices_digit = torch.where(data.targets == digit)[0]
data = Subset(data, indices_digit)

filename = f'../examples/saved_models/mnist_{digit}_ve_euler_{epochs}e.pth'

if os.path.isfile(filename):
    model.load(filename)
else:
    model.train(data, epochs=epochs)
    # Tip: Save the models for them to be accessible through the dashboard
    model.save(filename)

print(evaluate_generative_model(model, data, num_samples=16, batch_size=32))
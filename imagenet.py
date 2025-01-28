import numpy as np
import re
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from scipy.spatial.distance import cosine
import os
import argparse
from models import ResNetN
import keras
from utils import custom_functions as func
from utils import custom_callbacks as cb
import math

# Variáveis globais para armazenar pesos iniciais e distâncias cosseno
initial_weights = None
cosine_distances_mean = []
cosine_distances_concat = []

def save_initial_weights(model, layer_names):
    global initial_weights
    initial_weights = [model.get_layer(name).get_weights()[0] for name in layer_names]

def calculate_cosine_distance(weights1, weights2):
    distances = []
    # for w1, w2 in zip(weights1, weights2):
    #     flat_w1 = np.concatenate([w.flatten() for w in w1])
    #     flat_w2 = np.concatenate([w.flatten() for w in w2])
    #     distance = cosine(flat_w1, flat_w2)
    distance = np.array([cosine(w1.flatten(), w2.flatten()) for w1, w2 in zip(weights1, weights2)])
    distances.append(distance)
    
    return distances

def check_rotation_angle_criterion(model, layer_names, window_size=5, epoch=6, epochs=200):
    global initial_weights, cosine_distances_mean, cosine_distances_concat
        
    # w_0_concat = np.concatenate([w.reshape(-1) for w in initial_weights])

    # Obtém os pesos atuais do modelo
    current_weights = [model.get_layer(name).get_weights()[0] for name in layer_names]
    # w_i_concat = np.concatenate([w.reshape(-1) for w in current_weights])

    # Calcula a distância cosseno entre os pesos atuais e os pesos iniciais (mean)
    cosine_distance_mean = np.mean(calculate_cosine_distance(initial_weights, current_weights))
    cosine_distances_mean.append(cosine_distance_mean)

    print(f"\nMean cosine distance: {cosine_distance_mean:.4f}", flush=True)

    # Calcula a distância cosseno entre os pesos atuais e os pesos iniciais (concatenados)
    # cosine_distance_concat = calculate_cosine_distance(w_0_concat, w_i_concat)
    # cosine_distances_concat.append(cosine_distance_concat)

    # Verifica se temos pelo menos 'window_size' distâncias para calcular a média
    if len(cosine_distances_mean) >= window_size:
        # Pega os últimos 'window_size' pontos
        recent_distances = cosine_distances_mean[-window_size:]
        # epochs = np.arange(len(cosine_distances_mean) - window_size + 1, len(cosine_distances_mean) + 1)

        # Supondo que epocas_janela e valores_janela são suas variáveis x e y
        recent_distances = np.array(recent_distances).reshape(-1, 1)
        epochs_ndarray = np.arange(1, epochs+1).reshape(-1, 1)  

        # Criando o normalizador
        scaler = MinMaxScaler(feature_range=(0, 1))

        # Normalizando os dados
        recent_distances_norm = recent_distances
        epochs_norm = scaler.fit_transform(epochs_ndarray)

        recent_distances_norm_1d = recent_distances_norm.flatten()
        epochs_norm_1d = epochs_norm.flatten()

        # Selecionar a janela atual de 5 pontos  
        epochs_window = epochs_norm_1d[epoch-window_size:epoch]  

        print(epochs_window)
        print(recent_distances_norm_1d)
        
        # Ajusta uma linha de regressão linear aos pontos
        slope, intercept = np.polyfit(epochs_window, recent_distances_norm_1d, 1)

        # Converte a inclinação para um ângulo em graus
        angle = np.degrees(np.arctan(slope))
        print(f"\nAngle: {angle:.2f}°", flush=True)
        
        # Verifica se o ângulo é menor que 45°
        if angle < 45:
            return True

    return False

def get_kernel_layer_names(model):
    layer_names = []
    for l in model.layers:
        if len(l.weights) > 0:
            if 'kernel' in l.weights[0].name:
                layer_names.append(l.name)
    return layer_names

if __name__ == '__main__':
    seed = 12227
    func.set_seeds(seed)  # Set seeds for repeatability

    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', type=str, default='ResNet56')
    parser.add_argument('--dataset', type=str, default='imagenet5')
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--verbose', type=int, default=2, help='Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch')
    parser.add_argument('--baseline', type=str, default='False', help='True = All epochs k = 3, False = Criterion will be implemented')

    args = parser.parse_args()
    architecture = args.architecture
    dataset_name = args.dataset
    weights_dir = args.weights if args.weights != '' else f'./weights/{dataset_name}/{architecture}'
    model_name = args.model_name
    verbose = args.verbose
    criterion_met = False if args.baseline == 'False' else True

    model_name = architecture.split('/')[-1] if model_name == '' else model_name
    print(f"{model_name} {dataset_name}", flush=True)

    # Create directory for saving weights if it doesn't exist
    os.makedirs(weights_dir, exist_ok=True)

    match = re.search(r'(\d+)', dataset_name)
    if match:
        num_classes = int(match.group(1))
    else:
        raise ValueError("Unsupported dataset. Dataset name should contain the number of classes, e.g., CIFAR10 or ImageNet30.")

    # Load the dataset
    data = np.load(f'/home/vm03/Datasets/imagenet{num_classes}_cls.npz')
    X_train, y_train, X_test, y_test = data['X_train'], data['y_train'], data['X_val'], data['y_val']
    
    # Convert y_train from int to one-hot encoding
    y_train = np.eye(num_classes)[y_train.reshape(-1)]
    y_test = np.eye(num_classes)[y_test.reshape(-1)]

    # Determinar num_classes baseado no dataset
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")
    num_classes = y_train.shape[1]

    # Determinar N_layers baseado na arquitetura
    match = re.search(r'ResNet(\d+)', architecture)
    if match:
        N_layers = int(match.group(1))
        print(f"\nNumber of Layers: {N_layers}", flush=True)
    else:
        raise ValueError(
            "Arquitetura não suportada. Suporta apenas formatos como ResNetXX, onde XX é o número de camadas.")

    # Load a model
    model = ResNetN.build_model(model_name, input_shape=(224, 224, 3), num_classes=num_classes, N_layers=N_layers)

    # Path to random starting weights
    random_weights_path = os.path.join(weights_dir, f'@random_starting_weights_{model_name}_.weights.h5')
    
    # Load or create the random starting weights using the seed
    func.load_or_create_weights(model, random_weights_path)

    # Configure the optimizer
    lr = 0.01
    sgd = keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    # Configure Data Augmentation
    datagen = func.generate_data_augmentation(X_train)

    # Set up learning rate scheduler
    lr_scheduler_callback = LearningRateScheduler(func.scheduler)
    
    callbacks = [lr_scheduler_callback]     # , checkpoint_callback

    # Set manual epoch loop configuration
    epochs = 200
    batch_size = 32

    # Repeat the data k times, datagen will transform
    k = 1
    y_aug = np.tile(y_train, (k, 1))
    X_aug = np.tile(X_train, (k, 1, 1, 1))
    
    print(f"{X_train.shape[0]} samples per epoch, grouped into batches of {batch_size}.", flush=True)

    # Get kernel layer names
    layer_names = get_kernel_layer_names(model)

    # Salva os pesos da inicialização se ainda não foram salvos
    if initial_weights is None:
        save_initial_weights(model, layer_names)

    epochs_annealing = []
    # Epoch loop
    for epoch in range(1, epochs+1):
        print(f"\nEpoch {epoch}/{epochs}", flush=True)
        model.fit(
            datagen.flow(X_aug, y_aug, batch_size=batch_size, seed=seed, shuffle=True),
            epochs=epoch, initial_epoch=epoch - 1,
            verbose=verbose, callbacks=callbacks,
            validation_data = (X_test, y_test),
            validation_freq=5
        )
        
        if epoch in epochs_annealing:
             # Salva os pesos do modelo nas epocas que serão aplicados o annealing
            weights_path = os.path.join(weights_dir, f'{model_name}_{dataset_name}_epoch_{epoch:02d}.weights.h5')
            print(f"\nSaving model weights. Epoch {epoch}.")
            model.save_weights(weights_path)

        # Check if the criterion is met after the current epoch
        if not criterion_met:
            if check_rotation_angle_criterion(model, layer_names, window_size = 5, epoch=epoch, epochs=epochs):
                print(f"Criterion met at epoch {epoch}. Adjusting k value.")
                k = 1
                # Recalcula os dados aumentados com base no novo valor de k
                y_aug = np.tile(y_train, (k, 1))
                X_aug = np.tile(X_train, (k, 1, 1, 1))
                # Limpa as distâncias cosseno para a próxima janela
                cosine_distances_mean = []
                # cosine_distances_concat = []

                # Atualiza a variável de controle para que este bloco não seja mais acessado
                criterion_met = True

                # Salva os pesos do modelo ao atingir o critério
                weights_path = os.path.join(weights_dir, f'{model_name}_{dataset_name}_epoch_{epoch:02d}.weights.h5')
                print(f"\nSaving model weights. Epoch {epoch}.")
                model.save_weights(weights_path)

                epochs_diff = epochs - epoch
                annealing_ratio = [0.99, 0.95, 0.90, 0.85, 0.80, 0.50, 0.875] 

                for ratio in annealing_ratio:  
                    annealing_epoch = epoch + math.ceil(epochs_diff * ratio)
                    epochs_annealing.append(annealing_epoch) 
                    print(f"\nAnnealing ratio: {ratio}. Annealing epoch: {annealing_epoch} ")
        else:
            # Get dinamic coreset (10%)
            X_coreset, y_coreset = func.subsampling(X_train, y_train, p=0.1, random_state=None)
            datagen = func.generate_data_augmentation(X_coreset)

            # Data augmentation with k=1
            k = 1
            y_aug = np.tile(y_coreset, (k, 1))
            X_aug = np.tile(X_coreset, (k, 1, 1, 1))

    # Evaluate model after training
    y_pred = model.predict(X_test, verbose=0)
    accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    print(f'Final accuracy: {accuracy:.4f}')

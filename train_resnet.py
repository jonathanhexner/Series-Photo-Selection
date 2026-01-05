import os
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from sklearn import metrics
from data.dataloader import make_loader
from models.ResNet50 import make_network
from utils.lr_scheduler import LR_Scheduler
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Windows
import matplotlib.pyplot as plt
import random
import sys
sys.path.append(r'D:\sim-bench')
from sim_bench.utils.model_inspection import inspect_model_output
from sim_bench.utils import batch_logger


def reset_rng_seeds(seed):
    """Reset all random number generator seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def plot_training_curves(history, output_dir):
    """Plot and save training/validation curves."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150)
    print(f"Training curves saved to {output_dir / 'training_curves.png'}")
    plt.close()

def loader_indices_to_df(loader, K_batches=10, N_images=None, tag="A"):
    rows = []
    it = iter(loader)

    for batch_id in range(K_batches):
        batch = next(it)
        idxs = batch[0]  # because we return (idx, x)

        if N_images is not None:
            idxs = idxs[:N_images]

        for pos_in_batch, idx in enumerate(idxs.tolist()):
            rows.append({
                "loader": tag,
                "batch_id": batch_id,
                "pos_in_batch": pos_in_batch,
                "global_pos": batch_id * len(batch[0]) + pos_in_batch,
                "image_idx": idx,
            })

    return pd.DataFrame(rows)


def main():
    # Set random seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)  # For Python's random module used in make_shuffle_path.py
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Create timestamped output directory
    output_dir = Path(f"outputs/siamese_e2e/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Batch logger output path
    batch_log_path = output_dir / "telemetry" / 'batch_predictions.csv'

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=str(output_dir / 'tensorboard'))

    best_pred = 0.0
    best_acc = 0.0
    best_macro = 0.0
    best_micro = 0.0
    lr = 0.00001
    num_epochs = 2
    batch_size = 8  # Reduced for CPU training

    # Telemetry configuration
    telemetry_config = {
        'enabled': True,
        'collect_every_n': 10,  # Collect metrics every 200 batches

        # Enable/disable individual metrics
        'track_gradients': True,
        'track_weight_delta': True,
        'track_learning_rates': True,
        'track_holdout_logits': True,
        'track_batch_stats': True,

        # Holdout settings
        'holdout_size': 50  # Number of validation pairs to track
    }

    # Create inspection loaders FIRST (before model init affects RNG)
    _, _, inspect_trainloader, inspect_valloader = make_loader(batch_size=batch_size, seed=seed)
    inspect_train_batch = next(iter(inspect_trainloader))
    inspect_val_batch = next(iter(inspect_valloader))

    # Print first batch
    print("\nFirst training batch:")
    df_train = pd.DataFrame({
        'image1': inspect_train_batch['image1'],
        'image2': inspect_train_batch['image2']
    })
    print(df_train)

    print("\nFirst validation batch:")
    df_val = pd.DataFrame({
        'image1': inspect_val_batch['image1'],
        'image2': inspect_val_batch['image2']
    })
    print(df_val)

    # Reset RNG seeds before creating training loaders (ensures same shuffle order)
    reset_rng_seeds(seed)

    # Create training loaders (for actual training)
    train_data, val_data, trainloader, valloader = make_loader(batch_size=batch_size, seed=seed)

    model = make_network()

    # Dump model state after creation
    from sim_bench.training.model_comparison import dump_model_to_csv
    dump_model_to_csv(model, output_dir / 'model_state_after_creation.csv')

    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")  # Force CPU usage
    print(f"Using device: {device}")
    model.to(device)
    criterion.to(device)
    train_params = [{'params': model.get_1x_lr_params(), 'lr': lr},
                    {'params': model.get_10x_lr_params(), 'lr': lr * 10}]
    optimizer = optim.SGD(train_params, momentum=0.9, weight_decay=5e-4, nesterov=False)
    scheduler = LR_Scheduler(mode='step', base_lr=lr, num_epochs=num_epochs, iters_per_epoch=len(trainloader), lr_step=25)

    # Initialize training history for plotting
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    # Reset seeds again before inspection to ensure same shuffle order    
    reset_rng_seeds(seed)

    # Dump model state before inspection
    from sim_bench.training.model_comparison import dump_model_to_csv
    dump_model_to_csv(model, output_dir / 'model_state_before_inspection.csv')

    # Inspect model output before training
    print("\n=== Model Output Inspection (Before Training) ===")
    df_inspect_train = inspect_model_output(
        model=model,
        loader=trainloader,
        device=device,
        save_path=output_dir / 'model_inspection_train.csv'
    )
    df_inspect_val = inspect_model_output(
        model=model,
        loader=valloader,
        device=device,
        save_path=output_dir / 'model_inspection_val.csv'
    )
    print("Inspection complete. Starting training...\n")

    # Initialize telemetry system
    from sim_bench import telemetry
    config_with_telemetry = {
        'output_dir': output_dir,
        'telemetry': telemetry_config
    }
    telemetry.init(config_with_telemetry, valloader)
    print("Telemetry initialized\n")
    reset_rng_seeds(seed)
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        acc = 0.0
        micro = 0.0
        macro = 0.0
        count = 0
        model.train()
        for batch_idx, batch in enumerate(trainloader):
            dataA = batch['img1'].to(device)
            dataB = batch['img2'].to(device)
            target = batch['winner'].to(device)
            scheduler(optimizer, batch_idx, epoch, best_pred)
            optimizer.zero_grad()
            pred = model(dataA, dataB)
            batch_logger.log_batch_predictions(batch_log_path, batch_idx, epoch, batch['image1'], batch['image2'], target, pred)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            # Record telemetry after optimizer.step()
            from sim_bench import telemetry
            telemetry.record(model, optimizer, batch_idx + 1, epoch, device, batch)

            predict = torch.argmax(pred, 1)

            # Log batch progression every 100 batches
            if (batch_idx + 1) % 100 == 0:
                batch_acc = torch.eq(predict, target).sum().double().item() / target.size(0) * 100
                print(f'  Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(trainloader)}], '
                      f'Loss: {loss.item():.4f}, Batch Acc: {batch_acc:.2f}%')
            a = metrics.accuracy_score(target.cpu(), predict.cpu())
            b = metrics.f1_score(target.cpu(), predict.cpu(), average='micro')
            c = metrics.f1_score(target.cpu(), predict.cpu(), average='macro')
            acc += a
            micro += b
            macro += c
            count += 1
            correct = torch.eq(predict, target).sum().double().item()
            running_loss += loss.item()
            running_correct += correct
            running_total += target.size(0)
        loss = running_loss * batch_size / running_total
        accuracy = 100 * running_correct / running_total
        acc /= count
        micro /= count
        macro /= count
        writer.add_scalar('scalar/loss_train', loss, epoch)
        writer.add_scalar('scalar/accuracy_train', accuracy, epoch)
        writer.add_scalar('scalar/acc_train', acc, epoch)
        writer.add_scalar('scalar/micro_train', micro, epoch)
        writer.add_scalar('scalar/macro_train', macro, epoch)
        history['train_loss'].append(loss)
        history['train_acc'].append(accuracy)
        print('Training ',
              'Epoch[%d /50],loss = %.6f,accuracy=%.4f %%, acc = %.4f, micro = %.4f, macro = %.4f' %
              (epoch + 1, loss, accuracy, acc, micro, macro))
        model.eval()
        with torch.no_grad():
            running_loss = 0.0
            running_correct = 0
            running_total = 0
            acc = 0.0
            micro = 0.0
            macro = 0.0
            count = 0
            for batch_idx, batch in enumerate(valloader):
                dataA = batch['img1'].to(device)
                dataB = batch['img2'].to(device)
                target = batch['winner'].to(device)
                pred = model(dataA, dataB)
                loss = criterion(pred, target)
                predict = torch.argmax(pred, 1)

                # Log batch progression every 100 batches
                if (batch_idx + 1) % 100 == 0:
                    batch_acc = torch.eq(predict, target).sum().double().item() / target.size(0) * 100
                    print(f'  Validation Batch [{batch_idx+1}/{len(valloader)}], '
                          f'Loss: {loss.item():.4f}, Batch Acc: {batch_acc:.2f}%')

                a = metrics.accuracy_score(target.cpu(), predict.cpu())
                b = metrics.f1_score(target.cpu(), predict.cpu(), average='micro')
                c = metrics.f1_score(target.cpu(), predict.cpu(), average='macro')
                correct = torch.eq(predict, target).sum().double().item()
                running_loss += loss.item()
                running_correct += correct
                running_total += target.size(0)
                acc += a
                micro += b
                macro += c
                count += 1
            loss = running_loss * batch_size / running_total
            accuracy = 100 * running_correct / running_total
            acc /= count
            micro /= count
            macro /= count
            history['val_loss'].append(loss)
            history['val_acc'].append(accuracy)
            if acc > best_acc:
                best_acc = acc
            if micro > best_micro:
                best_micro = micro
            if macro > best_macro:
                best_macro = macro
            if accuracy > best_pred:
                best_pred = accuracy
            print('best results: ', 'best_acc = %.4f, best_micro = %.4f, best_macro = %.4f, best_pred = %.4f' %
                  (best_acc, best_micro, best_macro, best_pred,))
            writer.add_scalar('scalar/loss_val', loss, epoch)
            writer.add_scalar('scalar/accuracy_val', accuracy, epoch)
            writer.add_scalar('scalar/acc_val', acc, epoch)
            writer.add_scalar('scalar/micro_val', micro, epoch)
            writer.add_scalar('scalar/macro_val', macro, epoch)
            print('Valing',
                  '    Epoch[%d /50],loss = %.6f,accuracy=%.4f %%, acc = %.4f, micro = %.4f, macro = %.4f, running_total=%d,running_correct=%d' %
                  (epoch + 1, loss, accuracy, acc, micro, macro, running_total, running_correct))

    # Plot and save training curves
    plot_training_curves(history, output_dir)

    # Close TensorBoard writer
    writer.close()


if __name__ == "__main__":
    main()
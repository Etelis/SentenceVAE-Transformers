import os
import matplotlib.pyplot as plt
import torch

def plot_metrics(data, labels, title, ylabel, filename, directory='plots'):
    """
    Plots training and validation metrics.

    Args:
        data (List[List[float]]): A list of lists containing metric data for each plot.
        labels (List[str]): Labels for each plot line.
        title (str): The title of the plot.
        ylabel (str): The label for the y-axis.
        filename (str): The filename to save the plot as.
        directory (str): The directory to save the plot in.
    """
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Plotting
    plt.figure(figsize=(10, 5))
    for metric, label in zip(data, labels):
        plt.plot(metric, label=label)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(directory, filename))
    plt.close()

def save_model(model, path, filename):
    """
    Saves the model's state dictionary to a file.

    Args:
        model (torch.nn.Module): The model to save.
        path (str): The directory to save the model in.
        filename (str): The filename to use for saving the model.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path, filename))
    print(f"Model saved to {os.path.join(path, filename)}")

def load_model(model, path):
    """
    Loads a model's state dictionary from a file.

    Args:
        model (torch.nn.Module): The model to load state into.
        path (str): The path to the model file to load.
    """
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}")
    return model

def create_summary_writer(logdir):
    """
    Creates a TensorBoard SummaryWriter.

    Args:
        logdir (str): The directory to store TensorBoard logs.

    Returns:
        SummaryWriter: A TensorBoard SummaryWriter object.
    """
    from torch.utils.tensorboard import SummaryWriter
    return SummaryWriter(logdir)

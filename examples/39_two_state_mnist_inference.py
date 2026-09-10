# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""aihwkit example 39: MNIST inference on bit-sliced two-state ReRAM cells.

A small MLP is trained in FP32 with plain PyTorch and then converted with
``convert_to_analog`` into IR-drop tiles (``TorchInferenceRPUConfigIRDropT``)
whose unit cell is described by ``BinaryDeviceConductanceConverter``: every
weight is a signed-magnitude integer, its magnitude bits are stored one per
binary (HRS/LRS) cell in a positive and a negative array, and the periphery
sums the cells with their significance ``f_k``. Every layer is split into
``CROSSBAR_SIZE x CROSSBAR_SIZE`` arrays.

The test accuracy is reported for

* ideal cells: exact HRS/LRS conductances and negligible wire resistance,
  i.e. the digital INT reference of the same quantized network,
* + programming spread: Gaussian deviation of every programmed cell,
* + IR drop: wire resistance between neighbouring cells along each column.

The converter carries the programming spread; the noise model only hands the
converter to the tile (``program_analog_weights`` is never called).
"""
# pylint: disable=invalid-name

import os

# Imports from PyTorch.
import torch
from torch import nn
from torch.nn.functional import pad
from torchvision import datasets, transforms

# Imports from aihwkit.
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.inference import BinaryDeviceConductanceConverter, PCMLikeNoiseModel
from aihwkit.simulator.configs import TorchInferenceRPUConfigIRDropT
from aihwkit.simulator.configs.utils import (
    BoundManagementType,
    WeightClipType,
    NoiseManagementType,
    WeightRemapType,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path where the dataset and the FP32 checkpoint are stored.
PATH_DATASET = os.path.join("data", "DATASET")
PATH_CHECKPOINT = os.path.join("data", "two_state_mnist_fp32.pt")

# Crossbar array size (rows = columns). The 784 MNIST inputs are zero-padded
# to a multiple of the rows so that every row block is a full array (the
# IR-drop tile requires equally sized row blocks).
CROSSBAR_SIZE = 32
INPUT_SIZE = -(-784 // CROSSBAR_SIZE) * CROSSBAR_SIZE
HIDDEN_SIZES = [256, 128]
OUTPUT_SIZE = 10

# Unit cell: number of magnitude cells per array (INT4 -> 4 cells, 31 levels),
# conductances (uS) and relative programming spread. Replace with measured values.
N_BITS = 4
G_LRS, G_HRS = 20.0, 0.2
PROGRAMMING_SPREAD = 0.02
# Wire resistance between neighbouring cells in Ohms.
WIRE_RESISTANCE = 0.35
# Input DAC resolution (bits); the ADC is ideal.
INPUT_BITS = 6

# FP32 training.
EPOCHS = 3
BATCH_SIZE = 128
# Independent programming draws per stage (mean +- std is reported).
REPEATS = 1


def load_images():
    """Load the MNIST images, flattened and zero-padded to ``INPUT_SIZE``."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: pad(x.view(-1), (0, INPUT_SIZE - 784)))]
    )
    train_set = datasets.MNIST(PATH_DATASET, download=True, train=True, transform=transform)
    test_set = datasets.MNIST(PATH_DATASET, download=True, train=False, transform=transform)
    train_data = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_data = torch.utils.data.DataLoader(test_set, batch_size=500, shuffle=False)
    return train_data, test_data


def create_digital_network():
    """Create the FP32 MLP."""
    return nn.Sequential(
        nn.Linear(INPUT_SIZE, HIDDEN_SIZES[0]),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZES[0], HIDDEN_SIZES[1]),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZES[1], OUTPUT_SIZE),
    )


@torch.no_grad()
def test_accuracy(model, test_data):
    """Return the test accuracy in percent."""
    model.eval()
    correct, total = 0, 0
    for images, labels in test_data:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        correct += (model(images).argmax(1) == labels).sum().item()
        total += labels.numel()
    return 100.0 * correct / total


def train_digital_network(model, train_data, test_data):
    """Train the FP32 network (or load a previously trained checkpoint)."""
    if os.path.exists(PATH_CHECKPOINT):
        model.load_state_dict(torch.load(PATH_CHECKPOINT, map_location=DEVICE))
        print("Loaded FP32 checkpoint", PATH_CHECKPOINT)
        return
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(EPOCHS):
        model.train()
        for images, labels in train_data:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            nn.functional.cross_entropy(model(images), labels).backward()
            optimizer.step()
        accuracy = test_accuracy(model, test_data)
        print("FP32 epoch {}: test accuracy {:.2f} %".format(epoch + 1, accuracy))
    os.makedirs(os.path.dirname(PATH_CHECKPOINT), exist_ok=True)
    torch.save(model.state_dict(), PATH_CHECKPOINT)


def create_rpu_config(g_converter, wire_resistance):
    """IR-drop inference tile with the two-state unit cell."""
    rpu_config = TorchInferenceRPUConfigIRDropT()
    rpu_config.modifier.type = None
    # Weights: per-column scaling so that every column uses the full +-L range;
    # layers are split into CROSSBAR_SIZE x CROSSBAR_SIZE arrays.
    rpu_config.mapping.digital_bias = True
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.max_input_size = CROSSBAR_SIZE
    rpu_config.mapping.max_output_size = CROSSBAR_SIZE
    rpu_config.remap.type = WeightRemapType.NONE
    rpu_config.clip.type = WeightClipType.NONE
    # Inputs: INPUT_BITS DAC, every input vector scaled to its abs-max.
    rpu_config.forward.is_perfect = False
    rpu_config.forward.inp_bound = 1.0
    rpu_config.forward.inp_res = 1.0 / (2**INPUT_BITS - 2)
    rpu_config.forward.noise_management = NoiseManagementType.ABS_MAX
    rpu_config.forward.bound_management = BoundManagementType.NONE
    # Outputs: ideal ADC, no output noise.
    rpu_config.forward.out_res = -1.0
    rpu_config.forward.out_bound = 0.0
    rpu_config.forward.out_noise = 0.0
    # IR drop with exact read voltage, exact PWM integration and no CCO-ADC floor.
    rpu_config.forward.ir_drop = 1.0
    rpu_config.forward.ir_drop_rs = wire_resistance
    rpu_config.forward.ir_drop_v_read = 0.4
    rpu_config.forward.ir_drop_integration_sum = True
    rpu_config.forward.adc_quantization = False
    # The tile takes the unit cell from noise_model.g_converter. The PCM noise of
    # the carrier never applies because program_analog_weights is not called.
    rpu_config.noise_model = PCMLikeNoiseModel(g_converter=g_converter)
    rpu_config.drift_compensation = None
    return rpu_config


def analog_accuracy(model, test_data, spread, wire_resistance, seed):
    """Convert the FP32 model to analog and return its test accuracy."""
    torch.manual_seed(seed)  # programming draw
    g_converter = BinaryDeviceConductanceConverter(
        n_bits=N_BITS, g_lrs=G_LRS, g_hrs=G_HRS, g_lrs_std=spread * G_LRS, g_hrs_std=spread * G_HRS
    )
    analog_model = convert_to_analog(model, create_rpu_config(g_converter, wire_resistance))
    return test_accuracy(analog_model.to(DEVICE).eval(), test_data)


def main():
    """Train the FP32 network and evaluate it on two-state ReRAM arrays."""
    torch.manual_seed(0)
    train_data, test_data = load_images()
    model = create_digital_network().to(DEVICE)
    train_digital_network(model, train_data, test_data)

    n_levels = BinaryDeviceConductanceConverter(n_bits=N_BITS).n_levels
    print(
        "\nMNIST MLP {}-{}-{}-{} | arrays of {n} x {n} | sign-magnitude, {} magnitude cells per "
        "array ({} levels) | LRS {:g} uS, HRS {:g} uS | {}-bit inputs | ideal ADC | {}".format(
            INPUT_SIZE,
            HIDDEN_SIZES[0],
            HIDDEN_SIZES[1],
            OUTPUT_SIZE,
            N_BITS,
            n_levels,
            G_LRS,
            G_HRS,
            INPUT_BITS,
            DEVICE,
            n=CROSSBAR_SIZE,
        )
    )
    print("FP32 digital accuracy: {:.2f} %\n".format(test_accuracy(model, test_data)))

    stages = [
        ("ideal cells (digital INT)", 0.0, 1e-9),
        ("+ prog spread {:g} %".format(100 * PROGRAMMING_SPREAD), PROGRAMMING_SPREAD, 1e-9),
        ("+ IR drop {:g} Ohm".format(WIRE_RESISTANCE), PROGRAMMING_SPREAD, WIRE_RESISTANCE),
    ]
    print("{:<28s} {:>24s}".format("stage", "accuracy (mean +- std) %"))
    for name, spread, wire_resistance in stages:
        accs = torch.tensor(
            [
                analog_accuracy(model, test_data, spread, wire_resistance, seed)
                for seed in range(REPEATS)
            ]
        )
        std = accs.std().item() if REPEATS > 1 else 0.0
        print("{:<28s} {:>16.2f} +- {:<5.2f}".format(name, accs.mean().item(), std))


if __name__ == "__main__":
    main()

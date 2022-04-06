# Copyright 2021 qclib project.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
import pickle as pkl
from scipy import stats
from qiskit import Aer
from qiskit.providers.aer.noise import  NoiseModel
from qiskit.test.mock import FakeJakarta, FakeMontreal, FakeMumbai
from qiskit.utils import QuantumInstance
import matplotlib.pyplot as plt

def check_create_dir(dir_path):
  if not os.path.exists(dir_path): 
      os.mkdir(dir_path)

def sample_from_distribution(dist_type, n_samples, loc, scale, random_state=7):
  """
    Samples random variables based on the chosen distribution
    Args: 
      dist_type: str The name of the distribution from which the data is to be 
                 sampled. Must be in `['lognormal', 'normal', 'laplace', 'semicircular']`
      n_samples: int The total number of random variables to be sampled
      loc: float The expected value of the distribution
      scale: float The standard deviation of the values sampled
      random_state: seed or generator instance for sampling data
  """
  assert dist_type in ['lognormal', 'normal', 'laplace', 'semicircular']

  if dist_type == 'lognormal': 
    return stats.lognorm.rvs(
                          s=1, 
                          loc=loc, 
                          scale=scale, 
                          size=(n_samples, ), 
                          random_state=random_state
                        )
  elif dist_type == 'normal':
    return stats.norm.rvs(
                          loc=loc, 
                          scale=scale, 
                          size=(n_samples, ), 
                          random_state=random_state
                        )
  elif dist_type == 'laplace': 
    return stats.laplace.rvs(
                              loc=loc, 
                              scale=scale, 
                              size=(n_samples, ), 
                              random_state=random_state
                            )
  elif dist_type == 'semicircular': 
    return stats.semicircular.rvs(
                                    loc=loc, 
                                    scale=scale, 
                                    size=(n_samples, ), 
                                    random_state=random_state
                                  )

def serialize_model(file_name, model):
  with open(file_name, 'wb') as target_file: 
    pkl.dump(model, target_file)

def load_model(file_name): 
  model = None
  with open(file_name, 'rb') as target_file: 
    model = pkl.load(target_file)
  return model

def build_quantum_instance(instance_type, seed=7, n_shots=2000, device='CPU'):
  """
    Builts the quantum instance needed for the simulation
    Args:
      instance_type: str Name of the backend type
      seed: int The value of the seed
      n_shots: int Total number of shots in the simulations
  """
  if instance_type == 'statevector_simulator':
    aer_backend = Aer.get_backend(instance_type)
    aer_backend.set_options(device=device)
    return QuantumInstance(aer_backend, 
                           seed_transpiler=seed, 
                           seed_simulator=seed, 
                           shots=n_shots)
  elif instance_type == 'qasm_simulator':
    aer_backend = Aer.get_backend(instance_type)
    aer_backend.set_options(device=device)
    return QuantumInstance(aer_backend, 
                           seed_transpiler=seed, 
                           seed_simulator=seed,
                           shots=n_shots)
  elif instance_type == 'fake_jakarta':
    qasm_simulator = Aer.get_backend('qasm_simulator')
    qasm_simulator.set_options(device=device)
    fake_device = FakeJakarta()
    noise_model = NoiseModel.from_backend(fake_device)
    basis_gates = noise_model.basis_gates
    coupling_map = fake_device.configuration().coupling_map
    return QuantumInstance(qasm_simulator, 
                           noise_model=noise_model,
                           coupling_map=coupling_map, 
                           basis_gates=basis_gates,
                           seed_simulator=seed,
                           seed_transpiler=seed,
                           shots=n_shots)
  
  elif instance_type == 'fake_montreal':
    qasm_simulator = Aer.get_backend('qasm_simulator')
    qasm_simulator.set_options(device=device)
    fake_device = FakeMontreal()
    noise_model = NoiseModel.from_backend(fake_device)
    basis_gates = noise_model.basis_gates
    coupling_map = fake_device.configuration().coupling_map
    return QuantumInstance(qasm_simulator, 
                           noise_model=noise_model,
                           coupling_map=coupling_map, 
                           basis_gates=basis_gates,
                           seed_simulator=seed,
                           seed_transpiler=seed,
                           shots=n_shots)
  
  elif instance_type == 'fake_mumbai':
    qasm_simulator = Aer.get_backend('qasm_simulator')
    qasm_simulator.set_options(device=device)
    fake_device = FakeMumbai()
    noise_model = NoiseModel.from_backend(fake_device)
    basis_gates = noise_model.basis_gates
    coupling_map = fake_device.configuration().coupling_map
    return QuantumInstance(qasm_simulator, 
                           noise_model=noise_model,
                           coupling_map=coupling_map, 
                           basis_gates=basis_gates,
                           seed_simulator=seed,
                           seed_transpiler=seed,
                           shots=n_shots)

def compute_estimated_pmf(dist_type: str, loc: float, scale: float, x_points, n_qubits: int):
  """
    Computes the estimated PMF according to the cumulative distribution
    functions of the distribution type selected

    Args: 
      dist_type: String, Selected distribution
      loc: Float, The expected value of the selected distribution
      scale: Float, The scale of the selected distribution
      x_points: Array, The collection of points on which the distirbution is 
                discretized
      n_qubits: The total number of qubits used in the quantum circuit
  """

  distance = (x_points.max() - x_points.min()) / (2**n_qubits - 1)
  delta = distance * .5

  if dist_type == 'lognormal':
    xcdf_plus = stats.lognorm.cdf(x_points + delta, s=1, scale=scale)
    xcdf_minus = stats.lognorm.cdf(x_points - delta, s=1, scale=scale)
  elif dist_type == 'laplace':
    xcdf_plus = stats.laplace.cdf(x_points + delta, loc=loc, scale=scale)
    xcdf_minus = stats.laplace.cdf(x_points - delta, loc=loc, scale=scale)
  elif dist_type == 'normal':
    xcdf_plus = stats.norm.cdf(x_points + delta, loc=loc, scale=scale)
    xcdf_minus = stats.norm.cdf(x_points - delta, loc=loc, scale=scale)
  elif dist_type == 'semicircular':
    xcdf_plus = stats.semicircular.cdf(x_points + delta, loc=loc, scale=scale)
    xcdf_minus = stats.semicircular.cdf(x_points - delta, loc=loc, scale=scale)
  else: 
    raise Exception('Unmapped distribution')

  return xcdf_plus - xcdf_minus

def extract_statevector_from_generator(quantum_instance, generator):
  """
    Initializes quantum circuit with generator's parameters in order to extract a statevector

    Args: 
      quantum_instance: QuantumInstance object containing a statevector backend
      generator: QuantumGenerator instance for generating states
    Returns: 
      The statevector generated by the generator's circuit
  """
  circ = generator.construct_circuit(generator.parameter_values)
  result = quantum_instance.execute(circ)
  return result.get_statevector(circ)

def fidelity(input_state, state_vector):
  """
    Computes the fidelity of a generated quantum state

    Args:
      input_state: list or np.ndarray, The squared version of the pmf
      state_vector: list or np.ndarray, The generated statevector
  """    
  return np.abs(np.conj(input_state).dot(state_vector))**2

def plot_probs(x_pmf, gen_probs, computed_fidelity: float, x=None, experiment_dir: str = None):
  """
    Plot the frequencies of the estimated PMF of the distribution and the probability
    distribution loaded by the GAN
    Args: 
      x_pmf: Array, the frequencies of the estimated PMF
      gen_probs: Array, the frequencies generated by  the qGAN
      qgan: QGAN object containing the quantum generator
      quantum_instance: QuantumInstance object containing a statevector backend
      x: the collection of points defined for the estimated PMF
      experiment_dir: String, the path where the plot is to be saved
  """

  ## Plotting results
  plt.rcParams["font.family"] = "Times New Roman"
  fig = plt.figure(figsize=(10, 10))

  x = list(range(len(x_pmf))) if x is None else x
  plt.plot(x, x_pmf, color='darkred', label='fidelity=1.0') # target
  plt.plot(x, gen_probs, color='blue', label=f'fidelity={round(computed_fidelity,4)}')
  plt.xticks(fontsize=24)
  plt.yticks(fontsize=24)
  plt.legend(loc='best', fontsize=24)
  if experiment_dir:
    plt.savefig(os.path.join(experiment_dir, 'frequencies.pdf'))
  plt.show()

def plot_relative_entropy(rel_entr, experiment_dir: str = None):
  """
    Plot the relative entropy (KL-divergenge) between the frequencies of the  probability 
    distribution loaded by the GAN and the target distribution of the data throughout the 
    training epochs
    Args: 
      rel_entr: Array, Computed relative entropy per-epoch
      experiment_dir: String, the path where the plot is to be saved
  """
  plt.rcParams["font.family"] = "Times New Roman"
  fig = plt.figure(figsize=(10, 10))

  x = list(range(len(rel_entr)))
  plt.plot(x, rel_entr, color='darkred', label='relative_entropy')
  plt.xticks(fontsize=24)
  plt.yticks(fontsize=24)
  plt.legend(loc='best', fontsize=24)
  if experiment_dir:
    plt.savefig(os.path.join(experiment_dir, 'relative_entropy.pdf'))
  plt.show()

def plot_gan_losses(g_loss, d_loss, experiment_dir: str = None):
  """
    Plot the generator and discriminator losses throughout the training
    epochs.
    Args: 
      g_loss: Array, Generator loss
      d_loss: Array, Discriminator loss
      experiment_dir: String, the path where the plot is to be saved
  """
  # Plotting generator and discriminator losses
  plt.rcParams["font.family"] = "Times New Roman"
  fig = plt.figure(figsize=(10, 10))

  plt.plot(g_loss, color='darkred', label='generator-loss')
  plt.plot(d_loss, color='darkblue', label='discriminator-loss')
  plt.xticks(fontsize=24)
  plt.yticks(fontsize=24)
  plt.legend(loc='best', fontsize=24)
  plt.xlabel('epoch')
  plt.ylabel('loss')
  if experiment_dir:
    plt.savefig(os.path.join(experiment_dir, 'losses.pdf'))
  plt.show()
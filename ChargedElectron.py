import numpy as np
#Created by Arctic Framework 2023, and if you want to use it for commercial use or as a corperate tool for some reason please send me an email so that I can put it on my resume!#
class QuantumSimulator:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.state = np.zeros(2 ** num_qubits, dtype=np.complex64)
        self.state[0] = 1.0  # Initializes with |0...0> which i think kinda looks like a bird:D


        self.gates = {

            "V": np.array([[1, -1j], [-1j, 1]], dtype=np.complex64) / np.sqrt(2),

            "U": np.array([[1, 0], [0, 1]], dtype=np.complex64),  # Placeholder for custom unitary gate
            "U1": np.array([[1, 0], [0, np.exp(1j * np.pi / 2)]], dtype=np.complex64),  # U1 gate
            "U2": np.array([[1, -1], [1, 1]], dtype=np.complex64) / np.sqrt(2),  # U2 gate
            "U3": np.array([[1, -1j], [1j, 1]], dtype=np.complex64) / np.sqrt(2),  # U3 gate
            "CSWAP": np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0, 0, 0],
                               [0, 0, 0, 1, 0, 0, 0, 0],
                               [0, 0, 0, 0, 1, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.complex64),  # Controlled SWAP gate


            "CCNOT": np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0, 0, 0],
                               [0, 0, 0, 1, 0, 0, 0, 0],
                               [0, 0, 0, 0, 1, 0, 0, 0],
                               [0, 0, 0, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 1],
                               [0, 0, 0, 0, 0, 0, 1, 0]], dtype=np.complex64),

            # Toffoli (CCX) gate
            "CCX": np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0, 0],
                             [0, 0, 0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 1],
                             [0, 0, 0, 0, 0, 0, 1, 0]], dtype=np.complex64),
            "I": np.array([[1, 0], [0, 1]], dtype=np.complex64),  # Identity gate
            "X": np.array([[0, 1], [1, 0]], dtype=np.complex64),  # Pauli-X gate
            "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex64),  # Pauli-Y gate
            "Z": np.array([[1, 0], [0, -1]], dtype=np.complex64),  # Pauli-Z gate
            "H": np.array([[1, 1], [1, -1]], dtype=np.complex64) / np.sqrt(2),  # Hadamard gate
            "S": np.array([[1, 0], [0, 1j]], dtype=np.complex64),  # Phase gate (S gate)
            "T": np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex64),  # T gate
            "RX": np.array([[np.cos(np.pi / 4), -1j * np.sin(np.pi / 4)],
                            [-1j * np.sin(np.pi / 4), np.cos(np.pi / 4)]], dtype=np.complex64),  # X-rotation gate
            "RY": np.array([[np.cos(np.pi / 4), -np.sin(np.pi / 4)],
                            [np.sin(np.pi / 4), np.cos(np.pi / 4)]], dtype=np.complex64),  # Y-rotation gate
            "RZ": np.array([[np.exp(-1j * np.pi / 8), 0],
                            [0, np.exp(1j * np.pi / 8)]], dtype=np.complex64),  # Z-rotation gate
            "CNOT": np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex64),  # CNOT gate
            "CPHASE": np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, np.exp(1j * np.pi / 4)]],
                               dtype=np.complex64),
            "CX": np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex64),
            # Controlled-X gate
            "CY": np.array([[1, 0, 0, 0], [0, 1j, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1j]], dtype=np.complex64),
            # Controlled-Y gate
            "CZ": np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=np.complex64),
            # Controlled-Z gate
            "SWAP": np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.complex64),  # SWAP gate
            "PHASE": np.array([[1, 0], [0, np.exp(1j * np.pi / 8)]], dtype=np.complex64),  # Phase gate
            "SQRT_X": np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=np.complex64) / 2,  # Square root of X gate
            "SQRT_Y": np.array([[1 + 1j, -1 + 1j], [1 - 1j, 1 + 1j]], dtype=np.complex64) / 2,  # Square root of Y gate
            "W": np.array([[1, 1], [-1, 1]], dtype=np.complex64) / np.sqrt(2),  # W gate
            "SW": np.array([[1, -1], [1, 1]], dtype=np.complex64) / np.sqrt(2),  # Square root of W gate
            "CS": np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1j]], dtype=np.complex64),
            # Controlled-S gate
            "CCZ": np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0, 0],
                             [0, 0, 0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, np.exp(1j * np.pi / 4)],
                             [0, 0, 0, 0, 0, 0, np.exp(1j * np.pi / 4), 0]], dtype=np.complex64),  # Controlled-CCZ gate
            "U4": np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, np.exp(1j * np.pi / 8), 0],
                            [0, 0, 0, np.exp(1j * np.pi / 4)]], dtype=np.complex64),  # U4 gate
            "U5": np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, np.exp(1j * np.pi / 16), 0],
                            [0, 0, 0, np.exp(1j * np.pi / 8)]], dtype=np.complex64),  # U5 gate
            "U6": np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, np.exp(1j * np.pi / 32), 0],
                            [0, 0, 0, np.exp(1j * np.pi / 16)]], dtype=np.complex64),  # U6 gate
            "U7": np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, np.exp(1j * np.pi / 64), 0],
                            [0, 0, 0, np.exp(1j * np.pi / 32)]], dtype=np.complex64),  # U7 gate
            "U8": np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, np.exp(1j * np.pi / 128), 0],
                            [0, 0, 0, np.exp(1j * np.pi / 64)]], dtype=np.complex64),  # U8 gate
            "U9": np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, np.exp(1j * np.pi / 256), 0],
                            [0, 0, 0, np.exp(1j * np.pi / 128)]], dtype=np.complex64),  # U9 gate
            "U10": np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, np.exp(1j * np.pi / 512), 0],
                             [0, 0, 0, np.exp(1j * np.pi / 256)]], dtype=np.complex64),  # U10 gate
            "U11": np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, np.exp(1j * np.pi / 1024), 0],
                             [0, 0, 0, np.exp(1j * np.pi / 512)]], dtype=np.complex64),  # U11 gate
            "U12": np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, np.exp(1j * np.pi / 2048), 0],
                             [0, 0, 0, np.exp(1j * np.pi / 1024)]], dtype=np.complex64),  # U12 gate
            "U13": np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, np.exp(1j * np.pi / 4096), 0],
                             [0, 0, 0, np.exp(1j * np.pi / 2048)]], dtype=np.complex64),  # U13 gate
            "U14": np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, np.exp(1j * np.pi / 8192), 0],
                             [0, 0, 0, np.exp(1j * np.pi / 4096)]], dtype=np.complex64),  # U14 gate
            "U15": np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, np.exp(1j * np.pi / 16384), 0],
                             [0, 0, 0, np.exp(1j * np.pi / 8192)]], dtype=np.complex64),  # U15 gate
            "U16": np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, np.exp(1j * np.pi / 32768), 0],
                             [0, 0, 0, np.exp(1j * np.pi / 16384)]], dtype=np.complex64),  # U16 gate
            "U17": np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, np.exp(1j * np.pi / 65536), 0],
                             [0, 0, 0, np.exp(1j * np.pi / 32768)]], dtype=np.complex64),  # U17 gate
            "U18": np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, np.exp(1j * np.pi / 131072), 0],
                             [0, 0, 0, np.exp(1j * np.pi / 65536)]], dtype=np.complex64),
            "CNOT13": np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                                [0, 1, 0, 0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0, 0, 0],
                                [0, 0, 0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 1],
                                [0, 0, 0, 0, 0, 0, 1, 0]], dtype=np.complex64),
            "CPhaseShift": np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, np.exp(1j * np.pi / 4)]], dtype=np.complex64),
            "Fredkin": np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 1, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 1, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 1, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 1, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 1, 0],
                                 [0, 0, 0, 0, 0, 1, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.complex64),
            "Toffoli2": np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 1, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 1, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 1, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 1, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 1, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 1],
                                  [0, 0, 0, 0, 0, 0, 1, 0]], dtype=np.complex64),
            "Toffoli3": np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.complex64)
        }

    def measure(self):
        normalized_state = self.state / np.linalg.norm(self.state)
        probabilities = np.abs(normalized_state) ** 2
        outcome = np.random.choice(range(2 ** self.num_qubits), p=probabilities)
        return format(outcome, f"0{self.num_qubits}b")

    def save_state(self, filename):
        np.save(filename, self.state)

    def load_state(self, filename):
        self.state = np.load(filename)

    def entangle_qubits(self, qubit_indices):
        num_qubits = len(qubit_indices)
        entanglement_gate = np.zeros((2 ** num_qubits, 2 ** num_qubits), dtype=np.complex64)
        for i in range(2 ** num_qubits):
            entanglement_gate[i, i] = 1 / np.sqrt(2 ** num_qubits)
        self.apply_gate(entanglement_gate, qubit_indices)

    def visualize_state(self):
        state_magnitudes = np.abs(self.state)
        for i, magnitude in enumerate(state_magnitudes):
            binary = format(i, f"0{self.num_qubits}b")
            print(f"|{binary}>: {magnitude:.3f}")

    def is_entangled(self, qubit1, qubit2):
        state_density_matrix = np.outer(self.state, self.state.conj())
        reduced_indices = [2 ** (self.num_qubits - qubit - 1) for qubit in range(self.num_qubits) if
                           qubit != qubit1 and qubit != qubit2]
        remaining_indices = list(set(range(2 ** self.num_qubits)) - set(reduced_indices))
        permuted_indices = reduced_indices + remaining_indices
        reduced_density_matrix = state_density_matrix[permuted_indices, :][:, permuted_indices]

        entanglement_measure = np.linalg.norm(reduced_density_matrix)
        entangled = entanglement_measure > 1e-10
        return entangled

    def add_custom_gate(self, gate_name, matrix):
        self.gates[gate_name] = matrix

    def visualize_circuit(self, circuit):
        circuit_flowchart = ""

        # Create a dictionary to keep track of qubit positions
        qubit_positions = {qubit: index for index, qubit in enumerate(range(self.num_qubits))}

        for gate in circuit:
            gate_name, target_qubits = gate[0], gate[1:]
            gate_str = f"{gate_name} --> "
            for target_qubit in target_qubits:
                gate_str += f"Q{target_qubit} "
            gate_str = gate_str.strip()
            circuit_flowchart += gate_str + "\n"

        # Print the qubit and wire layout
        qubits_str = "Qubits: "
        wires_str = "Wires: "
        for qubit in range(self.num_qubits):
            qubits_str += f"Q{qubit} "
            position = qubit_positions[qubit]
            wires_str += f"{'|' * position} "
        circuit_flowchart += qubits_str + "\n"
        circuit_flowchart += wires_str

        print(circuit_flowchart)

    def visualize_qubit_circuits(self):
        qubit_circuit_chart = ""
        for qubit in range(self.num_qubits):
            qubit_circuit_chart += f"Q{qubit}: "
            circuits_with_qubit = [circuit for circuit in self.circuits if qubit in [gate[1] for gate in circuit]]
            circuit_names = [", ".join([gate[0] for gate in circuit]) for circuit in circuits_with_qubit]
            qubit_circuit_chart += " | ".join(circuit_names) + "\n"
        print(qubit_circuit_chart)
    def apply_gate(self, gate_matrix, target_qubits):
        gate_indices = [2 ** (self.num_qubits - qubit - 1) for qubit in target_qubits]
        gate_tensor = gate_matrix
        for qubit in range(self.num_qubits):
            if 2 ** (self.num_qubits - qubit - 1) in gate_indices:
                continue
            gate_tensor = np.kron(gate_tensor, np.eye(2, dtype=np.complex64))

        gate_indices.sort()
        identity_indices = set(range(2 ** self.num_qubits)) - set(gate_indices)
        permuted_indices = gate_indices + list(identity_indices)
        self.state = np.matmul(np.transpose(gate_tensor[permuted_indices, :]), self.state)

    def apply_single_qubit_gate(self, gate_name, target_qubit):
        gate_matrix = self.gates[gate_name]
        self.apply_gate(gate_matrix, [target_qubit])

    def apply_circuit(self, circuit):
        for gate in circuit:
            gate_name, target_qubits = gate[0], gate[1:]
            gate_matrix = self.gates[gate_name]
            self.apply_gate(gate_matrix, target_qubits)

    def cphase(self, target_qubit, control_qubit):
        control_matrix = np.eye(2, dtype=np.complex64)
        control_matrix[1, 1] = -1
        control_gate = np.kron(np.eye(2 ** target_qubit), control_matrix)

        self.apply_gate(control_gate, [control_qubit, target_qubit])

    def add_wires(self, num_wires):
        self.num_qubits += num_wires
        self.state = np.zeros(2 ** self.num_qubits, dtype=np.complex64)
        self.state[0] = 1.0

# Quantum teleportation example
teleportation_circuit = [
    ("H", 0),
    ("CNOT", 0, 1),
    ("CNOT", 1, 2),
    ("H", 0),
    ("H", 1),
    ("CNOT", 1, 2),
    ("CNOT", 0, 1),
    ("H", 0),
    ("H", 1),
    ("CNOT", 1, 2)
]
cphase_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, np.exp(1j * np.pi / 4)]], dtype=np.complex64)

simulator = QuantumSimulator(3)
print("Initial state:", simulator.state)
simulator.apply_circuit(teleportation_circuit)
print("State after teleportation circuit:", simulator.state)

# Quantum algorithm example: Quantum Phase Estimation
phase_estimation_circuit = [
    ("H", 0),
    ("H", 1),
    ("CNOT", 1, 2),
    ("CPHASE", 0, 2),  # Use your actual unitary gate here
    ("CNOT", 1, 2),
    ("H", 0),
    ("H", 1)
]

if __name__ == "__main__":
    simulator = QuantumSimulator(3)

    while True:
        print("\nOptions:")
        print("1. Apply a single-qubit gate")
        print("2. Apply a circuit")
        print("3. Visualize state")
        print("4. Measure")
        print("5. Check entanglement")
        print("6. Add wires")
        print("7. Remove wires")
        print("8. Entangle qubits")
        print("9. Untangle qubits")
        print("10. Add custom gate")
        print("11. Remove custom gate")
        print("12. Apply single-qubit gate from predefined set")
        print("13. Save state")
        print("14. Load state")
        print("15. Clear circuit")
        print("0. Quit")

        choice = input("Enter your choice: ")

        if choice == "1":
            gate_name = input("Enter the gate name: ")
            target_qubit = int(input("Enter the target qubit: "))
            simulator.apply_single_qubit_gate(gate_name, target_qubit)
            print(f"Applied {gate_name} gate to Q{target_qubit}")
        elif choice == '2':
            circuit_str = input("Enter the circuit as a space-separated list (e.g., H 0 CX 0 1): ")
            circuit_list = circuit_str.split()
            if len(circuit_list) % 2 == 0:
                circuit = [(circuit_list[i], int(circuit_list[i + 1])) for i in range(0, len(circuit_list), 2)]
                simulator.apply_circuit(circuit)
                print("Applied custom circuit.")
            else:
                print("Invalid circuit format. Each gate name should be followed by a target qubit number.")
        elif choice == "3":
            print("Current state:")
            simulator.visualize_state()
        elif choice == "4":
            outcome = simulator.measure()
            print("Measurement outcome:", outcome)
        elif choice == "5":
            qubit1 = int(input("Enter the first qubit: "))
            qubit2 = int(input("Enter the second qubit: "))
            entangled = simulator.is_entangled(qubit1, qubit2)
            print(f"Are qubits Q{qubit1} and Q{qubit2} entangled?", entangled)
        elif choice == "6":
            num_wires = int(input("Enter the number of wires to add: "))
            simulator.add_wires(num_wires)
            print(f"Added {num_wires} wires. New state:")
            simulator.visualize_state()
        elif choice == "7":
            num_wires = int(input("Enter the number of wires to remove: "))
            simulator.remove_wires(num_wires)
            print(f"Removed {num_wires} wires. New state:")
            simulator.visualize_state()
        elif choice == "8":
            qubit_indices = input("Enter the qubit indices to entangle (space-separated): ").split()
            qubit_indices = [int(qubit) for qubit in qubit_indices]
            simulator.entangle_qubits(qubit_indices)
            print(f"Qubits {', '.join([f'Q{qubit}' for qubit in qubit_indices])} entangled.")
        elif choice == "9":
            qubit_indices = input("Enter the qubit indices to untangle (space-separated): ").split()
            qubit_indices = [int(qubit) for qubit in qubit_indices]
            simulator.untangle_qubits(qubit_indices)
            print(f"Qubits {', '.join([f'Q{qubit}' for qubit in qubit_indices])} untangled.")
        elif choice == "10":
            gate_name = input("Enter the custom gate name: ")
            gate_matrix = np.array([[1, 0], [0, 1]], dtype=np.complex64)  # Replace with your custom gate matrix
            simulator.add_custom_gate(gate_name, gate_matrix)
            print(f"Custom gate '{gate_name}' added.")
        elif choice == "11":
            gate_name = input("Enter the custom gate name to remove: ")
            simulator.remove_custom_gate(gate_name)
            print(f"Custom gate '{gate_name}' removed.")
        elif choice == "12":
            print("Available single-qubit gates:")
            print(", ".join(simulator.gates.keys()))
            gate_name = input("Enter the gate name from the predefined set: ")
            target_qubit = int(input("Enter the target qubit: "))
            simulator.apply_single_qubit_gate(gate_name, target_qubit)
            print(f"Applied {gate_name} gate to Q{target_qubit}")
        elif choice == "13":
            filename = input("Enter the filename to save state: ")
            simulator.save_state(filename)
            print("State saved to", filename)
        elif choice == "14":
            filename = input("Enter the filename to load state: ")
            simulator.load_state(filename)
            print("State loaded from", filename)
        elif choice == "15":
            simulator.clear_circuit()
            print("Circuit cleared.")
        elif choice == "0":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter a valid option.")


import bluequbit
from qiskit.circuit.library import quantum_volume
import time

try:
    bq = bluequbit.init("lEiTmm6zeLxxZ6q3aKBMsxwhrdnDr7vF")
    
    num_qubits = 4
    depth = 4
    qc = quantum_volume(num_qubits, depth, seed=42)
    qc.measure_all()
    
    print("Testing bq.run with shots in options...")
    try:
        options = {"mps_bond_dimension": 16, "shots": 100}
        job = bq.run(qc, device="mps.cpu", options=options)
        print(f"Job ID: {job.job_id}")
        print(f"Run time: {job.run_time_ms} ms")
        result = job.result()
        print(f"Counts: {result.get_counts()}")
        print("Success with shots in options!")
    except Exception as e:
        print(f"Failed with shots in options: {e}")

    print("\nTesting bq.run with shots as kwarg...")
    try:
        options = {"mps_bond_dimension": 16}
        job = bq.run(qc, device="mps.cpu", options=options, shots=100)
        print(f"Job ID: {job.job_id}")
        print(f"Run time: {job.run_time_ms} ms")
        result = job.result()
        print(f"Counts: {result.get_counts()}")
        print("Success with shots as kwarg!")
    except Exception as e:
        print(f"Failed with shots as kwarg: {e}")

except Exception as e:
    print(f"General failure: {e}")

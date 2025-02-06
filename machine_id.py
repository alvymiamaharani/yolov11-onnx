import machineid
import hashlib


def get_machine_id():
    machine_id = machineid.id()
    return hashlib.sha256(machine_id.encode()).hexdigest()


current_machine_id = get_machine_id()
print(current_machine_id)

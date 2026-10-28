from pyfedappwrap.engine.runtime import FedDBEngine

from app import PatientUmapExportAPP

engine = FedDBEngine()

engine.register(PatientUmapExportAPP())

if __name__ == "__main__":
    engine.start()
    engine.wait_until_stop()

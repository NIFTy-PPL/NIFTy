from mpi4py import MPI
import coverage

cov = None

def pytest_sessionstart(session):
    global cov
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0 and size > 1:
        cov = coverage.Coverage(data_file=".coverage.cl_mpi", source=["nifty.cl"])
        cov.start()

def pytest_sessionfinish(session, exitstatus):
    global cov
    if cov is not None:
        cov.stop()
        cov.save()

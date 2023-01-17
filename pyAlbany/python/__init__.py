__version__ = '1.1.0'
def version():
    return 'PyAlbany version: ' + __version__

from PyAlbany.AlbanyInterface import finalizeKokkos

class Kokkos_Finalizer():
    def __del__(self):
        finalizeKokkos()

kf = Kokkos_Finalizer()
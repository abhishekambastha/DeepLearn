import tarfile
import io
import os

def get_file_progress_file_object_class(on_progress):
    class FileProgressFileObject(tarfile.ExFileObject):
        def read(self, size, *args):
          on_progress(self.name, self.position, self.size)
          return tarfile.ExFileObject.read(self, size, *args)
    return FileProgressFileObject

class TestFileProgressFileObject(tarfile.ExFileObject):
    def read(self, size, *args):
      on_progress(self.name, self.position, self.size)
      return tarfile.ExFileObject.read(self, size, *args)

class ProgressFileObject(io.FileIO):
    def __init__(self, path, *args, **kwargs):
        self._total_size = os.path.getsize(path)
        io.FileIO.__init__(self, path, *args, **kwargs)

    def read(self, size):
        print("Overall process: %d of %d" %(self.tell(), self._total_size))
        return io.FileIO.read(self, size)

def on_progress(filename, position, total_size):
    print("%s: %d of %s" %(filename, position, total_size))

tarfile.TarFile.fileobject = get_file_progress_file_object_class(on_progress)
tar = tarfile.open(fileobj=ProgressFileObject("notMNIST_small.tar.gz"))
tar.extractall()
tar.close()

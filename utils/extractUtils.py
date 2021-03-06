import os
import io
import progressbar
import tarfile

class ProgressFileObject(io.FileIO):
    def __init__(self, path, *args, **kwargs):
        self._total_size = os.path.getsize(path)
        io.FileIO.__init__(self, path, *args, **kwargs)
        widgets = ["    ", progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
        self.pbar = progressbar.ProgressBar(widgets = widgets)

    def dlProgress(self, currSize, totalSize):
        if self.pbar.maxval is None:
            self.pbar.maxval = totalSize
            self.pbar.start()
        self.pbar.update(min(currSize, totalSize))

    def read(self, size):
        self.dlProgress(self.tell(), self._total_size)
        return io.FileIO.read(self, size)

class extract:
    def untar(self, filename):
        print('Extracting %s' % filename)
        fileobj = ProgressFileObject(filename)
        tar = tarfile.open(fileobj=fileobj)
        tar.extractall()
        tar.close()
        fileobj.pbar.finish()

    def maybe_extract(filename, force=False):
        root = os.path.splitext(os.path.splitext(filename)[0])[0]
        if os.path.isdir(root) and not force:
            print('%s already present - Skipping Extraction of %s' % (root, filename))
        else:
            self.untar(filename)
        data_folders = [os.path.join(root,d) for d in sorted(os.listdir(root))]
        print(data_folders)
        if len(data_folders) != num_classes:
            raise Exception('Expected %d folders, one per class. Found %d instead' % (num_classes, len(data_folders)))
        print(data_folders)
        return data_folders

def test():
    e = extract()
    filename = './notMNIST_large.tar.gz'
    e.untar(filename)
    print("Finished")

if __name__=='__main__':
    test()

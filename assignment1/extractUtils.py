# -*- coding: utf-8 -*- 
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

    def maybe_extract(self, filename, force=False):
        root = os.path.splitext(os.path.splitext(filename)[0])[0]
        if os.path.isdir(root) and not force:
            print('%s already present - Skipping Extraction of %s' % (root, filename))
        else:
            self.untar(filename)
        data_folders = [os.path.join(root,d) for d in sorted(os.listdir(root))]
        data_folders = [d for d in data_folders if os.path.isdir(d)]
        return data_folders

def extractTar(filename=None):
    e = extract()
    if filename is None:
    	filename = './notMNIST_large.tar.gz'
    e.untar(filename)

if __name__=='__main__':
    extractTar()


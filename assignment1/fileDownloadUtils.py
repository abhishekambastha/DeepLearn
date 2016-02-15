import os
import progressbar
from six.moves.urllib.request import urlretrieve

class download:
    def __init__(self):
        widgets = ["    ", progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.FileTransferSpeed()]
        self.pbar = progressbar.ProgressBar(widgets = widgets)

    def dlProgress(self, count, blockSize, totalSize):
        if self.pbar.maxval is None:
            self.pbar.maxval = totalSize
            self.pbar.start()

        self.pbar.update(min(count*blockSize, totalSize))

    def get(self, url, filename):
        print('Downloading: %s' % filename)
        returnFilename = urlretrieve(url, filename, reporthook=self.dlProgress)
        self.pbar.finish()
        print('\nFinished downloading %s, Total Size: %s bytes\n' % (filename, os.stat(filename).st_size))
	return returnFilename

    def maybe_download(self, url, filename, expected_bytes, force=False):
        """Download a file if not present, and make sure it's the right size."""
        if force or not os.path.exists(filename):
            self.get(url+filename, filename)
            statinfo = os.stat(filename)
            if statinfo.st_size == expected_bytes:
                print('Found and verified', filename)
            else:
                raise Exception('Failed to verify' + filename + '. Can you get to it with a browser?')
        else:
            print('File %s is already present' % filename)
        return filename


def downloadTar(filename=None):
    d = download()
    url = 'http://yaroslavvb.com/upload/notMNIST/'
    if filename is None:
        filename = 'notMNIST_small.tar.gz'
    d.get(url+ filename, filename)
    print("Finished")

if __name__=='__main__':
    downloadTar()

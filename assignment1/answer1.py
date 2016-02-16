import numpy as np
import fileDownloadUtils
import extractUtils
import matplotlib.pyplot as plt
import os
from progressbar import ProgressBar
from six.moves import cPickle as pickle
from scipy import ndimage
import hashlib as h
from sklearn import linear_model

#Download Files
url = 'http://yaroslavvb.com/upload/notMNIST/'
d = fileDownloadUtils.download()
train_filename = d.maybe_download(url, 'notMNIST_large.tar.gz', 247336696)
test_filename = d.maybe_download(url, 'notMNIST_small.tar.gz', 8458043)

#Extracting Files
num_classes = 10
e = extractUtils.extract()
train_folder = e.maybe_extract(train_filename)
test_folder = e.maybe_extract(test_filename)

print('Train Folder %s\n' % train_folder)
print('Test Folder %s\n' % test_folder)

#Problem 1: Display the images

image_size = 28
pixel_depth = 255.0

def load_letter(folder, min_num_images):
    image_files = [d for d in os.listdir(folder)]
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size), dtype=np.float32)

    image_index = 0
    print(folder)
    pbar = ProgressBar()

    for image in pbar(image_files):
        image_file = os.path.join(folder, image)
        try:
            image_data = (ndimage.imread(image_file).astype(float) - pixel_depth/2)/pixel_depth
            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s'% str(image_data.shape))
            dataset[image_index,: ,:] = image_data
            image_index += 1
        except IOError as e:
            print('Could not read:' , image_file, ':', e, 'it is ok. Skipping')

    num_images = image_index
    dataset = dataset[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Fewer images than expected %d < %d' % (num_images, min_num_images))

    print('Full dataset tensor:', dataset.shape)
    print('Mean: ', np.mean(dataset))
    print('Standard Deviation: ', np.std(dataset))
    return dataset

def maybe_pickle(data_folders, min_num_images_per_class, force=False):
    dataset_names = []
    print('Debug pickle: %s' % str(data_folders))
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            print('%s already present. Skipping pickling' % set_filename)
        else:
            print('Pickling %s' % set_filename)
            dataset = load_letter(folder, min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unbale to save data to', set_filename, ':', e)
    return dataset_names

train_datasets = maybe_pickle(train_folder, 45000)
test_datasets = maybe_pickle(test_folder, 1800)




def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
  num_classes = len(pickle_files)
  valid_dataset, valid_labels = make_arrays(valid_size, image_size)
  train_dataset, train_labels = make_arrays(train_size, image_size)
  vsize_per_class = valid_size // num_classes
  tsize_per_class = train_size // num_classes
  start_v, start_t = 0, 0
  end_v, end_t = vsize_per_class, tsize_per_class
  end_l = vsize_per_class+tsize_per_class
  for label, pickle_file in enumerate(pickle_files):
    try:
      with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)
        # let's shuffle the letters to have random validation and training set
        np.random.shuffle(letter_set)
        if valid_dataset is not None:
          valid_letter = letter_set[:vsize_per_class, :, :]
          valid_dataset[start_v:end_v, :, :] = valid_letter
          valid_labels[start_v:end_v] = label
          start_v += vsize_per_class
          end_v += vsize_per_class
        train_letter = letter_set[vsize_per_class:end_l, :, :]
        train_dataset[start_t:end_t, :, :] = train_letter
        train_labels[start_t:end_t] = label
        start_t += tsize_per_class
        end_t += tsize_per_class
    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise
  return valid_dataset, valid_labels, train_dataset, train_labels


train_size = 200000
valid_size = 10000
test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
  train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)

def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)


def get_duplicates(dataset1, dataset2):
    duplicates = 0
    dataset1_hashes = [h.md5(img_database).hexdigest() for img_database in dataset1]
    dataset2_hashes = [h.md5(img_database).hexdigest() for img_database in dataset2]
    s = set(dataset2_hashes)
    pbar = ProgressBar()
    index_list = []
    for index, h_dataset1 in enumerate(pbar(dataset1_hashes)):
        if h_dataset1 in s:
            duplicates += 1
            index_list.append(index)
    return duplicates, index_list


num_overlap_test, test_overlap = get_duplicates(test_dataset, train_dataset)
num_overlap_valid, valid_overlap = get_duplicates(valid_dataset, train_dataset)
num_overlap_cross, cross_overlap = get_duplicates(test_dataset, valid_dataset)

def sanitize_dataset(dataset, label, index_list):
    index_set = set(index_list)
    sanitized_dataset = []
    sanitized_label = []
    for index, data in enumerate(dataset):
        if index not in index_set:
            sanitized_dataset.append(data)
            sanitized_label.append(label[index])
    sanitized_dataset_array = np.ndarray((len(sanitized_dataset), dataset.shape[1], dataset.shape[2]), dtype=np.float32)
    sanitized_label_array = np.ndarray(len(sanitized_dataset), dtype=np.int32)
    sanitized_dataset_array[:, :, :] = sanitized_dataset
    sanitized_label_array[:] = sanitized_label

    return sanitized_dataset_array, sanitized_label_array


test_dataset_sanitized, test_label_sanitized = sanitize_dataset(test_dataset, test_labels, test_overlap)
valid_dataset_sanitized, valid_label_sanitized = sanitize_dataset(valid_dataset, valid_labels, valid_overlap)

#Saving the result so far ...

pickle_file = 'notMNIST.pickle'
try:
    f = open(pickle_file, 'wb')
    save = {
        'train_dataset' : train_dataset,
        'train_labels' : train_labels,
        'valid_dataset' : valid_dataset_sanitized,
        'valid_labels' : valid_label_sanitized,
        'test_dataset' : test_dataset_sanitized,
        'test_labels' : test_label_sanitized,
    }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    print('Unable to save data to %s: %s' % (pickle_file, str(e)))
    raise



#Training
print('Training ...')
logreg = linear_model.LogisticRegression(verbose=1, n_jobs=-1)
samples = train_dataset.shape[0]
t = train_dataset[:samples, :, :].reshape(samples, 28*28)
logreg.fit(t, train_labels[:samples].flatten())
print('End.')

Z = logreg.predict(test_dataset_sanitized[:,:,:].reshape(test_dataset_sanitized.shape[0], 28*28))
print('\n\n')

correct = 0
incorrect = 0
for x,y in enumerate(Z):
    if y == test_label_sanitized[x]:
        correct += 1
    else:
        incorrect += 1
print('\n\n')
print('Correct predictions', correct)
print('Incorrect predictions', incorrect)
print(correct*1.0/(correct+incorrect))

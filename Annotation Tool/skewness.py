# Import libraries
try:
    import os
    import matplotlib.pyplot as plt
except ModuleNotFoundError as err:
    print(err)

# Define the directories: Change the annotation_dir
root_dir = os.path.dirname(os.getcwd())
annotation_dir = r'Dataset\xlg31025_20200902_1800\xlg31025_20200902_1800_annots\Annotation_OD_refined'
base_dir = os.path.join(root_dir, annotation_dir)

# Dict variable to hold the classes and the number of txt files
count = {'total_instances': 0,
         'person': 0,
         'bicycle': 0,
         'motorcycle': 0,
         'car': 0,
         'train': 0,
         'truck': 0,
         'bus': 0}


# Loop through the directory to find text files and parse them to get the count
class Skewness:
    """
    A real world dataset is not necessarily diverse. It is generally skewed and it is needed
    to find out which class has more images and which have the least amount of images.

    This is important because according to the skewness, we can assign the weights to the classes
    accordingly when building a neural network.

        read_files:
            Reads the files ending with '.txt' and stores the data in a list.
                :returns -> list of data

        parse_files:
            Parses the list and extracts the instances of the classes in files.
                :param -> list of data (from read_files function).
                :returns -> A dictionary of the names of the classes and number of instances of those
                        particular classes in the data.

        visualize_histogram:
            Plot histograms of the instances of the objects in image. Represents total number of a particular
            object in the dataset.
                :param -> Dictionary returned from parse_files
                :returns -> Histogram of total instances of all the objects belonging to all classes.
    """

    def __init__(self, base_dir):
        self.base_dir = base_dir

    def read_files(self):
        read_files = []
        for file in os.listdir(base_dir):
            if file.endswith('.txt'):
                with open(os.path.join(self.base_dir, file), 'r') as f:
                    text = f.readlines()
                    read_files.append(text)
        return read_files

    def parse_files(self, files: list):
        for idx in files:
            for jdx in idx:
                label = jdx.split(',')[-1].split('\n')[0]
                if label == '1':
                    count['person'] += 1
                elif label == '2':
                    count['bicycle'] += 1
                elif label == '3':
                    count['motorcycle'] += 1
                elif label == '4':
                    count['car'] += 1
                elif label == '5':
                    count['train'] += 1
                elif label == '6':
                    count['truck'] += 1
                elif label == '7':
                    count['bus'] += 1
        count['total_instances'] = sum(list(count.values())[1:])
        return count

    # This function should only be used when one encounters a train label in histogram to find out the index and thus
    # rectifying it using the XLabelTool
    def error(self, files: list):
        error_labels = []
        for idx, value in enumerate(files):
            for jdx, val in enumerate(value):
                labels = val.split(',')[-1].split('\n')[0]
                if labels == '5':
                    error_labels.append(idx)
        return error_labels

    def visualize_histogram(self, dict_of_classes):
        # Bar plot with x axis and keys and y axis as values and set the first bar to red
        hists = plt.bar(dict_of_classes.keys(), dict_of_classes.values(), 0.4, color='g')
        hists[0].set_color('r')

        # Loop through bar chart to add text on top of the bar. text :-> total instances of the object
        for rect in hists:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2., 1 * height,
                     '%d' % int(height),
                     ha='center', va='bottom')

        # Label x and y axis.
        plt.xlabel('Classes')
        plt.ylabel('Number of instances of objects in the dataset')
        plt.tight_layout()
        plt.show()


# Pipeline for calling the class
def pipeline():
    skewness = Skewness(base_dir)
    files = skewness.read_files()
    count = skewness.parse_files(files=files)
    skewness.visualize_histogram(count)


if __name__ == '__main__':
    pipeline()


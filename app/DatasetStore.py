from Dataset import Dataset

class DatasetStore:

    def __init__(self):
        self.available = [
            {
                "name": "EMBL Raw",
                "slices": "data/EM/EMBL/raw/",
                "labels": "data/EM/EMBL/labels/"
            },
            {
                "name": "EMBL Test",
                "slices": "data/EM/EMBL/test/",
                "labels": "data/EM/EMBL/test_labels/"
            },
            {
                "name": "EMBL Validation",
                "slices": "data/EM/EMBL/val/",
                "labels": "data/EM/EMBL/val_labels/"
            },
            {
                "name": "EMBL Training",
                "slices": "data/EM/EMBL/train/",
                "labels": "data/EM/EMBL/train_labels/"
            }
        ]
        self.selected = "EMBL Raw"

    def set_selection(self, name):
        self.selected = name

    def get_names_available(self):
        return [d["name"] for d in self.available]

    def get_selected_dataset(self):
        return self.get_dataset(self.selected)

    def get_dataset(self, name):
        metadata = [d for d in self.available if d["name"] == name][0]
        return Dataset(
            slices_folder=metadata["slices"], 
            labels_folder=metadata["labels"])
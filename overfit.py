from run_trainings import train_perceiver
from mapillary_data_loader.load_mapillary import MapillaryDatasetPerceiver, make_dataloader

class OverFitDset:

    def __init__(self, dset, train):
        self._full_dset = dset
        self._l = 3200 if train else 1

    def __len__(self):
        return self._l

    def __getitem__(self, index):
        return self._full_dset[0]

def overfit_perceiver():
    train_dset = make_dataloader(OverFitDset(MapillaryDatasetPerceiver(max_size=40000, train=True), True), 16, True)
    eval_dset = make_dataloader(OverFitDset(MapillaryDatasetPerceiver(max_size=40000, train=True), False), 1, True)
    train_perceiver(train_dset, eval_dset)


#def overfit_


if __name__ == "__main__":
    overfit_perceiver()

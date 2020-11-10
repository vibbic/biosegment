import { DatasetState } from './state';
import { getStoreAccessors } from 'typesafe-vuex';
import { State } from '../state';

export const getters = {
    datasets: (state: DatasetState) => state.datasets,
    oneDataset: (state: DatasetState) => (datasetId: number) => {
        const filteredDatasets = state.datasets.filter((dataset) => dataset.id === datasetId);
        if (filteredDatasets.length > 0) {
            return { ...filteredDatasets[0] };
        }
    },
};

const { read } = getStoreAccessors<DatasetState, State>('');

export const readOneDataset = read(getters.oneDataset);
export const readDatasets = read(getters.datasets);

import { Dataset } from '@/interfaces';
import { DatasetState } from './state';
import { getStoreAccessors } from 'typesafe-vuex';
import { State } from '../state';

export const mutations = {
    setDatasets(state: DatasetState, payload: Dataset[]) {
        state.datasets = payload;
    },
    setDataset(state: DatasetState, payload: Dataset) {
        const datasets = state.datasets.filter((dataset: Dataset) => dataset.id !== payload.id);
        datasets.push(payload);
        state.datasets = datasets;
    },
    deleteDataset(state: DatasetState, payload: Dataset) {
        const datasets = state.datasets.filter((dataset: Dataset) => dataset.id !== payload.id);
        state.datasets = datasets;
    },
};

const { commit } = getStoreAccessors<DatasetState, State>('');

export const commitSetDataset = commit(mutations.setDataset);
export const commitSetDatasets = commit(mutations.setDatasets);
export const commitDeleteDataset = commit(mutations.deleteDataset);

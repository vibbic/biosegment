import { Model } from '@/api';
import { ModelState } from './state';
import { getStoreAccessors } from 'typesafe-vuex';
import { State } from '../state';

export const mutations = {
    setModels(state: ModelState, payload: Model[]) {
        state.models = payload;
    },
    setModel(state: ModelState, payload: Model) {
        const models = state.models.filter((model: Model) => model.id !== payload.id);
        models.push(payload);
        state.models = models;
    },
    deleteModel(state: ModelState, payload: Model) {
        const models = state.models.filter((model: Model) => model.id !== payload.id);
        state.models = models;
    },
};

const { commit } = getStoreAccessors<ModelState, State>('');

export const commitSetModel = commit(mutations.setModel);
export const commitSetModels = commit(mutations.setModels);
export const commitDeleteModel = commit(mutations.deleteModel);
